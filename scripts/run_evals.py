"""Run the eval checks, one process each, and print one summary.

    python scripts/run_evals.py                          # every check that exists
    python scripts/run_evals.py --only routing           # just these
    python scripts/run_evals.py --only routing,memory,persona   # Feature 2's Done-when

Each check (scripts/check_<name>.py) runs as its own subprocess with `--json`, so a
crash in one cannot take down the others and no state leaks between them. The runner
reads each record back through the contract (scripts/eval_contract.py) and never
parses printed text; the checks' own printouts stream to the console as the detail,
and the table at the end is the summary.

Exit code, same precedence as a single check:

    1  any check regressed
    2  otherwise, any check could not run, any check named in --only is not built,
       or no check ran at all (an empty run is never a pass)
    0  otherwise

Order matters on a 12 GB card: Ollama keeps a model loaded for minutes after use, so a
check inherits whatever ran before it (mistral was resident during routing only because
the ingestion tests ran first). The runner does NOT clean that up — it records what was
loaded at each check's START, so the leftovers are visible rather than sanitised away.

One line per run is appended to eval_logs/runs.jsonl (timestamp, commit, dirty flag,
and each check's full record), so runs compare across commits.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.eval_contract import (BOUNDARY, PASS, REGRESSION_VERDICTS,  # noqa: E402
                                   CheckRecord, record_passed)
from scripts.eval_metrics import gpu_used_mib, ollama_vram  # noqa: E402

#: every check the runner knows. A name whose script does not exist yet is NOT-BUILT,
#: so Steps 4-5 only have to add the file.
REGISTRY = {
    "routing": "scripts/check_routing.py",
    "memory": "scripts/check_memory.py",
    "persona": "scripts/check_persona.py",
}

DEFAULT_LOG = "eval_logs/runs.jsonl"
DEFAULT_TIMEOUT_S = 1200

ST_PASS = "PASS"
ST_REGRESSION = "REGRESSION"
ST_COULD_NOT_RUN = "COULD-NOT-RUN"
ST_NOT_BUILT = "NOT-BUILT"

DISAGREE = "exit code and record disagree"


@dataclass
class CheckOutcome:
    check: str
    status: str
    exit_code: Optional[int] = None
    reason: Optional[str] = None
    #: set when the exit code and the record tell different stories
    note: Optional[str] = None
    record: Optional[CheckRecord] = None
    start_vram_models: Optional[dict] = None
    start_gpu_mib: Optional[int] = None


# ==========================================================================
# pure logic — no processes, no files; this is what CI pins
# ==========================================================================

def classify(exit_code: Optional[int], record: Optional[CheckRecord],
             failure: Optional[str] = None) -> tuple:
    """(status, reason, note) for one check that was launched.

    `failure` is set by the launcher when the process never finished normally (timed
    out, could not start) or its record could not be read. The stricter reading always
    wins: a disagreement between exit code and record is flagged, never trusted.
    """
    if failure:
        return ST_COULD_NOT_RUN, failure, None
    if exit_code not in (0, 1, 2):
        return ST_COULD_NOT_RUN, f"unexpected exit code {exit_code}", None
    if record is None:
        #: a crash exits 1 like a regression would — but it skips writing the record,
        #: and that is what tells them apart
        return ST_COULD_NOT_RUN, f"no record written (exit {exit_code})", None
    if record.error is not None or exit_code == 2:
        note = None if (record.error is not None and exit_code == 2) else DISAGREE
        return ST_COULD_NOT_RUN, record.error or "exit 2", note
    if not record.cases:
        return ST_COULD_NOT_RUN, "record has no cases", (DISAGREE if exit_code == 0 else None)
    passed = record_passed(record)
    if exit_code == 1 or not passed:
        failing = [c.id for c in record.cases if c.verdict in REGRESSION_VERDICTS]
        reason = ("FAIL: " + ", ".join(failing)) if failing else f"exit {exit_code}"
        note = DISAGREE if (exit_code == 0 or passed) else None
        return ST_REGRESSION, reason, note
    return ST_PASS, None, None


def overall_exit(outcomes: list, requested: Optional[set] = None) -> int:
    """1 regression > 2 could-not-run > 0. `requested` = names given to --only."""
    requested = requested or set()
    if any(o.status == ST_REGRESSION for o in outcomes):
        return 1
    if any(o.status == ST_COULD_NOT_RUN for o in outcomes):
        return 2
    if any(o.status == ST_NOT_BUILT and o.check in requested for o in outcomes):
        return 2
    if not any(o.status == ST_PASS for o in outcomes):
        return 2
    return 0


def _short(name: str) -> str:
    return name[:-len(":latest")] if name.endswith(":latest") else name


def _models(models: Optional[dict]) -> str:
    if models is None:
        return "?"
    if not models:
        return "(none)"
    return ", ".join(f"{_short(n)} {mib}" for n, mib in sorted(models.items()))


def _median(values) -> Optional[float]:
    present = [v for v in values if v is not None]
    return statistics.median(present) if present else None


def summarize(outcome: CheckOutcome) -> dict:
    """The numbers one table row shows. Nulls are skipped, never averaged in as 0."""
    cases = outcome.record.cases if outcome.record else []
    walls = [c.wall_seconds for c in cases]
    seen = {}
    for case in cases:
        for name, mib in (case.vram_models or {}).items():
            seen[name] = max(mib, seen.get(name, 0))
    gpu = [c.vram_gpu_mib for c in cases if c.vram_gpu_mib is not None]
    return {
        "passed": sum(c.verdict == PASS for c in cases),
        "boundary": sum(c.verdict == BOUNDARY for c in cases),
        "total": len(cases),
        "wall_median": _median(walls),
        "wall_max": max(walls) if walls else None,
        "model_median": _median(c.model_seconds for c in cases),
        "peak_gpu_mib": max(gpu) if gpu else None,
        #: None when no case could read /api/ps at all, {} when read fine but empty
        "models_during": seen if any(c.vram_models is not None for c in cases) else None,
    }


def _secs(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:.1f}"


ROW = "{:<9}{:<15}{:<12}{:<15}{:<11}{:<11}{:<24}{}"


def format_table(outcomes: list, commit: Optional[str], dirty: Optional[bool],
                 timestamp: str, exit_code: int) -> str:
    head = f"commit {commit[:7] if commit else 'unknown'}" + (" (dirty)" if dirty else "")
    lines = ["=" * 72, f"EVAL SUMMARY   {head}   {timestamp}", "=" * 72,
             ROW.format("check", "status", "cases", "wall med/max", "model med",
                        "peak GPU", "loaded at start", "loaded during")]
    for o in outcomes:
        if o.record is None:
            #: nothing measured — say why instead of printing a row of dashes
            lines.append(f"{o.check:<9}{o.status:<15}{o.reason or ''}")
            continue
        s = summarize(o)
        cases = f"{s['passed']}/{s['total']}" + (f" +{s['boundary']}bdy" if s["boundary"] else "")
        lines.append(ROW.format(
            o.check, o.status, cases,
            f"{_secs(s['wall_median'])} / {_secs(s['wall_max'])}s",
            f"{_secs(s['model_median'])}s",
            "-" if s["peak_gpu_mib"] is None else f"{s['peak_gpu_mib']} MiB",
            _models(o.start_vram_models), _models(s["models_during"])))
        if o.reason:
            lines.append(f"{'':<9}-> {o.reason}")
    for o in outcomes:
        if o.note:
            lines.append(f"{'':<9}!! {o.check}: {o.note}")
    regressions = sum(o.status == ST_REGRESSION for o in outcomes)
    cannot = sum(o.status == ST_COULD_NOT_RUN for o in outcomes)
    missing = sum(o.status == ST_NOT_BUILT for o in outcomes)
    #: NOT-BUILT is counted so an exit 2 from a named-but-missing check explains itself
    lines.append(f"RESULT   {regressions} regressed, {cannot} could not run, "
                 f"{missing} not built -> exit {exit_code}")
    lines.append("=" * 72)
    return "\n".join(lines)


def build_log_line(outcomes: list, commit: Optional[str], dirty: Optional[bool],
                   timestamp: str, only: list, exit_code: int) -> dict:
    return {
        "timestamp": timestamp,
        "commit": commit,
        "dirty": dirty,
        "only": only,
        "exit_code": exit_code,
        "checks": [{
            "check": o.check,
            "status": o.status,
            "exit_code": o.exit_code,
            "reason": o.reason,
            "note": o.note,
            "start_vram_models": o.start_vram_models,
            "start_gpu_mib": o.start_gpu_mib,
            "record": o.record.to_dict() if o.record else None,
        } for o in outcomes],
    }


# ==========================================================================
# I/O — launching checks, git, the log
# ==========================================================================

def run_check(name: str, script, timeout_s: float = DEFAULT_TIMEOUT_S,
              workdir: Optional[str] = None) -> CheckOutcome:
    """Launch one check as its own process and turn what it left behind into an outcome."""
    script_path = Path(script) if Path(script).is_absolute() else REPO / script
    if not script_path.exists():
        return CheckOutcome(check=name, status=ST_NOT_BUILT,
                            reason=f"{Path(script).as_posix()} not found")

    #: BEFORE the check starts: what an earlier check (or test run) left loaded
    start_models, start_gpu = ollama_vram(), gpu_used_mib()

    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    with tempfile.TemporaryDirectory(dir=workdir) as tmp:
        record_path = Path(tmp) / f"{name}.json"
        exit_code, failure, record = None, None, None
        try:
            #: output is NOT captured: the check's printout is the detail and streams live
            done = subprocess.run([sys.executable, str(script_path), "--json", str(record_path)],
                                  cwd=str(REPO), env=env, timeout=timeout_s, check=False)
            exit_code = done.returncode
        except subprocess.TimeoutExpired:
            failure = f"timed out after {timeout_s:g}s"
        except Exception as exc:
            failure = f"could not start: {type(exc).__name__}: {exc}"

        if failure is None and record_path.exists():
            try:
                record = CheckRecord.from_dict(
                    json.loads(record_path.read_text(encoding="utf-8")))
            except Exception as exc:
                failure = f"unreadable record (exit {exit_code}): {type(exc).__name__}: {exc}"

    status, reason, note = classify(exit_code, record, failure)
    return CheckOutcome(check=name, status=status, exit_code=exit_code, reason=reason,
                        note=note, record=record, start_vram_models=start_models,
                        start_gpu_mib=start_gpu)


def git_state() -> tuple:
    """(commit sha, dirty) — (None, None) when git is unavailable."""
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO), capture_output=True,
                             text=True, timeout=10, check=True).stdout.strip()
        porcelain = subprocess.run(["git", "status", "--porcelain"], cwd=str(REPO),
                                   capture_output=True, text=True, timeout=10,
                                   check=True).stdout
        return sha or None, bool(porcelain.strip())
    except Exception:
        return None, None


def append_log(line: dict, path) -> bool:
    """Never raises: losing a log line must not change the run's exit code."""
    try:
        target = Path(path) if Path(path).is_absolute() else REPO / path
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(line, ensure_ascii=False) + "\n")
        return True
    except Exception as exc:
        print(f"warning: could not append to {path}: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return False


def parse_only(value: Optional[str], registry: dict) -> tuple:
    """(names to run, names explicitly requested, unknown names)."""
    if value is None:
        return list(registry), set(), []
    names = list(dict.fromkeys(n.strip() for n in value.split(",") if n.strip()))
    unknown = [n for n in names if n not in registry]
    return names, set(names), unknown


def main(argv=None, registry: Optional[dict] = None) -> int:
    registry = registry if registry is not None else REGISTRY
    parser = argparse.ArgumentParser(prog="scripts/run_evals.py",
                                     description="Run the eval checks and summarise them.")
    parser.add_argument("--only", default=None,
                        help=f"comma-separated checks to run (known: {', '.join(registry)})")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S,
                        help=f"seconds per check before it is killed (default {DEFAULT_TIMEOUT_S})")
    parser.add_argument("--log", default=DEFAULT_LOG, help=f"JSONL run log (default {DEFAULT_LOG})")
    args = parser.parse_args(argv)

    names, requested, unknown = parse_only(args.only, registry)
    if unknown or not names:
        print(f"unknown check(s): {', '.join(unknown) or '(none given)'}; "
              f"known: {', '.join(registry)}", file=sys.stderr)
        return 2

    outcomes = []
    for name in names:
        print("#" * 72)
        print(f"# CHECK {name}")
        print("#" * 72, flush=True)
        outcomes.append(run_check(name, registry[name], args.timeout))

    code = overall_exit(outcomes, requested)
    commit, dirty = git_state()
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print()
    print(format_table(outcomes, commit, dirty, timestamp, code))
    append_log(build_log_line(outcomes, commit, dirty, timestamp, names, code), args.log)
    return code


if __name__ == "__main__":
    sys.exit(main())
