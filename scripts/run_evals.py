"""Run the eval checks, one process each, and print one summary.

    python scripts/run_evals.py                          # every check that exists
    python scripts/run_evals.py --only routing           # just these
    python scripts/run_evals.py --only routing,memory,persona   # Feature 2's Done-when
    python scripts/run_evals.py --model qwen3.5:9b       # Feature 3 bake-off: another model
    python scripts/run_evals.py --model qwen3.5:9b --num-ctx 8192   # ...at a pinned context

`--model` is passed to every check as its own `--model`, which replaces both the
generation model and the memory model. Without it the checks run voice_main.VOICE_MODEL,
exactly as before. The table header names the model the checks actually reported, so a
pasted table cannot be mistaken for a different model's run. `--num-ctx` works the same
way (default: the live voice_main.NUM_CTX) and the header reads `model X @ ctx N`: one
model at 8192 and at 32768 is otherwise two indistinguishable tables.

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

`--clean` (Feature 3) is the opt-in opposite, for a measured run: unload every model and
wait until /api/ps is empty, take the IDLE GPU reading, then warm the candidate once at
the checks' own num_ctx so no case pays a cold load or a reload.

`--with-voice` (implies --clean) is the "fits with voice" test. It starts
scripts/voice_ballast.py, which holds voice_main's real Whisper, Kokoro and Silero in VRAM
and does one round trip; the LLM is warmed AFTER it, last, as voice_main.boot() does, so
Ollama loads against what voice really leaves free. After the checks the ballast does a
second round trip, and a FIT verdict is printed and logged. FIT FAIL is LOUD but does not
change the exit code: a model that does not fit is not a code regression.

One line per run is appended to eval_logs/runs.jsonl (timestamp, commit, dirty flag,
and each check's full record), so runs compare across commits.
"""
import argparse
import json
import os
import statistics
import subprocess
import queue
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.eval_contract import (BOUNDARY, PASS, REGRESSION_VERDICTS,  # noqa: E402
                                   CheckRecord, record_passed)
from scripts.eval_metrics import (gpu_total_mib, gpu_used_mib, offload_mib,  # noqa: E402
                                  ollama_models, ollama_vram)
import scripts.ollama_control as ollama_control  # noqa: E402

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

#: --with-voice: the FIT verdict's three gates (docs/feature3_model_bakeoff.md).
#: Headroom is what the desktop's own VRAM use may grow by mid-session (it moved between
#: 1.6 and 2.8 GB on 24-25 Nov) without pushing a model that "fit" off the card.
FIT_HEADROOM_MIB = 1024
#: the second voice round trip (after the checks) may be at most this much slower than
#: the first. A JUDGEMENT, labelled as one: it exists to catch Windows quietly paging GPU
#: memory into system RAM, which slows everything by multiples rather than percent.
SLOWDOWN_LIMIT = 1.5

BALLAST_SCRIPT = "scripts/voice_ballast.py"
#: loading Whisper + Kokoro cold, plus the first round trip, took ~20-30 s here
BALLAST_READY_TIMEOUT_S = 300
BALLAST_PASS_TIMEOUT_S = 120


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


@dataclass
class RunInfo:
    """What one run_evals invocation measured OUTSIDE the checks. All None on a plain run."""
    model: Optional[str] = None
    num_ctx: Optional[int] = None
    clean: bool = False
    with_voice: bool = False
    #: idle = after unload_all, before anything of ours loads; voice = ballast ready
    #: (after its first round trip); post = after the checks and the second round trip
    idle_gpu_mib: Optional[int] = None
    voice_gpu_mib: Optional[int] = None
    post_gpu_mib: Optional[int] = None
    total_gpu_mib: Optional[int] = None
    #: unload_all's result: {} clean, {...} left over, None unreadable
    leftover: Optional[dict] = None
    warm: Optional[dict] = None
    ballast_stages: Optional[list] = None
    pre_pass: Optional[dict] = None
    post_pass: Optional[dict] = None
    #: /api/ps after the checks, so an offload that began late is still seen
    final_models: Optional[dict] = None
    fit: Optional[dict] = None
    #: set when the run could not be measured at all (dirty start, ballast never ready)
    error: Optional[str] = None
    notes: list = field(default_factory=list)


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


def _models(models: Optional[dict], offload: Optional[dict] = None) -> str:
    """`llama3.1 5987`, or `llama3.1 10093 (+671 CPU)` when part of it spilled."""
    if models is None:
        return "?"
    if not models:
        return "(none)"
    spill = offload or {}
    return ", ".join(f"{_short(n)} {mib}" + (f" (+{spill[n]} CPU)" if spill.get(n) else "")
                     for n, mib in sorted(models.items()))


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
    #: worst CPU spill per model over the check's cases. The FIT gate wants this empty.
    offload = {}
    for case in cases:
        for name, mib in offload_mib(case.model_size_mib, case.vram_models).items():
            offload[name] = max(mib, offload.get(name, 0))
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
        "offload": offload,
    }


def _secs(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:.1f}"


ROW = "{:<9}{:<15}{:<12}{:<15}{:<11}{:<11}{:<24}{}"


def reported(outcomes: list, field: str) -> list:
    """The distinct values of one record field (`model`, `num_ctx`) the checks' own
    records report, in check order. None (a v1-v3 record has no num_ctx) is skipped."""
    return list(dict.fromkeys(getattr(o.record, field) for o in outcomes
                              if o.record is not None and getattr(o.record, field) is not None))


def reported_models(outcomes: list) -> list:
    return reported(outcomes, "model")


def _field_note(outcomes: list, field: str, flag: str, asked, plural: str) -> Optional[str]:
    """Two ways a bake-off table lies about a setting: the checks disagree with each
    other, or a check ran something other than the flag it was given."""
    seen = reported(outcomes, field)
    if len(seen) > 1:
        return f"checks ran different {plural}: {', '.join(map(str, seen))}"
    if asked is not None and seen and seen[0] != asked:
        return f"asked for {flag} {asked} but the checks ran {seen[0]}"
    return None


def model_note(outcomes: list, model: Optional[str]) -> Optional[str]:
    """Why the header's model can't be trusted, or None when it can."""
    return _field_note(outcomes, "model", "--model", model, "models")


def ctx_note(outcomes: list, num_ctx: Optional[int]) -> Optional[str]:
    """Why the header's context can't be trusted, or None when it can."""
    return _field_note(outcomes, "num_ctx", "--num-ctx", num_ctx, "num_ctx values")


def run_peak_gpu(outcomes: list, run: Optional["RunInfo"]) -> Optional[int]:
    """Highest whole-GPU reading of the run: every case, the ballast, the final read."""
    readings = [c.vram_gpu_mib for o in outcomes if o.record for c in o.record.cases]
    if run is not None:
        readings += [run.voice_gpu_mib, run.post_gpu_mib]
        readings += [(p or {}).get("gpu_mib") for p in (run.pre_pass, run.post_pass)]
    present = [r for r in readings if r is not None]
    return max(present) if present else None


def fit_verdict(outcomes: list, run: "RunInfo", headroom_mib: int = FIT_HEADROOM_MIB,
                slowdown_limit: float = SLOWDOWN_LIMIT) -> dict:
    """Does the model fit WITH voice loaded? Pure. Three gates, all must hold:

    1. nothing on the CPU: every model, in every case and in the final /api/ps read,
       has its whole size on the GPU (size == size_vram);
    2. headroom: the run's peak GPU use stays FIT_HEADROOM_MIB under the card's total;
    3. voice still works and is not slowed: the second round trip succeeds and takes at
       most `slowdown_limit` times the first.

    Missing evidence is a FAIL, never a pass: a gate that could not be read cannot vouch.
    """
    reasons = []
    offload = {}
    for o in outcomes:
        if o.record is None:
            continue
        for name, mib in summarize(o)["offload"].items():
            offload[name] = max(mib, offload.get(name, 0))
    final = run.final_models or {}
    for name, mib in offload_mib({n: m["size"] for n, m in final.items()},
                                 {n: m["vram"] for n, m in final.items()}).items():
        offload[name] = max(mib, offload.get(name, 0))
    seen_any = run.final_models is not None or any(
        c.model_size_mib is not None for o in outcomes if o.record for c in o.record.cases)
    if offload:
        reasons.append("on the CPU: " + ", ".join(f"{_short(n)} {mib} MiB"
                                                  for n, mib in sorted(offload.items())))
    elif not seen_any:
        reasons.append("no /api/ps size reading, so CPU offload cannot be ruled out")

    peak, total = run_peak_gpu(outcomes, run), run.total_gpu_mib
    if peak is None or total is None:
        reasons.append("no GPU reading (nvidia-smi), so headroom is unknown")
    elif peak > total - headroom_mib:
        reasons.append(f"peak {peak} MiB leaves {total - peak} MiB of {total}; "
                       f"needs {headroom_mib} MiB headroom")

    pre, post = run.pre_pass or {}, run.post_pass or {}
    slowdown = None
    if not pre.get("ok"):
        reasons.append(f"first voice round trip failed: {pre.get('error') or 'no result'}")
    if not post.get("ok"):
        reasons.append("voice round trip after the checks failed: "
                       f"{post.get('error') or 'no result'}")
    elif pre.get("ok") and pre.get("round_trip_ms") and post.get("round_trip_ms") is not None:
        slowdown = post["round_trip_ms"] / pre["round_trip_ms"]
        if slowdown > slowdown_limit:
            reasons.append(f"voice round trip {slowdown:.2f}x slower after the checks "
                           f"({pre['round_trip_ms']:.0f} -> {post['round_trip_ms']:.0f} ms; "
                           f"limit {slowdown_limit}x)")
    return {"passed": not reasons, "reasons": reasons, "offload": offload,
            "peak_gpu_mib": peak, "total_gpu_mib": total, "headroom_mib": headroom_mib,
            "slowdown": slowdown, "slowdown_limit": slowdown_limit}


def _vram_line(outcomes: list, run: "RunInfo") -> Optional[str]:
    parts = []
    if run.idle_gpu_mib is not None:
        parts.append(f"idle {run.idle_gpu_mib}")
    if run.voice_gpu_mib is not None:
        parts.append(f"+voice {run.voice_gpu_mib}")
    peak = run_peak_gpu(outcomes, run)
    if peak is not None and (run.clean or run.with_voice):
        parts.append(f"peak {peak}" + (f" / {run.total_gpu_mib}" if run.total_gpu_mib else ""))
    return ("VRAM     " + " | ".join(parts) + " MiB") if parts else None


def format_table(outcomes: list, commit: Optional[str], dirty: Optional[bool],
                 timestamp: str, exit_code: int, model: Optional[str] = None,
                 num_ctx: Optional[int] = None, run: Optional["RunInfo"] = None) -> str:
    head = f"commit {commit[:7] if commit else 'unknown'}" + (" (dirty)" if dirty else "")
    #: what the records say ran; the requested model only when no record came back
    head += f"   model {', '.join(reported_models(outcomes)) or model or 'unknown'}"
    ctx = ", ".join(map(str, reported(outcomes, "num_ctx"))) or num_ctx or "unknown"
    head += f" @ ctx {ctx}"
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
            _models(o.start_vram_models), _models(s["models_during"], s["offload"])))
        if o.reason:
            lines.append(f"{'':<9}-> {o.reason}")
    for o in outcomes:
        if o.note:
            lines.append(f"{'':<9}!! {o.check}: {o.note}")
    for label, note in (("model", model_note(outcomes, model)),
                        ("ctx", ctx_note(outcomes, num_ctx))):
        if note:
            lines.append(f"{'':<9}!! {label}: {note}")
    for o in outcomes:
        spill = summarize(o)["offload"] if o.record else {}
        if spill:
            lines.append(f"{'':<9}!! offload: {o.check}: " + ", ".join(
                f"{_short(n)} {mib} MiB on the CPU" for n, mib in sorted(spill.items())))
    run = run or RunInfo()
    for note in run.notes:
        lines.append(f"{'':<9}!! {note}")
    if run.error:
        lines.append(f"{'':<9}!! run: {run.error}")
    vram = _vram_line(outcomes, run)
    if vram:
        lines.append(vram)
    if run.fit is not None:
        if run.fit["passed"]:
            lines.append(f"FIT      PASS - all on GPU, peak {run.fit['peak_gpu_mib']} of "
                         f"{run.fit['total_gpu_mib']} MiB, voice "
                         + (f"{run.fit['slowdown']:.2f}x" if run.fit["slowdown"] else "ok"))
        else:
            #: loud on purpose: the exit code stays clean, so this line is the alarm
            lines.append("FIT      *** FAIL *** this model does NOT fit with voice loaded")
            for reason in run.fit["reasons"]:
                lines.append(f"{'':<9}- {reason}")
    regressions = sum(o.status == ST_REGRESSION for o in outcomes)
    cannot = sum(o.status == ST_COULD_NOT_RUN for o in outcomes)
    missing = sum(o.status == ST_NOT_BUILT for o in outcomes)
    #: NOT-BUILT is counted so an exit 2 from a named-but-missing check explains itself
    lines.append(f"RESULT   {regressions} regressed, {cannot} could not run, "
                 f"{missing} not built -> exit {exit_code}"
                 + ("   [FIT FAIL]" if run.fit is not None and not run.fit["passed"] else ""))
    lines.append("=" * 72)
    return "\n".join(lines)


def build_log_line(outcomes: list, commit: Optional[str], dirty: Optional[bool],
                   timestamp: str, only: list, exit_code: int,
                   run: Optional["RunInfo"] = None) -> dict:
    run = run or RunInfo()
    return {
        "timestamp": timestamp,
        "commit": commit,
        "dirty": dirty,
        "only": only,
        "exit_code": exit_code,
        #: always present (null / false on a plain run), so every line has one shape
        "settings": {"model": run.model, "num_ctx": run.num_ctx, "clean": run.clean,
                     "with_voice": run.with_voice},
        "vram": {"idle_gpu_mib": run.idle_gpu_mib, "voice_gpu_mib": run.voice_gpu_mib,
                 "post_gpu_mib": run.post_gpu_mib, "total_gpu_mib": run.total_gpu_mib,
                 "peak_gpu_mib": run_peak_gpu(outcomes, run),
                 "final_models": run.final_models},
        "clean": None if not run.clean else {"leftover": run.leftover, "warm": run.warm},
        "voice": None if not run.with_voice else {"stages": run.ballast_stages,
                                                  "pre_pass": run.pre_pass,
                                                  "post_pass": run.post_pass},
        #: the verdict itself, so a FIT FAIL is in the record and not only on screen
        "fit": run.fit,
        "error": run.error,
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

def check_command(script_path, record_path, model: Optional[str] = None,
                  num_ctx: Optional[int] = None) -> list:
    """The argv for one check. `--model` / `--num-ctx` are added only when asked for, so
    a plain run launches exactly the command it always did."""
    cmd = [sys.executable, str(script_path), "--json", str(record_path)]
    if model:
        cmd += ["--model", model]
    if num_ctx is not None:
        cmd += ["--num-ctx", str(num_ctx)]
    return cmd


def run_check(name: str, script, timeout_s: float = DEFAULT_TIMEOUT_S,
              workdir: Optional[str] = None, model: Optional[str] = None,
              num_ctx: Optional[int] = None) -> CheckOutcome:
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
            done = subprocess.run(check_command(script_path, record_path, model, num_ctx),
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


class BallastError(RuntimeError):
    """The voice ballast could not be started, or stopped answering."""


class VoiceBallast:
    """Runs scripts/voice_ballast.py --serve as a child and speaks its line protocol.

    Its stderr is inherited, so its loading lines stream to the console like a check's
    printout; only `BALLAST `-prefixed stdout lines are messages, read on a thread so
    every wait has a timeout. `stop()` is safe to call more than once and always leaves
    no child behind: quit, then terminate, then kill.
    """

    def __init__(self, cmd: Optional[list] = None,
                 ready_timeout_s: float = BALLAST_READY_TIMEOUT_S,
                 pass_timeout_s: float = BALLAST_PASS_TIMEOUT_S):
        self.cmd = cmd or [sys.executable, str(REPO / BALLAST_SCRIPT), "--serve"]
        self.ready_timeout_s = ready_timeout_s
        self.pass_timeout_s = pass_timeout_s
        self.process = None
        self._messages: queue.Queue = queue.Queue()

    def start(self) -> dict:
        """Launch and wait for `ready`. Returns the ready message; raises BallastError."""
        from scripts.voice_ballast import parse_message

        env = dict(os.environ)
        env.setdefault("PYTHONIOENCODING", "utf-8")
        try:
            self.process = subprocess.Popen(self.cmd, cwd=str(REPO), env=env, text=True,
                                            encoding="utf-8", stdin=subprocess.PIPE,
                                            stdout=subprocess.PIPE, bufsize=1)
        except Exception as exc:
            raise BallastError(f"could not start: {type(exc).__name__}: {exc}") from exc

        def pump(stream):
            for line in stream:
                message = parse_message(line.rstrip("\n"))
                if message is not None:
                    self._messages.put(message)
            self._messages.put(None)            # EOF: the child is gone

        threading.Thread(target=pump, args=(self.process.stdout,), daemon=True).start()
        return self._expect("ready", self.ready_timeout_s)

    def run_pass(self) -> dict:
        """One round trip. Never raises: a dead or silent ballast is a failed pass."""
        try:
            self.process.stdin.write("pass\n")
            self.process.stdin.flush()
            return self._expect("pass", self.pass_timeout_s)
        except Exception as exc:
            return {"ok": False, "error": f"ballast: {exc}"}

    def _expect(self, event: str, timeout_s: float) -> dict:
        try:
            message = self._messages.get(timeout=timeout_s)
        except queue.Empty:
            raise BallastError(f"no {event!r} within {timeout_s:g}s") from None
        if message is None:
            raise BallastError(f"exited before {event!r} (exit {self.process.poll()})")
        if message.get("event") == "error":
            raise BallastError(message.get("error") or "reported an error")
        if message.get("event") != event:
            raise BallastError(f"expected {event!r}, got {message.get('event')!r}")
        return message

    def stop(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        try:
            self.process.stdin.write("quit\n")
            self.process.stdin.flush()
            self.process.stdin.close()
            self.process.wait(timeout=15)
        except Exception:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except Exception:
                self.process.kill()


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
    #: a flag, never an env var: the model belongs in the command that made the numbers
    parser.add_argument("--model", default=None,
                        help="Ollama model every check runs, for generation AND memory "
                             "(default: each check's own, voice_main.VOICE_MODEL)")
    parser.add_argument("--num-ctx", type=int, default=None,
                        help="context window every check sends (default: each check's own, "
                             "voice_main.NUM_CTX). The bake-off pins this for every model.")
    parser.add_argument("--clean", action="store_true",
                        help="unload every model and wait for /api/ps to empty, take the idle "
                             "GPU reading, then warm the model at the checks' num_ctx")
    parser.add_argument("--with-voice", action="store_true",
                        help="hold the real voice models (scripts/voice_ballast.py) during the "
                             "checks and print a FIT verdict. Implies --clean.")
    parser.add_argument("--slowdown-limit", type=float, default=SLOWDOWN_LIMIT,
                        help="FIT: how much slower the voice round trip may get after the "
                             f"checks (default {SLOWDOWN_LIMIT}x, a judgement)")
    args = parser.parse_args(argv)

    names, requested, unknown = parse_only(args.only, registry)
    if unknown or not names:
        print(f"unknown check(s): {', '.join(unknown) or '(none given)'}; "
              f"known: {', '.join(registry)}", file=sys.stderr)
        return 2

    run = RunInfo(model=args.model, num_ctx=args.num_ctx,
                  clean=args.clean or args.with_voice, with_voice=args.with_voice)
    outcomes = []
    ballast = VoiceBallast() if run.with_voice else None
    try:
        if prepare(run, ballast):
            for name in names:
                print("#" * 72)
                print(f"# CHECK {name}")
                print("#" * 72, flush=True)
                outcomes.append(run_check(name, registry[name], args.timeout,
                                          model=args.model, num_ctx=args.num_ctx))
            if ballast is not None:
                finish_voice(run, ballast, outcomes, args.slowdown_limit)
    finally:
        if ballast is not None:
            ballast.stop()

    #: a run that could not be measured is exit 2 even with no check run, as before
    code = 2 if run.error else overall_exit(outcomes, requested)
    commit, dirty = git_state()
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print()
    print(format_table(outcomes, commit, dirty, timestamp, code, model=args.model,
                       num_ctx=args.num_ctx, run=run))
    append_log(build_log_line(outcomes, commit, dirty, timestamp, names, code, run), args.log)
    return code


def _defaults(run: RunInfo) -> tuple:
    """(model, num_ctx, keep_alive) the checks will actually use for this run."""
    import voice_main
    return (run.model or voice_main.VOICE_MODEL, run.num_ctx or voice_main.NUM_CTX,
            voice_main.KEEP_ALIVE)


def prepare(run: RunInfo, ballast: Optional[VoiceBallast]) -> bool:
    """--clean / --with-voice set-up, in voice_main.boot()'s order. False = do not run.

    unload all -> idle reading -> (ballast loads + first round trip) -> warm the LLM LAST,
    so Ollama loads against what voice really leaves free and any spill is a real one.
    """
    if not run.clean:
        return True
    run.total_gpu_mib = gpu_total_mib()
    print("#" * 72)
    print("# CLEAN START" + (" + VOICE" if run.with_voice else ""))
    print("#" * 72, flush=True)
    run.leftover = ollama_control.unload_all()
    if run.leftover is None:
        run.error = "could not read /api/ps to unload models (is Ollama running?)"
        return False
    if run.leftover:
        run.error = ("models still loaded after unloading: "
                     + ", ".join(_short(n) for n in run.leftover) + " - start is not clean")
        return False
    run.idle_gpu_mib = gpu_used_mib()
    print(f"  unloaded   /api/ps empty; idle GPU {run.idle_gpu_mib} MiB "
          f"of {run.total_gpu_mib}", flush=True)

    if ballast is not None:
        print("  ballast    loading Whisper + Kokoro + Silero (no embedder: the check has it)",
              flush=True)
        try:
            ready = ballast.start()
        except BallastError as exc:
            run.error = f"voice ballast: {exc}"
            return False
        run.ballast_stages = ready.get("stages")
        run.voice_gpu_mib = ready.get("gpu_mib")
        run.pre_pass = ballast.run_pass()
        #: the reference pass: the ballast already did its cold warm-up before `ready`,
        #: so this one is warm and the FIT slowdown gate compares warm with warm
        print(f"  voice      {run.voice_gpu_mib} MiB with voice loaded; reference round trip "
              + (f"{run.pre_pass.get('round_trip_ms', 0):.0f} ms" if run.pre_pass.get("ok")
                 else f"FAILED: {run.pre_pass.get('error')}"), flush=True)

    model, num_ctx, keep_alive = _defaults(run)
    run.warm = ollama_control.warm(model, num_ctx, keep_alive)
    if run.warm["error"]:
        #: not fatal: the checks hit the same problem and report it with their own exit 2
        run.notes.append(f"warm: {model} @ ctx {num_ctx} failed: {run.warm['error']}")
    else:
        print(f"  warmed     {model} @ ctx {num_ctx} in {run.warm['seconds']:.1f}s", flush=True)
    return True


def finish_voice(run: RunInfo, ballast: VoiceBallast, outcomes: list,
                 slowdown_limit: float = SLOWDOWN_LIMIT) -> None:
    """After the checks: the second round trip, the final readings, the FIT verdict."""
    run.post_pass = ballast.run_pass()
    run.final_models = ollama_models()
    run.post_gpu_mib = gpu_used_mib()
    run.fit = fit_verdict(outcomes, run, slowdown_limit=slowdown_limit)


if __name__ == "__main__":
    sys.exit(main())
