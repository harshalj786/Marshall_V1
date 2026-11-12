"""The result contract every eval check writes, and the one rule that reads it.

A check script (scripts/check_routing.py today; memory and persona later) prints a
human report AND, given `--json <path>`, writes one `CheckRecord`. The runner reads
records back with `CheckRecord.from_dict` and never parses printed text, so rewording a
printout cannot break aggregation.

Exit codes, shared by every check and NOT stored in the record (the runner takes them
from the process):

    0  every must-pass case passed
    1  at least one case regressed (FAIL or ERROR)
    2  the check could not run (Ollama down, model not pulled, agent would not build)

**2 is never a pass.** A record from a run that could not happen carries `error` and an
empty `cases` list, and `record_passed` treats both as failure. Without the non-empty
rule an empty list would read as "no regressions", the quiet way a skipped check turns
into a green one.

`passed` is deliberately not a stored field. It is derived here, once, so a record can
never claim a verdict its own cases contradict.

Schema version 2 (v1 = the first four CaseResult fields only; still readable):

    CheckRecord
      schema_version : int             bumped when a field changes meaning or is added
      check          : str             "routing"
      model          : str             "llama3.1"
      error          : str | None      set only when the check could not run (exit 2)
      cases          : list[CaseResult]

    CaseResult
      id             : str             stable per case, e.g. the route name "math"
      verdict        : str             one of VERDICTS
      wall_seconds   : float           wall clock for the case, unrounded
      error          : str | None      exception text, for an ERROR verdict
      -- v2, filled by scripts/eval_metrics.py; None = the reading could not be taken --
      model_seconds  : float | None    Ollama's own total_duration, summed over the turn
      load_seconds   : float | None    Ollama's load_duration, summed (see is_cold_load)
      prompt_tokens  : int | None      prompt_eval_count, summed
      output_tokens  : int | None      eval_count, summed
      vram_models    : dict | None     {loaded model: size_vram MiB} from /api/ps;
                                       {} = read fine, nothing loaded
      vram_gpu_mib   : int | None      nvidia-smi memory.used, whole GPU

`wall_seconds - model_seconds` is the remainder (tools, embeddings, graph, HTTP, and any
LLM call whose reply is not in the thread). It is computed by readers, never stored —
see the caveat at the top of scripts/eval_metrics.py before reading it as tool time.
"""
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

SCHEMA_VERSION = 2
#: versions from_dict accepts. A v1 record simply lacks the v2 fields, which default None.
READABLE_VERSIONS = (1, 2)

#: Ollama reports ~0.09 s of load_duration on a model that is ALREADY loaded (measured
#: 2026-11-12, llama3.1 warm), so "non-zero" cannot mean cold. A real 8B load from disk
#: takes seconds. The threshold is a judgement, which is why it lives here and records
#: store the raw seconds: change this and every old record is re-judged, not frozen.
COLD_LOAD_THRESHOLD_S = 1.0

PASS = "PASS"
FAIL = "FAIL"
#: a known, recorded limitation — reported, and not a regression on its own
BOUNDARY = "KNOWN-BOUNDARY"
ERROR = "ERROR"

VERDICTS = (PASS, FAIL, BOUNDARY, ERROR)
#: the verdicts that mean something broke
REGRESSION_VERDICTS = (FAIL, ERROR)


@dataclass
class CaseResult:
    id: str
    verdict: str
    wall_seconds: float
    error: Optional[str] = None
    model_seconds: Optional[float] = None
    load_seconds: Optional[float] = None
    prompt_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    vram_models: Optional[dict] = None
    vram_gpu_mib: Optional[int] = None

    def __post_init__(self):
        if self.verdict not in VERDICTS:
            raise ValueError(f"unknown verdict {self.verdict!r}; expected one of {VERDICTS}")


@dataclass
class CheckRecord:
    check: str
    model: str
    cases: list = field(default_factory=list)
    error: Optional[str] = None
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict:
        #: fixed key order, so two records diff cleanly
        return {
            "schema_version": self.schema_version,
            "check": self.check,
            "model": self.model,
            "error": self.error,
            "cases": [asdict(case) for case in self.cases],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CheckRecord":
        if data.get("schema_version") not in READABLE_VERSIONS:
            raise ValueError(f"unsupported schema_version {data.get('schema_version')!r}; "
                             f"this reader understands {READABLE_VERSIONS}")
        #: read into the current shape: a v1 record's missing fields become None, and the
        #: object is a v2 record from here on (to_dict writes it back as v2)
        return cls(
            check=data["check"],
            model=data["model"],
            error=data.get("error"),
            cases=[CaseResult(**case) for case in data.get("cases", [])],
        )


def record_passed(record: CheckRecord) -> bool:
    """The single pass rule: it ran, it ran something, and nothing regressed."""
    return (record.error is None
            and bool(record.cases)
            and not any(case.verdict in REGRESSION_VERDICTS for case in record.cases))


def is_cold_load(case: CaseResult, threshold_s: float = COLD_LOAD_THRESHOLD_S):
    """True / False, or None when the case has no load reading to judge."""
    if case.load_seconds is None:
        return None
    return case.load_seconds >= threshold_s


def write_record(record: CheckRecord, path) -> bool:
    """Write `record` as JSON. True on success.

    Never raises: like voice/telemetry.py, the instrumentation may lose its own data but
    never changes the check's result. A failure is reported on stderr and the caller's
    exit code stands.
    """
    try:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(record.to_dict(), indent=2, ensure_ascii=False) + "\n",
                          encoding="utf-8")
        return True
    except Exception as exc:
        print(f"warning: could not write {path}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return False
