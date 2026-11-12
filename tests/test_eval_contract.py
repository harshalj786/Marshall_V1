"""The eval contract's pass rule and record format. No model needed, so this runs in CI.

`record_passed` is the rule the runner and every check trust, so it is pinned here
before anything is built on it. The empty-cases test is the important one: it is what
keeps a check that could not run (exit 2) from reading as "no regressions".
"""
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.eval_contract import (BOUNDARY, ERROR, FAIL, PASS, SCHEMA_VERSION,
                                   CaseResult, CheckRecord, is_cold_load, record_passed,
                                   write_record)


def _record(*verdicts, error=None):
    return CheckRecord(check="routing", model="llama3.1", error=error,
                       cases=[CaseResult(id=f"case-{i}", verdict=v, wall_seconds=1.0)
                              for i, v in enumerate(verdicts)])


# --------------------------------------------------------------------------
# record_passed
# --------------------------------------------------------------------------

def test_empty_case_list_is_not_a_pass():
    assert record_passed(_record()) is False


def test_could_not_run_is_not_a_pass():
    """Exit 2's record: error set, no cases."""
    assert record_passed(_record(error="Ollama is not reachable")) is False


def test_error_field_fails_even_with_passing_cases():
    assert record_passed(_record(PASS, PASS, error="aborted part-way")) is False


@pytest.mark.parametrize("bad", [FAIL, ERROR])
def test_a_regression_verdict_fails_the_record(bad):
    assert record_passed(_record(PASS, bad, PASS)) is False


def test_all_pass_passes():
    assert record_passed(_record(PASS, PASS, PASS)) is True


def test_known_boundary_alone_does_not_fail_the_record():
    """Same as routing's no-tool row: reported, never a regression on its own."""
    assert record_passed(_record(PASS, BOUNDARY)) is True


# --------------------------------------------------------------------------
# format guards
# --------------------------------------------------------------------------

def test_unknown_verdict_is_rejected():
    with pytest.raises(ValueError, match="unknown verdict"):
        CaseResult(id="x", verdict="pass", wall_seconds=1.0)


def test_unknown_schema_version_is_rejected():
    data = _record(PASS).to_dict()
    data["schema_version"] = SCHEMA_VERSION + 1
    with pytest.raises(ValueError, match="schema_version"):
        CheckRecord.from_dict(data)


def test_round_trip_is_unchanged():
    original = _record(PASS, BOUNDARY, ERROR)
    original.cases[2].error = "TimeoutError: timed out"
    restored = CheckRecord.from_dict(json.loads(json.dumps(original.to_dict())))
    assert restored == original
    assert restored.to_dict() == original.to_dict()


def test_a_v1_record_still_reads_with_the_v2_fields_null():
    v1 = {"schema_version": 1, "check": "routing", "model": "llama3.1", "error": None,
          "cases": [{"id": "math", "verdict": "PASS", "wall_seconds": 4.0, "error": None}]}
    record = CheckRecord.from_dict(v1)
    case = record.cases[0]
    assert (case.id, case.verdict, case.wall_seconds) == ("math", PASS, 4.0)
    assert case.model_seconds is None and case.vram_models is None and case.vram_gpu_mib is None
    assert record_passed(record) is True


def test_v2_fields_round_trip():
    original = _record(PASS)
    case = original.cases[0]
    case.model_seconds, case.load_seconds = 2.5, 0.09
    case.prompt_tokens, case.output_tokens = 430, 32
    case.vram_models = {"llama3.1:latest": 5211, "mistral:latest": 4906}
    case.vram_gpu_mib = 11269
    assert CheckRecord.from_dict(json.loads(json.dumps(original.to_dict()))) == original


@pytest.mark.parametrize("load, expected", [(None, None), (0.09, False), (1.0, True), (6.2, True)])
def test_is_cold_load_uses_the_threshold_not_nonzero(load, expected):
    """A warm llama3.1 reports ~0.09 s of load_duration, so 'non-zero' is not 'cold'."""
    case = CaseResult(id="x", verdict=PASS, wall_seconds=1.0, load_seconds=load)
    assert is_cold_load(case) is expected


def test_write_record_round_trips_through_disk(tmp_path):
    original = _record(PASS, FAIL)
    path = tmp_path / "nested" / "routing.json"
    assert write_record(original, path) is True
    assert CheckRecord.from_dict(json.loads(path.read_text(encoding="utf-8"))) == original


def test_write_record_never_raises(tmp_path):
    """Telemetry must not change the exit code: a failed write returns False."""
    blocker = tmp_path / "a_file"
    blocker.write_text("x")
    assert write_record(_record(PASS), blocker / "routing.json") is False
