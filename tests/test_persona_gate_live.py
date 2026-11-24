"""The persona check, as a local gate: llama3.1 must still sound like Marshall.

    pytest tests/test_persona_gate_live.py -v
    python scripts/check_persona.py           # same cases, readable output, exit code

The model-free half (leak grader, spoken-form rules, persona traps) is
tests/test_persona_check.py and runs in CI. THIS file needs the real model, so it is
`needs_llama31`: CI deselects it, and it skips — never passes — without Ollama + llama3.1.

**No drift.** The fixture calls the script's own `_run`, so this gate and
`python scripts/check_persona.py` execute the same code.

**Its own stores.** Run by hand, the check opens `db/voice_vdb` (the LIVE document index)
and writes under `db/`. Here both are redirected into tmp from the test side, without
changing the check.

**What turns the gate red.** Only a must-pass case (identity, tool-turn) missing. A
boundary missing is the 23 Nov baseline and xfails with its known gap; a lucky boundary
pass is allowed and is NOT "fixed".
"""
import functools
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import voice_main
from scripts import check_persona as cp
from scripts.eval_contract import BOUNDARY, PASS, CheckRecord


#: same probe as tests/test_routing_matrix.py, restated rather than imported because that
#: is this repo's pattern for it
def _ollama_up() -> bool:
    try:
        import urllib.request
        urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=3)
        return True
    except Exception:
        return False


def _has_llama31() -> bool:
    from scripts.check_routing import model_available
    return model_available(voice_main.VOICE_MODEL)


#: the reason string is what conftest.py reads to add needs_ollama + needs_llama31
needs_llama31 = pytest.mark.skipif(
    not (_ollama_up() and _has_llama31()),
    reason=f"needs Ollama on :11434 with {voice_main.VOICE_MODEL} pulled")

MUST_PASS = [c for c in cp.CASES if c.must_pass]
BOUNDARIES = [c for c in cp.CASES if not c.must_pass]


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """Run the whole persona check once, through the script's own `_run`, on tmp stores."""
    import scripts.check_routing as routing

    root = tmp_path_factory.mktemp("persona_gate")
    record = CheckRecord(check="persona", model=voice_main.VOICE_MODEL)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cp, "CHECK_USER_HISTORY_PATH", str(root / "user_history"))
        mp.setattr(cp, "CHECK_SUMMARY_PATH", str(root / "summaries"))
        mp.setattr(routing, "build_check_agent",
                   functools.partial(routing.build_check_agent,
                                     vector_db_path=str(root / "vdb")))
        code = cp._run(cp.parse_args([]), record)
    return {"code": code, "record": record,
            "by_id": {result.id: result for result in record.cases}}


@needs_llama31
def test_the_check_ran(run):
    """Exit 2 means 'could not run' — a gate that silently measured nothing."""
    assert run["record"].error is None, run["record"].error
    assert run["code"] != 2
    assert set(run["by_id"]) == {c.id for c in cp.CASES}


@needs_llama31
@pytest.mark.parametrize("case", MUST_PASS, ids=lambda c: c.id)
def test_must_pass_case_passes(run, case):
    result = run["by_id"][case.id]
    assert result.verdict == PASS, f"{case.id}: {result.verdict} — {result.detail}"


@needs_llama31
@pytest.mark.parametrize("case", BOUNDARIES, ids=lambda c: c.id)
def test_boundary_case_is_no_worse_than_baseline(run, case):
    """A miss is the baseline (xfail). A pass is allowed and is not a fix."""
    result = run["by_id"][case.id]
    if result.verdict == BOUNDARY:
        pytest.xfail(f"known boundary ({cp.BASELINE_DATE}): {case.known_gap}")
    assert result.verdict == PASS, f"{case.id}: {result.verdict} — {result.detail}"
