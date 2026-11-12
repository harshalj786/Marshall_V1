"""scripts/run_evals.py with no model, so this runs in CI.

Two halves. The pure logic (classify, overall_exit, the table, the log line) is fed
hand-built records. The launch-and-collect path — the runner's riskiest code — is
exercised for REAL: tiny fake check scripts are written into tmp_path and run as actual
subprocesses (pass, FAIL + exit 1, crash with no record, hang past the timeout, exit 0
with a FAIL record). Only the two VRAM readers are stubbed, so the start snapshot is
deterministic and its timing can be witnessed.
"""
import json
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import scripts.run_evals as runner
from scripts.eval_contract import BOUNDARY, ERROR, FAIL, PASS, CaseResult, CheckRecord
from scripts.run_evals import (DISAGREE, ST_COULD_NOT_RUN, ST_NOT_BUILT, ST_PASS,
                               ST_REGRESSION, CheckOutcome, build_log_line, classify,
                               format_table, overall_exit, run_check, summarize)


def _record(*verdicts, error=None, **case_fields):
    return CheckRecord(check="routing", model="llama3.1", error=error, cases=[
        CaseResult(id=f"c{i}", verdict=v, wall_seconds=float(i + 1), **case_fields)
        for i, v in enumerate(verdicts)])


# ==========================================================================
# classify — every row of the approved table
# ==========================================================================

def test_clean_pass():
    assert classify(0, _record(PASS, PASS)) == (ST_PASS, None, None)


def test_boundary_alone_still_passes():
    assert classify(0, _record(PASS, BOUNDARY))[0] == ST_PASS


@pytest.mark.parametrize("bad", [FAIL, ERROR])
def test_regression_names_the_failing_cases(bad):
    status, reason, note = classify(1, _record(PASS, bad))
    assert (status, reason, note) == (ST_REGRESSION, "FAIL: c1", None)


def test_crash_without_a_record_is_could_not_run_not_a_regression():
    """An uncaught exception exits 1, exactly like a regression. The missing record is
    what tells them apart."""
    status, reason, _ = classify(1, None)
    assert status == ST_COULD_NOT_RUN
    assert reason == "no record written (exit 1)"


def test_exit_2_with_error_record_is_could_not_run():
    assert classify(2, _record(error="Ollama is not reachable")) == (
        ST_COULD_NOT_RUN, "Ollama is not reachable", None)


def test_launcher_failure_wins():
    assert classify(None, None, failure="timed out after 1s")[:2] == (
        ST_COULD_NOT_RUN, "timed out after 1s")


def test_unexpected_exit_code_is_could_not_run():
    assert classify(-9, _record(PASS))[0] == ST_COULD_NOT_RUN


def test_empty_cases_is_could_not_run_and_exit_0_is_flagged():
    status, _, note = classify(0, _record())
    assert status == ST_COULD_NOT_RUN and note == DISAGREE


@pytest.mark.parametrize("exit_code, record, expected", [
    (0, _record(PASS, FAIL), ST_REGRESSION),           # exit says pass, record says FAIL
    (1, _record(PASS, PASS), ST_REGRESSION),           # exit says fail, record says pass
    (0, _record(PASS, error="half-run"), ST_COULD_NOT_RUN),  # error set but exit 0
    (2, _record(PASS), ST_COULD_NOT_RUN),              # exit 2 but no error in record
])
def test_disagreement_takes_the_stricter_reading_and_is_flagged(exit_code, record, expected):
    status, _, note = classify(exit_code, record)
    assert status == expected
    assert note == DISAGREE


# ==========================================================================
# overall_exit — precedence
# ==========================================================================

def _o(status, check="x"):
    return CheckOutcome(check=check, status=status)


@pytest.mark.parametrize("statuses, expected", [
    ([ST_PASS], 0),
    ([ST_PASS, ST_REGRESSION, ST_COULD_NOT_RUN], 1),   # regression outranks could-not-run
    ([ST_PASS, ST_COULD_NOT_RUN], 2),
    ([ST_COULD_NOT_RUN], 2),
    ([], 2),                                          # empty run is never a pass
])
def test_exit_precedence(statuses, expected):
    assert overall_exit([_o(s) for s in statuses]) == expected


def test_not_built_is_ignored_on_a_plain_run_but_fails_when_named():
    outcomes = [_o(ST_PASS, "routing"), _o(ST_NOT_BUILT, "memory")]
    assert overall_exit(outcomes, requested=set()) == 0
    assert overall_exit(outcomes, requested={"routing", "memory"}) == 2


def test_only_not_built_checks_is_not_a_pass():
    assert overall_exit([_o(ST_NOT_BUILT, "memory")]) == 2


# ==========================================================================
# table and log line
# ==========================================================================

def test_summary_skips_nulls_instead_of_counting_them_as_zero():
    record = _record(PASS, PASS, BOUNDARY)
    record.cases[0].model_seconds = 2.0
    record.cases[2].model_seconds = 4.0           # c1 stays None
    record.cases[0].vram_models = {"llama3.1:latest": 5211}
    record.cases[2].vram_models = {"llama3.1:latest": 5211, "mistral:latest": 4906}
    record.cases[2].vram_gpu_mib = 11755
    s = summarize(CheckOutcome(check="routing", status=ST_PASS, record=record))
    assert (s["passed"], s["boundary"], s["total"]) == (2, 1, 3)
    assert s["wall_median"] == 2.0 and s["wall_max"] == 3.0
    assert s["model_median"] == 3.0               # median of 2.0 and 4.0, not of 0/2/4
    assert s["peak_gpu_mib"] == 11755
    assert s["models_during"] == {"llama3.1:latest": 5211, "mistral:latest": 4906}


def test_summary_with_every_reading_null_shows_dashes():
    outcome = CheckOutcome(check="routing", status=ST_PASS, record=_record(PASS))
    s = summarize(outcome)
    assert s["model_median"] is None and s["peak_gpu_mib"] is None and s["models_during"] is None
    table = format_table([outcome], "f425cbd" + "0" * 33, False, "T", 0)
    assert "-s" in table and "?" in table


def test_table_shows_status_reason_snapshot_and_result():
    passed = CheckOutcome(check="routing", status=ST_PASS, exit_code=0,
                          record=_record(PASS, BOUNDARY, vram_models={"llama3.1:latest": 5211},
                                         vram_gpu_mib=7412),
                          start_vram_models={"mistral:latest": 4906})
    missing = CheckOutcome(check="memory", status=ST_NOT_BUILT,
                           reason="scripts/check_memory.py not found")
    table = format_table([passed, missing], "f425cbd" + "0" * 33, True, "2026-11-12T12:00:00+00:00", 0)
    assert "commit f425cbd (dirty)" in table
    assert "1/2 +1bdy" in table
    assert "mistral 4906" in table              # loaded at start: the leftover is visible
    assert "llama3.1 5211" in table
    assert "7412 MiB" in table
    assert "NOT-BUILT" in table and "scripts/check_memory.py not found" in table
    assert "0 regressed, 0 could not run, 1 not built -> exit 0" in table


def test_disagreement_note_is_printed():
    outcome = CheckOutcome(check="routing", status=ST_REGRESSION, record=_record(FAIL),
                           reason="FAIL: c0", note=DISAGREE)
    assert f"!! routing: {DISAGREE}" in format_table([outcome], None, None, "T", 1)


def test_log_line_shape_embeds_the_full_record():
    record = _record(PASS)
    outcome = CheckOutcome(check="routing", status=ST_PASS, exit_code=0, record=record,
                           start_vram_models={}, start_gpu_mib=900)
    line = build_log_line([outcome, _o(ST_NOT_BUILT, "memory")], "abc", False, "T",
                          ["routing", "memory"], 0)
    assert set(line) == {"timestamp", "commit", "dirty", "only", "exit_code", "checks"}
    first, second = line["checks"]
    assert set(first) == {"check", "status", "exit_code", "reason", "note",
                          "start_vram_models", "start_gpu_mib", "record"}
    assert CheckRecord.from_dict(first["record"]) == record
    assert first["start_vram_models"] == {} and first["start_gpu_mib"] == 900
    assert second["record"] is None
    json.dumps(line)                             # must serialise as-is


# ==========================================================================
# the real subprocess path — fake check scripts, no model
# ==========================================================================

def _record_json(*verdicts, error=None):
    return json.dumps({"schema_version": 2, "check": "fake", "model": "fake-model",
                       "error": error,
                       "cases": [{"id": f"c{i}", "verdict": v, "wall_seconds": 0.1}
                                 for i, v in enumerate(verdicts)]})


def _fake_check(tmp_path, name, body):
    """A check script that honours `--json PATH`, like the real ones."""
    script = tmp_path / f"check_{name}.py"
    script.write_text(textwrap.dedent(f"""\
        import sys, time
        from pathlib import Path
        out = Path(sys.argv[sys.argv.index("--json") + 1])
        Path(__file__).with_suffix(".started").write_text("x")
        {body}
        """), encoding="utf-8")
    return script


@pytest.fixture
def snapshot(monkeypatch):
    """Stub the readers, and witness that the snapshot is taken BEFORE launch."""
    #: calls = one entry per snapshot: the fake checks that had ALREADY started by then
    state = SimpleNamespace(calls=[], dir=None)

    def fake_vram():
        state.calls.append(sorted(p.name for p in state.dir.glob("*.started"))
                           if state.dir else [])
        return {"leftover:latest": 1234}

    monkeypatch.setattr(runner, "ollama_vram", fake_vram)
    monkeypatch.setattr(runner, "gpu_used_mib", lambda: 4321)
    return state


def test_real_subprocess_pass(tmp_path, snapshot):
    snapshot.dir = tmp_path
    script = _fake_check(tmp_path, "pass",
                         f"out.write_text({_record_json(PASS, BOUNDARY)!r}); sys.exit(0)")
    outcome = run_check("fake", script, timeout_s=60)
    assert outcome.status == ST_PASS and outcome.exit_code == 0
    assert [c.verdict for c in outcome.record.cases] == [PASS, BOUNDARY]
    #: the snapshot was taken before the check started (no .started marker yet)
    assert snapshot.calls == [[]]
    assert outcome.start_vram_models == {"leftover:latest": 1234}
    assert outcome.start_gpu_mib == 4321


def test_real_subprocess_fail_exit_1(tmp_path, snapshot):
    script = _fake_check(tmp_path, "fail",
                         f"out.write_text({_record_json(PASS, FAIL)!r}); sys.exit(1)")
    outcome = run_check("fake", script, timeout_s=60)
    assert (outcome.status, outcome.reason, outcome.note) == (ST_REGRESSION, "FAIL: c1", None)


def test_real_subprocess_crash_without_record(tmp_path, snapshot):
    script = _fake_check(tmp_path, "crash", "raise RuntimeError('check blew up')")
    outcome = run_check("fake", script, timeout_s=60)
    assert outcome.status == ST_COULD_NOT_RUN
    assert outcome.exit_code == 1
    assert outcome.reason == "no record written (exit 1)"
    assert outcome.record is None


def test_real_subprocess_hang_is_killed_at_the_timeout(tmp_path, snapshot):
    script = _fake_check(tmp_path, "hang", "time.sleep(60)")
    outcome = run_check("fake", script, timeout_s=1)
    assert outcome.status == ST_COULD_NOT_RUN
    assert outcome.reason == "timed out after 1s"


def test_real_subprocess_exit_0_with_a_fail_record_is_flagged(tmp_path, snapshot):
    script = _fake_check(tmp_path, "liar",
                         f"out.write_text({_record_json(PASS, FAIL)!r}); sys.exit(0)")
    outcome = run_check("fake", script, timeout_s=60)
    assert outcome.status == ST_REGRESSION
    assert outcome.note == DISAGREE


def test_real_subprocess_unreadable_record(tmp_path, snapshot):
    script = _fake_check(tmp_path, "junk", "out.write_text('not json'); sys.exit(0)")
    outcome = run_check("fake", script, timeout_s=60)
    assert outcome.status == ST_COULD_NOT_RUN
    assert outcome.reason.startswith("unreadable record (exit 0)")


def test_missing_script_is_not_built_and_takes_no_snapshot(tmp_path, snapshot):
    outcome = run_check("memory", tmp_path / "check_memory.py")
    assert outcome.status == ST_NOT_BUILT
    assert snapshot.calls == []


# ==========================================================================
# main — the whole command, on a fake registry
# ==========================================================================

def _registry(tmp_path):
    return {
        "good": str(_fake_check(tmp_path, "good",
                                f"out.write_text({_record_json(PASS)!r}); sys.exit(0)")),
        "bad": str(_fake_check(tmp_path, "bad",
                               f"out.write_text({_record_json(FAIL)!r}); sys.exit(1)")),
        "absent": str(tmp_path / "check_absent.py"),
    }


def test_main_plain_run_ignores_not_built(tmp_path, snapshot, capsys):
    registry = _registry(tmp_path)
    del registry["bad"]
    log = tmp_path / "runs.jsonl"
    assert runner.main(["--log", str(log)], registry=registry) == 0
    out = capsys.readouterr().out
    assert "EVAL SUMMARY" in out and "NOT-BUILT" in out
    line = json.loads(log.read_text(encoding="utf-8").strip())
    assert [c["status"] for c in line["checks"]] == [ST_PASS, ST_NOT_BUILT]
    assert line["exit_code"] == 0


def test_main_named_missing_check_exits_2(tmp_path, snapshot):
    assert runner.main(["--only", "good,absent", "--log", str(tmp_path / "l.jsonl")],
                       registry=_registry(tmp_path)) == 2


def test_main_regression_exits_1_and_appends_one_line_per_run(tmp_path, snapshot):
    log = tmp_path / "runs.jsonl"
    registry = _registry(tmp_path)
    assert runner.main(["--only", "good,bad", "--log", str(log)], registry=registry) == 1
    assert runner.main(["--only", "good", "--log", str(log)], registry=registry) == 0
    lines = log.read_text(encoding="utf-8").splitlines()
    assert [json.loads(l)["exit_code"] for l in lines] == [1, 0]


def test_main_unknown_name_is_a_usage_error(tmp_path, snapshot, capsys):
    assert runner.main(["--only", "rotuing", "--log", str(tmp_path / "l.jsonl")],
                       registry=_registry(tmp_path)) == 2
    assert "unknown check(s): rotuing" in capsys.readouterr().err
    assert not (tmp_path / "l.jsonl").exists()   # nothing ran, nothing logged
