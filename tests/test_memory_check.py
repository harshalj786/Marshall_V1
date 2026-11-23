"""scripts/check_memory.py with no model, so this runs in CI.

What needs llama3.1 (actually running the seven cases) is local-only for now; its
`needs_llama31` pytest copy is Feature 2 Step 6. What is pinned here is everything that
decides a verdict without a model:

  * the case table itself — above all the PROMPT GUARD: no graded fact may already be in
    the system prompt or the bound tools' descriptions, or a case passes for the wrong
    reason ("Pune" and "RAG agent" are both in the prompt's statement examples);
  * `grade`, one rule at a time;
  * k-of-n: when to stop, how attempts fold into one verdict, how metrics combine.
"""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts import check_memory as cm
from scripts.check_memory import (CASES, COLOURS, DISTRACTORS, RECALL, Attempt, MemoryCase,
                                  aggregate, all_seeds, contains, grade, run_k_of_n, settled)
from scripts.eval_contract import BOUNDARY, ERROR, FAIL, PASS


# --------------------------------------------------------------------------
# the case table
# --------------------------------------------------------------------------

def _prompt_and_tool_text() -> str:
    """Everything the model sees before the user speaks: the system prompt the check's
    agent is built with (recall enabled, no session) AND every bound tool's description."""
    from pipeline_lg import AGENT_SYSTEM_PROMPT
    from tools.tools import (calculator, make_recall_user_memory, make_search_current_doc,
                             web_search, wikipedia_lookup)

    tools = [calculator, wikipedia_lookup, web_search,
             make_search_current_doc(None), make_recall_user_memory(lambda *a: None)]
    return "\n".join([AGENT_SYSTEM_PROMPT] + [t.name + " " + t.description for t in tools])


def _graded_phrases():
    for case in CASES:
        for phrase in case.tool_returned + case.answer_has:
            yield case.id, phrase


@pytest.mark.parametrize("case_id, phrase", list(_graded_phrases()))
def test_no_graded_fact_is_already_in_the_prompt(case_id, phrase):
    """THE prompt guard. 'Pune' would pass this check for the wrong reason: it is in the
    prompt's own example, so the answer could name it without any recall at all."""
    assert not contains(_prompt_and_tool_text(), phrase), (
        f"{case_id}: {phrase!r} is already in the system prompt or a tool description, "
        f"so a correct answer would prove nothing about memory")


def test_the_guard_would_catch_the_prompt_facts():
    """The guard is only worth something if it fires on the known traps."""
    text = _prompt_and_tool_text()
    assert contains(text, "Pune") and contains(text, "Harshal") and contains(text, "RAG agent")


def test_pune_is_an_answer_lacks_trap():
    ranking = next(c for c in CASES if c.id == "ranking")
    assert "Pune" in ranking.answer_lacks


def test_case_ids_are_unique_and_there_are_seven():
    ids = [c.id for c in CASES]
    assert len(ids) == len(set(ids)) == 7


def test_every_graded_phrase_has_a_source():
    """A phrase no seed or told statement contains could never be returned: the case would
    fail forever and look like a regression."""
    for case in CASES:
        sources = " ".join(case.seeds + case.told_in_thread)
        for phrase in case.tool_returned + case.answer_has:
            assert contains(sources, phrase), f"{case.id}: nothing seeds {phrase!r}"


def test_k_of_n_only_on_flaky_cases():
    for case in CASES:
        k, n = case.runs
        assert 1 <= k <= n
        if not case.flaky:
            assert (k, n) == (1, 1), f"{case.id} is solid and must run once"


def test_enough_seeds_that_the_top_four_must_rank():
    """Default k=4 for recall_user_memory: with <= 4 seeds the tool returns all of them
    whatever the query, and 'the tool returned the fact' proves nothing."""
    assert len(all_seeds()) >= 10
    assert len(set(all_seeds())) == len(all_seeds())


def test_the_unknown_target_is_seeded_nowhere():
    """If any seed mentions a colour, 'I don't know your favourite colour' becomes wrong."""
    unknown = next(c for c in CASES if c.id == "unknown")
    assert not unknown.must_pass, "Feature 5 is not built: a miss is KNOWN-BOUNDARY"
    everything = " ".join(all_seeds() + tuple(s for c in CASES for s in c.told_in_thread))
    for colour in COLOURS:
        assert not contains(everything, colour), f"a statement mentions {colour!r}"
        assert colour in unknown.answer_lacks


def test_carry_over_is_told_not_seeded():
    """It is the one case that exercises the gate's write, so seeding it would test nothing:
    its fact must be absent from the shared seeds, and it must take >= 5 turns to tell."""
    carry = next(c for c in CASES if c.id == "carry-over")
    assert not carry.seeds and len(carry.told_in_thread) >= 5
    assert not any(contains(" ".join(all_seeds()), p) for p in carry.answer_has)


def test_distractors_are_seeded_but_never_graded():
    graded = [p for _, p in _graded_phrases()]
    for statement in DISTRACTORS:
        assert statement in all_seeds()
        assert not any(contains(statement, p) for p in graded)


# --------------------------------------------------------------------------
# grade
# --------------------------------------------------------------------------

PET = MemoryCase("pet", "q", require_tool=RECALL, tool_returned=("Biscuit",),
                 answer_has=("Biscuit",), answer_lacks=("Pune",))
GOOD_RETURN = [(RECALL, "My dog is called Biscuit.\n\nI'm learning Japanese.")]


def test_grade_passes_when_every_rule_holds():
    assert grade(PET, [RECALL], GOOD_RETURN, "Your dog is called Biscuit!") is None


def test_grade_tool_not_called():
    reason = grade(PET, ["wikipedia_lookup"], [], "Biscuit")
    assert f"{RECALL} not called (called: wikipedia_lookup)" in reason


def test_grade_right_tool_wrong_snippets():
    """Called recall, but ranking missed: the store did not return the fact."""
    reason = grade(PET, [RECALL], [(RECALL, "I'm learning Japanese.")], "Biscuit")
    assert reason == "tool did not return 'Biscuit'"


def test_grade_ignores_other_tools_returns():
    """The fact must come back from the REQUIRED tool, not from a web page."""
    reason = grade(PET, [RECALL, "web_search"],
                   [(RECALL, "nothing"), ("web_search", "Biscuit the dog")], "Biscuit")
    assert reason == "tool did not return 'Biscuit'"


def test_grade_answer_lacks_the_fact():
    assert grade(PET, [RECALL], GOOD_RETURN, "I'm not sure.") == "answer lacks 'Biscuit'"


def test_grade_answer_lacks_trap():
    reason = grade(PET, [RECALL], GOOD_RETURN, "Biscuit, and you live in Pune")
    assert reason == "answer contains 'Pune'"


def test_grade_names_every_problem_not_just_the_first():
    reason = grade(PET, [], [], "you live in Pune")
    assert reason.count(";") == 3


def test_grade_answer_any():
    case = MemoryCase("unknown", "q", answer_any=("don't know", "haven't told"),
                      answer_lacks=COLOURS)
    assert grade(case, [], [], "You haven't told me your favourite colour.") is None
    assert grade(case, [], [], "It's blue!") == ("answer has no 'don't know'-style phrase; "
                                                "answer contains 'blue'")


@pytest.mark.parametrize("text, phrase, expected", [
    ("you shared that", "red", False),       # word start, not substring
    ("Red is nice", "red", True),            # case-insensitive
    ("allergic to peanuts", "peanut", True), # only the left edge is anchored
    ("", "x", False),
    (None, "x", False),
])
def test_contains(text, phrase, expected):
    assert contains(text, phrase) is expected


# --------------------------------------------------------------------------
# k-of-n
# --------------------------------------------------------------------------

@pytest.mark.parametrize("passes, done, settled_", [
    (3, 3, True),    # k reached early
    (2, 3, False),   # still open
    (0, 3, True),    # 0 + 2 left < 3: unreachable
    (1, 3, False),   # 1 + 2 left = 3: still reachable
    (2, 4, False),   # 2 + 1 left = 3: still reachable
    (1, 4, True),    # 1 + 1 left < 3
])
def test_settled(passes, done, settled_):
    assert settled(passes, done, 3, 5) is settled_


FLAKY = MemoryCase("f", "q", flaky=True, k=3, n=5)
SOLID = MemoryCase("s", "q")


def _scripted(pattern):
    """A run_attempt that plays back 'P'ass / 'M'iss / 'E'rror and records the calls."""
    played = []

    def run(number):
        played.append(number)
        mark = pattern[number - 1]
        return Attempt(passed=mark == "P",
                       reason=None if mark != "M" else f"missed {number}",
                       error=f"boom {number}" if mark == "E" else None,
                       wall_seconds=float(number))
    return run, played


@pytest.mark.parametrize("pattern, ran", [
    ("PPPPP", 3),   # stops at k: a clean run costs 3, not 5
    ("MMMPP", 3),   # 0 of 3 -> can no longer reach 3, stop
    ("PMPMP", 5),   # needs every attempt
    ("PPMMM", 5),
    ("MPPMP", 5),
    ("PMMPP", 5),
])
def test_run_k_of_n_stops_as_soon_as_settled(pattern, ran):
    run, played = _scripted(pattern)
    attempts = run_k_of_n(FLAKY, run)
    assert played == list(range(1, ran + 1))
    assert len(attempts) == ran


def test_a_solid_case_runs_once():
    run, played = _scripted("MPPPP")
    assert len(run_k_of_n(SOLID, run)) == 1 and played == [1]


def _verdict(case, pattern):
    run, _ = _scripted(pattern)
    return aggregate(case, run_k_of_n(case, run))


def test_aggregate_three_of_five_passes():
    result = _verdict(FLAKY, "PMPMP")
    assert (result.verdict, result.attempts, result.passes, result.required) == (PASS, 5, 3, 3)
    assert result.detail == "a2: missed 2; a4: missed 4"


def test_aggregate_two_of_five_fails():
    result = _verdict(FLAKY, "PPMMM")
    assert (result.verdict, result.passes) == (FAIL, 2)


def test_a_clean_pass_has_no_detail():
    result = _verdict(FLAKY, "PPPPP")
    assert (result.verdict, result.attempts, result.detail) == (PASS, 3, None)


def test_a_missed_boundary_case_is_known_boundary_not_fail():
    boundary = MemoryCase("unknown", "q", must_pass=False)
    assert _verdict(boundary, "M").verdict == BOUNDARY
    assert _verdict(boundary, "P").verdict == PASS


def test_every_attempt_erroring_is_error():
    result = _verdict(FLAKY, "EEEEE")
    assert result.verdict == ERROR and result.attempts == 3
    assert result.error == "boom 1"
    assert result.detail.startswith("a1: ERROR boom 1")


def test_an_errored_attempt_counts_as_a_miss_not_as_error():
    result = _verdict(FLAKY, "PEPMP")
    assert (result.verdict, result.passes, result.attempts) == (PASS, 3, 5)
    assert "a2: ERROR boom 2" in result.detail and "a4: missed 4" in result.detail


def test_solid_case_counters_are_one_of_one():
    result = _verdict(SOLID, "P")
    assert (result.attempts, result.passes, result.required) == (1, 1, 1)


def test_metrics_fold_median_max_and_last():
    attempts = [
        Attempt(passed=True, wall_seconds=2.0,
                metrics={"model_seconds": 1.0, "load_seconds": 2.6, "prompt_tokens": 400,
                         "output_tokens": 10, "vram_models": {"llama3.1:latest": 5211},
                         "vram_gpu_mib": 7000}),
        Attempt(passed=True, wall_seconds=4.0,
                metrics={"model_seconds": None, "load_seconds": 0.1, "prompt_tokens": 500,
                         "output_tokens": 30, "vram_models": None, "vram_gpu_mib": 7700}),
        Attempt(passed=True, wall_seconds=3.0,
                metrics={"model_seconds": 2.0, "load_seconds": 0.2, "prompt_tokens": 600,
                         "output_tokens": 20, "vram_models": None, "vram_gpu_mib": None}),
    ]
    result = aggregate(FLAKY, attempts)
    assert result.wall_seconds == 3.0                     # median
    assert result.model_seconds == pytest.approx(1.5)     # median, null skipped
    assert result.load_seconds == 2.6                     # max: the cold load stays visible
    assert (result.prompt_tokens, result.output_tokens) == (500, 20)
    assert result.vram_gpu_mib == 7700                    # max
    assert result.vram_models == {"llama3.1:latest": 5211}  # last non-null reading


def test_all_null_metrics_stay_null():
    result = aggregate(SOLID, [Attempt(passed=True, wall_seconds=1.0, metrics={})])
    assert result.model_seconds is None and result.vram_models is None
    assert result.prompt_tokens is None


# --------------------------------------------------------------------------
# could-not-run path
# --------------------------------------------------------------------------

def test_no_ollama_is_exit_2_with_an_error_record(monkeypatch, tmp_path):
    import json

    import scripts.check_routing as routing
    monkeypatch.setattr(routing, "ollama_up", lambda: False)
    out = tmp_path / "memory.json"
    assert cm.main(["--json", str(out)]) == 2
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["check"] == "memory" and data["cases"] == []
    assert "Ollama is not reachable" in data["error"]


# --------------------------------------------------------------------------
# the 23 Nov baseline (known boundaries)
# --------------------------------------------------------------------------

BASELINE_BOUNDARIES = {"pet", "two-facts", "paraphrase", "carry-over"}


def test_must_pass_set_is_the_baseline():
    """Changing which cases must pass is a decision, not a drive-by: this pins it.
    Two layers: ranking grades retrieve -> answer, born grades retrieval only.
    Feature 5 flips the four baseline boundaries back to must_pass=True."""
    must_pass = {c.id for c in CASES if c.must_pass}
    assert must_pass == {"ranking", "born"}
    assert {c.id for c in CASES if c.retrieval_only} == {"born"}
    assert {c.id for c in CASES if not c.must_pass} == BASELINE_BOUNDARIES | {"unknown"}


@pytest.mark.parametrize("case_id", sorted(BASELINE_BOUNDARIES))
def test_baseline_boundaries_say_why_they_miss(case_id):
    case = next(c for c in CASES if c.id == case_id)
    assert case.known_gap and "\n" not in case.known_gap


def test_a_boundary_miss_carries_its_reason_into_detail():
    case = MemoryCase("gap", "q", must_pass=False, known_gap="routes to Wikipedia")
    run, _ = _scripted("M")
    result = aggregate(case, run_k_of_n(case, run))
    assert result.verdict == BOUNDARY
    assert result.detail == (f"known boundary ({cm.BASELINE_DATE} baseline): "
                             f"routes to Wikipedia; a1: missed 1")


def test_a_boundary_case_that_passes_has_no_gap_note():
    """If Feature 5 work makes it pass, the record must not still claim the gap."""
    case = MemoryCase("gap", "q", must_pass=False, known_gap="routes to Wikipedia")
    run, _ = _scripted("P")
    assert aggregate(case, run_k_of_n(case, run)).detail is None


# --------------------------------------------------------------------------
# retrieval-only (born)
# --------------------------------------------------------------------------

def test_a_retrieval_only_case_grades_no_answer_phrase():
    """If it graded the answer it would not be retrieval-only, and the label would lie."""
    for case in CASES:
        if case.retrieval_only:
            assert case.require_tool and case.tool_returned
            assert not (case.answer_has or case.answer_any or case.answer_lacks), case.id


def test_born_passes_on_retrieval_even_when_the_answer_denies_it():
    """The observed 23 Nov behaviour: right snippet, denying answer. Retrieval held."""
    born = next(c for c in CASES if c.id == "born")
    denial = "However, I don't have any information about your birth year."
    assert grade(born, [RECALL], [(RECALL, "I was born in 1998.")], denial) is None


def test_born_fails_when_retrieval_breaks():
    born = next(c for c in CASES if c.id == "born")
    assert grade(born, [], [], "1998") == ("recall_user_memory not called (called: nothing); "
                                           "tool did not return '1998'")
    assert grade(born, [RECALL], [(RECALL, "I drive a Maruti Swift.")], "x") == (
        "tool did not return '1998'")


def test_a_retrieval_only_record_says_so_even_when_green():
    """So a green born is never read as 'the answer used the fact'."""
    case = MemoryCase("born", "q", require_tool=RECALL, tool_returned=("1998",),
                      retrieval_only=True)
    run, _ = _scripted("P")
    result = aggregate(case, run_k_of_n(case, run))
    assert (result.verdict, result.detail) == (PASS, cm.RETRIEVAL_ONLY_NOTE)

    run, _ = _scripted("M")
    result = aggregate(case, run_k_of_n(case, run))
    assert result.verdict == FAIL
    assert result.detail == f"{cm.RETRIEVAL_ONLY_NOTE}; a1: missed 1"
