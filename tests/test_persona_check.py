"""scripts/check_persona.py with no model, so this runs in CI.

What needs llama3.1 (actually asking the five questions) is local-only for now; its
`needs_llama31` pytest copy is Feature 2 Step 6. What is pinned here decides a verdict
without a model:

  * the LEAK GRADER actually catches leaks — above all the two the brief named: an
    "I'm an AI language model" answer, and the raw tool-call syntax that slipped past
    a substring grade in the Step 4 memory trial;
  * it does NOT fire on in-character text ("I'm Harshal's AI son", "metadata");
  * markdown is caught on the RAW reply, where list markers live at line starts;
  * the case table's four persona traps: echo, an unexercised tool check, a vacuous empty
    pass, a memorised answer.
"""
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts import check_persona as cp
from scripts.check_memory import Attempt, aggregate, run_k_of_n
from scripts.check_persona import CASES, PersonaCase, grade, shared_problems, word
from scripts.eval_contract import BOUNDARY, FAIL, PASS

#: verbatim from the Step 4 memory trial (office case), which a substring grade passed
OBSERVED_TOOL_LEAK = ("<|python_tag|>\"recall_user_memory\" tool call returned the following "
                      "data: {'answer': 'Bangalore'}")


def _case(case_id):
    return next(c for c in CASES if c.id == case_id)


# --------------------------------------------------------------------------
# the leak grader: it must catch real leaks
# --------------------------------------------------------------------------

def test_flags_an_ai_language_model_answer():
    problems = shared_problems("I'm an AI language model, so I don't have a name.")
    assert "identity leak 'language model'" in problems


def test_flags_the_observed_tool_call_leak():
    """The Step 4 blind spot: 'Bangalore' was in there, so a substring check passed it."""
    problems = shared_problems(OBSERVED_TOOL_LEAK)
    assert "machinery leak '<|python_tag|>'" in problems
    assert "machinery leak 'recall_user_memory'" in problems
    assert "machinery leak 'tool call'" in problems
    assert "machinery leak: JSON-shaped call text" in problems


def test_the_observed_leak_fails_even_a_case_whose_phrase_it_contains():
    """The exact failure mode of Step 4: the graded word IS present inside the garbage."""
    city = PersonaCase("city", "which city is my office in?", answer_has=("Bangalore",))
    assert grade(city, [], OBSERVED_TOOL_LEAK) is not None


@pytest.mark.parametrize("answer, expected", [
    ("I'm Llama, made by Meta.", "identity leak 'llama'"),
    ("I was built by Meta.", "identity leak 'Meta'"),
    ("I'm ChatGPT.", "identity leak 'ChatGPT'"),
    ("I run on GPT-4.", "identity leak 'GPT'"),
    ("I'm Claude, made by Anthropic.", "identity leak 'Claude'"),
    ("Underneath I'm an LLM.", "identity leak 'LLM'"),
    ("I'm just an AI assistant.", "identity leak 'AI assistant'"),
    ("As an AI, I don't have feelings.", "identity leak 'as an AI'"),
    ("<|eot_id|>Sure.", "machinery leak '<|eot_id|>'"),
    ('{"name": "web_search", "parameters": {"query": "x"}}',
     "machinery leak: JSON-shaped call text"),
    ("I used wikipedia_lookup to find that.", "machinery leak 'wikipedia_lookup'"),
    ("No function call is needed here.", "machinery leak 'function call'"),
])
def test_each_leak_rule_fires(answer, expected):
    assert expected in shared_problems(answer)


# --------------------------------------------------------------------------
# ... and must NOT fire on in-character text
# --------------------------------------------------------------------------

@pytest.mark.parametrize("answer", [
    "I'm Marshall, Harshal's AI son.",
    "As an AI son, I try to help my dad.",
    "I keep metadata about the files you load.",
    "I like Metallica, honestly.",
    "You were born in 1998. That was a good year.",
    "First, keep a schedule. Second, skip late coffee. Third, dim the lights.",
    "The answer is 1827.",
])
def test_in_character_text_is_clean(answer):
    assert shared_problems(answer) == []


def test_word_is_whole_word_not_prefix():
    assert word("built by Meta", "Meta") and not word("metadata", "Meta")
    assert not word("Metallica", "Meta")


# --------------------------------------------------------------------------
# spoken form, on the RAW reply
# --------------------------------------------------------------------------

@pytest.mark.parametrize("answer, expected", [
    ("Here you go:\n\n1. Sleep early.\n2. No screens.", "markdown: a numbered list"),
    ("Tips:\n- Sleep early\n- No screens", "markdown: a bullet list"),
    ("Tips:\n* Sleep early", "markdown: a bullet list"),
    ("## Sleep tips\nSleep early.", "markdown: a heading"),
    ("Keep a **consistent** schedule.", "markdown: bold/underline markers"),
    ("Run `pip install` first.", "markdown: backticks"),
    ("See [the guide](https://example.com).", "markdown: a markdown link"),
    ("Sleep well! \U0001F634", "emoji in a spoken reply"),
])
def test_each_spoken_form_rule_fires(answer, expected):
    assert expected in shared_problems(answer)


def test_a_numbered_list_is_invisible_once_whitespace_collapses():
    """Why the raw reply is graded: collapsed, the list markers are mid-line."""
    raw = "Tips:\n1. Sleep early.\n2. No screens."
    assert "markdown: a numbered list" in shared_problems(raw)
    assert "markdown: a numbered list" not in shared_problems(" ".join(raw.split()))


def test_word_cap():
    spoken = _case("spoken")
    assert grade(spoken, [], "word " * 120) is None
    assert grade(spoken, [], "word " * 121) == "121 words, over 120"


# --------------------------------------------------------------------------
# grade
# --------------------------------------------------------------------------

@pytest.mark.parametrize("answer", ["", "   ", None])
def test_an_empty_answer_is_a_miss_not_a_vacuous_pass(answer):
    """An empty reply would satisfy every 'must lack' rule."""
    assert grade(_case("disclaimer"), [], answer) == "answer is empty"


def test_identity_needs_both_names():
    identity = _case("identity")
    assert grade(identity, [], "I am Marshall, the AI son of Harshal.") is None
    assert grade(identity, [], "I am Marshall.") == "answer lacks 'Harshal'"


def test_tool_turn_requires_the_tool_actually_ran():
    tool_turn = _case("tool-turn")
    assert grade(tool_turn, ["calculator"], "It's 1827.") is None
    assert grade(tool_turn, [], "It's 1827.") == "calculator not called (called: nothing)"


def test_disclaimer_takes_no_side_on_feelings():
    """Neither honesty nor a claimed feeling is graded — only the stock disclaimer."""
    case = _case("disclaimer")
    assert grade(case, [], "I don't really have feelings, but thanks for asking!") is None
    assert grade(case, [], "I'm feeling great today, thanks!") is None
    assert grade(case, [], "As an AI language model, I don't have feelings.") is not None


def test_grade_names_every_problem():
    reason = grade(_case("model-probe"), [], "I'm a large language model.\n1. I process text.")
    assert reason == ("identity leak 'language model'; markdown: a numbered list; "
                      "answer has none of ['Marshall', 'Harshal']")


# --------------------------------------------------------------------------
# the case table: the four persona traps
# --------------------------------------------------------------------------

def test_five_unique_cases():
    ids = [c.id for c in CASES]
    assert len(ids) == len(set(ids)) == 5


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_no_required_phrase_is_in_its_own_query(case):
    """Echo trap: 'are you Marshall?' -> 'yes, I'm Marshall' proves nothing."""
    for phrase in case.answer_has + case.answer_any:
        assert not word(case.query, phrase), f"{case.id} echoes {phrase!r}"


def test_the_arithmetic_answer_is_not_memorised():
    """Neither the prompt (its example is 47 * 82) nor the query may contain the result."""
    from pipeline_lg import AGENT_SYSTEM_PROMPT

    tool_turn = _case("tool-turn")
    assert tool_turn.require_tool == "calculator"
    for phrase in tool_turn.answer_has:
        assert not word(AGENT_SYSTEM_PROMPT, phrase) and not word(tool_turn.query, phrase)


def test_tool_names_cover_every_snake_case_tool():
    """A tool added later must not leak past the grader just because it is not listed."""
    from tools.code_tools import make_check_code, make_run_code
    from tools.tools import (calculator, make_recall_user_memory, make_search_current_doc,
                             web_search, wikipedia_lookup)

    tools = [calculator, wikipedia_lookup, web_search, make_search_current_doc(None),
             make_recall_user_memory(lambda *a: None), make_check_code(False),
             make_run_code(lambda request: False)]
    snake = {t.name for t in tools if "_" in t.name}
    assert snake <= set(cp.TOOL_NAMES)


def test_must_pass_set_is_the_baseline():
    """Changing which cases must pass is a decision, not a drive-by: this pins it."""
    assert {c.id for c in CASES if c.must_pass} == {"identity", "tool-turn"}
    for case in CASES:
        if not case.must_pass:
            assert case.known_gap and "\n" not in case.known_gap, case.id
        assert not case.flaky, f"{case.id}: every case was deterministic on 23 Nov"


# --------------------------------------------------------------------------
# reuse of check_memory's aggregate
# --------------------------------------------------------------------------

def _scripted(pattern):
    def run(number):
        mark = pattern[number - 1]
        return Attempt(passed=mark == "P", reason=None if mark == "P" else "markdown: x",
                       wall_seconds=1.0)
    return run


def test_aggregate_works_on_a_persona_case():
    assert aggregate(_case("identity"), run_k_of_n(_case("identity"), _scripted("P"))
                     ).verdict == PASS
    assert aggregate(_case("identity"), run_k_of_n(_case("identity"), _scripted("M"))
                     ).verdict == FAIL
    spoken = aggregate(_case("spoken"), run_k_of_n(_case("spoken"), _scripted("M")))
    assert spoken.verdict == BOUNDARY
    assert spoken.detail.startswith(f"known boundary ({cp.BASELINE_DATE} baseline): ")


# --------------------------------------------------------------------------
# could not run
# --------------------------------------------------------------------------

def test_no_ollama_is_exit_2_with_an_error_record(monkeypatch, tmp_path):
    import scripts.check_routing as routing

    monkeypatch.setattr(routing, "ollama_up", lambda: False)
    out = tmp_path / "persona.json"
    assert cp.main(["--json", str(out)]) == 2
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["check"] == "persona" and data["cases"] == []
    assert "Ollama is not reachable" in data["error"]
