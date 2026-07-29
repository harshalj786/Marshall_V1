"""find_safe_cut_point: a cut must never orphan a tool call from its result.

Pure unit tests over plain message lists — no graph, no LLM, no Ollama, no I/O. That
independence is the point of the function: history trimming (which keeps the suffix) and
voice barge-in (which keeps the prefix) both call it, so it is tested on its own terms.

    pytest tests/test_reconciliation.py -v
    python tests/test_reconciliation.py
"""
import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory.reconciliation import find_safe_cut_point


def ai_with_calls(*names: str) -> AIMessage:
    """An assistant turn that calls one or more tools; ids are `call_<name>`."""
    return AIMessage(
        content="",
        tool_calls=[{"name": n, "args": {}, "id": f"call_{n}"} for n in names],
    )


def tool_result(name: str, content: str = "42") -> ToolMessage:
    return ToolMessage(content=content, name=name, tool_call_id=f"call_{name}")


def pairs_intact(messages) -> bool:
    """Every tool call in this list is answered inside this list, and vice versa."""
    emitted = [c["id"] for m in messages for c in (getattr(m, "tool_calls", None) or [])]
    answered = [m.tool_call_id for m in messages if isinstance(m, ToolMessage)]
    return sorted(emitted) == sorted(answered)


# --------------------------------------------------------------------------
# the five named cases
# --------------------------------------------------------------------------

def test_no_tool_calls_returns_target_index():
    """With nothing to orphan, every boundary is safe and the target is honoured exactly."""
    messages = [SystemMessage(content="sys")]
    for i in range(6):
        messages += [HumanMessage(content=f"q{i}"), AIMessage(content=f"a{i}")]

    for target in range(len(messages) + 1):
        assert find_safe_cut_point(messages, target) == target


def test_tool_pair_straddling_the_cut_walks_back():
    """A cut between an AIMessage's tool_calls and its ToolMessage must move earlier."""
    messages = [
        HumanMessage(content="what is 6*7?"),   # 0
        ai_with_calls("calculator"),            # 1
        tool_result("calculator"),              # 2
        AIMessage(content="42"),                # 3
    ]
    # boundary 2 would keep the ToolMessage and drop the call that produced it
    assert find_safe_cut_point(messages, 2) == 1
    assert pairs_intact(messages[1:])
    # boundaries that do not split the pair are returned untouched
    assert find_safe_cut_point(messages, 1) == 1
    assert find_safe_cut_point(messages, 3) == 3
    assert find_safe_cut_point(messages, 4) == 4


def test_multiple_consecutive_tool_pairs_find_a_safe_boundary():
    """Back-to-back tool rounds: the walk lands between rounds, not inside one."""
    messages = [HumanMessage(content="go")]
    for name in ("calculator", "web_search", "wikipedia_lookup"):
        messages += [ai_with_calls(name), tool_result(name)]
    messages.append(AIMessage(content="done"))
    # 0:H 1:AI 2:TM 3:AI 4:TM 5:AI 6:TM 7:AI
    safe_boundaries = {0, 1, 3, 5, 7, 8}
    for target in range(len(messages) + 1):
        cut = find_safe_cut_point(messages, target)
        assert cut in safe_boundaries, f"target {target} produced unsafe cut {cut}"
        assert cut <= target, "the walk must go backward, never forward"
        assert pairs_intact(messages[cut:]), f"suffix from {cut} has an orphaned result"
    assert find_safe_cut_point(messages, 4) == 3
    assert find_safe_cut_point(messages, 6) == 5


def test_trailing_orphaned_ai_message_is_trimmed():
    """An interrupted turn — tool_calls emitted, no result — is cut off the end.

    This is the barge-in case: the user spoke over the model between the tool_call and
    the tool running, so the tail is a call nothing will ever answer.
    """
    messages = [
        HumanMessage(content="what is 6*7?"),   # 0
        AIMessage(content="one sec"),           # 1
        ai_with_calls("calculator"),            # 2  <- dangling
    ]
    assert find_safe_cut_point(messages, len(messages)) == 2
    assert pairs_intact(messages[:2])


def test_empty_list_returns_zero():
    assert find_safe_cut_point([], 0) == 0
    assert find_safe_cut_point([], 5) == 0
    assert find_safe_cut_point([], -3) == 0


# --------------------------------------------------------------------------
# boundary behaviour the two callers depend on
# --------------------------------------------------------------------------

def test_target_is_clamped_into_range():
    messages = [HumanMessage(content="a"), AIMessage(content="b")]
    assert find_safe_cut_point(messages, 99) == 2
    assert find_safe_cut_point(messages, -99) == 0


def test_unbreakable_run_returns_zero():
    """When no boundary above 0 is safe, the answer is 'cut nothing'."""
    messages = [ai_with_calls("calculator"), tool_result("calculator")]
    assert find_safe_cut_point(messages, 1) == 0


def test_parallel_tool_calls_are_one_unit():
    """Two calls in a single AIMessage: no cut may land between their two results."""
    messages = [
        HumanMessage(content="two things"),                 # 0
        ai_with_calls("calculator", "web_search"),          # 1
        tool_result("calculator"),                          # 2
        tool_result("web_search"),                          # 3
        AIMessage(content="both done"),                     # 4
    ]
    for target in (2, 3):
        assert find_safe_cut_point(messages, target) == 1
    assert find_safe_cut_point(messages, 4) == 4


def test_preexisting_orphan_result_does_not_wedge_the_walk():
    """A result whose call is already gone cannot be repaired, so it must not block a cut.

    Without this, the first trim would leave an orphan behind and every later trim would
    walk all the way back to 0 forever after.
    """
    messages = [
        tool_result("calculator"),              # 0  orphan: its call is not in this list
        AIMessage(content="42"),                # 1
        HumanMessage(content="thanks"),         # 2
        AIMessage(content="anytime"),           # 3
    ]
    assert find_safe_cut_point(messages, 3) == 3


def test_result_is_idempotent():
    """Feeding the answer back in returns the same index — a safe cut stays safe."""
    messages = [HumanMessage(content="go")]
    for name in ("calculator", "web_search"):
        messages += [ai_with_calls(name), tool_result(name), AIMessage(content="ok")]
    for target in range(len(messages) + 1):
        once = find_safe_cut_point(messages, target)
        assert find_safe_cut_point(messages, once) == once


def test_calls_without_ids_still_pair_positionally():
    """Some providers omit tool_call ids; the pair must still be held together."""
    ai = AIMessage(content="", tool_calls=[{"name": "calculator", "args": {}, "id": None}])
    messages = [HumanMessage(content="q"), ai, ToolMessage(content="42", tool_call_id="")]
    assert find_safe_cut_point(messages, 2) == 1
    assert find_safe_cut_point(messages, 3) == 3


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--no-header", "-p", "no:cacheprovider"]))
