"""H2: AgentGraph's message list must stop growing without bound.

Covers the `summarize_and_trim` cycle — the gate's cadence, the replacement of the message
list, the tool-pair boundary, the failure contract (`{}`, retry next turn), and the
summary's survival across sessions.

No Ollama needed: the graph is built for real (nodes, edges, checkpointer, ToolNode with
the real `calculator`) and only the two model handles are replaced — `agent.llm` for the
ReAct loop and `agent.summary_llm` for the compressor. Everything the tests assert is
graph behaviour, so faking the model is what makes the behaviour observable.

    pytest tests/test_summarization_cycle.py -v
    python tests/test_summarization_cycle.py
"""
import inspect
import json
import shutil
import sys
import tempfile
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline_lg import (
    SUMMARY_FILE_NAME,
    SUMMARY_OPEN_TAG,
    AgentGraph,
    is_summary_message,
)

DOC = "Marshall is a local RAG framework. It loads documents and answers questions about them.\n"


# --------------------------------------------------------------------------
# fakes
# --------------------------------------------------------------------------

class PlainChat:
    """Tool-bound LLM stand-in: one plain reply per call, so a turn is exactly 2 messages."""

    def __init__(self, reply="ok"):
        self.reply = reply
        self.seen: list[list] = []

    def invoke(self, messages, **kwargs):
        self.seen.append(list(messages))
        return AIMessage(content=self.reply)


class ToolCallingChat:
    """Calls `calculator` on every user turn, then answers — 4 messages per turn."""

    def __init__(self):
        self.calls = 0

    def invoke(self, messages, **kwargs):
        if isinstance(messages[-1], ToolMessage):
            return AIMessage(content="42")
        self.calls += 1
        return AIMessage(content="", tool_calls=[{
            "name": "calculator",
            "args": {"expression": "6*7"},
            "id": f"call_{self.calls}",
        }])


class CountingSummarizer:
    """Returns SUMMARY_1, SUMMARY_2, ... and records the prompt it was given."""

    def __init__(self):
        self.calls = 0
        self.prompts: list[str] = []

    def invoke(self, messages, **kwargs):
        self.calls += 1
        self.prompts.append(messages[0].content)
        return AIMessage(content=f"SUMMARY_{self.calls}")


class BoomSummarizer:
    def invoke(self, messages, **kwargs):
        raise RuntimeError("ollama connection refused")


class BlankSummarizer:
    def invoke(self, messages, **kwargs):
        return AIMessage(content="   ")


# --------------------------------------------------------------------------
# fixtures — construction costs ~8s (embeddings + Chroma), so build as few as possible
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tmp_root():
    tmp = Path(tempfile.mkdtemp(prefix="marshall_h2_"))
    (tmp / "doc.txt").write_text(DOC, encoding="utf-8")
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


def build_agent(tmp_root: Path, name: str, **kwargs) -> AgentGraph:
    kwargs.setdefault("summary_persist_path", str(tmp_root / "summaries"))
    return AgentGraph(
        file_path=str(tmp_root / "doc.txt"),
        mode_docs="txt",
        mode_retreiver="simple",
        persist_directory_path=str(tmp_root / f"vdb_{name}"),
        user_history_directory_path=str(tmp_root / "mem"),
        memory_every_n_turns=0,          # isolate the trim cycle from the long-term-memory gate
        **kwargs,
    )


@pytest.fixture(scope="module")
def agent(tmp_root):
    """One shared graph; tests keep their state apart with their own thread_id."""
    return build_agent(tmp_root, "main")


def run_turns(agent: AgentGraph, thread_id: str, n: int, start: int = 1) -> None:
    cfg = {"configurable": {"thread_id": thread_id}}
    for i in range(start, start + n):
        agent.invoke(f"question number {i}", config=cfg)


def summary_messages(messages) -> list:
    return [m for m in messages if is_summary_message(m)]


def pairs_intact(messages) -> bool:
    emitted = [c["id"] for m in messages for c in (getattr(m, "tool_calls", None) or [])]
    answered = [m.tool_call_id for m in messages if isinstance(m, ToolMessage)]
    return sorted(emitted) == sorted(answered)


# --------------------------------------------------------------------------
# the gate's cadence
# --------------------------------------------------------------------------

def test_seven_turns_accumulate_without_triggering(agent, monkeypatch):
    """Both conditions must hold. At turn 7 the list is exactly 14 — not yet longer."""
    summarizer = CountingSummarizer()
    monkeypatch.setattr(agent, "llm", PlainChat())
    monkeypatch.setattr(agent, "summary_llm", summarizer)

    run_turns(agent, "cadence", 7)

    assert summarizer.calls == 0, "the cycle fired before the list outgrew the keep window"
    assert agent.get_running_summary("cadence") == ""
    assert summary_messages(agent.get_history("cadence")) == []
    assert len(agent.get_history("cadence")) == 15      # system + 7 turns x 2
    assert agent.get_summary_turns_since("cadence") == 7


def test_cycle_fires_once_both_conditions_hold(agent, monkeypatch):
    """Turn 8: 16 messages at the gate and 8 turns since the last compression."""
    summarizer = CountingSummarizer()
    monkeypatch.setattr(agent, "llm", PlainChat())
    monkeypatch.setattr(agent, "summary_llm", summarizer)

    run_turns(agent, "fires", 8)

    assert summarizer.calls == 1
    assert agent.get_running_summary("fires") == "SUMMARY_1"
    assert agent.get_summary_turns_since("fires") == 0, "the counter must reset on success"


def test_gate_is_disabled_by_zero(agent):
    """0 disables, mirroring memory_every_n_turns. The guard sits before the comparison."""
    bare = AgentGraph.__new__(AgentGraph)
    bare.summarize_every_n_turns = 0
    bare.summary_keep_recent = 14
    bare.memory_every_n_turns = 0
    state = {"messages": [HumanMessage(content="x")] * 40, "summary_turns_since": 99}
    assert bare._should_summarize(state) == "agent"

    bare.summarize_every_n_turns = -3
    assert bare._should_summarize(state) == "agent"


def test_trim_wins_when_both_gates_are_due(agent):
    """One router owns both decisions, so they can never fan out in parallel."""
    bare = AgentGraph.__new__(AgentGraph)
    bare.summarize_every_n_turns = 7
    bare.summary_keep_recent = 14
    bare.memory_every_n_turns = 5
    state = {
        "messages": [HumanMessage(content="x")] * 40,
        "summary_turns_since": 7,
        "turn_count": 35,                                  # divisible by 5: memory gate is due too
    }
    assert bare._should_summarize(state) == "summarize_and_trim"
    assert bare._memory_gate(state) == "summarize", (
        "the memory gate must still be due — summarize_and_trim routes into it afterwards")


# --------------------------------------------------------------------------
# what the trim leaves behind
# --------------------------------------------------------------------------

def test_summary_leads_the_list_and_is_a_human_message(agent, monkeypatch):
    monkeypatch.setattr(agent, "llm", PlainChat())
    monkeypatch.setattr(agent, "summary_llm", CountingSummarizer())

    run_turns(agent, "shape", 8)
    messages = agent.get_history("shape")

    assert isinstance(messages[0], SystemMessage), "the agent's own prompt must survive the trim"
    assert is_summary_message(messages[1]), "the summary must lead the conversation"
    assert isinstance(messages[1], HumanMessage), (
        "H1: LLM-generated text derived from untrusted turns must not hold system authority")
    assert SUMMARY_OPEN_TAG in messages[1].content
    assert "SUMMARY_1" in messages[1].content
    assert not any(isinstance(m, SystemMessage) for m in messages[1:])


def test_list_is_replaced_not_appended(agent, monkeypatch):
    """After a cycle the list is exactly: system + summary + the keep window."""
    monkeypatch.setattr(agent, "llm", PlainChat())
    monkeypatch.setattr(agent, "summary_llm", CountingSummarizer())

    run_turns(agent, "replace", 8)
    # the trim runs before the agent replies, so the reply sits on top of the 16 kept
    assert len(agent.get_history("replace")) == 1 + 1 + agent.summary_keep_recent + 1


def test_growth_is_bounded_over_many_cycles(agent, monkeypatch):
    """The point of the exercise: length oscillates instead of climbing."""
    monkeypatch.setattr(agent, "llm", PlainChat())
    monkeypatch.setattr(agent, "summary_llm", CountingSummarizer())

    run_turns(agent, "bounded", 40)
    messages = agent.get_history("bounded")

    # unbounded would be 1 + 80; the ceiling here is one full inter-cycle interval on top
    # of the keep window
    assert len(messages) <= 2 + agent.summary_keep_recent + 2 * agent.summarize_every_n_turns
    assert len(summary_messages(messages)) == 1


def test_second_cycle_updates_the_summary_rather_than_appending_one(agent, monkeypatch):
    """Turn 15 is the next cycle (turn 8 reset the counter), and it must replace, not stack."""
    summarizer = CountingSummarizer()
    monkeypatch.setattr(agent, "llm", PlainChat())
    monkeypatch.setattr(agent, "summary_llm", summarizer)

    run_turns(agent, "update", 15)
    messages = agent.get_history("update")

    assert summarizer.calls == 2
    assert len(summary_messages(messages)) == 1, "a second summary block was appended"
    assert "SUMMARY_2" in messages[1].content
    assert "SUMMARY_1" not in messages[1].content
    assert agent.get_running_summary("update") == "SUMMARY_2"
    # the previous summary reaches the compressor as previous_summary, and only there —
    # transcribing it as well would nest one summary inside the next
    second = summarizer.prompts[1]
    assert "<previous_summary>\nSUMMARY_1" in second
    assert second.count("SUMMARY_1") == 1


def test_compressor_sees_the_dropped_turns_and_not_the_system_prompt(agent, monkeypatch):
    summarizer = CountingSummarizer()
    monkeypatch.setattr(agent, "llm", PlainChat("an answer"))
    monkeypatch.setattr(agent, "summary_llm", summarizer)

    run_turns(agent, "transcript", 8)
    prompt = summarizer.prompts[0]

    assert "question number 1" in prompt, "the dropped turn never reached the compressor"
    assert "You are Marshall" not in prompt, "the system prompt is ours, not conversation"


# --------------------------------------------------------------------------
# tool pairs at the boundary
# --------------------------------------------------------------------------

def test_tool_pair_at_the_trim_boundary_is_not_orphaned(agent, monkeypatch):
    """The naive cut lands between a tool_call and its result; the node must walk back."""
    monkeypatch.setattr(agent, "summary_llm", CountingSummarizer())
    monkeypatch.setattr(agent, "summary_keep_recent", 6)

    messages = [SystemMessage(content="sys"), HumanMessage(content="q0"), AIMessage(content="a0")]
    for i in range(3):
        messages += [
            HumanMessage(content=f"q{i + 1}"),
            AIMessage(content="", tool_calls=[
                {"name": "calculator", "args": {"expression": "1+1"}, "id": f"call_{i}"}]),
            ToolMessage(content="2", name="calculator", tool_call_id=f"call_{i}"),
            AIMessage(content="2"),
        ]
    # 15 messages, keep 6 -> the naive boundary is 9, the ToolMessage of the second round
    assert isinstance(messages[9], ToolMessage), "test setup no longer exercises the walk-back"

    out = agent._summarize_and_trim_node(
        {"messages": messages, "running_summary": ""},
        {"configurable": {"thread_id": "toolcut"}},
    )
    replacement = out["messages"][1:]          # drop the RemoveMessage sentinel

    assert pairs_intact(replacement), "a tool result was left without its call"
    assert not isinstance(replacement[2], ToolMessage), (
        "the kept window starts on an orphaned tool result")
    assert len(replacement) == 2 + 7, (
        "the walk-back should keep one message more than asked, not fewer")


def test_tool_loop_survives_repeated_cycles(agent, monkeypatch):
    """End to end with the real ToolNode: 4 messages a turn, several cycles, pairs intact."""
    monkeypatch.setattr(agent, "llm", ToolCallingChat())
    monkeypatch.setattr(agent, "summary_llm", CountingSummarizer())
    # 11, not the default 14: with 4-message turns that is the window whose first naive
    # boundary lands on a ToolMessage, so the loop really does exercise the walk-back
    monkeypatch.setattr(agent, "summary_keep_recent", 11)

    run_turns(agent, "toolloop", 20)
    messages = agent.get_history("toolloop")

    assert pairs_intact(messages)
    assert len(summary_messages(messages)) == 1
    assert not isinstance(messages[2], ToolMessage)
    assert any(isinstance(m, ToolMessage) for m in messages), "the tool loop never ran"


# --------------------------------------------------------------------------
# failure contract — H4's pattern
# --------------------------------------------------------------------------

def test_llm_failure_leaves_the_messages_alone(agent, monkeypatch):
    monkeypatch.setattr(agent, "llm", PlainChat())
    monkeypatch.setattr(agent, "summary_llm", BoomSummarizer())

    run_turns(agent, "boom", 8)
    messages = agent.get_history("boom")

    assert summary_messages(messages) == []
    assert agent.get_running_summary("boom") == ""
    assert len(messages) == 1 + 8 * 2, "the failed cycle ate messages it could not summarise"
    assert agent.get_summary_turns_since("boom") == 8, (
        "the counter must keep climbing so the next turn retries")


def test_failed_cycle_retries_on_the_next_turn(agent, monkeypatch):
    monkeypatch.setattr(agent, "llm", PlainChat())
    monkeypatch.setattr(agent, "summary_llm", BoomSummarizer())
    run_turns(agent, "retry", 8)
    assert agent.get_running_summary("retry") == ""

    monkeypatch.setattr(agent, "summary_llm", CountingSummarizer())
    run_turns(agent, "retry", 1, start=9)

    assert agent.get_running_summary("retry") == "SUMMARY_1"
    assert agent.get_summary_turns_since("retry") == 0


def test_the_turn_still_answers_when_the_cycle_fails(agent, monkeypatch):
    monkeypatch.setattr(agent, "llm", PlainChat("here is your answer"))
    monkeypatch.setattr(agent, "summary_llm", BoomSummarizer())

    run_turns(agent, "answer", 7)
    answer = agent.invoke("question number 8", config={"configurable": {"thread_id": "answer"}})

    assert answer == "here is your answer", "a summariser failure cost the user their turn"


def test_blank_summary_is_refused(agent, monkeypatch):
    """An empty completion would trade real turns for nothing at all."""
    monkeypatch.setattr(agent, "llm", PlainChat())
    monkeypatch.setattr(agent, "summary_llm", BlankSummarizer())

    run_turns(agent, "blank", 8)

    assert summary_messages(agent.get_history("blank")) == []
    assert len(agent.get_history("blank")) == 1 + 8 * 2


def test_unwritable_persist_path_does_not_raise(agent, tmp_root):
    """Durability can fail; the turn cannot. The summary is already in state by then."""
    blocker = tmp_root / "not_a_directory"
    blocker.write_text("i am a file", encoding="utf-8")
    bare = AgentGraph.__new__(AgentGraph)
    bare.summary_persist_path = str(blocker)

    bare._persist_summary("t", "still fine")               # must not raise
    assert bare.load_persisted_summary("t") == ""


# --------------------------------------------------------------------------
# across sessions
# --------------------------------------------------------------------------

def test_summary_is_written_to_disk(agent, monkeypatch, tmp_root):
    monkeypatch.setattr(agent, "llm", PlainChat())
    monkeypatch.setattr(agent, "summary_llm", CountingSummarizer())

    run_turns(agent, "ondisk", 8)

    path = tmp_root / "summaries" / SUMMARY_FILE_NAME
    assert path.exists(), "nothing was persisted"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["threads"]["ondisk"]["summary"] == "SUMMARY_1"
    assert data["threads"]["ondisk"]["updated_at"]
    assert agent.load_persisted_summary("ondisk") == "SUMMARY_1"
    assert agent.load_persisted_summary("a-thread-that-never-ran") == ""


def test_a_new_session_loads_the_persisted_summary(agent, monkeypatch, tmp_root):
    """MemorySaver dies with the process; the file is all the next session inherits."""
    monkeypatch.setattr(agent, "llm", PlainChat())
    monkeypatch.setattr(agent, "summary_llm", CountingSummarizer())
    run_turns(agent, "session", 8)
    assert agent.get_running_summary("session") == "SUMMARY_1"

    revived = build_agent(tmp_root, "revived")             # fresh graph, fresh checkpointer
    monkeypatch.setattr(revived, "llm", PlainChat("second session"))
    assert revived.get_history("session") == [], "the new session started with old state"

    answer = revived.invoke("what were we doing?", config={"configurable": {"thread_id": "session"}})
    messages = revived.get_history("session")

    assert answer == "second session"
    assert revived.get_running_summary("session") == "SUMMARY_1"
    assert isinstance(messages[0], SystemMessage)
    assert is_summary_message(messages[1]) and "SUMMARY_1" in messages[1].content

    # and an unknown thread in the same file simply starts fresh — no error
    revived.invoke("hello", config={"configurable": {"thread_id": "brand-new"}})
    assert summary_messages(revived.get_history("brand-new")) == []


def test_missing_file_starts_fresh(tmp_root, monkeypatch):
    bare = AgentGraph.__new__(AgentGraph)
    bare.summary_persist_path = str(tmp_root / "nothing_here")
    assert bare.load_persisted_summary("default") == ""

    bare.summary_persist_path = None                        # persistence off entirely
    assert bare.load_persisted_summary("default") == ""
    bare._persist_summary("default", "dropped on the floor")


def test_corrupt_file_starts_fresh_and_is_replaced(tmp_root):
    directory = tmp_root / "corrupt"
    directory.mkdir(exist_ok=True)
    (directory / SUMMARY_FILE_NAME).write_text("{not json at all", encoding="utf-8")
    bare = AgentGraph.__new__(AgentGraph)
    bare.summary_persist_path = str(directory)

    assert bare.load_persisted_summary("default") == "", "a corrupt file must not stop a session"
    bare._persist_summary("default", "written over the wreckage")
    assert bare.load_persisted_summary("default") == "written over the wreckage"


def test_threads_do_not_share_a_summary(tmp_root):
    directory = tmp_root / "multi"
    bare = AgentGraph.__new__(AgentGraph)
    bare.summary_persist_path = str(directory)

    bare._persist_summary("alpha", "alpha's history")
    bare._persist_summary("beta", "beta's history")

    assert bare.load_persisted_summary("alpha") == "alpha's history"
    assert bare.load_persisted_summary("beta") == "beta's history", (
        "writing one thread's summary erased another's")


# --------------------------------------------------------------------------
# the existing graph must be untouched
# --------------------------------------------------------------------------

def test_existing_node_signatures_unchanged():
    assert list(inspect.signature(AgentGraph._count_turn_node).parameters) == ["self", "state"]
    assert list(inspect.signature(AgentGraph._agent_node).parameters) == ["self", "state"]
    assert list(inspect.signature(AgentGraph._memory_gate).parameters) == ["self", "state"]
    assert list(inspect.signature(
        AgentGraph._summarize_user_history_node).parameters) == ["self", "state", "config"]


def test_graph_shape(agent):
    graph = agent.graph.get_graph()
    assert "summarize_and_trim" in graph.nodes
    edges = {(e.source, e.target) for e in graph.edges}
    assert ("count_turn", "summarize_and_trim") in edges
    assert ("summarize_and_trim", "agent") in edges
    assert ("summarize_and_trim", "summarize_user_history") in edges
    assert ("count_turn", "agent") in edges, "the ordinary path must still exist"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--no-header", "-p", "no:cacheprovider"]))
