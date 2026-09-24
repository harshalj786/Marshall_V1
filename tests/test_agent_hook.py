"""The `agent_hook` seam: one caller-supplied node on the agent's outgoing edge.

    pytest tests/test_agent_hook.py -v

Two things have to be true for this seam to be worth having, and they are what this file
pins. First, a hook that says "proceed" must be INVISIBLE: the graph has to route exactly
as it does with no hook at all, because otherwise installing one silently changes the
agent's behaviour and every routing measurement taken without it stops being meaningful.
Second, a verdict the graph has no target for must be LOUD, because the alternative — a
verdict nobody recognises quietly falling through to the end of the turn — is
indistinguishable from a normal finish and would be found only by noticing that answers
had gone missing.

No Ollama here, and no marker: `agent.llm` is replaced with a scripted stand-in, which is
this repo's pattern for graph-behaviour tests (see tests/test_summarization_cycle.py).
That is deliberate for routing SHAPE — a real model would make the comparison a
measurement of llama3.1 rather than of the wiring. What the live model routes to is
tests/test_routing_matrix.py's job, and it stays that way.
"""
import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline_lg import AGENT_HOOK_VERDICTS, AgentGraph

DOC = "Marshall is a local RAG framework. It loads documents and answers questions.\n"


class ToolCallingChat:
    """Calls `calculator` once per user turn, then answers: two generations per turn."""

    def __init__(self):
        self.generations = 0

    def invoke(self, messages, **kwargs):
        self.generations += 1
        if isinstance(messages[-1], ToolMessage):
            return AIMessage(content="42")
        return AIMessage(content="", tool_calls=[{
            "name": "calculator",
            "args": {"expression": "6*7"},
            "id": f"call_{self.generations}",
        }])


class RecordingHook:
    """A hook node that records how often it ran and a router with a fixed verdict."""

    def __init__(self, verdict):
        self.verdict = verdict
        self.node_calls = 0
        self.router_calls = 0

    def node(self, state):
        self.node_calls += 1
        return {}                      # writes nothing: the seam must work empty-handed

    def router(self, state):
        self.router_calls += 1
        return self.verdict

    def as_pair(self):
        #: a plain tuple, not the AgentHook NamedTuple — a caller must not need the import
        return (self.node, self.router)


@pytest.fixture(scope="module")
def tmp_root(tmp_path_factory):
    #: pytest owns the lifetime. A manual rmtree here could not succeed: Chroma keeps the
    #: system for a directory registered for the life of the process, so on Windows the
    #: delete raises WinError 32 and ignore_errors=True silently left the tree behind.
    tmp = tmp_path_factory.mktemp("marshall_hook")
    (tmp / "doc.txt").write_text(DOC, encoding="utf-8")
    return tmp


def build_agent(tmp_root: Path, **kwargs) -> AgentGraph:
    #: one shared index for every agent here — the second construction opens what the
    #: first wrote, so the embedding cost is paid once for the module
    return AgentGraph(
        file_path=str(tmp_root / "doc.txt"),
        mode_docs="txt",
        mode_retreiver="simple",
        persist_directory_path=str(tmp_root / "vdb"),
        user_history_directory_path=str(tmp_root / "mem"),
        memory_every_n_turns=0,
        summarize_every_n_turns=0,
        summary_persist_path=str(tmp_root / "sum"),
        **kwargs,
    )


def shape(messages):
    """Compare-able view of a message list: ids and object identity are not behaviour."""
    return [(type(m).__name__,
             m.content,
             tuple(c["name"] for c in getattr(m, "tool_calls", None) or ()))
            for m in messages]


def run_turn(agent, thread_id, query="what is 6*7?"):
    agent.invoke(query, config={"configurable": {"thread_id": thread_id}})
    return agent.get_history(thread_id)


# --------------------------------------------------------------------------
# 1. "proceed" is indistinguishable from no hook at all
# --------------------------------------------------------------------------

def test_proceed_routes_identically_to_no_hook(tmp_root, monkeypatch):
    plain = build_agent(tmp_root)
    monkeypatch.setattr(plain, "llm", ToolCallingChat())
    plain_history = run_turn(plain, "hook-plain")

    hook = RecordingHook("proceed")
    hooked = build_agent(tmp_root, agent_hook=hook.as_pair())
    monkeypatch.setattr(hooked, "llm", ToolCallingChat())
    hooked_history = run_turn(hooked, "hook-proceed")

    assert shape(hooked_history) == shape(plain_history)

    # the tool really was called on both paths — otherwise "identical" could mean
    # "identically did nothing", which would pass this test for the wrong reason
    assert any(isinstance(m, ToolMessage) for m in plain_history)
    assert [m for m in plain_history if isinstance(m, AIMessage)][-1].content == "42"

    # fires once per agent generation, including the intermediate tool-calling one
    assert hook.node_calls == 2
    assert hook.router_calls == 2


def test_hook_state_defaults_to_empty_and_marshall_leaves_it_alone(tmp_root, monkeypatch):
    hook = RecordingHook("proceed")
    agent = build_agent(tmp_root, agent_hook=hook.as_pair())
    monkeypatch.setattr(agent, "llm", ToolCallingChat())
    config = {"configurable": {"thread_id": "hook-state"}}
    agent.invoke("what is 6*7?", config=config)

    state = agent.graph.get_state(config).values
    assert state["hook_state"] == {}


# --------------------------------------------------------------------------
# 2. an unknown verdict fails loudly
# --------------------------------------------------------------------------

def test_an_unknown_verdict_raises_instead_of_ending_the_turn(tmp_root, monkeypatch):
    hook = RecordingHook("banana")
    agent = build_agent(tmp_root, agent_hook=hook.as_pair())
    monkeypatch.setattr(agent, "llm", ToolCallingChat())

    with pytest.raises(ValueError) as excinfo:
        run_turn(agent, "hook-unknown")

    message = str(excinfo.value)
    assert "banana" in message, "the rejected verdict must be named"
    for verdict in AGENT_HOOK_VERDICTS:
        assert verdict in message, f"the error must list {verdict!r} as an option"


def test_a_hook_that_is_not_callable_is_rejected_at_construction(tmp_root):
    with pytest.raises(TypeError, match="callable"):
        build_agent(tmp_root, agent_hook=("not a callable", lambda state: "proceed"))

    with pytest.raises(TypeError, match="pair"):
        build_agent(tmp_root, agent_hook=object())
