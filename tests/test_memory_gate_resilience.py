"""H4 regression: a long-term-memory failure must never destroy the user's turn.

ChatRagWithMemoryGraph._summarize_user_history_node used to call UserMemory
unguarded, so an Ollama hiccup or a Chroma write error on turn 5 raised straight
out of .invoke() and lost the turn. AgentGraph had the guard; the chat graph did
not. These tests pin the fixed behaviour and the parity between the two.

Runs under pytest, or standalone:

    pytest tests/test_memory_gate_resilience.py -v
    python tests/test_memory_gate_resilience.py

The unit tests need no LLM. The two end-to-end tests generate through Ollama
(mistral) and are skipped automatically when it is not reachable.
"""
import inspect
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pipeline_lg
from pipeline_lg import AgentGraph, ChatRagWithMemoryGraph

DOC = "Marshall is a local RAG framework. It loads documents and answers questions about them.\n"


def _ollama_up() -> bool:
    try:
        import urllib.request
        urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=3)
        return True
    except Exception:
        return False


needs_ollama = pytest.mark.skipif(not _ollama_up(), reason="Ollama not reachable on :11434")


class _BoomOnInvoke:
    """UserMemory stand-in that builds fine and fails when compressing."""
    def __init__(self, *a, **kw):
        pass

    def invoke(self, *a, **kw):
        raise RuntimeError("ollama connection refused")


class _BoomOnInit:
    """UserMemory stand-in that fails during construction — i.e. on the Chroma write."""
    def __init__(self, *a, **kw):
        raise RuntimeError("chroma write failed")


class _Works:
    def __init__(self, *a, **kw):
        pass

    def invoke(self, *a, **kw):
        return "SENTINEL_SUMMARY"


@pytest.fixture(scope="module")
def graph():
    tmp = Path(tempfile.mkdtemp(prefix="marshall_h4_"))
    (tmp / "doc.txt").write_text(DOC, encoding="utf-8")
    g = ChatRagWithMemoryGraph(
        file_path=str(tmp / "doc.txt"),
        mode_docs="txt",
        mode_retreiver="simple",
        persist_directory_path=str(tmp / "vdb"),
        user_history_directory_path=str(tmp / "mem"),
    )
    g._test_tmp = tmp
    yield g
    shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------
# signature guarantees — the fix must not have changed any public shape
# --------------------------------------------------------------------------

def test_chat_graph_node_signature_unchanged():
    params = list(inspect.signature(
        ChatRagWithMemoryGraph._summarize_user_history_node).parameters)
    assert params == ["self", "state"], (
        "the chat graph's node must NOT gain AgentGraph's `config` parameter")


def test_agent_graph_node_signature_unchanged():
    params = list(inspect.signature(
        AgentGraph._summarize_user_history_node).parameters)
    assert params == ["self", "state", "config"], (
        "AgentGraph still needs `config` to resolve thread_id for its memory cache")


# --------------------------------------------------------------------------
# the guard itself — no LLM needed
# --------------------------------------------------------------------------

def test_compression_failure_is_swallowed(graph, monkeypatch):
    """An LLM failure while compressing must not propagate."""
    monkeypatch.setattr(pipeline_lg, "UserMemory", _BoomOnInvoke)
    assert graph._summarize_user_history_node({"messages": [], "query": "hi"}) == {}


def test_construction_failure_is_swallowed(graph, monkeypatch):
    """Construction is what writes to Chroma, so it must be inside the guard too."""
    monkeypatch.setattr(pipeline_lg, "UserMemory", _BoomOnInit)
    assert graph._summarize_user_history_node({"messages": [], "query": "hi"}) == {}


def test_empty_dict_not_blank_summary(graph, monkeypatch):
    """{} means 'no update' under the overwrite reducer; a blank string would erase."""
    monkeypatch.setattr(pipeline_lg, "UserMemory", _BoomOnInvoke)
    out = graph._summarize_user_history_node({"messages": [], "query": "hi"})
    assert "user_history_summary" not in out, (
        "returning the key with a falsy value would wipe a good previous summary")


def test_success_path_still_returns_summary(graph, monkeypatch):
    """The guard must not swallow the happy path."""
    monkeypatch.setattr(pipeline_lg, "UserMemory", _Works)
    out = graph._summarize_user_history_node({"messages": [], "query": "hi"})
    assert out == {"user_history_summary": "SENTINEL_SUMMARY"}


def test_both_graphs_survive_the_same_failure(graph, monkeypatch):
    """Parity: the divergence H4 describes must be gone."""
    monkeypatch.setattr(pipeline_lg, "UserMemory", _BoomOnInvoke)
    chat = graph._summarize_user_history_node({"messages": [], "query": "hi"})

    agent = AgentGraph.__new__(AgentGraph)          # no ingestion; only the node is exercised
    agent._build_user_memory = lambda messages, thread_id="default": _BoomOnInvoke()
    agent_out = AgentGraph._summarize_user_history_node(
        agent, {"messages": []}, {"configurable": {"thread_id": "t"}})

    assert chat == agent_out == {}


# --------------------------------------------------------------------------
# end to end — the turn actually survives (needs Ollama/mistral)
# --------------------------------------------------------------------------

@needs_ollama
def test_turn_survives_gate_failure(graph, monkeypatch):
    """Turn 5 trips the gate. With memory broken the user must still get an answer."""
    cfg = {"configurable": {"thread_id": "h4-survive"}}
    graph.invoke({"query": "what is Marshall?"}, config=cfg)      # turn 1, gate quiet
    graph.graph.update_state(cfg, {"turn_count": 4})              # next turn trips it

    monkeypatch.setattr(pipeline_lg, "UserMemory", _BoomOnInvoke)
    answer = graph.invoke({"query": "what does it do?"}, config=cfg)

    assert answer and answer.strip(), "the turn was lost to a memory failure"
    snap = graph.graph.get_state(cfg).values
    assert snap["turn_count"] == 5, "the gate should have fired on this turn"
    assert not snap.get("user_history_summary"), "no summary should have been written"


@needs_ollama
def test_previous_summary_survives_a_later_failure(graph, monkeypatch):
    """A good summary must not be erased by a later failed gate hit."""
    cfg = {"configurable": {"thread_id": "h4-preserve"}}
    graph.invoke({"query": "what is Marshall?"}, config=cfg)
    graph.graph.update_state(cfg, {"turn_count": 4, "user_history_summary": "GOOD_PROFILE"})

    monkeypatch.setattr(pipeline_lg, "UserMemory", _BoomOnInvoke)
    answer = graph.invoke({"query": "what does it do?"}, config=cfg)

    assert answer and answer.strip()
    snap = graph.graph.get_state(cfg).values
    assert snap["user_history_summary"] == "GOOD_PROFILE", (
        "a failed gate hit overwrote a previously good profile")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--no-header", "-p", "no:cacheprovider"]))
