"""H6 regression: reading long-term memory must not write to it.

`UserMemory.__init__` used to call `add_documents` as a construction side effect, so
*every* path that merely opened memory also wrote to it — including `recall_user_memory`,
whose whole job is to read. `docs/architecture_review.md` H6/R5 flagged it as "safe today
only because of the `_memory_cache` fingerprint and content-hash ids — both of which are
`AgentGraph`'s doing, not `UserMemory`'s".

**It was not safe.** `scripts/check_routing.py` caught it in the wild: asked *"what is my
name?"*, the recall route routed correctly and then returned the string `"what is my
name?"` — the user's own question. `_build_user_memory` had indexed the query into the
collection on the way in, and at `k=1` the nearest neighbour to "my name" was that fresh
row rather than the seeded *"My name is Harshal."* The answer still looked right, because
the name is also in the system prompt (`pipeline_lg.py:396`) — which is exactly how a bug
like this survives a human reading the transcript.

The fix (R5's shape): construction opens the collection and writes nothing; `.index(...)`
is the only write; the recall tool gets a read-only handle. These tests pin all three.

    pytest tests/test_memory_read_path.py -v
    python tests/test_memory_read_path.py

No LLM and no Ollama: every assertion here is about Chroma and the message list.
Compression (`UserMemory.invoke`) is the only part that needs a model, and it is not
under test — H4 already covers it.
"""
import sys
import warnings
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from langchain_core.messages import AIMessage, HumanMessage

from memory.long_term_memory import UserMemory

DOC = "Marshall is a local RAG framework. It loads documents and answers questions.\n"

SEED = "My name is Harshal."
QUERY = "what is my name?"


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    #: pytest owns the lifetime -- see the note in tests/test_agent_hook.py
    tmp = tmp_path_factory.mktemp("marshall_h6")
    (tmp / "doc.txt").write_text(DOC, encoding="utf-8")
    return tmp


def _memory(path, name, messages=None, **kwargs):
    return UserMemory(
        chat_history=list(messages or []),
        collection_name=name,
        persist_directory_path=str(path),
        **kwargs,
    )


def _count(path, name) -> int:
    """Rows actually on disk. Goes to chromadb directly — a langchain `Chroma` would
    get-or-create the collection it is being asked about."""
    import chromadb
    try:
        return chromadb.PersistentClient(path=str(path)).get_collection(name).count()
    except Exception:
        return 0


# --------------------------------------------------------------------------
# construction reads; only .index() writes
# --------------------------------------------------------------------------

def test_construction_writes_nothing(workspace):
    """The whole finding, in one assertion."""
    _memory(workspace, "c_writes_nothing", [HumanMessage(content=SEED)])
    assert _count(workspace, "c_writes_nothing") == 0, (
        "constructing a UserMemory wrote to Chroma — H6 is back")


def test_index_writes(workspace):
    memory = _memory(workspace, "c_index_writes", [HumanMessage(content=SEED)])
    assert memory.index() == 1
    assert _count(workspace, "c_index_writes") == 1


def test_index_defaults_to_its_own_chat_history(workspace):
    memory = _memory(workspace, "c_index_default",
                     [HumanMessage(content=SEED), HumanMessage(content="I live in Pune.")])
    assert memory.index() == 2


def test_index_accepts_an_explicit_message_list(workspace):
    """The gate hands in the list it filtered, rather than rebuilding the object."""
    memory = _memory(workspace, "c_index_explicit")
    assert memory.index([HumanMessage(content=SEED)]) == 1
    assert _count(workspace, "c_index_explicit") == 1


def test_index_stores_only_human_messages(workspace):
    """Marshall's own replies are not facts about the user. `ai_messages` stays unused."""
    memory = _memory(workspace, "c_index_humans",
                     [HumanMessage(content=SEED), AIMessage(content="Nice to meet you!")])
    assert memory.index() == 1
    assert _count(workspace, "c_index_humans") == 1


def test_index_of_nothing_is_a_no_op(workspace):
    memory = _memory(workspace, "c_index_empty")
    assert memory.index() == 0
    assert _count(workspace, "c_index_empty") == 0


def test_index_is_idempotent(workspace):
    """Content-hash ids, unchanged: re-indexing a growing conversation must not duplicate."""
    messages = [HumanMessage(content=SEED)]
    _memory(workspace, "c_idempotent", messages).index()
    _memory(workspace, "c_idempotent", messages).index()
    _memory(workspace, "c_idempotent", messages + [HumanMessage(content="I live in Pune.")]).index()
    assert _count(workspace, "c_idempotent") == 2, "content-hash upsert stopped deduplicating"


def test_a_reader_sees_what_an_indexer_wrote(workspace):
    """Two handles, one collection — the cross-session path recall depends on."""
    _memory(workspace, "c_roundtrip", [HumanMessage(content=SEED)]).index()
    assert SEED in _memory(workspace, "c_roundtrip").retreival("my name", k=1)


# --------------------------------------------------------------------------
# the witness: the read path no longer poisons its own search
# --------------------------------------------------------------------------

def test_reading_does_not_index_the_query(workspace):
    """The exact failure `scripts/check_routing.py` caught.

    A reader built from the live conversation — which is what `_build_user_memory` hands
    the recall tool — must return the seeded fact, not the question that asked for it.
    """
    name = "c_witness"
    _memory(workspace, name, [HumanMessage(content=SEED)]).index()

    reader = _memory(workspace, name, [HumanMessage(content=QUERY)])
    found = reader.retreival(QUERY, k=1)

    assert SEED in found, f"recall returned {found!r} instead of the seeded fact"
    assert QUERY not in found, "the query was indexed and retrieved back as a memory"
    assert _count(workspace, name) == 1, "the read path wrote a row"


def test_the_pre_fix_behaviour_is_what_it_looked_like(workspace):
    """Pin the failure itself, so the witness above cannot be read as luck.

    Indexing the query first — which is precisely what the constructor used to do — makes
    `k=1` return that query. This test does *not* exercise the fixed path; it demonstrates
    that the collection ordering the fix relies on is real and not an artifact of the
    embedding model happening to prefer the seed.
    """
    name = "c_prefix"
    _memory(workspace, name, [HumanMessage(content=SEED)]).index()
    _memory(workspace, name, [HumanMessage(content=QUERY)]).index()   # the old side effect

    found = _memory(workspace, name).retreival(QUERY, k=1)
    assert QUERY in found, (
        "indexing the query no longer wins at k=1 — the witness test above is now "
        "passing for a different reason than the fix")


# --------------------------------------------------------------------------
# C2 must survive the move
# --------------------------------------------------------------------------

def test_durability_warning_still_fires_without_a_path():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        UserMemory(chat_history=[HumanMessage(content=SEED)], collection_name="c2_check")
    assert any("in-memory only" in str(w.message) for w in caught), (
        "C2's durability warning was lost when the write moved out of __init__")


def test_no_durability_warning_with_a_path(workspace):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _memory(workspace, "c2_quiet", [HumanMessage(content=SEED)]).index()
    assert not [w for w in caught if "in-memory only" in str(w.message)]


# --------------------------------------------------------------------------
# the graph funnel: the gate writes, recall does not
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def agent(workspace):
    """A real AgentGraph. Needs no Ollama — `ChatOllama` construction does not connect."""
    from pipeline_lg import AgentGraph
    return AgentGraph(
        file_path=str(workspace / "doc.txt"),
        mode_docs="txt",
        mode_retreiver="simple",
        persist_directory_path=str(workspace / "vdb"),
        user_history_directory_path=str(workspace / "agent_mem"),
        user_id="h6",
    )


def _agent_count(agent) -> int:
    return _count(agent.user_history_directory_path, agent.user_memory_collection)


def test_gate_path_still_indexes(agent):
    """The passive gate is the write path and must stay one."""
    before = _agent_count(agent)
    agent._build_user_memory([HumanMessage(content="I work on a RAG agent.")], "h6-gate")
    assert _agent_count(agent) == before + 1


def test_recall_path_does_not_index(agent):
    before = _agent_count(agent)
    memory = agent._read_user_memory([HumanMessage(content=QUERY)], "h6-read")
    assert memory is not None
    assert _agent_count(agent) == before, "the read-only funnel wrote to the collection"


def test_a_cached_read_only_handle_is_still_indexed_when_the_gate_runs(agent):
    """The cache must not let a read-only build satisfy a later write.

    Both funnels share `_memory_cache`, keyed by thread and fingerprinted by message
    count. Returning a cached handle that was built read-only would silently skip the
    gate's write for that fingerprint — the passive memory would just stop, with nothing
    to show for it.
    """
    thread, messages = "h6-cache", [HumanMessage(content="I am learning LangGraph.")]
    before = _agent_count(agent)

    agent._read_user_memory(messages, thread)
    assert _agent_count(agent) == before

    agent._build_user_memory(messages, thread)          # same fingerprint, cache hit
    assert _agent_count(agent) == before + 1, (
        "a cached read-only handle swallowed the gate's write")


def test_the_recall_tool_itself_writes_nothing(agent):
    """End to end through the bound tool, which is where H6 actually bit."""
    recall = next(tool for tool in agent.tools if tool.name == "recall_user_memory")
    before = _agent_count(agent)

    result = recall.func(
        query="my name", k=1,
        state={"messages": [HumanMessage(content=QUERY)]},
        config={"configurable": {"thread_id": "h6-tool"}},
    )

    assert isinstance(result, str)
    assert _agent_count(agent) == before, "recall_user_memory wrote while reading"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--no-header", "-p", "no:cacheprovider"]))
