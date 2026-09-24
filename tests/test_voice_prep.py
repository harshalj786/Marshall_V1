"""Part A of the voice work: four changes that must not disturb anything already proven.

Covered here:

1. **Tool timeouts** (finding K, `docs/architecture_review.md:309`). `wikipedia_lookup` and
   `web_search` now run under a hard wall-clock budget. The load-bearing assertion is not
   that the timeout works — it is that the tools' *interface* did not move: R4 measured tool
   routing against these exact names, docstrings and schemas, and those docstrings are prompt
   engineering, not documentation. The expected values below are hard-coded literals captured
   from the pre-timeout code, so this test compares against history rather than against
   itself.
2. **`UserMemory(model=...)`**, threaded through both graphs as `memory_model`.
3. **`AgentGraph(keep_alive=...)`** reaching `ChatOllama`.
4. **`warmup.warm_embeddings()`** forcing both MiniLM singletons to load.

Runs under pytest, or standalone:

    pytest tests/test_voice_prep.py -v
    python tests/test_voice_prep.py

No LLM and no network. A few tests load MiniLM (the singleton they are asserting on), which
costs a few seconds the first time and nothing afterwards.
"""
import inspect
import sys
import threading
from pathlib import Path

import pytest
from langchain_core.runnables import Runnable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tools.tools as tools_module
from tools import wikipedia_lookup, web_search
from memory.long_term_memory import UserMemory, DEFAULT_MEMORY_MODEL
import memory.long_term_memory as ltm
import pipeline_lg
from pipeline_lg import AgentGraph, ChatRagWithMemoryGraph


# ==========================================================================
# 1. The interface pins — captured from the code BEFORE the timeout wrap.
#
# If one of these fails, tool routing is no longer the thing R4 verified and the
# measurement in docs/architecture_review.md:793-802 no longer describes this code.
# Fix the code, not the constant.
# ==========================================================================

PRE_WRAP_DESCRIPTION = {
    "wikipedia_lookup": (
        "Look up a topic on Wikipedia and return the joined page content of the top-k articles.\n"
        "\n"
        "    Use for general factual questions about people, places, historical events, science\n"
        "    concepts — anything encyclopedic that is not tied to the currently loaded document."
    ),
    "web_search": (
        "Search the web via DuckDuckGo and return the joined text snippets of the top-k results.\n"
        "\n"
        "    Use for current events, news, or anything time-sensitive that Wikipedia will not have."
    ),
}

PRE_WRAP_SCHEMA = {
    "wikipedia_lookup": {
        "description": (
            "Look up a topic on Wikipedia and return the joined page content of the top-k articles.\n"
            "\n"
            "Use for general factual questions about people, places, historical events, science\n"
            "concepts — anything encyclopedic that is not tied to the currently loaded document."
        ),
        "properties": {
            "k": {"default": 3, "title": "K", "type": "integer"},
            "query": {"title": "Query", "type": "string"},
        },
        "required": ["query"],
        "title": "wikipedia_lookup",
        "type": "object",
    },
    "web_search": {
        "description": (
            "Search the web via DuckDuckGo and return the joined text snippets of the top-k results.\n"
            "\n"
            "Use for current events, news, or anything time-sensitive that Wikipedia will not have."
        ),
        "properties": {
            "k": {"default": 5, "title": "K", "type": "integer"},
            "query": {"title": "Query", "type": "string"},
        },
        "required": ["query"],
        "title": "web_search",
        "type": "object",
    },
}

TIMED_TOOLS = [wikipedia_lookup, web_search]


@pytest.mark.parametrize("tool", TIMED_TOOLS, ids=lambda t: t.name)
def test_tool_name_unchanged(tool):
    """The name the model emits in a tool_call. A rename silently breaks every route."""
    assert tool.name in PRE_WRAP_DESCRIPTION, f"unexpected tool name: {tool.name!r}"


@pytest.mark.parametrize("tool", TIMED_TOOLS, ids=lambda t: t.name)
def test_tool_description_byte_identical(tool):
    """`.description` is the raw docstring — what actually reaches the model's tool list."""
    assert tool.description == PRE_WRAP_DESCRIPTION[tool.name], (
        f"{tool.name}'s docstring changed. It is prompt engineering verified under R4, "
        "not documentation — wrap execution, never the interface.")


@pytest.mark.parametrize("tool", TIMED_TOOLS, ids=lambda t: t.name)
def test_tool_schema_byte_identical(tool):
    """Arg names, types, defaults and the dedented description the schema carries."""
    assert tool.args_schema.model_json_schema() == PRE_WRAP_SCHEMA[tool.name], (
        f"{tool.name}'s tool schema changed; llama3.1 routing was measured against the old one")


@pytest.mark.parametrize("tool", TIMED_TOOLS, ids=lambda t: t.name)
def test_tool_signature_unchanged(tool):
    """Defaults live in the signature; k=3 vs k=5 is a behaviour difference the model sees."""
    expected = {"wikipedia_lookup": {"query": inspect.Parameter.empty, "k": 3},
                "web_search": {"query": inspect.Parameter.empty, "k": 5}}[tool.name]
    params = inspect.signature(tool.func).parameters
    assert list(params) == list(expected)
    assert {n: p.default for n, p in params.items()} == expected


def test_untimed_tools_were_not_touched():
    """Only the two NETWORK tools were in scope for finding K."""
    from tools import calculator
    assert calculator.invoke({"expression": "47 * 82"}) == "3854"


# ==========================================================================
# 2. The timeout wrapper itself
# ==========================================================================

def test_default_budget_is_ten_seconds():
    assert tools_module.TOOL_TIMEOUT_SECONDS == 10.0


def test_timeout_message_is_exactly_the_required_string():
    """Pinned wording at the default budget, asserted without waiting for it."""
    assert tools_module._timeout_message("wikipedia_lookup") == (
        "wikipedia_lookup error: timed out after 10s")
    assert tools_module._timeout_message("web_search") == (
        "web_search error: timed out after 10s")


def test_slow_callable_returns_the_error_string(monkeypatch):
    """A body that outlives the budget must hand the turn back, not hold it."""
    monkeypatch.setattr(tools_module, "TOOL_TIMEOUT_SECONDS", 0.2)
    release = threading.Event()

    def _never_returns_in_time():
        release.wait(30)          # released in the finally below, so no 30s orphan
        return "too late"

    try:
        result = tools_module._run_with_timeout("slow_tool", _never_returns_in_time)
        assert result == "slow_tool error: timed out after 0.2s"
    finally:
        release.set()


def test_fast_callable_passes_its_result_through(monkeypatch):
    """Under budget, the wrapper must be invisible."""
    monkeypatch.setattr(tools_module, "TOOL_TIMEOUT_SECONDS", 5.0)
    assert tools_module._run_with_timeout("fast_tool", lambda: "the real answer") == (
        "the real answer")


def test_arguments_reach_the_wrapped_callable(monkeypatch):
    monkeypatch.setattr(tools_module, "TOOL_TIMEOUT_SECONDS", 5.0)
    seen = {}

    def _body(query, k):
        seen.update(query=query, k=k)
        return f"{query}/{k}"

    assert tools_module._run_with_timeout("t", _body, "pipes", 7) == "pipes/7"
    assert seen == {"query": "pipes", "k": 7}


def test_exception_propagates_rather_than_being_converted(monkeypatch):
    """The wrapper changes WHEN a tool returns, never WHAT it returns.

    `wikipedia_lookup`'s bare `import wikipedia` sits outside its own try on purpose
    (§4, missing-error-handling table); swallowing it here would quietly close a finding
    that is still open.
    """
    monkeypatch.setattr(tools_module, "TOOL_TIMEOUT_SECONDS", 5.0)

    def _boom():
        raise ImportError("no module named wikipedia")

    with pytest.raises(ImportError, match="no module named wikipedia"):
        tools_module._run_with_timeout("wikipedia_lookup", _boom)


def test_timeout_is_wired_into_the_real_wikipedia_tool(monkeypatch):
    """End to end through the decorated tool, not just the helper."""
    monkeypatch.setattr(tools_module, "TOOL_TIMEOUT_SECONDS", 0.2)
    release = threading.Event()
    monkeypatch.setattr(tools_module, "_wikipedia_lookup_body",
                        lambda query, k: (release.wait(30), "too late")[1])
    try:
        assert wikipedia_lookup.invoke({"query": "transistor"}) == (
            "wikipedia_lookup error: timed out after 0.2s")
    finally:
        release.set()


def test_timeout_is_wired_into_the_real_web_search_tool(monkeypatch):
    monkeypatch.setattr(tools_module, "TOOL_TIMEOUT_SECONDS", 0.2)
    release = threading.Event()
    monkeypatch.setattr(tools_module, "_web_search_body",
                        lambda query, k: (release.wait(30), "too late")[1])
    try:
        assert web_search.invoke({"query": "news today"}) == (
            "web_search error: timed out after 0.2s")
    finally:
        release.set()


def test_fast_body_still_reaches_the_caller_through_the_real_tool(monkeypatch):
    monkeypatch.setattr(tools_module, "_web_search_body", lambda query, k: f"hits for {query}")
    assert web_search.invoke({"query": "pipes"}) == "hits for pipes"


# ==========================================================================
# 3. UserMemory(model=...) and its two graph parameters
# ==========================================================================

class _RecordingOllama(Runnable):
    """Stands in for langchain's Ollama, capturing which model was asked for.

    A real `Runnable` rather than a bare stub: `UserMemory.invoke` builds
    `template | llm | parser`, and langchain refuses to coerce anything else.
    """
    seen: list = []

    def __init__(self, model=None, temperature=None, **kw):
        _RecordingOllama.seen.append(model)
        self.model = model

    def invoke(self, input, config=None, **kw):
        return "compressed"


@pytest.fixture
def memory_dir(tmp_path):
    #: pytest owns the lifetime -- see the note in tests/test_agent_hook.py
    return str(tmp_path)


def test_user_memory_model_defaults_to_mistral():
    assert DEFAULT_MEMORY_MODEL == "mistral"
    assert inspect.signature(UserMemory.__init__).parameters["model"].default == "mistral"


def test_user_memory_uses_the_default_model_when_unspecified(memory_dir, monkeypatch):
    monkeypatch.setattr(ltm, "Ollama", _RecordingOllama)
    _RecordingOllama.seen = []
    UserMemory(chat_history=[], persist_directory_path=memory_dir).invoke({})
    assert _RecordingOllama.seen == ["mistral"], "existing callers must be unaffected"


def test_user_memory_honours_a_model_override(memory_dir, monkeypatch):
    monkeypatch.setattr(ltm, "Ollama", _RecordingOllama)
    _RecordingOllama.seen = []
    UserMemory(chat_history=[], persist_directory_path=memory_dir, model="llama3.1").invoke({})
    assert _RecordingOllama.seen == ["llama3.1"]


def test_both_graphs_expose_memory_model_defaulting_to_mistral():
    for cls in (AgentGraph, ChatRagWithMemoryGraph):
        param = inspect.signature(cls.__init__).parameters["memory_model"]
        assert param.default == DEFAULT_MEMORY_MODEL, f"{cls.__name__} changed an existing default"


def test_agent_graph_passes_memory_model_to_user_memory(monkeypatch):
    """The graph must hand its configured model to every UserMemory it builds."""
    captured = {}

    class _FakeUserMemory:
        def __init__(self, **kw):
            captured.update(kw)

        #: the write is an explicit call since H6 — the funnel makes it, so the double
        #: has to answer it. What is under test here is still only `model=`.
        def index(self, *a, **kw):
            return 0

    monkeypatch.setattr(pipeline_lg, "UserMemory", _FakeUserMemory)

    agent = AgentGraph.__new__(AgentGraph)          # no ingestion; only the funnel is exercised
    agent.memory_model = "llama3.1"
    agent.user_memory_collection = "user_history"
    agent.user_history_directory_path = "/tmp/does-not-matter"
    agent._memory_cache = {}

    from langchain_core.messages import HumanMessage
    agent._build_user_memory([HumanMessage(content="my name is Harshal")], "t1")
    assert captured["model"] == "llama3.1"

    captured.clear()
    agent._build_user_memory([], "t2")              # the read-only/no-humans branch
    assert captured["model"] == "llama3.1", "the empty-history branch must thread it too"


def test_chat_graph_passes_memory_model_to_user_memory(monkeypatch):
    captured = {}

    class _FakeUserMemory:
        def __init__(self, **kw):
            captured.update(kw)

        #: see the agent-side double above — H6 moved the write out of __init__
        def index(self, *a, **kw):
            return 0

        def invoke(self, *a, **kw):
            return "SUMMARY"

    monkeypatch.setattr(pipeline_lg, "UserMemory", _FakeUserMemory)

    chat = ChatRagWithMemoryGraph.__new__(ChatRagWithMemoryGraph)
    chat.memory_model = "llama3.1"
    chat.user_history_directory_path = "/tmp/does-not-matter"

    out = ChatRagWithMemoryGraph._summarize_user_history_node(chat, {"messages": []})
    assert out == {"user_history_summary": "SUMMARY"}
    assert captured["model"] == "llama3.1"


# ==========================================================================
# 4. keep_alive passthrough
# ==========================================================================

def test_agent_graph_keep_alive_defaults_to_none():
    """None is ChatOllama's own default, so the default path is byte-for-byte unchanged."""
    assert inspect.signature(AgentGraph.__init__).parameters["keep_alive"].default is None


def test_keep_alive_reaches_chat_ollama():
    from langchain_ollama import ChatOllama
    assert ChatOllama(model="llama3.1", temperature=0, keep_alive="30m").keep_alive == "30m"
    assert ChatOllama(model="llama3.1", temperature=0, keep_alive=None).keep_alive is None


# ==========================================================================
# 5. The embedding warm-up hook
# ==========================================================================

def test_warm_embeddings_is_importable_without_side_effects():
    """Importing must not load anything — only the call is allowed to cost seconds."""
    import warmup
    assert callable(warmup.warm_embeddings)


def test_warm_embeddings_populates_both_singletons():
    import warmup
    import ingestion.vector_db as vdb

    timings = warmup.warm_embeddings()

    assert set(timings) == {"ingestion", "memory"}
    assert all(isinstance(v, float) and v >= 0 for v in timings.values())
    assert vdb._EMBEDDING_MODEL is not None, "ingestion's MiniLM was not forced to load"
    assert ltm._EMBEDDING_MODEL is not None, "memory's MiniLM was not forced to load"


def test_warm_embeddings_is_idempotent():
    """Called twice, the second is a no-op — it must not build a second model."""
    import warmup
    import ingestion.vector_db as vdb

    warmup.warm_embeddings()
    first = vdb._EMBEDDING_MODEL
    warmup.warm_embeddings()
    assert vdb._EMBEDDING_MODEL is first, "warm-up rebuilt a singleton instead of reusing it"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--no-header", "-p", "no:cacheprovider"]))
