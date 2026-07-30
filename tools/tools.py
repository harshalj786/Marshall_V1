"""Tools available to the Marshall agent.

Static tools importable directly:
    calculator, wikipedia_lookup, web_search

Factories (built at agent-init because they close over live agent state):
    make_search_current_doc(retrival) -> BaseTool
    make_recall_user_memory(build_memory) -> BaseTool
"""
import ast
import operator
import threading
from typing import Annotated, Optional

from langchain_core.tools import tool, BaseTool
from langchain_core.runnables import RunnableConfig
from langchain_community.retrievers import WikipediaRetriever
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langgraph.prebuilt import InjectedState


#: hard wall-clock budget for one network tool call (finding K, docs/architecture_review.md:309).
#: `wikipedia_lookup` was measured at 53-82 s inside the agent against 22.6 s standalone under
#: R4, which in §10 terms is that many seconds of dead air on a turn that routed correctly. A
#: tool that gives up is strictly better than one that answers late: the model can narrate a
#: timeout, but nothing downstream can un-stall a turn that already blocked.
TOOL_TIMEOUT_SECONDS = 10.0


def _timeout_message(tool_name: str) -> str:
    #: `:g` so the default budget reads "10s" rather than "10.0s". Split out from
    #: `_run_with_timeout` so the exact wording can be asserted without a test that waits.
    return f"{tool_name} error: timed out after {TOOL_TIMEOUT_SECONDS:g}s"


def _run_with_timeout(tool_name: str, func, *args, **kwargs) -> str:
    """Run `func` under `TOOL_TIMEOUT_SECONDS`, returning a tool-style error string on expiry.

    The worker is NOT killed — Python cannot interrupt a thread, and the blocking call is
    down inside urllib/requests — so what unblocks is the turn, not the socket. The orphan
    finishes into a result nobody reads.

    A daemon thread rather than `ThreadPoolExecutor` for exactly that reason: the pool's
    workers are non-daemon and joined by an atexit hook, so a single hung lookup would hold
    the process open at shutdown — and the voice loop's Ctrl+C must not hang behind it.

    Exceptions are re-raised in the calling thread rather than converted, so this wrapper
    changes WHEN a tool returns and nothing about WHAT it returns. That matters here: the
    tools already convert their own failures to `"<name> error: ..."` strings, and
    `wikipedia_lookup`'s bare `import wikipedia` deliberately still escapes to `ToolNode`.
    """
    outcome: list[tuple[bool, object]] = []

    def _target() -> None:
        try:
            outcome.append((True, func(*args, **kwargs)))
        except BaseException as exc:      # re-raised below, never swallowed
            outcome.append((False, exc))

    worker = threading.Thread(target=_target, name=f"marshall-{tool_name}", daemon=True)
    worker.start()
    worker.join(TOOL_TIMEOUT_SECONDS)
    if worker.is_alive():
        return _timeout_message(tool_name)
    succeeded, value = outcome[0]
    if succeeded:
        return value
    raise value


_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
_ALLOWED_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval_ast(node):
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Num):
        return node.n
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](_eval_ast(node.left), _eval_ast(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return _ALLOWED_UNARYOPS[type(node.op)](_eval_ast(node.operand))
    raise ValueError(f"unsupported expression element: {type(node).__name__}")


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression and return the numeric result as a string.

    Supports +, -, *, /, **, %, //, parentheses, and unary +/-. No variables, no function calls,
    no imports. Use this whenever the user asks for a calculation."""
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval_ast(tree)
        return str(result)
    except Exception as e:
        return f"calculator error: {e}"


def _wikipedia_lookup_body(query: str, k: int) -> str:
    #: the whole body — both attempts — shares ONE budget, because the retry is part of what
    #: made this slow: two serial network round-trips with no delay between them.
    import wikipedia
    wikipedia.set_user_agent("Marshall-V1/0.1 (educational RAG agent)")
    last_err: Optional[Exception] = None
    for _ in range(2):
        try:
            retriever = WikipediaRetriever(top_k_results=k, lang="en")
            docs = retriever.invoke(query)
            if docs:
                return "\n\n".join(d.page_content for d in docs)
        except Exception as e:
            last_err = e
    if last_err is not None:
        return f"wikipedia_lookup error: {last_err}"
    return f"no Wikipedia results for: {query}"


@tool
def wikipedia_lookup(query: str, k: int = 3) -> str:
    """Look up a topic on Wikipedia and return the joined page content of the top-k articles.

    Use for general factual questions about people, places, historical events, science
    concepts — anything encyclopedic that is not tied to the currently loaded document."""
    #: name, docstring and signature above are byte-identical to the pre-timeout version and
    #: must stay that way: R4 measured tool routing against exactly these, and the docstrings
    #: are load-bearing prompt engineering rather than documentation. The timeout wraps
    #: EXECUTION only. tests/test_voice_prep.py pins the interface against the old values.
    return _run_with_timeout("wikipedia_lookup", _wikipedia_lookup_body, query, k)


def _web_search_body(query: str, k: int) -> str:
    try:
        wrapper = DuckDuckGoSearchAPIWrapper(max_results=k)
        results = wrapper.results(query, max_results=k)
        if not results:
            return f"no web results for: {query}"
        parts = []
        for r in results:
            title = r.get("title", "")
            snippet = r.get("snippet", "") or r.get("body", "")
            link = r.get("link", "") or r.get("href", "")
            parts.append(f"{title}\n{snippet}\n{link}")
        return "\n\n".join(parts)
    except Exception as e:
        return f"web_search error: {e}"


@tool
def web_search(query: str, k: int = 5) -> str:
    """Search the web via DuckDuckGo and return the joined text snippets of the top-k results.

    Use for current events, news, or anything time-sensitive that Wikipedia will not have."""
    #: interface frozen for the same reason as `wikipedia_lookup` above.
    return _run_with_timeout("web_search", _web_search_body, query, k)


def make_search_current_doc(retrival) -> BaseTool:
    """Build a `search_current_doc` tool bound to a live Retrival instance.

    The returned tool searches ONLY the document currently loaded into the agent's
    Retrival index — not Wikipedia, not the web.
    """

    @tool
    def search_current_doc(query: str, k: int = 4) -> str:
        """Search the currently loaded document for the given query and return joined chunks.

        Use this when the user's question is about the specific document Marshall was loaded
        with (invoices, PDFs, web pages ingested at startup)."""
        try:
            result = retrival.invoke({"query": query, "k": k})
            docs = result.get("context", [])
            if not docs:
                return f"no chunks matched: {query}"
            return "\n\n".join(d.page_content for d in docs)
        except Exception as e:
            return f"search_current_doc error: {e}"

    return search_current_doc


def make_recall_user_memory(build_memory) -> BaseTool:
    """Build a `recall_user_memory` tool over the user's own long-term memory.

    `build_memory(messages, thread_id) -> UserMemory | None` is called on demand so the
    tool always reads the live conversation and stays scoped to the calling thread.
    `state` and `config` are injected by LangGraph and hidden from the LLM's tool schema,
    so the model only ever sees `query` and `k`.
    """

    @tool
    def recall_user_memory(query: str,
                           k: int = 4,
                           state: Annotated[dict, InjectedState] = None,
                           config: RunnableConfig = None) -> str:
        """Look up a fact the user ASKED you to remember from an earlier conversation.

        Returns verbatim snippets of the user's own past messages — their name, family,
        job, location, preferences, past projects.

        Call this ONLY when the user ASKS a question about themselves that you cannot
        already answer from the visible conversation, e.g. "what is my name?", "where do
        I live?", "what did I tell you about my job?", "do you remember my project?".

        Only a QUESTION can justify this tool. Do NOT call it when:
        - the user is TELLING you something about themselves — "my name is X", "I live in
          Y", "I work on Z" are statements, so there is nothing to look up;
        - the answer is already visible earlier in this conversation;
        - they greet you or make small talk ("hi", "hello there", "how are you?");
        - the question is about the world, the loaded document, or current events — use
          `wikipedia_lookup`, `search_current_doc`, or `web_search` instead.

        After calling it, quote the actual detail you found back to the user as part of a
        natural reply."""
        try:
            messages = (state or {}).get("messages", [])
            thread_id = ((config or {}).get("configurable") or {}).get("thread_id", "default")
            memory = build_memory(messages, thread_id)
            if memory is None:
                return "no user memory yet: the user has not told you anything about themselves."
            context = memory.retreival(query, k)
            if not context or not context.strip():
                return f"no user memory matched: {query}"
            return context
        except Exception as e:
            return f"recall_user_memory error: {e}"

    return recall_user_memory
