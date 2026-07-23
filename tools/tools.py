"""Tools available to the Marshall agent.

Static tools importable directly:
    calculator, wikipedia_lookup, web_search

Factory (built at agent-init because it closes over a live Retrival instance):
    make_search_current_doc(retrival) -> BaseTool
"""
import ast
import operator
from typing import Optional

from langchain_core.tools import tool, BaseTool
from langchain_community.retrievers import WikipediaRetriever
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper


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


@tool
def wikipedia_lookup(query: str, k: int = 3) -> str:
    """Look up a topic on Wikipedia and return the joined page content of the top-k articles.

    Use for general factual questions about people, places, historical events, science
    concepts — anything encyclopedic that is not tied to the currently loaded document."""
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
def web_search(query: str, k: int = 5) -> str:
    """Search the web via DuckDuckGo and return the joined text snippets of the top-k results.

    Use for current events, news, or anything time-sensitive that Wikipedia will not have."""
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
