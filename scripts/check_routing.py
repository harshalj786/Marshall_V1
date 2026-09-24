"""Routing regression gate for the voice-configured agent.

    python scripts/check_routing.py              # all six routes
    python scripts/check_routing.py --route math # one route, for a fast retry
    pytest tests/test_routing_matrix.py -v       # the same matrix, as a gate

**The gap this closes.** `voice_main.py --check` loads every model, probes the mic and
runs a TTS -> STT round trip — and never sends a single prompt to the LLM. `boot()`
constructs the `AgentGraph`, which opens the document index and builds `ChatOllama`
(`pipeline_lg.py:657`), but nothing invokes it. So `--check` can pass on a machine where
llama3.1 was never pulled, or where it answers but no longer routes tools. This script is
the other half: the model actually answers, and the tool it chose is read back off the
graph state rather than inferred from the prose.

**Why it exists at all.** `docs/architecture_review.md` R4 measured this table once, with
a throwaway harness, and the review says so in as many words: *"no test in the repository
exercises tool routing... Routing is verified as of this commit and will silently rot."*
Every other `AgentGraph` test monkeypatches `agent.llm`, deliberately — faking the model
is what makes graph behaviour assertable — which is exactly why nothing else can catch a
routing regression. This makes R4's table permanent.

**What counts as a pass.** The tool was *selected*. A correctly-routed `wikipedia_lookup`
that returns `"wikipedia_lookup error: timed out after 10s"` still passes: finding K's
timeout (`tools/tools.py:27`) bounds network tools at 10 s, and whether Wikipedia answered
today says nothing about whether the model routed. What is asserted is the `tool_calls` on
the `AIMessage`, never the answer text — the system prompt itself contains "Harshal"
(`pipeline_lg.py:396`), so an answer naming him proves nothing about `recall_user_memory`.

**The failure this is really hunting.** An `AIMessage` that *describes* calling a tool in
prose — `<function=calculator`, or a JSON blob — with no structured `tool_calls` behind it.
That is the mistral symptom the code comments keep warning about, and it is invisible to a
human reading the transcript, because the reply looks like an answer. `narrated_tool_calls`
names it explicitly and fails the route.

**Two R4 nuances, so a correct result does not read as a failure.**

1. *The no-tool route is a category boundary, not a regression.* R4 measured
   *"what do you think makes a good explanation?"* at 0/2 on both the current and the
   pre-H1 prompt: it reads factually enough to pull an 8B model to Wikipedia, while `hi`,
   `how are you?` and *"do you prefer tea or coffee?"* all correctly take no tool. It is
   reported as KNOWN-BOUNDARY and does not, alone, fail the run. A prose-narrated call on
   that route still does — that is an emission defect, not a category call.
2. *Never reduce the bound tool list to reproduce a routing bug.* R4: llama3.1 emits proper
   `tool_calls` with four or five tools bound and falls back to prose JSON with exactly
   **one** — mangling the name into the description's first word. `AgentGraph` always binds
   five, so the real system never hits it, but a minimal one-tool reproduction shows a
   failure that does not exist.

Nothing here modifies routing: no prompt edits, no tool renames, no graph changes. The
document index is the voice path's own (`db/voice_vdb`, already ingested, so this is a
read); long-term memory is a dedicated store seeded by this script, so a run neither
depends on nor pollutes what the live voice sessions remember.
"""
import argparse
import json
import re
import shutil
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

#: every agent knob comes from here rather than being restated, so the day the voice path
#: changes model or keep_alive, this check follows it instead of quietly testing the old one
import voice_main

#: Long-term memory gets its own directory rather than the voice path's `db/voice_user_history`.
#: The recall route needs a KNOWN fact to find, which means seeding — and seeding into the
#: live store would write check fixtures into the user's real profile. Separate store, seeded
#: here, deterministic on a machine that has never run a voice session.
CHECK_USER_HISTORY_PATH = "db/routing_check_user_history"
#: same reasoning for the conversation summary. One turn per thread never trips the
#: summariser, but pointing it at the voice path's summaries would be a write waiting to happen.
CHECK_SUMMARY_PATH = "db/routing_check_summaries"

#: seeded as the user's own past messages, which is what `UserMemory` stores. Statements,
#: not questions: this is the earlier conversation `recall_user_memory` is supposed to reach
#: back into. Content-hash ids make re-seeding idempotent, so re-running never duplicates.
SEED_MESSAGES = (
    "My name is Harshal.",
    "I live in Pune.",
    "I am building Marshall, a local RAG agent, in my spare time.",
)

OLLAMA_TAGS_URL = "http://127.0.0.1:11434/api/tags"


# ==========================================================================
# the matrix
# ==========================================================================

@dataclass(frozen=True)
class Route:
    """One row of R4's routing table."""
    name: str
    thread_id: str
    query: str
    #: the tool that must appear in a real `tool_calls`; None means "answer directly"
    expected_tool: Optional[str]
    #: False only for the no-tool row — see the module docstring, nuance 1
    must_pass: bool = True


#: R4's table verbatim (`docs/architecture_review.md:793`). Each route gets its own
#: thread_id so no history is shared — a route that inherited the previous answer would be
#: measuring the conversation, not the routing. None of these is `voice-main`, so a run
#: never touches the live voice conversation.
ROUTES = (
    Route("math", "route-math",
          "what is 47 times 82?", "calculator"),
    Route("document", "route-doc",
          "what does the document say about OPVC pipes?", "search_current_doc"),
    Route("user", "route-user",
          "what is my name?", "recall_user_memory"),
    Route("encyclopedic", "route-wiki",
          "who invented the transistor?", "wikipedia_lookup"),
    Route("web", "route-web",
          "what are the top news stories today?", "web_search"),
    #: R4's table calls this row "no tool"; hyphenated here so `--route no-tool` and the
    #: pytest id do not need quoting
    Route("no-tool", "route-none",
          "what do you think makes a good explanation?", None, must_pass=False),
)

PASS = "PASS"
FAIL = "FAIL"
BOUNDARY = "KNOWN-BOUNDARY"
ERROR = "ERROR"


@dataclass
class RouteResult:
    """What one route actually did. Built by `run_route`, asserted by the pytest gate."""
    route: Route
    verdict: str
    #: [(tool_name, json args)] in call order, off `AIMessage.tool_calls`
    calls: list = field(default_factory=list)
    #: [(tool_name, first line of what it returned)] off the `ToolMessage`s
    returns: list = field(default_factory=list)
    #: [(tool_name, excerpt)] where the reply DESCRIBED a call it never made
    narrated: list = field(default_factory=list)
    answer: str = ""
    seconds: float = 0.0
    error: Optional[str] = None

    @property
    def tools_called(self) -> list:
        return [name for name, _ in self.calls]

    @property
    def is_regression(self) -> bool:
        """Does this route failing mean routing broke? The boundary row says no."""
        return self.verdict in (FAIL, ERROR)


# ==========================================================================
# reading the trace
# ==========================================================================

def observed_calls(history) -> list:
    """Every real tool call in a thread's history, in order.

    Reads `AIMessage.tool_calls` — the structured field the `ToolNode` dispatches on — so
    this sees what LangGraph saw, not what the reply claims.
    """
    calls = []
    for message in history:
        for call in getattr(message, "tool_calls", None) or []:
            name = call.get("name", "") if isinstance(call, dict) else getattr(call, "name", "")
            args = call.get("args", {}) if isinstance(call, dict) else getattr(call, "args", {})
            calls.append((name, json.dumps(args, ensure_ascii=False, default=str)))
    return calls


def observed_returns(history) -> list:
    """What each tool handed back, one short line each.

    Printed rather than asserted: a tool that timed out or rate-limited still routed
    correctly, and the whole point of separating these two is to keep a network flake from
    reading as a routing regression.
    """
    from langchain_core.messages import ToolMessage

    returns = []
    for message in history:
        if isinstance(message, ToolMessage):
            text = " ".join(str(message.content or "").split())
            returns.append((getattr(message, "name", "?"), text))
    return returns


def narrated_tool_calls(text: str, tool_names) -> list:
    """Tool calls WRITTEN OUT in prose. Returns [(tool_name, excerpt)].

    The failure mode: the model wants a tool, and instead of emitting `tool_calls` it types
    the call into its reply — `<function=calculator`, `{"name": "web_search", ...}`,
    `calculator(expression="47*82")`. LangGraph sees no tool call, so the loop ends and the
    text ships to the user as an answer. R4 reproduced it with a single tool bound; the
    code comments in `pipeline_lg.py` reach it by advertising an unbound tool in the prompt.

    Deliberately narrow. A name alone is not a hit — the reply may legitimately say "I used
    the calculator" — so a call-shaped token must follow it: a bracket, or the `name` key of
    a serialised call. `run_route` only lets a hit FAIL a route when the turn produced no
    real `tool_calls` at all, which is the shape the review actually warns about.
    """
    hits = []
    haystack = text or ""
    for name in tool_names:
        escaped = re.escape(name)
        patterns = (
            #: llama3.1's own function-call token, leaked into the content
            rf"<function\s*=\s*{escaped}",
            #: a serialised tool call pasted into the reply, single or double quoted
            rf"[\"']name[\"']\s*:\s*[\"']{escaped}[\"']",
            #: `calculator(...)`, `calculator[...]`, `calculator{...}` — a call, not a mention
            rf"\b{escaped}\s*[\(\[{{]",
        )
        for pattern in patterns:
            match = re.search(pattern, haystack, re.IGNORECASE)
            if match:
                start = max(0, match.start() - 20)
                excerpt = " ".join(haystack[start:match.end() + 60].split())
                hits.append((name, excerpt))
                break
    return hits


def _final_narration(history, tool_names) -> list:
    """Scan only AI messages that carry no structured calls of their own."""
    from langchain_core.messages import AIMessage

    hits = []
    for message in history:
        if isinstance(message, AIMessage) and not getattr(message, "tool_calls", None):
            hits.extend(narrated_tool_calls(message.content, tool_names))
    return hits


# ==========================================================================
# the agent under test
# ==========================================================================

def build_check_agent(file_path: Optional[str] = None,
                      fresh: bool = False,
                      vector_db_path: Optional[str] = None,
                      user_history_path: Optional[str] = None,
                      summary_path: Optional[str] = None):
    """The voice path's agent, with long-term memory pointed at a dedicated store.

    Every other argument is `voice_main`'s own constant, imported rather than copied: same
    llama3.1 for generation AND the memory gate, same keep_alive, same document. What this
    measures is therefore the agent the voice loop talks to, not a lookalike built for it.

    The three directories are parameters, defaulting to the constants above, for ONE reason:
    run by hand this script should use its own stores as it always has, but the pytest gate
    in tests/test_routing_matrix.py must not write anywhere the live voice pipeline or a
    hand-run of this script reads. `db/voice_vdb` in particular is the live document index —
    with it as the default, a suite running while a voice session is live put two processes
    on one Chroma directory. The gate passes its own tmp paths; nothing else changes.
    """
    from pipeline_lg import AgentGraph

    user_history = user_history_path or CHECK_USER_HISTORY_PATH
    if fresh:
        shutil.rmtree(user_history, ignore_errors=True)
    document = file_path or voice_main.DEFAULT_DOCUMENT
    return AgentGraph(
        file_path=document,
        mode_docs="pdf" if str(document).lower().endswith(".pdf") else "txt",
        persist_directory_path=vector_db_path or voice_main.VECTOR_DB_PATH,
        model=voice_main.VOICE_MODEL,
        memory_model=voice_main.VOICE_MODEL,
        keep_alive=voice_main.KEEP_ALIVE,
        user_history_directory_path=user_history,
        summary_persist_path=summary_path or CHECK_SUMMARY_PATH,
    )


def seed_memory(agent) -> None:
    """Give `recall_user_memory` something to find, before the user route runs.

    Written through the agent's OWN collection name and directory rather than a hardcoded
    pair, so the seed cannot land in a collection the recall tool does not read — the
    silent failure that would make a working route look like a regression.
    """
    from langchain_core.messages import HumanMessage
    from memory.long_term_memory import UserMemory

    Path(agent.user_history_directory_path).mkdir(parents=True, exist_ok=True)
    #: `.index()` is explicit since H6 — constructing a UserMemory opens the collection and
    #: writes nothing. Without it the seed would never reach Chroma and the recall route
    #: would fail for a reason that has nothing to do with routing.
    UserMemory(
        chat_history=[HumanMessage(content=text) for text in SEED_MESSAGES],
        collection_name=agent.user_memory_collection,
        persist_directory_path=agent.user_history_directory_path,
        model=voice_main.VOICE_MODEL,
    ).index()


def run_route(agent, route: Route) -> RouteResult:
    """Invoke one route on its own thread and read back what it did. Never raises."""
    tool_names = [tool.name for tool in agent.tools]
    started = time.perf_counter()
    try:
        answer = agent.invoke(
            route.query, config={"configurable": {"thread_id": route.thread_id}})
    except Exception as exc:
        return RouteResult(route=route, verdict=ERROR,
                           seconds=time.perf_counter() - started,
                           error=f"{type(exc).__name__}: {exc}")
    seconds = time.perf_counter() - started
    history = agent.get_history(route.thread_id)
    result = RouteResult(
        route=route,
        verdict=FAIL,
        calls=observed_calls(history),
        returns=observed_returns(history),
        narrated=_final_narration(history, tool_names),
        answer=" ".join(str(answer or "").split()),
        seconds=seconds,
    )

    #: a narrated call only condemns the turn when NOTHING real was emitted. With a real
    #: call in hand the turn worked, and the same pattern in the final answer is far more
    #: likely to be the model quoting itself than the emission bug.
    narration_is_the_failure = bool(result.narrated) and not result.calls

    if route.expected_tool is None:
        if narration_is_the_failure:
            result.verdict = FAIL
        elif result.calls:
            #: R4 measured this row at 0/2. Reported, and not a regression on its own.
            result.verdict = BOUNDARY
        else:
            result.verdict = PASS
    else:
        if route.expected_tool in result.tools_called:
            result.verdict = PASS
        else:
            result.verdict = FAIL
    return result


# ==========================================================================
# preflight
# ==========================================================================

def ollama_up() -> bool:
    try:
        urllib.request.urlopen(OLLAMA_TAGS_URL, timeout=3)
        return True
    except Exception:
        return False


def model_available(model: str) -> bool:
    """Is `model` pulled? Tag-insensitive: llama3.1 is served as `llama3.1:latest`."""
    try:
        with urllib.request.urlopen(OLLAMA_TAGS_URL, timeout=3) as response:
            tags = json.load(response)
    except Exception:
        return False
    wanted = model.split(":")[0]
    return any(str(entry.get("name", "")).split(":")[0] == wanted
               for entry in tags.get("models", []))


# ==========================================================================
# output
# ==========================================================================

def print_result(result: RouteResult) -> None:
    route = result.route
    expected = route.expected_tool or "no tool (answer directly)"
    print(f"  route      {route.name}")
    print(f"  query      {route.query!r}")
    print(f"  thread     {route.thread_id}")
    print(f"  expected   {expected}"
          f"{'' if route.must_pass else '   [boundary row - see R4]'}")
    if result.error:
        print(f"  error      {result.error}")
    for name, args in result.calls:
        print(f"  called     {name}({args})")
    if not result.calls and not result.error:
        print("  called     - nothing -")
    for name, text in result.returns:
        #: truncated hard: one wikipedia_lookup result can be tens of thousands of chars
        print(f"  returned   {name}: {text[:100]}{'...' if len(text) > 100 else ''}")
    for name, excerpt in result.narrated:
        marker = "PROSE!" if not result.calls else "prose "
        print(f"  {marker}     {name} described in the reply: ...{excerpt}...")
    if result.answer:
        print(f"  answer     {result.answer[:160]}{'...' if len(result.answer) > 160 else ''}")
    print(f"  timing     {result.seconds:.1f}s")
    print(f"  verdict    {result.verdict}")
    print()


def print_summary(results: list, elapsed: float) -> bool:
    """Print the closing block. Returns True when routing is intact."""
    must_pass = [r for r in results if r.route.must_pass]
    correct = [r for r in must_pass if r.verdict == PASS]
    regressions = [r for r in results if r.is_regression]
    boundaries = [r for r in results if r.verdict == BOUNDARY]
    narrated = [r for r in results if r.narrated and not r.calls]

    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"  must-pass routes   {len(correct)}/{len(must_pass)} routed correctly")
    for result in boundaries:
        print(f"  known boundary     {result.route.name} -> "
              f"{', '.join(result.tools_called)}  (R4 measured 0/2; not a regression)")
    print(f"  prose-narrated     {len(narrated)}"
          f"{'  <- the mistral-style failure' if narrated else ''}")
    print(f"  elapsed            {elapsed:.1f}s")
    print()

    ok = not regressions
    if ok:
        print(f"ROUTING OK - {voice_main.VOICE_MODEL} answers and selects the right tool "
              f"on all {len(must_pass)}.")
    else:
        print("ROUTING REGRESSION - " + ", ".join(
            f"{r.route.name} (expected {r.route.expected_tool}, got "
            f"{', '.join(r.tools_called) or 'nothing'})" for r in regressions))
        print("Suspect the model first: see the memory note on tool-calling models, and "
              "confirm the prompt still describes exactly the bound tools.")
    print("=" * 72)
    return ok


# ==========================================================================
# cli
# ==========================================================================

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="scripts/check_routing.py",
        description="Verify the voice-configured agent still routes each query to the "
                    "right tool (R4's table).")
    parser.add_argument("--file", default=None,
                        help=f"document to load (default: {voice_main.DEFAULT_DOCUMENT})")
    parser.add_argument("--route", default=None, choices=[r.name for r in ROUTES],
                        help="run a single route instead of all six")
    parser.add_argument("--fresh", action="store_true",
                        help=f"delete {CHECK_USER_HISTORY_PATH} and re-seed it first")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if not ollama_up():
        print("Ollama is not reachable on :11434 - start it and try again.", file=sys.stderr)
        return 2
    if not model_available(voice_main.VOICE_MODEL):
        print(f"{voice_main.VOICE_MODEL} is not pulled: ollama pull {voice_main.VOICE_MODEL}",
              file=sys.stderr)
        return 2
    document = args.file or voice_main.DEFAULT_DOCUMENT
    if not Path(document).exists():
        print(f"no such document: {document}", file=sys.stderr)
        return 2

    print("=" * 72)
    print("ROUTING CHECK")
    print("=" * 72)
    print(f"  model      {voice_main.VOICE_MODEL} (generation and memory gate)")
    print(f"  document   {document}")
    print(f"  doc index  {voice_main.VECTOR_DB_PATH}")
    print(f"  memory     {CHECK_USER_HISTORY_PATH}  (seeded here, not the live voice store)")
    print()

    started = time.perf_counter()
    try:
        agent = build_check_agent(document, fresh=args.fresh)
        seed_memory(agent)
    except Exception as exc:
        print(f"could not build the agent: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    bound = [tool.name for tool in agent.tools]
    print(f"  tools      {len(bound)} bound: {', '.join(bound)}")
    #: R4's trap, stated where someone about to "simplify" this would read it
    print("             (llama3.1 narrates calls as prose when only ONE tool is bound - "
          "never trim this list to reproduce a routing bug)")
    print(f"  seeded     {len(SEED_MESSAGES)} user messages, so recall has something to find")
    print()

    routes = [r for r in ROUTES if args.route is None or r.name == args.route]
    results = []
    for index, route in enumerate(routes, start=1):
        print("-" * 72)
        print(f"ROUTE {index}/{len(routes)}")
        print("-" * 72)
        result = run_route(agent, route)
        results.append(result)
        print_result(result)

    return 0 if print_summary(results, time.perf_counter() - started) else 1


if __name__ == "__main__":
    sys.exit(main())
