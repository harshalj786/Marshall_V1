"""R4's routing table, as a gate: llama3.1 must still pick the right tool.

    pytest tests/test_routing_matrix.py -v
    python tests/test_routing_matrix.py
    python scripts/check_routing.py          # same matrix, readable output, non-zero on a miss

`docs/architecture_review.md` R4 measured tool routing once, with a throwaway harness, and
closed with: *"no test in the repository exercises tool routing... Routing is verified as
of this commit and will silently rot; pinning even one route as a `needs_llama31`-marked
test would be the cheap fix."* This is that fix, for all six routes.

**Why nothing else catches this.** Every other `AgentGraph` test monkeypatches `agent.llm`
— `test_summarization_cycle.py` at 15 sites — because faking the model is what makes graph
behaviour assertable. That is the right call for graph tests and it is exactly why they are
blind to routing: they never ask the real model to choose. And `voice_main.py --check`,
which does load the agent, never sends it a prompt at all.

**So this one is different, and skips like it.** It needs Ollama up, llama3.1 pulled, the
document index built, and it spends a real minute or two doing it. Skipped means "not
verifiable on this machine", never "passing" — same contract as
`tests/test_voice_integration.py` and the `needs_ollama` tests. Each route runs on its own
`thread_id` (`route-math`, `route-doc`, ...), never `voice-main`, and long-term memory is a
dedicated seeded store, so a run neither reads nor writes what live voice sessions remember.

The harness lives in `scripts/check_routing.py` so the runnable script and this gate can
never drift into measuring two different things.
"""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import voice_main
from scripts.check_routing import (
    BOUNDARY,
    PASS,
    ROUTES,
    build_check_agent,
    narrated_tool_calls,
    run_route,
    seed_memory,
)


#: same probe as tests/test_ingestion_retrieval_split.py and test_memory_gate_resilience.py,
#: restated rather than imported because that is this repo's pattern for it
def _ollama_up() -> bool:
    try:
        import urllib.request
        urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=3)
        return True
    except Exception:
        return False


def _has_llama31() -> bool:
    """Reachable is not enough: these tests are about one specific model's choices.

    The chat graphs run on mistral, so an Ollama with only mistral pulled passes
    `_ollama_up` and would fail every route here for a reason that is not a regression.
    """
    from scripts.check_routing import model_available
    return model_available(voice_main.VOICE_MODEL)


needs_llama31 = pytest.mark.skipif(
    not (_ollama_up() and _has_llama31()),
    reason=f"needs Ollama on :11434 with {voice_main.VOICE_MODEL} pulled")

MUST_PASS = [route for route in ROUTES if route.must_pass]
BOUNDARY_ROUTE = next(route for route in ROUTES if not route.must_pass)


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    """Build the agent once, seed memory once, run all six routes once.

    Module-scoped because the cost is the model, not the assertions: a per-test agent would
    re-open the index and re-seed six times over. Each route still gets its own thread, so
    sharing the agent shares no conversation.

    Every directory is this run's own tmp, and that is the point: the script's defaults are
    `db/voice_vdb` (the LIVE voice document index) and `db/routing_check_*`, which a hand-run
    of scripts/check_routing.py also uses. A suite that wrote there raced the live pipeline
    for one Chroma directory and mutated data a manual run depends on. The index is rebuilt
    from the same `voice_main.DEFAULT_DOCUMENT` with the same loader and chunking, so what is
    measured is unchanged — it costs one ingest of that document per module (~6s), which is
    small beside the model time these routes already spend.

    A tmp store is also fresh by construction, so the determinism `fresh=True` used to buy —
    no rows from an earlier run, in particular the routes' own questions from before H6 —
    now comes from the directory itself.
    """
    root = tmp_path_factory.mktemp("routing_check")
    agent = build_check_agent(
        vector_db_path=str(root / "vdb"),
        user_history_path=str(root / "user_history"),
        summary_path=str(root / "summaries"),
    )
    seed_memory(agent)
    return {route.name: run_route(agent, route) for route in ROUTES}


# --------------------------------------------------------------------------
# the five that must pass
# --------------------------------------------------------------------------

@needs_llama31
@pytest.mark.parametrize("route", MUST_PASS, ids=lambda route: route.name)
def test_route_selects_the_expected_tool(results, route):
    result = results[route.name]
    assert result.error is None, f"{route.name} raised: {result.error}"
    assert route.expected_tool in result.tools_called, (
        f"{route.name}: {route.query!r} expected {route.expected_tool}, "
        f"got {result.tools_called or 'no tool call at all'}")


@needs_llama31
@pytest.mark.parametrize("route", MUST_PASS, ids=lambda route: route.name)
def test_route_emits_a_structured_call_not_prose(results, route):
    """The mistral-style failure, named explicitly.

    A reply that writes the call out — `<function=calculator`, a JSON blob — with no
    `tool_calls` behind it looks like an answer in the transcript and is a silent miss.
    Only a hit with NO real call is condemned; the same shape alongside a real call is the
    model quoting its own work.
    """
    result = results[route.name]
    assert not (result.narrated and not result.calls), (
        f"{route.name}: the reply described a tool call in prose and emitted none — "
        f"{result.narrated}")


@needs_llama31
@pytest.mark.parametrize("route", MUST_PASS, ids=lambda route: route.name)
def test_route_reaches_an_answer(results, route):
    """Routing is the subject, but a route that ends in silence is still broken.

    Deliberately not asserting the CONTENT: a correctly-routed `wikipedia_lookup` may
    return `"wikipedia_lookup error: timed out after 10s"` under finding K's 10 s budget,
    and the model narrating that timeout is a network fact, not a routing regression.
    """
    result = results[route.name]
    assert result.answer.strip(), f"{route.name} produced no answer"


# --------------------------------------------------------------------------
# H6's witness, live: recall must return the memory, not the question
# --------------------------------------------------------------------------

@needs_llama31
def test_recall_returns_the_seeded_fact_not_the_query(results):
    """The end-to-end half of `tests/test_memory_read_path.py`.

    Routing correctly to `recall_user_memory` was never the hard part — this route passed
    before H6 was fixed. What it returned was the user's own question, because
    `UserMemory.__init__` indexed the query on the way in and `k=1` then retrieved it. The
    reply still said "Your name is Harshal" for an unrelated reason (the system prompt
    names him), which is why only the tool's own output can witness this.
    """
    result = results["user"]
    returned = " ".join(text for name, text in result.returns
                        if name == "recall_user_memory")

    assert returned.strip(), "recall_user_memory was called but returned nothing"
    assert "Harshal" in returned, (
        f"recall returned {returned[:120]!r} — not the seeded fact")
    assert "what is my name" not in returned.lower(), (
        "the query came back as a memory: the read path is writing again (H6)")


# --------------------------------------------------------------------------
# the sixth: a category boundary, reported as one
# --------------------------------------------------------------------------

@needs_llama31
def test_no_tool_route_is_a_boundary_not_a_hard_regression(results):
    """R4 measured *"what do you think makes a good explanation?"* at 0/2 on both the
    current and the reconstructed pre-H1 prompt: it reads factually enough to pull an 8B
    model to `wikipedia_lookup`, while `hi`, `how are you?` and "tea or coffee?" all
    correctly take no tool. So a tool call here is xfailed — recorded, visible, and not a
    build failure — while prose narration and an empty answer still fail, because neither
    is a category judgement.
    """
    result = results[BOUNDARY_ROUTE.name]
    assert result.error is None, f"boundary route raised: {result.error}"
    assert result.answer.strip(), "the boundary route produced no answer at all"
    assert not (result.narrated and not result.calls), (
        f"the boundary route described a tool call in prose and emitted none — "
        f"{result.narrated}")
    if result.verdict == BOUNDARY:
        pytest.xfail(f"known R4 boundary: routed to {', '.join(result.tools_called)} "
                     f"instead of answering directly (measured 0/2 in R4, unchanged)")
    assert result.verdict == PASS


# --------------------------------------------------------------------------
# the detector itself — no model needed
# --------------------------------------------------------------------------

TOOL_NAMES = ["calculator", "wikipedia_lookup", "web_search",
              "search_current_doc", "recall_user_memory"]


@pytest.mark.parametrize("text", [
    '<function=calculator{"expression": "47 * 82"}</function>',
    '{"name": "web_search", "parameters": {"query": "news"}}',
    "I'll call calculator(expression='47*82') for you.",
    "functools[wikipedia_lookup[{'query': 'transistor'}]]",
])
def test_prose_narrated_calls_are_detected(text):
    """The four shapes R4 and the code comments actually saw. If this stops firing, the
    gate above stops catching the failure it exists for."""
    assert narrated_tool_calls(text, TOOL_NAMES), f"missed a narrated call in {text!r}"


@pytest.mark.parametrize("text", [
    "47 times 82 is 3,854.",
    "Bardeen, Brattain and Shockley invented the transistor.",
    "Your name is Harshal, and you live in Pune.",
    "",
])
def test_ordinary_answers_are_not_flagged(text):
    """A false positive here would fail a route that routed correctly, so the detector
    stays narrow: a bare name is a mention, only a call-shaped token is a call."""
    assert not narrated_tool_calls(text, TOOL_NAMES), f"false positive on {text!r}"


def test_every_route_has_a_dedicated_thread_and_none_is_voice_main():
    """The live voice conversation must never be a test fixture."""
    threads = [route.thread_id for route in ROUTES]
    assert "voice-main" not in threads
    assert len(set(threads)) == len(threads), "two routes would share a history"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--no-header", "-p", "no:cacheprovider"]))
