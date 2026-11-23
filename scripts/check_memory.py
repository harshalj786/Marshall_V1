"""Memory check: does the voice-configured agent recall the right fact about the user?

    python scripts/check_memory.py                  # all seven cases
    python scripts/check_memory.py --case pet       # one case, for a fast retry
    python scripts/check_memory.py --json out/memory.json   # also write the eval record
    python scripts/run_evals.py --only memory       # the same, through the runner

**What it tests.** The path a user actually depends on: the model CHOOSES
`recall_user_memory`, the Chroma `UserMemory` store RANKS the right snippet into the
top k, and the answer USES it. Each case is graded on all three, read off the graph state:
the tool in `AIMessage.tool_calls`, the fact in the `ToolMessage` the tool returned, and
the phrases in the final answer — except a `retrieval_only` case, which grades only the
first two (see below). `memory/facts.py` is not on this path at all (only its own
tests import it), so nothing here repeats tests/test_facts.py.

**Why the facts are what they are.** The system prompt already contains "Harshal" AND,
as statement examples, "I live in Pune" and "I work on a RAG agent"
(pipeline_lg.py:433, 448-450). A case built on any of those passes for the wrong reason. Every
graded phrase here is absent from the prompt and the bound tools' descriptions —
tests/test_memory_check.py enforces it — and "Pune" is an `answer_lacks` trap: an answer
naming it came from the prompt, not from memory.

**Two layers, two must-pass cases.** `ranking` grades the whole chain (retrieve -> answer).
`born` is RETRIEVAL-ONLY: recall chosen + the right snippet returned, answer not graded. On
llama3.1 the answer step drops a correctly retrieved fact on almost every question
("I don't have any information about...") — that defect is Feature 5's, and the cases that
hit it are known boundaries. `born` keeps the retrieval layer pinned regardless, and its
record says "retrieval-only" so a green `born` is never read as "the answer used the fact".

**Why eleven seeds.** `recall_user_memory` returns the top k (default 4)
verbatim snippets. With three seeds it returns all of them whatever the query, so "the
tool returned the fact" would prove nothing. Ten statements (case facts + distractors)
make the top k actually rank.

**Isolation.** Every attempt runs on its own fresh thread, so a fact can only come from
the store, never from earlier messages in the conversation. The store is this script's
own (`db/memory_check_user_history`), wiped and re-seeded on every run — never the live
voice store. Single-turn threads never fire the memory gate (turn 5) and recall is
read-only (H6), so the seeded store stays exactly the seeds across attempts and k-of-n
attempts stay independent.

`carry-over` is the exception, and the one case that tests the real conversational path:
five statements in thread A (the gate fires on turn 5 and WRITES them), then the question
in a fresh thread B. Because that case writes, each attempt gets a fresh store AND a fresh
agent — otherwise attempt 1's write would make attempts 2..n pass for free.

**k-of-n.** llama3.1 at temperature 0 still varies run to run. Cases marked `flaky` run up
to n times and need k passes (default 3 of 5), stopping early once the verdict is
settled. Solid cases run once. The record carries attempts/passes/required and, in
`detail`, why each missed attempt missed.

**Timing.** Measured with eval_metrics.TurnTimer (a LangChain callback), not a history
slice: it is the one approach a summariser rewrite cannot corrupt. Its model_seconds is
WIDER than routing's (it also counts gate/summariser calls) — see its docstring before
comparing the two.

Exit codes are the shared contract's: 0 nothing regressed, 1 a case regressed, 2 could
not run. Nothing here modifies the agent, the prompt or the memory code.
"""
import argparse
import re
import shutil
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import voice_main  # noqa: E402

from scripts.eval_contract import (BOUNDARY, ERROR, FAIL, PASS,  # noqa: E402
                                   CaseResult, CheckRecord, write_record)
from scripts.eval_metrics import TurnTimer, timer_metrics  # noqa: E402

CHECK_USER_HISTORY_PATH = "db/memory_check_user_history"
#: per-attempt stores for carry-over, one subdirectory each (see the module docstring)
CHECK_CARRY_PATH = "db/memory_check_carry"
CHECK_SUMMARY_PATH = "db/memory_check_summaries"

RECALL = "recall_user_memory"


# ==========================================================================
# the cases
# ==========================================================================

@dataclass(frozen=True)
class MemoryCase:
    id: str
    #: asked on a FRESH thread
    query: str
    #: must appear in AIMessage.tool_calls; None = not graded
    require_tool: Optional[str] = None
    #: each must appear in what the tool returned (all its calls this turn, joined)
    tool_returned: tuple = ()
    #: each must appear in the answer
    answer_has: tuple = ()
    #: at least one must appear in the answer
    answer_any: tuple = ()
    #: none may appear in the answer
    answer_lacks: tuple = ()
    #: statements seeded into the shared store that this case relies on
    seeds: tuple = ()
    #: statements said in an earlier thread instead of seeded (carry-over)
    told_in_thread: tuple = ()
    #: False -> a miss is KNOWN-BOUNDARY, not FAIL
    must_pass: bool = True
    #: for a must_pass=False case: WHY it misses today, one line. Prefixed to the record's
    #: `detail` whenever the case lands on KNOWN-BOUNDARY, so the reason travels with it.
    known_gap: str = ""
    #: True -> grade the tool layer only (recall chosen + snippet returned); the answer is
    #: NOT graded and the record's `detail` always says so (RETRIEVAL_ONLY_NOTE)
    retrieval_only: bool = False
    #: True -> k-of-n; otherwise the case runs once
    flaky: bool = False
    k: int = 3
    n: int = 5

    @property
    def runs(self) -> tuple:
        """(k, n) actually used: a solid case is 1-of-1."""
        return (self.k, self.n) if self.flaky else (1, 1)


#: statements no case grades on. They exist so the top 4 has to rank (module docstring).
#: Nothing colour-related may appear here or in any seed: the `unknown` case asks for it.
DISTRACTORS = (
    "I play the guitar on weekends.",
    "My favourite food is masala dosa.",
    "I'm learning Japanese.",
    "I wake up at six every morning.",
)

COLOURS = ("red", "blue", "green", "yellow", "purple", "orange", "pink", "black", "white")

#: BASELINE 2026-11-23: memory check 2/7 must-pass (ranking full-chain, born retrieval-only),
#: 4 known boundaries (pet, two-facts, paraphrase, carry-over) + unknown (Feature 5 target).
#: The boundaries are retrieval/model gaps — Feature 5's job, not this harness's — so they are
#: must_pass=False and a green run means "no worse than the 23 Nov baseline".
#: Feature 5 must flip them back to must_pass=True and make them pass (docs/progress.md).
BASELINE_DATE = "2026-11-23"

#: always at the front of a retrieval_only case's `detail`, pass or miss
RETRIEVAL_ONLY_NOTE = "retrieval-only: answer not graded"

CASES = (
    MemoryCase(
        "pet", "what's my dog called?",
        require_tool=RECALL, tool_returned=("Biscuit",), answer_has=("Biscuit",),
        seeds=("My dog is called Biscuit.",),
        must_pass=False,
        known_gap="recall returned the fact, but the answer often doesn't say it"),
    MemoryCase(
        "ranking", "where did I grow up?",
        require_tool=RECALL, tool_returned=("Nagpur",), answer_has=("Nagpur",),
        #: the prompt's own example city — an answer naming it did not come from memory
        answer_lacks=("Pune",),
        seeds=("I grew up in Nagpur.", "My office is in Bangalore.")),
    MemoryCase(
        "born", "what year was I born?",
        #: RETRIEVAL-ONLY. Trial 23 Nov, 5 warm runs: recall chosen and '1998' returned 5/5,
        #: answer denied knowing it 5/5. The answer side is Feature 5's (see pet/carry-over).
        require_tool=RECALL, tool_returned=("1998",),
        seeds=("I was born in 1998.",),
        retrieval_only=True),
    MemoryCase(
        "two-facts", "what's my sister's name, and what is she studying?",
        require_tool=RECALL,
        #: two DIFFERENT snippets must both make the top k: neither one answers alone
        tool_returned=("sister is called Ananya", "studying medicine"),
        answer_has=("Ananya", "medicine"),
        seeds=("My sister is called Ananya.", "Ananya is studying medicine."),
        must_pass=False,
        known_gap="model requests too small a k; the second fact gets crowded out"),
    MemoryCase(
        "paraphrase", "is there any food I should stay away from?",
        require_tool=RECALL, tool_returned=("peanut",), answer_has=("peanut",),
        seeds=("I'm allergic to peanuts.",),
        #: no word overlap with the seed, and the likeliest case to skip the tool
        flaky=True,
        must_pass=False,
        known_gap="routes to Wikipedia instead of memory"),
    MemoryCase(
        "unknown", "what's my favourite colour?",
        #: Feature 5's Done-when ("unknown fact -> I don't know"). Not built yet, so a miss
        #: is KNOWN-BOUNDARY: it documents the target without pretending it is met.
        answer_any=("don't know", "do not know", "haven't told", "have not told",
                    "not sure", "don't have", "do not have", "no information",
                    "didn't mention", "haven't mentioned", "don't remember", "not aware",
                    "didn't tell", "did not tell", "no record"),
        answer_lacks=COLOURS,
        must_pass=False),
    MemoryCase(
        "carry-over", "who is my favourite cricketer?",
        require_tool=RECALL, tool_returned=("Dravid",), answer_has=("Dravid",),
        #: five statements so the memory gate (every 5 turns) fires on the last one
        told_in_thread=(
            "I just got back from a trip to Goa.",
            "My favourite cricketer is Rahul Dravid.",
            "I've started running in the mornings.",
            "My laptop is a ThinkPad.",
            "I'm reading a book about Roman history.",
        ),
        flaky=True,
        must_pass=False,
        known_gap="the fact was stored and recall returned it, but the answer still denied "
                  "knowing it (model ignored retrieved context)"),
)


def all_seeds(cases=CASES) -> tuple:
    """Everything the shared store is seeded with, in a stable order, no duplicates."""
    out = []
    for case in cases:
        for statement in case.seeds:
            if statement not in out:
                out.append(statement)
    for statement in DISTRACTORS:
        if statement not in out:
            out.append(statement)
    return tuple(out)


# ==========================================================================
# grading (pure)
# ==========================================================================

def contains(text: str, phrase: str) -> bool:
    """Case-insensitive, anchored at a word start: 'red' must not match 'shared'.

    Only the left edge is anchored, so 'peanut' still matches 'peanuts'.
    """
    return re.search(r"(?<!\w)" + re.escape(phrase), text or "", re.IGNORECASE) is not None


def grade(case: MemoryCase, calls: list, returns: list, answer: str) -> Optional[str]:
    """Why this attempt missed, or None if it passed. Pure.

    `calls` is the tool names called this turn; `returns` is [(tool_name, text)].
    Every rule is checked, so the reason names every problem, not just the first.
    """
    problems = []
    if case.require_tool and case.require_tool not in calls:
        problems.append(f"{case.require_tool} not called"
                        f" (called: {', '.join(calls) or 'nothing'})")
    returned = " ".join(text for name, text in returns
                        if case.require_tool is None or name == case.require_tool)
    for phrase in case.tool_returned:
        if not contains(returned, phrase):
            problems.append(f"tool did not return {phrase!r}")
    for phrase in case.answer_has:
        if not contains(answer, phrase):
            problems.append(f"answer lacks {phrase!r}")
    if case.answer_any and not any(contains(answer, phrase) for phrase in case.answer_any):
        problems.append("answer has no 'don't know'-style phrase"
                        if case.id == "unknown" else
                        f"answer has none of {list(case.answer_any)}")
    for phrase in case.answer_lacks:
        if contains(answer, phrase):
            problems.append(f"answer contains {phrase!r}")
    return "; ".join(problems) or None


# ==========================================================================
# k-of-n (pure)
# ==========================================================================

@dataclass
class Attempt:
    """One run of one case."""
    passed: bool
    reason: Optional[str] = None      # why it missed (grade's text)
    error: Optional[str] = None       # exception text: the attempt could not complete
    wall_seconds: float = 0.0
    metrics: dict = field(default_factory=dict)
    calls: list = field(default_factory=list)
    returns: list = field(default_factory=list)
    answer: str = ""


def settled(passes: int, done: int, k: int, n: int) -> bool:
    """Is the verdict already decided? k reached, or k no longer reachable."""
    return passes >= k or passes + (n - done) < k


def run_k_of_n(case: MemoryCase, run_attempt: Callable[[int], Attempt]) -> list:
    """Run attempts 1..n, stopping as soon as the verdict is settled."""
    k, n = case.runs
    attempts = []
    passes = 0
    for number in range(1, n + 1):
        attempt = run_attempt(number)
        attempts.append(attempt)
        passes += attempt.passed
        if settled(passes, number, k, n):
            break
    return attempts


def _median(values):
    present = [v for v in values if v is not None]
    return statistics.median(present) if present else None


def _max(values):
    present = [v for v in values if v is not None]
    return max(present) if present else None


def aggregate(case: MemoryCase, attempts: list) -> CaseResult:
    """Fold a case's attempts into one CaseResult. Pure.

    Verdict: every attempt errored -> ERROR; passes >= k -> PASS; otherwise FAIL, or
    KNOWN-BOUNDARY for a must_pass=False case. An errored attempt counts as a miss.
    """
    k, _ = case.runs
    passes = sum(1 for a in attempts if a.passed)
    if attempts and all(a.error for a in attempts):
        verdict = ERROR
    elif passes >= k:
        verdict = PASS
    else:
        verdict = FAIL if case.must_pass else BOUNDARY

    misses = []
    for number, attempt in enumerate(attempts, start=1):
        if attempt.error:
            misses.append(f"a{number}: ERROR {attempt.error}")
        elif not attempt.passed:
            misses.append(f"a{number}: {attempt.reason or 'missed'}")

    metric = lambda key: [a.metrics.get(key) for a in attempts]  # noqa: E731
    vram_models = next((m for m in reversed(metric("vram_models")) if m is not None), None)
    median_int = lambda key: (None if _median(metric(key)) is None  # noqa: E731
                              else int(round(_median(metric(key)))))
    if verdict == BOUNDARY and case.known_gap:
        misses.insert(0, f"known boundary ({BASELINE_DATE} baseline): {case.known_gap}")
    if case.retrieval_only:
        misses.insert(0, RETRIEVAL_ONLY_NOTE)

    return CaseResult(
        id=case.id,
        verdict=verdict,
        wall_seconds=_median([a.wall_seconds for a in attempts]) or 0.0,
        error=next((a.error for a in attempts if a.error), None),
        model_seconds=_median(metric("model_seconds")),
        load_seconds=_max(metric("load_seconds")),
        prompt_tokens=median_int("prompt_tokens"),
        output_tokens=median_int("output_tokens"),
        vram_models=vram_models,
        vram_gpu_mib=_max(metric("vram_gpu_mib")),
        attempts=len(attempts),
        passes=passes,
        required=k,
        detail="; ".join(misses) or None,
    )


# ==========================================================================
# running against the agent
# ==========================================================================

def build_agent(user_history_path: str):
    """The voice path's agent (via check_routing), on THIS check's own stores."""
    from scripts.check_routing import build_check_agent

    return build_check_agent(user_history_path=user_history_path,
                             summary_path=CHECK_SUMMARY_PATH)


def seed_store(agent, statements) -> None:
    """Write `statements` as the user's own past messages, through the agent's OWN
    collection name and directory, so the seed lands exactly where recall reads."""
    from langchain_core.messages import HumanMessage
    from memory.long_term_memory import UserMemory

    Path(agent.user_history_directory_path).mkdir(parents=True, exist_ok=True)
    UserMemory(
        chat_history=[HumanMessage(content=text) for text in statements],
        collection_name=agent.user_memory_collection,
        persist_directory_path=agent.user_history_directory_path,
        model=voice_main.VOICE_MODEL,
    ).index()


def turn_trace(history) -> tuple:
    """(tool names called, [(tool, returned text)]) for a fresh thread's whole history.

    Safe to read the whole history ONLY because every graded turn is on a fresh thread:
    the history is exactly that one turn. Timing does not rely on this (TurnTimer).
    """
    from scripts.check_routing import observed_calls, observed_returns

    return ([name for name, _ in observed_calls(history)], observed_returns(history))


def ask(agent, case: MemoryCase, thread_id: str, timer: TurnTimer) -> Attempt:
    """Run the graded turn on a fresh thread and grade it. Never raises."""
    timer.reset()
    started = time.perf_counter()
    try:
        answer = agent.invoke(case.query, config={"configurable": {"thread_id": thread_id},
                                                  "callbacks": [timer]})
    except Exception as exc:
        return Attempt(passed=False, error=f"{type(exc).__name__}: {exc}",
                       wall_seconds=time.perf_counter() - started,
                       metrics=timer_metrics(None))
    seconds = time.perf_counter() - started
    #: after the timer, so the VRAM readings never land inside wall_seconds
    metrics = timer_metrics(timer)
    try:
        calls, returns = turn_trace(agent.get_history(thread_id))
    except Exception as exc:
        return Attempt(passed=False, error=f"could not read the trace: {exc}",
                       wall_seconds=seconds, metrics=metrics)
    answer = " ".join(str(answer or "").split())
    reason = grade(case, calls, returns, answer)
    return Attempt(passed=reason is None, reason=reason, wall_seconds=seconds,
                   metrics=metrics, calls=calls, returns=returns, answer=answer)


def run_seeded_attempt(agent, case: MemoryCase, number: int) -> Attempt:
    return ask(agent, case, f"mem-{case.id}-{number}", TurnTimer())


def run_carry_attempt(case: MemoryCase, number: int) -> Attempt:
    """Fresh store + fresh agent, tell the facts in thread A, ask in fresh thread B."""
    store = f"{CHECK_CARRY_PATH}/a{number}"
    shutil.rmtree(store, ignore_errors=True)
    try:
        agent = build_agent(store)
    except Exception as exc:
        return Attempt(passed=False, error=f"could not build the agent: {exc}")
    setup = TurnTimer()
    thread_a = f"mem-{case.id}-{number}-tell"
    try:
        for statement in case.told_in_thread:
            agent.invoke(statement, config={"configurable": {"thread_id": thread_a},
                                            "callbacks": [setup]})
    except Exception as exc:
        return Attempt(passed=False, error=f"telling the facts failed: {type(exc).__name__}: {exc}")
    #: one line of evidence that the callback sees calls the thread never holds: with the
    #: gate firing on turn 5 this is more model calls than thread A has AI replies
    replies = sum(1 for m in agent.get_history(thread_a) if getattr(m, "type", "") == "ai")
    print(f"    setup      {len(case.told_in_thread)} turns in {thread_a}: "
          f"{setup.call_count} model calls seen by TurnTimer, {replies} AI replies in the "
          f"thread (gate turn = {agent.memory_every_n_turns})")
    return ask(agent, case, f"mem-{case.id}-{number}-ask", TurnTimer())


# ==========================================================================
# output
# ==========================================================================

def print_attempt(number: int, attempt: Attempt) -> None:
    status = "pass" if attempt.passed else ("ERROR" if attempt.error else "miss")
    print(f"    a{number}  {status:<5}  {attempt.wall_seconds:5.1f}s  "
          f"called: {', '.join(attempt.calls) or 'nothing'}")
    for name, text in attempt.returns:
        print(f"           returned {name}: {text[:110]}{'...' if len(text) > 110 else ''}")
    if attempt.answer:
        print(f"           answer   {attempt.answer[:150]}"
              f"{'...' if len(attempt.answer) > 150 else ''}")
    if attempt.error:
        print(f"           error    {attempt.error}")
    elif attempt.reason:
        print(f"           why      {attempt.reason}")


def print_summary(results: list, elapsed: float) -> bool:
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    by_id = {c.id: c for c in CASES}
    for result in results:
        counts = f"{result.passes}/{result.attempts} (need {result.required})"
        print(f"  {result.id:<12}{result.verdict:<16}{counts}"
              f"{'   [retrieval-only]' if by_id[result.id].retrieval_only else ''}")
        if result.detail:
            print(f"  {'':<12}{result.detail}")
    print(f"  elapsed     {elapsed:.1f}s")
    regressions = [r for r in results if r.verdict in (FAIL, ERROR)]
    print()
    if regressions:
        print("MEMORY REGRESSION - " + ", ".join(r.id for r in regressions))
    else:
        print(f"MEMORY OK - {voice_main.VOICE_MODEL} recalls every must-pass fact.")
    print("=" * 72)
    return not regressions


# ==========================================================================
# cli
# ==========================================================================

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="scripts/check_memory.py",
        description="Verify the voice-configured agent recalls the right fact about the user.")
    parser.add_argument("--case", default=None, choices=[c.id for c in CASES],
                        help="run a single case instead of all of them")
    parser.add_argument("--json", default=None, metavar="PATH",
                        help="also write the eval record (scripts/eval_contract.py) to PATH")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    record = CheckRecord(check="memory", model=voice_main.VOICE_MODEL)
    code = _run(args, record)
    if args.json:
        write_record(record, args.json)
    return code


def _cannot_run(record: CheckRecord, message: str) -> int:
    print(message, file=sys.stderr)
    record.error = message
    return 2


def _run(args, record: CheckRecord) -> int:
    from scripts.check_routing import model_available, ollama_up

    if not ollama_up():
        return _cannot_run(record, "Ollama is not reachable on :11434 - start it and try again.")
    if not model_available(voice_main.VOICE_MODEL):
        return _cannot_run(record, f"{voice_main.VOICE_MODEL} is not pulled: "
                                   f"ollama pull {voice_main.VOICE_MODEL}")
    if not Path(voice_main.DEFAULT_DOCUMENT).exists():
        return _cannot_run(record, f"no such document: {voice_main.DEFAULT_DOCUMENT}")

    cases = [c for c in CASES if args.case is None or c.id == args.case]
    seeds = all_seeds()
    print("=" * 72)
    print("MEMORY CHECK")
    print("=" * 72)
    print(f"  model      {voice_main.VOICE_MODEL} (generation and memory gate)")
    print(f"  memory     {CHECK_USER_HISTORY_PATH}  (wiped + seeded here, not the live store)")
    print(f"  seeded     {len(seeds)} statements, so the top 4 has to rank")
    print()

    started = time.perf_counter()
    agent = None
    if any(not c.told_in_thread for c in cases):
        shutil.rmtree(CHECK_USER_HISTORY_PATH, ignore_errors=True)
        try:
            agent = build_agent(CHECK_USER_HISTORY_PATH)
            seed_store(agent, seeds)
        except Exception as exc:
            return _cannot_run(record, f"could not build the agent: {type(exc).__name__}: {exc}")

    results = []
    for index, case in enumerate(cases, start=1):
        k, n = case.runs
        print("-" * 72)
        print(f"CASE {index}/{len(cases)}  {case.id}   {case.query!r}"
              f"   [{k}-of-{n}]{'' if case.must_pass else '   [boundary: Feature 5 target]'}"
              f"{'   [retrieval-only: answer not graded]' if case.retrieval_only else ''}")
        print("-" * 72)

        def one(number, case=case):
            attempt = (run_carry_attempt(case, number) if case.told_in_thread
                       else run_seeded_attempt(agent, case, number))
            print_attempt(number, attempt)
            return attempt

        result = aggregate(case, run_k_of_n(case, one))
        results.append(result)
        record.cases.append(result)
        print(f"    verdict    {result.verdict}  ({result.passes}/{result.attempts}, "
              f"need {result.required})")
        print()

    return 0 if print_summary(results, time.perf_counter() - started) else 1


if __name__ == "__main__":
    sys.exit(main())
