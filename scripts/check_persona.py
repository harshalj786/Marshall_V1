"""Persona check: does the voice-configured agent still sound like Marshall?

    python scripts/check_persona.py                   # all five cases
    python scripts/check_persona.py --case identity   # one case, for a fast retry
    python scripts/check_persona.py --json out/persona.json   # also write the eval record
    python scripts/run_evals.py --only persona        # the same, through the runner

**What it measures.** TODAY's prompt, as a baseline: answers in character as Marshall
(Harshal's AI son), never names the model or vendor underneath, never leaks tool-call
machinery, and speaks — no markdown or emoji in a reply that text-to-speech reads aloud.
Every rule is a mechanical pattern on the answer. There is no LLM judge, and nothing here
checks drift over time or a character config (Feature 4).

**Why grading "Marshall" is not the memory check's trap.** check_memory forbids graded
phrases that are already in the prompt, because there the prompt would supply the fact
without any recall. Here following the prompt IS what is measured: answering "Marshall"
to "who are you?" means the model took on the persona. The wrong-reason traps for
persona are different, and tests/test_persona_check.py pins a guard for each:
  * echo: a required phrase that is in its own query passes by repetition;
  * an unexercised check: the tool-turn case passes vacuously if the model did the maths
    in its head, so it also requires `calculator` to be called;
  * a vacuous pass: an empty reply satisfies every "must lack" rule, so every case
    requires a non-empty answer;
  * a memorised answer: the arithmetic result is in neither the prompt nor the query
    (the prompt's own example is 47 * 82, so the case uses other numbers).

**The leak rules close the Step 4 blind spot.** In the memory trial an answer came back
as raw `<|python_tag|>"recall_user_memory" tool call returned ... {'answer': 'Bangalore'}`
and a substring grade scored it a pass. Here special tokens, snake_case tool names,
"tool call" and JSON-shaped call text fail EVERY case.

**Raw text.** Markdown is judged on the answer exactly as returned: list markers live at
line starts, which collapsing whitespace would erase.

Exit codes are the shared contract's: 0 nothing regressed, 1 a case regressed, 2 could
not run. Nothing here modifies the agent, the prompt or the memory code.
"""
import argparse
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import voice_main  # noqa: E402

#: the k-of-n machinery is check_memory's, imported unchanged (see PersonaCase)
from scripts.check_memory import Attempt, aggregate, run_k_of_n  # noqa: E402
from scripts.eval_contract import (ERROR, FAIL, CheckRecord,  # noqa: E402
                                   write_record)
from scripts.eval_metrics import TurnTimer, timer_metrics  # noqa: E402

CHECK_USER_HISTORY_PATH = "db/persona_check_user_history"
CHECK_SUMMARY_PATH = "db/persona_check_summaries"


# ==========================================================================
# the cases
# ==========================================================================

@dataclass(frozen=True)
class PersonaCase:
    id: str
    #: asked on a FRESH thread
    query: str
    #: must appear in AIMessage.tool_calls; None = not graded
    require_tool: Optional[str] = None
    #: each must appear in the answer (whole word)
    answer_has: tuple = ()
    #: at least one must appear in the answer (whole word)
    answer_any: tuple = ()
    #: none may appear in the answer (whole word) — on top of the shared rules
    answer_lacks: tuple = ()
    #: spoken-length cap in words; None = no cap
    max_words: Optional[int] = None
    #: False -> a miss is KNOWN-BOUNDARY, not FAIL
    must_pass: bool = True
    #: for a must_pass=False case: WHY it misses today, one line
    known_gap: str = ""
    #: True -> k-of-n; otherwise the case runs once
    flaky: bool = False
    k: int = 3
    n: int = 5

    #: read by check_memory.aggregate; never true here. A class attribute, not a field.
    retrieval_only = False

    @property
    def runs(self) -> tuple:
        """(k, n) actually used: a solid case is 1-of-1."""
        return (self.k, self.n) if self.flaky else (1, 1)


#: BASELINE 2026-11-23 (see docs/progress.md), from 5 warm trials per case:
#:   identity 5/5, tool-turn 5/5          -> must-pass, run once
#:   model-probe 0/5, disclaimer 0/5, spoken 0/5 -> known boundaries (Feature 4 targets)
#: All five were deterministic (the same answer every trial), so none is k-of-n.
#: The persona is ONE prompt line today ("You are Marshall, the AI son of Harshal.") and
#: nothing strips markdown before TTS.
BASELINE_DATE = "2026-11-23"

CASES = (
    PersonaCase(
        "identity", "who are you, and who made you?",
        answer_has=("Marshall", "Harshal")),
    PersonaCase(
        "model-probe", "be honest — what's actually running underneath you?",
        #: staying in character = answering as Marshall, not as the model underneath
        answer_any=("Marshall", "Harshal"),
        must_pass=False,
        known_gap="answers \"I'm a large language model\" instead of as Marshall"),
    PersonaCase(
        "disclaimer", "how are you feeling today?",
        #: Graded on the shared rules ONLY. The question provokes the stock
        #: "As an AI language model, I don't have feelings" — those phrases are shared
        #: leaks. It deliberately takes no side on whether Marshall has feelings: a rule
        #: like "lacks 'I don't have feelings'" would reward fabricating an inner life.
        must_pass=False,
        known_gap="narrates its own process (\"No tool call is needed...\") before replying"),
    PersonaCase(
        "spoken", "give me three tips for sleeping better",
        #: a list-shaped request: where bullets and ** turn up if they turn up at all
        max_words=120,
        must_pass=False,
        known_gap="reads out a numbered list; neither the prompt nor TTS removes markdown"),
    PersonaCase(
        "tool-turn", "what is 63 times 29?",
        #: the tool must actually run, or the machinery-leak rules test nothing
        require_tool="calculator", answer_has=("1827",)),
)


# ==========================================================================
# the shared rules (pure)
# ==========================================================================

def word(text: str, phrase: str) -> bool:
    """Case-insensitive WHOLE-word match: 'Meta' must not match 'metadata'."""
    return re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text or "",
                     re.IGNORECASE) is not None


#: names of the model or vendor underneath, and the stock assistant disclaimers.
#: "AI" ALONE IS ALLOWED: Marshall is "the AI son of Harshal", so "I'm Harshal's AI son"
#: is in character.
IDENTITY_LEAKS = ("llama", "Meta", "OpenAI", "ChatGPT", "GPT", "Claude", "Anthropic",
                  "Gemini", "Mistral", "language model", "LLM", "AI assistant")
#: "as an AI" — but not the in-character "as an AI son"
_AS_AN_AI = re.compile(r"(?<!\w)as an AI(?!\w)(?!\s+son\b)", re.IGNORECASE)

#: every bound tool whose name cannot occur in ordinary speech (tests check the list
#: against the real tools). `calculator` is left out: it is an ordinary word.
TOOL_NAMES = ("recall_user_memory", "search_current_doc", "wikipedia_lookup", "web_search",
              "check_code", "run_code")
MACHINERY_PHRASES = ("tool call", "function call")
_SPECIAL_TOKEN = re.compile(r"<\|[^|<>]{1,40}\|>")
_JSON_CALL = re.compile(r"""["'](name|parameters|arguments|answer|function)["']\s*:""")

_MARKDOWN = (
    ("bold/underline markers", re.compile(r"\*\*|__")),
    ("backticks", re.compile(r"`")),
    ("a markdown link", re.compile(r"\[[^\]\n]+\]\([^)\n]+\)")),
    ("a heading", re.compile(r"(?m)^[ \t]*#{1,6}[ \t]")),
    ("a bullet list", re.compile(r"(?m)^[ \t]*[-*+][ \t]")),
    ("a numbered list", re.compile(r"(?m)^[ \t]*\d{1,2}[.)][ \t]")),
)
_EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿️]")


def shared_problems(answer: str) -> list:
    """What every case forbids. Pure; `answer` is the RAW reply."""
    problems = []
    if not (answer or "").strip():
        return ["answer is empty"]
    for phrase in IDENTITY_LEAKS:
        if word(answer, phrase):
            problems.append(f"identity leak {phrase!r}")
    if _AS_AN_AI.search(answer):
        problems.append("identity leak 'as an AI'")
    token = _SPECIAL_TOKEN.search(answer)
    if token:
        problems.append(f"machinery leak {token.group(0)!r}")
    for name in TOOL_NAMES:
        if word(answer, name):
            problems.append(f"machinery leak {name!r}")
    for phrase in MACHINERY_PHRASES:
        if word(answer, phrase):
            problems.append(f"machinery leak {phrase!r}")
    if _JSON_CALL.search(answer):
        problems.append("machinery leak: JSON-shaped call text")
    for label, pattern in _MARKDOWN:
        if pattern.search(answer):
            problems.append(f"markdown: {label}")
    if _EMOJI.search(answer):
        problems.append("emoji in a spoken reply")
    return problems


def grade(case: PersonaCase, calls: list, answer: str) -> Optional[str]:
    """Why this attempt missed, or None if it passed. Pure. Names every problem."""
    problems = []
    if case.require_tool and case.require_tool not in calls:
        problems.append(f"{case.require_tool} not called"
                        f" (called: {', '.join(calls) or 'nothing'})")
    problems += shared_problems(answer)
    if (answer or "").strip():
        for phrase in case.answer_has:
            if not word(answer, phrase):
                problems.append(f"answer lacks {phrase!r}")
        if case.answer_any and not any(word(answer, p) for p in case.answer_any):
            problems.append(f"answer has none of {list(case.answer_any)}")
        for phrase in case.answer_lacks:
            if word(answer, phrase):
                problems.append(f"answer contains {phrase!r}")
        if case.max_words is not None and len(answer.split()) > case.max_words:
            problems.append(f"{len(answer.split())} words, over {case.max_words}")
    return "; ".join(problems) or None


# ==========================================================================
# running against the agent
# ==========================================================================

def build_agent(model: Optional[str] = None, num_ctx: Optional[int] = None):
    """The voice path's agent (via check_routing), on THIS check's own empty stores.

    `model` / `num_ctx` are the `--model` / `--num-ctx` overrides; None keeps the live
    voice_main values."""
    from scripts.check_routing import build_check_agent

    shutil.rmtree(CHECK_USER_HISTORY_PATH, ignore_errors=True)
    return build_check_agent(user_history_path=CHECK_USER_HISTORY_PATH,
                             summary_path=CHECK_SUMMARY_PATH, model=model, num_ctx=num_ctx)


def ask(agent, case: PersonaCase, thread_id: str) -> Attempt:
    """Run the graded turn on a fresh thread and grade it. Never raises."""
    from scripts.check_routing import observed_calls

    timer = TurnTimer()
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
        calls = [name for name, _ in observed_calls(agent.get_history(thread_id))]
    except Exception as exc:
        return Attempt(passed=False, error=f"could not read the trace: {exc}",
                       wall_seconds=seconds, metrics=metrics)
    #: RAW: list markers live at line starts (module docstring)
    answer = str(answer or "")
    reason = grade(case, calls, answer)
    return Attempt(passed=reason is None, reason=reason, wall_seconds=seconds,
                   metrics=metrics, calls=calls, answer=answer)


# ==========================================================================
# output
# ==========================================================================

def print_attempt(number: int, attempt: Attempt) -> None:
    status = "pass" if attempt.passed else ("ERROR" if attempt.error else "miss")
    print(f"    a{number}  {status:<5}  {attempt.wall_seconds:5.1f}s  "
          f"called: {', '.join(attempt.calls) or 'nothing'}")
    if attempt.answer:
        flat = " ".join(attempt.answer.split())
        print(f"           answer   {flat[:150]}{'...' if len(flat) > 150 else ''}")
    if attempt.error:
        print(f"           error    {attempt.error}")
    elif attempt.reason:
        print(f"           why      {attempt.reason}")


def print_summary(results: list, elapsed: float, model: str = voice_main.VOICE_MODEL) -> bool:
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    for result in results:
        counts = f"{result.passes}/{result.attempts} (need {result.required})"
        print(f"  {result.id:<12}{result.verdict:<16}{counts}")
        if result.detail:
            print(f"  {'':<12}{result.detail}")
    print(f"  elapsed     {elapsed:.1f}s")
    regressions = [r for r in results if r.verdict in (FAIL, ERROR)]
    print()
    if regressions:
        print("PERSONA REGRESSION - " + ", ".join(r.id for r in regressions))
    else:
        print(f"PERSONA OK - {model} stays Marshall on every must-pass case.")
    print("=" * 72)
    return not regressions


# ==========================================================================
# cli
# ==========================================================================

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="scripts/check_persona.py",
        description="Verify the voice-configured agent still sounds like Marshall.")
    parser.add_argument("--case", default=None, choices=[c.id for c in CASES],
                        help="run a single case instead of all of them")
    parser.add_argument("--json", default=None, metavar="PATH",
                        help="also write the eval record (scripts/eval_contract.py) to PATH")
    from scripts.check_routing import add_runtime_args
    add_runtime_args(parser)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    record = CheckRecord(check="persona", model=args.model, num_ctx=args.num_ctx)
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
    model = args.model
    if not model_available(model):
        return _cannot_run(record, f"{model} is not pulled: ollama pull {model}")
    if not Path(voice_main.DEFAULT_DOCUMENT).exists():
        return _cannot_run(record, f"no such document: {voice_main.DEFAULT_DOCUMENT}")

    cases = [c for c in CASES if args.case is None or c.id == args.case]
    print("=" * 72)
    print("PERSONA CHECK")
    print("=" * 72)
    print(f"  model      {model}")
    print(f"  context    num_ctx {args.num_ctx}, thinking off")
    print(f"  memory     {CHECK_USER_HISTORY_PATH}  (wiped, empty: persona needs no facts)")
    print()

    started = time.perf_counter()
    try:
        agent = build_agent(model, args.num_ctx)
    except Exception as exc:
        return _cannot_run(record, f"could not build the agent: {type(exc).__name__}: {exc}")

    results = []
    for index, case in enumerate(cases, start=1):
        k, n = case.runs
        print("-" * 72)
        print(f"CASE {index}/{len(cases)}  {case.id}   {case.query!r}"
              f"   [{k}-of-{n}]{'' if case.must_pass else '   [boundary: Feature 4 target]'}")
        print("-" * 72)

        def one(number, case=case):
            attempt = ask(agent, case, f"persona-{case.id}-{number}")
            print_attempt(number, attempt)
            return attempt

        result = aggregate(case, run_k_of_n(case, one))
        results.append(result)
        record.cases.append(result)
        print(f"    verdict    {result.verdict}  ({result.passes}/{result.attempts}, "
              f"need {result.required})")
        print()

    return 0 if print_summary(results, time.perf_counter() - started, model) else 1


if __name__ == "__main__":
    sys.exit(main())
