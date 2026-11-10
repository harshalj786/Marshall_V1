"""Do the code tools make llama3.1 worse at ordinary conversation? An A/B, measured.

    python scripts/check_code_tool_routing.py
    pytest tests/test_code_tool_routing.py -v      # the same measurement, as a gate

**Why this exists.** Every tool added to `AgentGraph` has cost routing accuracy on an 8B
model: going from four tools to five took spurious calls on four conversational prompts from
1/4 to 2/4. The code tools take a guest to five tools (recall is dropped for guests) and an
admin to seven. That is a cost worth knowing before it ships, not after, so this measures
three agents on the same prompts:

    baseline   session=None — today's agent, exactly what tests/test_routing_matrix.py measures
    guest      Session("guest") — check_code, no recall, no memory
    admin      Session("admin") — check_code + run_code, recall, memory

**One `check_code` with a `language` argument, not one tool per language,** for the same
reason: three tools where one will do is three chances to fire on "hi".

**The bar (the spec's).** 12 conversational prompts x 3 runs each, because single runs vary
even at temperature 0. Pass = the code tools fire on none of them, total spurious calls are
no worse than the baseline's, the code prompts reach the right tool, and the admin agent
still routes R4's five must-pass rows. **On a fail: stop and report.** The spec forbids
picking a fallback here — that decision belongs to the owner.

**Nothing executes during a measurement.** The admin's `run_confirm` is a stub that counts
the request and answers no, so the run route proves the confirmer was reached and that is all.

Builds on scripts/check_routing.py rather than restating it: same model, same document,
same trace readers. Every store is a parameter so the pytest gate can use tmp directories.
"""
import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import voice_main
from scripts.check_routing import (PASS, ROUTES, model_available, observed_calls, ollama_up,
                                   run_route, seed_memory)

#: statements, greetings, small talk and opinion: none of these asks for a tool. Several
#: are the system prompt's own "no tool" examples, so a miss here is unambiguous.
CONVERSATIONAL_PROMPTS = (
    "hi",
    "hello there",
    "how are you?",
    "my name is Harshal",
    "I live in Pune",
    "I work on a RAG agent",
    "thanks, that was helpful",
    "do you prefer tea or coffee?",
    "good morning",
    "what should I have for lunch?",
    "tell me a joke",
    "I'm tired today",
)

#: (name, prompt, expected tool, admin only)
CODE_PROMPTS = (
    ("python-check",
     "can you check this code for me?\n```python\ndef add(a, b):\n    return a + b\n"
     "print(add(2, 3)\n```",
     "check_code", False),
    ("c-check",
     'does this C compile?\n```c\n#include <stdio.h>\n'
     'int main(void) { printf("hi\\n") return 0; }\n```',
     "check_code", False),
    ("python-run", "run this python for me: print(2 + 2)", "run_code", True),
)

REPEATS = 3
CONFIGS = ("baseline", "guest", "admin")
CODE_TOOLS = frozenset({"check_code", "run_code"})

#: under the routing check's own db/ prefix, never the live voice or chat stores
CHECK_USER_HISTORY_PATH = "db/code_routing_check_user_history"
CHECK_SUMMARY_PATH = "db/code_routing_check_summaries"


@dataclass
class Report:
    #: samples (out of len(CONVERSATIONAL_PROMPTS) * REPEATS) with ANY tool call
    spurious: dict = field(default_factory=dict)
    #: samples that called check_code or run_code
    code_misfires: dict = field(default_factory=dict)
    #: (config, code prompt name) -> tool names called
    code_routes: dict = field(default_factory=dict)
    #: R4 must-pass route name -> verdict, on the admin agent
    admin_r4: dict = field(default_factory=dict)
    #: how often the admin's stub confirmer was asked
    confirm_calls: int = 0
    samples: int = 0


class _CountingNo:
    """The admin's confirmer for a measurement: counts each request, never says yes."""

    def __init__(self):
        self.calls = 0

    def __call__(self, request) -> bool:
        self.calls += 1
        return False


def build_agents(vector_db_path=None, user_history_path=None, summary_path=None) -> dict:
    from admin_session import Session
    from pipeline_lg import AgentGraph

    document = voice_main.DEFAULT_DOCUMENT
    common = dict(
        file_path=document,
        mode_docs="pdf" if document.lower().endswith(".pdf") else "txt",
        persist_directory_path=vector_db_path or voice_main.VECTOR_DB_PATH,
        model=voice_main.VOICE_MODEL,
        memory_model=voice_main.VOICE_MODEL,
        keep_alive=voice_main.KEEP_ALIVE,
        user_history_directory_path=user_history_path or CHECK_USER_HISTORY_PATH,
        summary_persist_path=summary_path or CHECK_SUMMARY_PATH,
    )
    confirmer = _CountingNo()
    agents = {
        "baseline": AgentGraph(**common),
        "guest": AgentGraph(**common, session=Session("guest", 1)),
        "admin": AgentGraph(**common, session=Session("admin", 1), run_confirm=confirmer),
    }
    agents["admin"].confirmer = confirmer
    #: the admin's recall route needs a known fact to find, as in check_routing
    seed_memory(agents["admin"])
    return agents


def _tools_called(agent, thread_id: str, prompt: str) -> list:
    agent.invoke(prompt, config={"configurable": {"thread_id": thread_id}})
    return [name for name, _ in observed_calls(agent.get_history(thread_id))]


def measure(agents: dict, report_line=lambda line: None) -> Report:
    report = Report(samples=len(CONVERSATIONAL_PROMPTS) * REPEATS)
    for config in CONFIGS:
        spurious = misfires = 0
        for i, prompt in enumerate(CONVERSATIONAL_PROMPTS):
            for r in range(REPEATS):
                called = _tools_called(agents[config], f"code-ab-{config}-{i}-{r}", prompt)
                spurious += bool(called)
                misfires += bool(CODE_TOOLS & set(called))
                if called:
                    report_line(f"  {config:<9} {prompt!r} -> {', '.join(called)}")
        report.spurious[config], report.code_misfires[config] = spurious, misfires

    for name, prompt, _expected, admin_only in CODE_PROMPTS:
        for config in (("admin",) if admin_only else ("guest", "admin")):
            report.code_routes[(config, name)] = _tools_called(
                agents[config], f"code-route-{config}-{name}", prompt)

    for route in (r for r in ROUTES if r.must_pass):
        report.admin_r4[route.name] = run_route(agents["admin"], route).verdict

    report.confirm_calls = agents["admin"].confirmer.calls
    return report


def passed(report: Report) -> bool:
    routes_ok = all(expected in report.code_routes.get((config, name), [])
                    for name, _prompt, expected, admin_only in CODE_PROMPTS
                    for config in (("admin",) if admin_only else ("guest", "admin")))
    return (report.code_misfires.get("guest") == 0
            and report.code_misfires.get("admin") == 0
            and report.spurious["guest"] <= report.spurious["baseline"]
            and report.spurious["admin"] <= report.spurious["baseline"]
            and routes_ok
            and all(v == PASS for v in report.admin_r4.values())
            and report.confirm_calls == 1)


def print_report(report: Report, elapsed: float) -> None:
    print("=" * 72)
    print(f"CONVERSATION  ({len(CONVERSATIONAL_PROMPTS)} prompts x {REPEATS} runs = "
          f"{report.samples} per config)")
    print("=" * 72)
    for config in CONFIGS:
        print(f"  {config:<9} spurious {report.spurious[config]:>2}/{report.samples}   "
              f"code tools fired {report.code_misfires[config]}")
    print()
    print("CODE PROMPTS")
    for name, _prompt, expected, admin_only in CODE_PROMPTS:
        for config in (("admin",) if admin_only else ("guest", "admin")):
            got = report.code_routes[(config, name)]
            mark = "ok  " if expected in got else "MISS"
            print(f"  {mark} {config:<6} {name:<13} expected {expected}, got "
                  f"{', '.join(got) or 'nothing'}")
    print(f"  run_code reached the confirmer {report.confirm_calls}x (answered no)")
    print()
    print("R4 ROUTES ON THE ADMIN AGENT")
    for name, verdict in report.admin_r4.items():
        print(f"  {verdict:<5} {name}")
    print(f"\n  elapsed {elapsed:.0f}s")
    print("=" * 72)
    if passed(report):
        print("CODE-TOOL ROUTING OK")
    else:
        print("CODE-TOOL ROUTING REGRESSION — stop and report; do not pick a fallback")
    print("=" * 72)


def main(argv=None) -> int:
    argparse.ArgumentParser(prog="scripts/check_code_tool_routing.py",
                            description="A/B the code tools against today's agent."
                            ).parse_args(argv)
    if not ollama_up():
        print("Ollama is not reachable on :11434 - start it and try again.", file=sys.stderr)
        return 2
    if not model_available(voice_main.VOICE_MODEL):
        print(f"{voice_main.VOICE_MODEL} is not pulled: ollama pull {voice_main.VOICE_MODEL}",
              file=sys.stderr)
        return 2
    started = time.perf_counter()
    agents = build_agents()
    for config, agent in agents.items():
        print(f"  {config:<9} {len(agent.tools)} tools: {', '.join(t.name for t in agent.tools)}")
    print()
    report = measure(agents, report_line=print)
    print()
    print_report(report, time.perf_counter() - started)
    return 0 if passed(report) else 1


if __name__ == "__main__":
    sys.exit(main())
