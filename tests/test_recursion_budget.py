"""AgentGraph's default recursion_limit must cover its own worst-case turn.

    pytest tests/test_recursion_budget.py -v

The worst realistic turn of the unhooked graph runs count_turn, both gate nodes, one
agent generation per tool round plus the final answer, and the tool node once per
round: 2N + 4 supersteps for N tool rounds. LangGraph raises when the count reaches
the limit, so the limit has to be at least 2N + 5.

Two things are pinned. The arithmetic: DEFAULT_RECURSION_LIMIT may never drop below
what MAX_TOOL_ROUNDS needs. And the arithmetic's premise: a real turn with both gates
due and N tool rounds is MEASURED to fail at 2N + 4 and pass at 2N + 5, so the formula
cannot go stale without this file noticing. No Ollama: the graph is built for real and
only the model handles are replaced, as in tests/test_summarization_cycle.py.
"""
import shutil
import sys
import tempfile
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.errors import GraphRecursionError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline_lg import DEFAULT_RECURSION_LIMIT, MAX_TOOL_ROUNDS, AgentGraph

DOC = "Marshall is a local RAG framework. It loads documents and answers questions.\n"


def worst_case_supersteps(tool_rounds: int) -> int:
    """count_turn + summarize_and_trim + summarize_user_history + agent x (N+1) + tools x N."""
    return 2 * tool_rounds + 4


class NToolCallsThenAnswer:
    """Calls `calculator` `n` times in a row, then answers: n + 1 generations per turn."""

    def __init__(self, n: int):
        self.n = n
        self.calls = 0

    def invoke(self, messages, **kwargs):
        if self.calls >= self.n:
            return AIMessage(content="42")
        self.calls += 1
        return AIMessage(content="", tool_calls=[{
            "name": "calculator",
            "args": {"expression": "6*7"},
            "id": f"call_{self.calls}",
        }])


class FixedSummarizer:
    def invoke(self, messages, **kwargs):
        return AIMessage(content="SUMMARY")


class FixedMemory:
    def invoke(self, _):
        return "PROFILE"


@pytest.fixture(scope="module")
def tmp_root():
    tmp = Path(tempfile.mkdtemp(prefix="marshall_budget_"))
    (tmp / "doc.txt").write_text(DOC, encoding="utf-8")
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


def build_agent(tmp_root: Path, tool_rounds: int) -> AgentGraph:
    """A graph on which BOTH gates fire on the second turn.

    memory_every_n_turns=1 makes the memory gate due on every turn from the first.
    summarize_every_n_turns=1 with summary_keep_recent=1 makes the trim due on the second
    turn, once the first turn's messages are on the list. The first turn is a plain reply
    so it is short; the measured turn is the second.
    """
    agent = AgentGraph(
        file_path=str(tmp_root / "doc.txt"),
        mode_docs="txt",
        mode_retreiver="simple",
        persist_directory_path=str(tmp_root / "vdb"),
        user_history_directory_path=str(tmp_root / "mem"),
        memory_every_n_turns=1,
        summarize_every_n_turns=1,
        summary_keep_recent=1,
        summary_persist_path=str(tmp_root / "sum"),
    )
    agent.summary_llm = FixedSummarizer()
    agent._build_user_memory = lambda messages, thread_id: FixedMemory()
    agent.llm = NToolCallsThenAnswer(0)
    agent.invoke("hello", config={"configurable": {"thread_id": "budget"}})
    agent.llm = NToolCallsThenAnswer(tool_rounds)
    return agent


def run_gate_turn(agent, limit: int):
    return agent.invoke("what is 6*7?", config={
        "configurable": {"thread_id": "budget"},
        "recursion_limit": limit,
    })


# --------------------------------------------------------------------------
# 1. the arithmetic: the default may never drop below what MAX_TOOL_ROUNDS needs
# --------------------------------------------------------------------------

def test_default_covers_the_worst_case_at_max_tool_rounds():
    needed = worst_case_supersteps(MAX_TOOL_ROUNDS) + 1
    assert DEFAULT_RECURSION_LIMIT >= needed, (
        f"DEFAULT_RECURSION_LIMIT={DEFAULT_RECURSION_LIMIT} is below the {needed} that "
        f"MAX_TOOL_ROUNDS={MAX_TOOL_ROUNDS} needs on a turn where both gates fire")


# --------------------------------------------------------------------------
# 2. the measurement: the formula is what the graph actually executes
# --------------------------------------------------------------------------

@pytest.mark.parametrize("tool_rounds", [1, 3])
def test_worst_case_turn_fails_at_the_count_and_passes_one_above(tmp_root, tool_rounds):
    count = worst_case_supersteps(tool_rounds)

    with pytest.raises(GraphRecursionError):
        run_gate_turn(build_agent(tmp_root, tool_rounds), count)

    assert run_gate_turn(build_agent(tmp_root, tool_rounds), count + 1) == "42"


def test_default_limit_runs_a_max_tool_rounds_gate_turn(tmp_root):
    agent = build_agent(tmp_root, MAX_TOOL_ROUNDS)
    assert agent.invoke("what is 6*7?",
                        config={"configurable": {"thread_id": "budget"}}) == "42"
