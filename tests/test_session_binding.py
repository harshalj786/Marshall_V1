"""`AgentGraph(session=, run_confirm=)`: which tools, which prompt and which memory a session gets.

    pytest tests/test_session_binding.py -v

No Ollama: `agent.llm` is replaced with a scripted stand-in, this repo's pattern for graph
tests (see tests/test_agent_hook.py). What the live model DOES with these tools is
tests/test_code_tool_routing.py's job.

The first test is the load-bearing one: with no session, the agent must be exactly today's
— same five tools, same prompt — because tests/test_routing_matrix.py and every other test
measure that agent.
"""
import json
import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from admin_session import Session, prompt_for_session, set_admin_password
from pipeline_lg import AGENT_SYSTEM_PROMPT, SUMMARY_FILE_NAME, AgentGraph

DOC = "Marshall is a local RAG framework. It loads documents and answers questions.\n"
TODAY = ["calculator", "wikipedia_lookup", "web_search", "search_current_doc",
         "recall_user_memory"]


class PlainChat:
    """Answers every turn without a tool call."""

    def invoke(self, messages, **kwargs):
        return AIMessage(content="ok")


@pytest.fixture(scope="module")
def tmp_root(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("marshall_session")
    (tmp / "doc.txt").write_text(DOC, encoding="utf-8")
    return tmp


def build(tmp_root, **kwargs):
    return AgentGraph(
        file_path=str(tmp_root / "doc.txt"),
        mode_docs="txt",
        mode_retreiver="simple",
        persist_directory_path=str(tmp_root / "vdb"),
        user_history_directory_path=str(tmp_root / "mem"),
        memory_every_n_turns=5,
        summarize_every_n_turns=0,
        summary_persist_path=str(tmp_root / "sum"),
        **kwargs,
    )


def names(agent):
    return [t.name for t in agent.tools]


def test_no_session_is_exactly_todays_agent(tmp_root):
    agent = build(tmp_root)
    assert names(agent) == TODAY
    assert agent.system_prompt == AGENT_SYSTEM_PROMPT
    assert agent.summary_persist_path == str(tmp_root / "sum")
    assert agent.memory_every_n_turns == 5


def test_the_default_prompt_mentions_no_code_tool():
    assert "check_code" not in AGENT_SYSTEM_PROMPT and "run_code" not in AGENT_SYSTEM_PROMPT


def test_guest_binds_check_code_only_and_keeps_nothing(tmp_root):
    agent = build(tmp_root, session=Session("guest", 1))
    assert names(agent) == ["calculator", "wikipedia_lookup", "web_search",
                            "search_current_doc", "check_code"]
    assert agent.summary_persist_path is None
    assert agent.memory_every_n_turns == 0
    assert agent.user_history_directory_path is None
    assert "run_code" not in agent.system_prompt
    assert "recall_user_memory" not in agent.system_prompt
    assert "check_code" in agent.system_prompt
    assert "Guest sessions started" not in agent.system_prompt


def test_guest_check_code_refuses_paths(tmp_root):
    agent = build(tmp_root, session=Session("guest", 1))
    check = next(t for t in agent.tools if t.name == "check_code")
    out = check.invoke({"path": str(tmp_root / "doc.txt")})
    assert out.startswith("check_code refused: file paths are not available in a guest session")


def test_guest_never_loads_a_stored_summary(tmp_root):
    (tmp_root / "sum").mkdir(exist_ok=True)
    (tmp_root / "sum" / SUMMARY_FILE_NAME).write_text(json.dumps(
        {"version": 1, "threads": {"g1": {"summary": "ADMIN SECRETS"}}}), encoding="utf-8")
    agent = build(tmp_root, session=Session("guest", 1))
    agent.llm = PlainChat()
    agent.invoke("hi", config={"configurable": {"thread_id": "g1"}})
    assert not any("ADMIN SECRETS" in str(m.content) for m in agent.get_history("g1"))


def test_admin_binds_both_code_tools_and_sees_the_guest_count(tmp_root):
    agent = build(tmp_root, session=Session("admin", 3), run_confirm=lambda r: False)
    assert names(agent) == TODAY + ["check_code", "run_code"]
    assert "`run_code`" in agent.system_prompt
    assert "Guest sessions started on this machine so far: 3" in agent.system_prompt
    assert agent.summary_persist_path == str(tmp_root / "sum")
    assert agent.memory_every_n_turns == 5


def test_admin_run_code_goes_through_the_given_confirmer(tmp_root):
    asked = []
    agent = build(tmp_root, session=Session("admin", 0),
                  run_confirm=lambda r: asked.append(r) or False)
    run = next(t for t in agent.tools if t.name == "run_code")
    out = run.invoke({"code": "print(1)", "language": "python"})
    assert out == "run_code: the user declined to run it; nothing was executed"
    assert [r.source for r in asked] == ["print(1)"]


def test_admin_without_a_confirmer_is_refused(tmp_root):
    with pytest.raises(ValueError, match="run_confirm"):
        build(tmp_root, session=Session("admin", 0))


def test_unknown_role_is_refused(tmp_root):
    with pytest.raises(ValueError):
        build(tmp_root, session=Session("root", 0), run_confirm=lambda r: False)


def test_the_password_never_reaches_the_model(tmp_root, tmp_path):
    h = str(set_admin_password("hunter2", str(tmp_path / "p.hash")))
    session = prompt_for_session(hash_path=h, guest_log_path=str(tmp_path / "g.jsonl"),
                                 getpass_fn=lambda _: "hunter2", report=lambda *a: None)
    agent = build(tmp_root, session=session, run_confirm=lambda r: False)
    agent.llm = PlainChat()
    agent.invoke("hello", config={"configurable": {"thread_id": "a1"}})
    blob = agent.system_prompt + json.dumps([str(m.content) for m in agent.get_history("a1")])
    assert "hunter2" not in blob and "password" not in blob.lower()
