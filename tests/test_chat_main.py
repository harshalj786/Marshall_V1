"""`chat_main.py`: the text entry point — session prompt first, then the agent, then the chat.

    pytest tests/test_chat_main.py -v

`AgentGraph` and the prompt are replaced with recorders: what is asserted is the wiring —
that the session reaches the agent and the terminal confirmer is what `run_code` gets —
not the agent itself, which tests/test_session_binding.py covers.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chat_main
import pipeline_lg
from admin_session import Session, terminal_confirm


class Recorder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.threads = []

    def run_chat(self, thread_id="default"):
        self.threads.append(thread_id)


def test_parse_args_defaults():
    args = chat_main.parse_args([])
    assert (args.file, args.thread) == (chat_main.DEFAULT_DOCUMENT, "chat-main")


def test_build_agent_passes_session_and_terminal_confirmer(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline_lg, "AgentGraph", Recorder)
    doc = tmp_path / "d.txt"
    doc.write_text("x")
    agent = chat_main.build_agent(str(doc), Session("admin", 2))
    assert agent.kwargs["session"] == Session("admin", 2)
    assert agent.kwargs["run_confirm"] is terminal_confirm
    assert agent.kwargs["model"] == agent.kwargs["memory_model"] == chat_main.CHAT_MODEL
    assert agent.kwargs["mode_docs"] == "txt"
    assert agent.kwargs["persist_directory_path"] == chat_main.VECTOR_DB_PATH
    assert agent.kwargs["user_history_directory_path"] == chat_main.USER_HISTORY_PATH
    assert agent.kwargs["summary_persist_path"] == chat_main.SUMMARY_PATH


def test_build_agent_reads_pdfs_as_pdf(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline_lg, "AgentGraph", Recorder)
    assert chat_main.build_agent("x.PDF", Session("guest", 1)).kwargs["mode_docs"] == "pdf"


def test_main_prompts_then_chats_on_the_named_thread(monkeypatch, tmp_path):
    doc = tmp_path / "d.txt"
    doc.write_text("x")
    built = []
    monkeypatch.setattr(chat_main, "prompt_for_session", lambda: Session("guest", 1))
    monkeypatch.setattr(chat_main, "build_agent",
                        lambda f, s, run_confirm=None: built.append((f, s)) or Recorder())
    monkeypatch.setattr(Recorder, "run_chat",
                        lambda self, thread_id="default": built.append(thread_id))
    assert chat_main.main(["--file", str(doc), "--thread", "t1"]) == 0
    assert built == [(str(doc), Session("guest", 1)), "t1"]


def test_main_refuses_a_missing_document_before_asking_for_a_password(monkeypatch, tmp_path):
    asked = []
    monkeypatch.setattr(chat_main, "prompt_for_session", lambda: asked.append(1))
    assert chat_main.main(["--file", str(tmp_path / "nope.pdf")]) == 2
    assert asked == []
