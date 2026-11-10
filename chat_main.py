"""Talk to Marshall by typing.

    python chat_main.py                          # over the default document
    python chat_main.py --file path/to/doc.pdf   # over a different document
    python chat_main.py --thread work            # a separate conversation

**The admin password comes first**, before the agent is built: Enter alone starts a guest
session. An admin session gets `run_code`, and every run is confirmed here in the
terminal with `[y/N]` — the model can ask to run code, it can never answer for you.
A guest session can check pasted code but not run it, and keeps no memory at all.
See `admin_session.py` and `docs/superpowers/specs/2026-11-10-code-check-and-run-design.md`.

Mirrors `voice_main.build_agent`: one model for generation AND the memory gate, so no turn
pays a model swap. Its own stores, separate from the voice path's, so typing and talking
never write into each other's conversation.
"""
import argparse
import sys
from pathlib import Path

from admin_session import Session, prompt_for_session, terminal_confirm

DEFAULT_DOCUMENT = "test_pdf's/Pipes_2025.pdf"

#: the only locally-pulled model that reliably emits tool_calls — see voice_main.VOICE_MODEL
CHAT_MODEL = "llama3.1"

VECTOR_DB_PATH = "db/chat_vdb"
USER_HISTORY_PATH = "db/chat_user_history"
SUMMARY_PATH = "db/chat_summaries"


def build_agent(file_path: str, session: Session, run_confirm=terminal_confirm):
    """Construct the AgentGraph for a typed session. Every difference is an argument."""
    import pipeline_lg

    return pipeline_lg.AgentGraph(
        file_path=file_path,
        mode_docs="pdf" if str(file_path).lower().endswith(".pdf") else "txt",
        persist_directory_path=VECTOR_DB_PATH,
        model=CHAT_MODEL,
        memory_model=CHAT_MODEL,
        user_history_directory_path=USER_HISTORY_PATH,
        summary_persist_path=SUMMARY_PATH,
        session=session,
        run_confirm=run_confirm,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="chat_main.py", description="Talk to Marshall by typing.")
    parser.add_argument("--file", default=DEFAULT_DOCUMENT,
                        help=f"document to load (default: {DEFAULT_DOCUMENT})")
    parser.add_argument("--thread", default="chat-main",
                        help="conversation thread_id (default: chat-main)")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    #: checked before the password prompt, so a typo'd path does not cost a login
    if not Path(args.file).exists():
        print(f"no such document: {args.file}", file=sys.stderr)
        return 2
    session = prompt_for_session()
    agent = build_agent(args.file, session)
    agent.run_chat(args.thread)
    return 0


if __name__ == "__main__":
    sys.exit(main())
