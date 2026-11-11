"""The admin password, the two session kinds, the guest counter, and the terminal confirmer.

    pytest tests/test_admin_session.py -v

No model, no terminal: `getpass` and `input` are injected. Every path is a tmp path — the
real hash lives in `db/admin/` and a test must never write there.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from admin_session import (Session, hash_password, password_is_set, prompt_for_session,
                           read_guest_count, record_guest_session, set_admin_password,
                           terminal_confirm, verify_password)
from tools.code_tools import RunRequest

NOOP = lambda *a, **k: None


def scripted(*answers):
    """A getpass stand-in: one answer per call, the last one repeating. Records the prompts."""
    answers = list(answers)
    calls = []

    def fn(prompt):
        calls.append(prompt)
        return answers.pop(0) if len(answers) > 1 else answers[0]
    fn.calls = calls
    return fn


def test_hash_is_argon2id_and_verifies():
    h = hash_password("hunter2")
    assert h.startswith("$argon2id$")
    assert verify_password(h, "hunter2")
    assert not verify_password(h, "hunter3")
    assert not verify_password("garbage", "hunter2")


def test_set_admin_password_writes_a_hash_not_the_password(tmp_path):
    p = set_admin_password("hunter2", str(tmp_path / "admin" / "password.hash"))
    assert "hunter2" not in p.read_text()
    assert password_is_set(str(p))


def test_no_password_set_means_guest_and_is_counted(tmp_path):
    log = str(tmp_path / "guests.jsonl")
    fn = scripted("never")
    s = prompt_for_session(hash_path=str(tmp_path / "none"), guest_log_path=log,
                           getpass_fn=fn, report=NOOP)
    assert s == Session("guest", 1) and not s.is_admin
    assert read_guest_count(log) == 1
    assert fn.calls == []          # nothing to check against, so nobody is asked


def test_correct_password_is_admin_and_counts_nothing(tmp_path):
    h = str(set_admin_password("hunter2", str(tmp_path / "p.hash")))
    log = str(tmp_path / "g.jsonl")
    s = prompt_for_session(hash_path=h, guest_log_path=log,
                           getpass_fn=scripted("hunter2"), report=NOOP)
    assert s == Session("admin", 0) and s.is_admin
    assert read_guest_count(log) == 0


def test_admin_sees_the_guest_count_so_far(tmp_path):
    h = str(set_admin_password("hunter2", str(tmp_path / "p.hash")))
    log = str(tmp_path / "g.jsonl")
    record_guest_session(log)
    record_guest_session(log)
    s = prompt_for_session(hash_path=h, guest_log_path=log,
                           getpass_fn=scripted("hunter2"), report=NOOP)
    assert s == Session("admin", 2)


def test_empty_entry_is_guest_after_one_prompt(tmp_path):
    h = str(set_admin_password("hunter2", str(tmp_path / "p.hash")))
    fn = scripted("")
    s = prompt_for_session(hash_path=h, guest_log_path=str(tmp_path / "g.jsonl"),
                           getpass_fn=fn, report=NOOP)
    assert s.role == "guest" and len(fn.calls) == 1


def test_three_wrong_attempts_fall_through_to_guest_counted_once(tmp_path):
    h = str(set_admin_password("hunter2", str(tmp_path / "p.hash")))
    fn = scripted("x")
    log = str(tmp_path / "g.jsonl")
    s = prompt_for_session(hash_path=h, guest_log_path=log, getpass_fn=fn, report=NOOP)
    assert s == Session("guest", 1)
    assert len(fn.calls) == 3
    assert read_guest_count(log) == 1


def test_wrong_then_right_is_admin(tmp_path):
    h = str(set_admin_password("hunter2", str(tmp_path / "p.hash")))
    assert prompt_for_session(hash_path=h, guest_log_path=str(tmp_path / "g.jsonl"),
                              getpass_fn=scripted("x", "hunter2"), report=NOOP).is_admin


def test_the_password_is_never_reported(tmp_path):
    h = str(set_admin_password("hunter2", str(tmp_path / "p.hash")))
    said = []
    prompt_for_session(hash_path=h, guest_log_path=str(tmp_path / "g.jsonl"),
                       getpass_fn=scripted("wrong1", "hunter2"), report=said.append)
    text = " ".join(map(str, said))
    assert "hunter2" not in text and "wrong1" not in text


def test_guest_count_accumulates_with_timestamps(tmp_path):
    log = str(tmp_path / "g.jsonl")
    assert record_guest_session(log) == 1
    assert record_guest_session(log) == 2
    for line in open(log, encoding="utf-8"):
        datetime.fromisoformat(json.loads(line)["started_at"])


def test_read_guest_count_is_zero_without_a_log(tmp_path):
    assert read_guest_count(str(tmp_path / "missing.jsonl")) == 0


@pytest.mark.parametrize("answer,expected", [
    ("y", True), ("yes", True), ("Y ", True), ("YES", True),
    ("n", False), ("", False), ("yeah", False), ("no", False),
])
def test_terminal_confirm_accepts_only_y_or_yes(answer, expected):
    req = RunRequest("python", "print(1)", "pasted")
    assert terminal_confirm(req, input_fn=lambda _: answer, output=NOOP) is expected


def test_terminal_confirm_shows_the_code_and_language():
    shown = []
    terminal_confirm(RunRequest("python", "print(1)", "pasted"),
                     input_fn=lambda _: "n", output=shown.append)
    text = "\n".join(shown)
    assert "print(1)" in text and "python" in text and "pasted" in text


def test_terminal_confirm_shows_every_line_that_would_run():
    # final review I3: "y" runs all of it, so all of it is shown — nothing elided
    shown = []
    source = "\n".join(f"x{i} = {i}" for i in range(100))
    terminal_confirm(RunRequest("python", source, "pasted"),
                     input_fn=lambda _: "n", output=shown.append)
    text = "\n".join(shown)
    assert "x0 = 0" in text and "x99 = 99" in text and "more lines" not in text


@pytest.mark.parametrize("hidden", ["\x1b[2K", "\r", "\x1b[1A", "\x08", "\x00", "\u202e"])
def test_terminal_confirm_escapes_control_characters(hidden):
    # final review I3: a carriage return or an ANSI escape could overwrite a line on screen
    shown = []
    source = f"print('safe'){hidden}import os; os.remove('x')"
    terminal_confirm(RunRequest("python", source, "pasted"),
                     input_fn=lambda _: "n", output=shown.append)
    text = "\n".join(shown)
    assert hidden not in text
    assert "import os; os.remove('x')" in text


def test_escape_for_terminal_keeps_tabs_and_printable_text():
    from admin_session import escape_for_terminal
    assert escape_for_terminal("a\tb = 'ü'") == "a\tb = 'ü'"
    assert escape_for_terminal("x\x1by") == "x\\x1by"


def test_terminal_confirm_eof_is_no():
    def eof(_):
        raise EOFError
    assert terminal_confirm(RunRequest("python", "x", "pasted"), input_fn=eof, output=NOOP) is False


def test_set_admin_password_script_rejects_mismatch_and_empty(tmp_path, monkeypatch):
    from scripts import set_admin_password as script
    target = tmp_path / "p.hash"
    monkeypatch.setattr(script.getpass, "getpass", scripted("a", "b"))
    assert script.main(["--path", str(target)]) == 1 and not target.exists()
    monkeypatch.setattr(script.getpass, "getpass", scripted(""))
    assert script.main(["--path", str(target)]) == 1 and not target.exists()
    monkeypatch.setattr(script.getpass, "getpass", scripted("hunter2"))
    assert script.main(["--path", str(target)]) == 0
    assert verify_password(target.read_text(encoding="utf-8").strip(), "hunter2")
