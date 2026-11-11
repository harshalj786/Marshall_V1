"""Who is talking to Marshall: an admin session or a guest session, decided once at startup.

    python scripts/set_admin_password.py       # once, by the owner

Both entry points (`chat_main.py`, `voice_main.py`) call `prompt_for_session()` BEFORE the
agent is built. The password is read with `getpass`, checked here against an Argon2id hash
(`docs/security.md` §2), and then forgotten. The only thing that leaves this module is a
`Session` — a role and a count — so the model never sees the password, never sees that one
was asked for, and has nothing to repeat. What the role changes is which tools and which
memory the agent is built with; that decision is `AgentGraph`'s.

Also here, because it is the other half of "the user decides, not the model":
`terminal_confirm`, the text-chat `[y/N]` that `run_code` must pass before anything runs.
"""
import getpass
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tools.code_tools import RunRequest

#: under `db/`, which .gitignore already excludes — the hash never reaches version control
ADMIN_HASH_PATH = "db/admin/password.hash"
#: one JSON line per guest session: a timestamp and nothing else. Nothing a guest SAYS is
#: ever written here; the owner asked to know how many guests there were, not what they said.
GUEST_LOG_PATH = "db/admin/guest_sessions.jsonl"
PASSWORD_ATTEMPTS = 3


@dataclass(frozen=True)
class Session:
    role: str            # "admin" | "guest"
    guest_count: int     # guest sessions recorded so far, this one included for a guest

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"


def hash_password(password: str) -> str:
    #: imported here so importing this module (and pipeline_lg through it) costs nothing
    from argon2 import PasswordHasher
    return PasswordHasher().hash(password)


def verify_password(encoded: str, password: str) -> bool:
    """True only for a match. A mismatch, a corrupt hash or an unreadable one is False."""
    from argon2 import PasswordHasher
    try:
        return PasswordHasher().verify(encoded, password)
    except Exception:
        return False


def set_admin_password(password: str, path: str = ADMIN_HASH_PATH) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(hash_password(password) + "\n", encoding="utf-8")
    return target


def password_is_set(path: str = ADMIN_HASH_PATH) -> bool:
    target = Path(path)
    return target.is_file() and bool(target.read_text(encoding="utf-8").strip())


def read_guest_count(path: str = GUEST_LOG_PATH) -> int:
    target = Path(path)
    if not target.is_file():
        return 0
    return sum(1 for line in target.read_text(encoding="utf-8").splitlines() if line.strip())


def record_guest_session(path: str = GUEST_LOG_PATH) -> int:
    """Append one guest start and return the new total."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "a", encoding="utf-8") as log:
        log.write(json.dumps({"started_at": datetime.now(timezone.utc).isoformat()}) + "\n")
    return read_guest_count(path)


def _guest(guest_log_path: str) -> Session:
    #: the single place a guest start is recorded, so every route to a guest session —
    #: no password set, Enter, three misses — counts exactly once
    return Session("guest", record_guest_session(guest_log_path))


def prompt_for_session(hash_path: str = ADMIN_HASH_PATH,
                       guest_log_path: str = GUEST_LOG_PATH,
                       attempts: int = PASSWORD_ATTEMPTS,
                       getpass_fn=getpass.getpass,
                       report=print) -> Session:
    """Ask for the admin password in the terminal, hidden. Enter alone means guest."""
    if not password_is_set(hash_path):
        report("no admin password is set — run scripts/set_admin_password.py; "
               "starting a guest session")
        return _guest(guest_log_path)
    encoded = Path(hash_path).read_text(encoding="utf-8").strip()
    for attempt in range(attempts):
        entered = getpass_fn("admin password (Enter for a guest session): ")
        if not entered:
            break
        if verify_password(encoded, entered):
            report("admin session")
            return Session("admin", read_guest_count(guest_log_path))
        left = attempts - attempt - 1
        if left:
            report(f"wrong password ({left} left)")
    session = _guest(guest_log_path)
    report(f"guest session (#{session.guest_count})")
    return session


def escape_for_terminal(text: str) -> str:
    """Make every character visible: control and format characters become escapes.

    The confirmation is only as good as what the user can SEE. A `\\r`, an ANSI erase-line
    or a right-to-left override in the code would let part of it overwrite or reorder
    another part on screen — and the model composes that code, possibly under injection.
    """
    return "".join(ch if ch == "\t" or ch.isprintable() else ascii(ch)[1:-1] for ch in text)


def terminal_confirm(request: "RunRequest", input_fn=input, output=print) -> bool:
    """Show ALL the code about to run and ask `[y/N]`. Only "y" or "yes" runs it; EOF is no.

    Nothing is elided, because "y" runs everything: approving a program whose tail the user
    never saw is not a confirmation. Split on "\\n" only, so a lone "\\r" is shown, not obeyed.
    """
    output(f"Marshall wants to run this {request.language} ({escape_for_terminal(request.origin)}):")
    output("-" * 60)
    for line in request.source.split("\n"):
        output(escape_for_terminal(line))
    output("-" * 60)
    try:
        answer = input_fn("run it? [y/N] ")
    except EOFError:
        return False
    return answer.strip().lower() in ("y", "yes")
