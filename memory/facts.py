"""Dated facts: a small knowledge graph of triples that keeps its history.

One SQLite table. Each row says "(subject, relation, object) was true from `valid_from`
until `valid_until`". A fact that stops being true is CLOSED — given an end date — never
deleted, so "where did I live last semester?" still has an answer after a move.

Pure stdlib. No agent, no Chroma, no network: this module is not wired into anything yet.

Dates are ISO strings, "YYYY-MM-DD", everywhere. They are compared as plain text, and
text order only matches calendar order in exactly that zero-padded form.
"""
import logging
import sqlite3
from datetime import date as _date, datetime

logger = logging.getLogger(__name__)

#: `valid_until` NULL means "still true". SQLite's implicit rowid is the only key.
SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    subject     TEXT NOT NULL,
    relation    TEXT NOT NULL,
    object      TEXT NOT NULL,
    valid_from  TEXT NOT NULL,
    valid_until TEXT,
    source      TEXT NOT NULL,
    CHECK (valid_until IS NULL OR valid_until >= valid_from)
)
"""

ONE = "one"
MANY = "many"

#: How many objects a subject can hold at once for each relation. ONE means a new fact
#: supersedes the open one; MANY means facts accumulate side by side.
RELATIONS = {
    "lives_in": ONE,
    "studies_at": ONE,
    "friend_of": MANY,
    "working_on": MANY,
}


def _one_open_index_sql() -> str:
    """The table-level half of the one-open-row rule, built from RELATIONS.

    A partial index's WHERE clause must be made of literals — no parameters, no
    subqueries — so the ONE relations are written into the statement as quoted strings.
    They come from this module's own dict, never from a caller.
    """
    ones = ", ".join("'" + name.replace("'", "''") + "'"
                     for name in sorted(RELATIONS) if RELATIONS[name] == ONE)
    return ("CREATE UNIQUE INDEX one_open_single ON facts (subject, relation) "
            f"WHERE valid_until IS NULL AND relation IN ({ones})")


def connect(db_path) -> sqlite3.Connection:
    """Open the fact store at `db_path`, creating the table on first use.

    The index is dropped and rebuilt on every open rather than created IF NOT EXISTS:
    its relation list is frozen into the database file when it is created, so an index
    left over from an older RELATIONS would silently not cover a relation added since.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    with conn:
        conn.execute(SCHEMA)
        conn.execute("DROP INDEX IF EXISTS one_open_single")
        conn.execute(_one_open_index_sql())
    return conn


def _iso_date(value: str) -> str:
    """The one gate every date passes through on its way into a query. Raises if not ISO.

    Parsing alone is not enough: strptime is lenient and reads "2026-7-1" as a valid
    date. So the parsed date is written back out and compared with what came in — only
    a string that was already strict, zero-padded YYYY-MM-DD survives the round trip.
    """
    if datetime.strptime(value, "%Y-%m-%d").strftime("%Y-%m-%d") != value:
        raise ValueError(f"date {value!r} is not in strict YYYY-MM-DD form")
    return value


def cardinality(relation: str) -> str:
    """Return ONE or MANY for `relation`. An unknown relation is MANY, and is logged.

    The default leans toward MANY because the two mistakes are not equal: a wrong ONE
    silently closes a fact that is still true, while a wrong MANY only leaves two open
    facts side by side — a conflict that can be seen and fixed.
    """
    try:
        return RELATIONS[relation]
    except KeyError:
        logger.warning("unknown relation %r: treating it as MANY. Add it to "
                       "memory.facts.RELATIONS to classify it.", relation)
        return MANY


def add_fact(conn: sqlite3.Connection, subject: str, relation: str, object: str,
             source: str, on_date: str = None) -> bool:
    """Record that (subject, relation, object) is true from `on_date` (default: today).

    Returns True if a row was inserted, False if the exact fact was already open.

    For a ONE relation the currently open row is closed first — given `on_date` as its
    end — and that close and the insert commit together or not at all. This is the only
    way a fact is superseded; `forget_fact` is for facts that should never have been here.

    Raises ValueError, writing nothing, if `on_date` is not strict YYYY-MM-DD or if it is
    earlier than the start of the ONE fact it would supersede.
    """
    on_date = _iso_date(_date.today().isoformat() if on_date is None else on_date)
    with conn:      # one transaction: commits on success, rolls back on any exception
        already_open = conn.execute(
            "SELECT 1 FROM facts WHERE subject = ? AND relation = ? AND object = ? "
            "AND valid_until IS NULL",
            (subject, relation, object)).fetchone()
        if already_open:
            return False
        if cardinality(relation) == ONE:
            open_row = conn.execute(
                "SELECT object, valid_from FROM facts "
                "WHERE subject = ? AND relation = ? AND valid_until IS NULL",
                (subject, relation)).fetchone()
            if open_row is not None and on_date < open_row["valid_from"]:
                # closing it at on_date would make it end before it began. Facts are
                # entered in date order; inserting into the past is not supported here.
                raise ValueError(
                    f"cannot start ({subject}, {relation}, {object}) on {on_date}: the "
                    f"current fact ({open_row['object']}) only began on "
                    f"{open_row['valid_from']}")
            conn.execute(
                "UPDATE facts SET valid_until = ? "
                "WHERE subject = ? AND relation = ? AND valid_until IS NULL",
                (on_date, subject, relation))
        conn.execute(
            "INSERT INTO facts (subject, relation, object, valid_from, valid_until, source) "
            "VALUES (?, ?, ?, ?, NULL, ?)",
            (subject, relation, object, on_date, source))
    return True


def current(conn: sqlite3.Connection, subject: str, relation: str) -> list:
    """The rows that are still true: `valid_until` is NULL."""
    return conn.execute(
        "SELECT subject, relation, object, valid_from, valid_until, source FROM facts "
        "WHERE subject = ? AND relation = ? AND valid_until IS NULL "
        "ORDER BY valid_from, rowid",
        (subject, relation)).fetchall()


def as_of(conn: sqlite3.Connection, subject: str, relation: str, date: str) -> list:
    """The rows that were true on `date`. The start day counts; the end day does not."""
    date = _iso_date(date)
    return conn.execute(
        "SELECT subject, relation, object, valid_from, valid_until, source FROM facts "
        "WHERE subject = ? AND relation = ? AND valid_from <= ? "
        "AND (valid_until IS NULL OR ? < valid_until) "
        "ORDER BY valid_from, rowid",
        (subject, relation, date, date)).fetchall()


def forget_fact(conn: sqlite3.Connection, subject: str, relation: str, object: str) -> int:
    """Hard-delete every row for this fact, open or closed. Returns how many were removed.

    Only for an explicit "forget". A fact that merely stopped being true is superseded
    through `add_fact`, which keeps its history; nothing here keeps any.
    """
    with conn:
        return conn.execute(
            "DELETE FROM facts WHERE subject = ? AND relation = ? AND object = ?",
            (subject, relation, object)).rowcount
