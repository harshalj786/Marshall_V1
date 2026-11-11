"""Dated facts: a changed fact is closed, never deleted — unless it is explicitly forgotten.

Pure stdlib over a throwaway SQLite file — no graph, no LLM, no Ollama, no Chroma.

    pytest tests/test_facts.py -v
    python tests/test_facts.py

The first four tests are the feature's done-when. The rest pin the two guards that keep
the table honest: dates must be strict YYYY-MM-DD, and a row may not end before it starts.
"""
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory import facts

ME = "harshal"


@pytest.fixture
def conn(tmp_path):
    connection = facts.connect(tmp_path / "facts.sqlite3")
    yield connection
    connection.close()


def all_rows(conn):
    """Every row in the table as (object, valid_from, valid_until), in insertion order."""
    return [tuple(row) for row in conn.execute(
        "SELECT object, valid_from, valid_until FROM facts ORDER BY rowid")]


def objects(rows):
    return [row["object"] for row in rows]


def move_hostels(conn):
    facts.add_fact(conn, ME, "lives_in", "Hostel A", "typed", on_date="2026-07-01")
    facts.add_fact(conn, ME, "lives_in", "Hostel B", "typed", on_date="2026-11-01")


# --------------------------------------------------------------------------
# the four done-when tests
# --------------------------------------------------------------------------

def test_moving_hostels_closes_the_old_row_and_opens_a_new_one(conn):
    move_hostels(conn)

    assert all_rows(conn) == [
        ("Hostel A", "2026-07-01", "2026-11-01"),   # closed on the day of the move, kept
        ("Hostel B", "2026-11-01", None),           # open
    ]
    assert objects(facts.current(conn, ME, "lives_in")) == ["Hostel B"]


def test_as_of_last_semester_returns_the_old_hostel_and_current_the_new(conn):
    move_hostels(conn)

    assert objects(facts.as_of(conn, ME, "lives_in", "2026-08-15")) == ["Hostel A"]
    assert objects(facts.current(conn, ME, "lives_in")) == ["Hostel B"]


def test_a_many_relation_closes_nothing(conn):
    facts.add_fact(conn, ME, "friend_of", "Asha", "typed", on_date="2026-07-01")
    facts.add_fact(conn, ME, "friend_of", "Ravi", "typed", on_date="2026-08-01")

    assert all_rows(conn) == [
        ("Asha", "2026-07-01", None),
        ("Ravi", "2026-08-01", None),
    ]
    assert objects(facts.current(conn, ME, "friend_of")) == ["Asha", "Ravi"]


def test_forget_removes_the_fact_entirely(conn):
    facts.add_fact(conn, ME, "lives_in", "Hostel A", "typed", on_date="2026-07-01")

    facts.forget_fact(conn, ME, "lives_in", "Hostel A")

    assert facts.current(conn, ME, "lives_in") == []
    assert facts.as_of(conn, ME, "lives_in", "2026-08-15") == []
    assert all_rows(conn) == [], "forget is a hard delete: no closed row may be left behind"


# --------------------------------------------------------------------------
# the guards
# --------------------------------------------------------------------------

@pytest.mark.parametrize("unpadded", ["2026-7-1", "2026-07-1"])
def test_an_unpadded_date_is_rejected_not_stored(conn, unpadded):
    """A lenient parser accepts these; stored as text they would sort in the wrong place."""
    with pytest.raises(ValueError):
        facts.add_fact(conn, ME, "lives_in", "Hostel A", "typed", on_date=unpadded)
    assert all_rows(conn) == []

    with pytest.raises(ValueError):
        facts.as_of(conn, ME, "lives_in", unpadded)


def test_backdating_a_one_relation_raises_and_writes_nothing(conn):
    facts.add_fact(conn, ME, "lives_in", "Hostel B", "typed", on_date="2026-11-01")
    before = all_rows(conn)

    with pytest.raises(ValueError):
        facts.add_fact(conn, ME, "lives_in", "Hostel A", "typed", on_date="2026-07-01")

    assert all_rows(conn) == before, "the open row must be neither closed nor joined"


def test_the_table_itself_refuses_a_row_that_ends_before_it_starts(conn):
    """The seatbelt under the backdating guard: it holds for code that skips add_fact."""
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO facts VALUES (?, ?, ?, ?, ?, ?)",
                     (ME, "lives_in", "Hostel A", "2026-11-01", "2026-07-01", "raw"))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--no-header", "-p", "no:cacheprovider"]))
