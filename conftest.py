"""Shared pytest setup: import bootstrap, and marker registration for the model-gated tests.

WHY THERE ARE NO SHARED FIXTURES HERE
-------------------------------------
All ten test modules were read looking for fixtures to lift. There is almost nothing
worth lifting, and the two things that look like duplication are both deliberate:

* `_ollama_up()` is defined three times (test_ingestion_retrieval_split.py:51,
  test_memory_gate_resilience.py:32, test_routing_matrix.py:50) with identical bodies.
  This is NOT an oversight — test_routing_matrix.py's own module docstring says it is
  "restated rather than imported because that is this repo's pattern for it". Hoisting
  it here would also move the skip decision out of the module that skips, and each copy
  is evaluated exactly once at import to produce a module-level `skipif`. Left alone.

* The module-scoped workspace fixtures all build a temp corpus, but each builds a
  different one for a different subsystem (ingestion index, memory store, agent with
  seeded history). They share a shape, not a body. Merging them would mean editing
  test bodies to match a common fixture, which is a refactor, not deduplication.

The one thing genuinely repeated in all ten files is the `sys.path` bootstrap below.
Those in-file copies are intentionally NOT removed: several modules advertise direct
invocation (`python tests/test_routing_matrix.py`), which never loads this conftest.
Re-inserting a path already on sys.path is a no-op in effect, so the two coexist.
"""
import sys
from pathlib import Path

import pytest

#: repo root — importable as `pipeline_lg`, `tools`, `voice`, `scripts`, `memory`.
#: pytest would put this on sys.path anyway by virtue of this file's location under
#: the default prepend import mode; doing it explicitly means the reason is written
#: down rather than being a side effect of where the file happens to sit.
REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def pytest_itemcollected(item):
    """Attach `needs_ollama` / `needs_llama31` by reading the skipif already on the test.

    The three modules that need a local model server each define a module-level alias
    (`needs_ollama = pytest.mark.skipif(...)` at test_memory_gate_resilience.py:41 and
    test_ingestion_retrieval_split.py:60; `needs_llama31 = pytest.mark.skipif(...)` at
    test_routing_matrix.py:69) and decorate individual tests with it. Those are `skipif`
    marks, so `-m needs_ollama` could not see them — the names existed but selection did
    not. This derives the selectable marker FROM that skipif rather than adding a second,
    independent declaration next to it, so the marker and the skip condition cannot drift
    apart: there is still exactly one place that decides whether a test needs Ollama.

    Deliberately not module-scoped. Only 2 of the 9 tests in test_memory_gate_resilience
    and 2 of the 18 in test_ingestion_retrieval_split are gated; marking whole modules
    would deselect fifteen tests that run fine with no model server at all.

    `needs_llama31` implies `needs_ollama` — a specific model is a stricter requirement
    than a reachable server — so `-m "not needs_ollama"` is the single predicate that
    excludes everything needing a model, and CI uses exactly that.

    Hook choice matters: `pytest_itemcollected` runs during collection, whereas `-m`
    deselection happens in `pytest_collection_modifyitems(tryfirst=True)`. Markers added
    from a plain `modifyitems` hook in a conftest would land after the filter had already
    run and would be invisible to `-m`.

    If these reason strings are ever reworded the coupling breaks — but it breaks safely.
    The marker stops being applied, `-m "not needs_ollama"` stops deselecting the test,
    and the untouched `skipif` still skips it. CI reports a skip instead of a
    deselection; it does not try to reach a model server that is not there.
    """
    for mark in item.iter_markers(name="skipif"):
        reason = str(mark.kwargs.get("reason") or "").lower()
        if "ollama" not in reason:
            continue
        item.add_marker(pytest.mark.needs_ollama)
        if "pulled" in reason:      # "...with llama3.1 pulled" — server AND model
            item.add_marker(pytest.mark.needs_llama31)
        return
