"""Eager loading of the process-wide embedding singletons.

`docs/architecture_review.md` §10 names what is left of the startup cost after R1 closed
the ingestion half: *"~4.8 s of the remaining 7.9 s is the MiniLM load, needed to embed
queries, and it is still synchronous and still ahead of the first word. Loading it eagerly
at process start ... is the next cut."* This module is that cut.

There are two independent MiniLM singletons — `ingestion/vector_db.py:_get_embedding_model`
(document retrieval) and `memory/long_term_memory.py:_get_embedding_model` (user memory) —
and a voice session hits the first on turn one and the second on the first memory-gate hit
or `recall_user_memory` call. Left lazy, each becomes dead air at a moment the user is
already waiting. Warming them costs the same seconds during boot, where nobody is listening.

Nothing here changes behaviour: both functions are already idempotent get-or-build, so
calling them early only decides WHEN the cost is paid. Existing callers are untouched and
a process that never calls `warm_embeddings()` behaves exactly as it does today.

This lives at top level rather than inside either package because it is the only thing that
needs both: as a leaf that imports the two and is imported by neither, it adds no edge to
the dependency graph §2 describes as acyclic.

    from warmup import warm_embeddings
    timings = warm_embeddings()      # {"ingestion": 4.81, "memory": 0.00}
"""
import time

#: short, meaningless, and never stored — this is a forward pass, not data. Embedding once
#: is what separates "weights loaded" from "ready to answer": the first embed_query also
#: pays tokenizer setup and, on CUDA, kernel compilation, which is otherwise charged to the
#: first thing the user says.
_PROBE_TEXT = "warm up"


def warm_embeddings(probe: bool = True) -> dict[str, float]:
    """Force both MiniLM singletons to load now. Returns seconds spent on each.

    Both keys are always present. A singleton already built reports ~0.0, which is what
    makes the return value readable as "what did boot actually cost" rather than a guess.

    `probe=False` builds the models without the extra forward pass — useful when the caller
    only wants the weights resident and is not about to embed anything.
    """
    from ingestion.vector_db import _get_embedding_model as _ingestion_embeddings
    from memory.long_term_memory import _get_embedding_model as _memory_embeddings

    timings: dict[str, float] = {}
    #: the two are separate objects on purpose (§2 "shared globals"), so both are warmed
    #: rather than one being assumed to cover the other
    for label, get_model in (("ingestion", _ingestion_embeddings),
                             ("memory", _memory_embeddings)):
        started = time.perf_counter()
        model = get_model()
        if probe:
            #: a warm-up must never be able to break a boot that would otherwise succeed —
            #: if the probe fails the weights are still loaded, which was the point
            try:
                model.embed_query(_PROBE_TEXT)
            except Exception:
                pass
        timings[label] = time.perf_counter() - started
    return timings
