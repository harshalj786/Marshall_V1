"""Ingestion as an explicit step: source -> chunks -> a vector collection on disk.

Until now this sequence had no home of its own. It lived inline in seven constructors
(`pipeline.py` x4, `pipeline_lg.py` x3), each of which loaded, chunked and embedded the
whole document before it could answer anything, because `Retrival` could not open a
collection without being handed the documents again — R1 in docs/architecture_review.md.

Ingest once:

    from ingestion.ingest import ingest
    ingest(file_path="test_pdf's/Pipes_2025.pdf", persist_directory_path="db/")

then open the result with no documents, no chunking and no embedding of the corpus:

    from retreival.retrieval import Retrival
    from ingestion.ingest import collection_name_for
    r = Retrival(mode="simple",
                 collection_name=collection_name_for("test_pdf's/Pipes_2025.pdf"),
                 path="db/")

The pipelines call `ensure_ingested`, which does the first only when the second would
otherwise fail — so their existing calling patterns keep working, and a second run over
an already-ingested document skips straight to the query.
"""
from typing import Literal, Optional

from ingestion.doc_loader import LoadDocs
from ingestion.chunking import Chunking
from ingestion.vector_db import (
    BuildKnowledge,
    collection_exists,
    sanitize_collection_name,
)

#: the values every pipeline has always hardcoded, in one place so ingesting by hand
#: produces the same chunks as ingesting through a pipeline. Different values mean
#: different chunk boundaries, hence different ids, hence a second copy of the corpus
#: alongside the first — see the R1 trade-off note in docs/architecture_review.md.
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50


def collection_name_for(file_path) -> str:
    """The collection a given source document is indexed under.

    Ingestion and retrieval address the same store only if they spell its name the same
    way, so both derive it here. The `f"VDB for {file_path}"` shape is what the pipelines
    have always passed, and sanitization is unchanged, so collections built before
    ingestion was split out still resolve to exactly the same name.
    """
    return sanitize_collection_name(f"VDB for {file_path}")


def _resolve_name(file_path, collection_name: Optional[str]) -> str:
    return sanitize_collection_name(collection_name) if collection_name else collection_name_for(file_path)


def ingest(file_path,
           mode_docs: Optional[str] = "pdf",
           glob_pattern: Optional[str] = None,
           mode_chunks: Optional[Literal["recursive", "semantic"]] = "recursive",
           persist_directory_path: Optional[str] = None,
           collection_name: Optional[str] = None,
           chunk_size: int = DEFAULT_CHUNK_SIZE,
           chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> str:
    """Load, chunk and index `file_path`. Returns the collection name it wrote to.

    Idempotent: chunk ids are content+metadata hashes, so ingesting the same document
    twice upserts the same rows rather than appending a second copy (C1).
    """
    name = _resolve_name(file_path, collection_name)
    loader = LoadDocs(file_path=file_path, mode=mode_docs)
    docs = loader.invoke({"glob_pattern": glob_pattern} if glob_pattern else {})
    chunker = Chunking(chunk_size=chunk_size, chunk_overlap=chunk_overlap, mode=mode_chunks)
    chunked = chunker.invoke({"docs": docs})
    builder = BuildKnowledge(docs=chunked, persist_directory_path=persist_directory_path)
    builder.invoke({"collection_name": name})
    return name


def ensure_ingested(file_path,
                    mode_docs: Optional[str] = "pdf",
                    glob_pattern: Optional[str] = None,
                    mode_chunks: Optional[Literal["recursive", "semantic"]] = "recursive",
                    persist_directory_path: Optional[str] = None,
                    collection_name: Optional[str] = None,
                    chunk_size: int = DEFAULT_CHUNK_SIZE,
                    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> str:
    """Ingest only if the collection is not already on disk. Returns the collection name.

    This is what makes a warm start free rather than merely idempotent: the existence
    check is a sqlite row count, so nothing is read from the source, nothing is chunked,
    and nothing is embedded. With `persist_directory_path=None` there is no disk to check
    and the store dies with the process, so ingestion always runs — the same behaviour
    those callers have always had.
    """
    name = _resolve_name(file_path, collection_name)
    if collection_exists(name, persist_directory_path):
        return name
    return ingest(
        file_path=file_path,
        mode_docs=mode_docs,
        glob_pattern=glob_pattern,
        mode_chunks=mode_chunks,
        persist_directory_path=persist_directory_path,
        collection_name=name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
