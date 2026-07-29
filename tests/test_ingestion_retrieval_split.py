"""R1 regression: retrieval must be able to open an index it did not build.

`Retrival.__init__` used to construct `BuildKnowledge(docs=...)` unconditionally, so
there was no way to query an existing collection without re-supplying the documents.
Every pipeline construction therefore re-ran load -> chunk -> embed -> upsert even when
the index was already on disk. These tests pin the split:

  * ingestion writes a collection that outlives the process that wrote it
  * retrieval opens it with no documents at all and returns relevant chunks
  * ingesting twice does not duplicate (C1, still true through the new entry point)
  * opening a collection that was never built fails loudly instead of returning nothing
  * both pipeline generations still construct through their existing calling patterns

Runs under pytest, or standalone:

    pytest tests/test_ingestion_retrieval_split.py -v
    python tests/test_ingestion_retrieval_split.py

No LLM is needed: every assertion is about the index and the constructors. The two
generation tests are skipped automatically when Ollama is not reachable.
"""
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from ingestion.ingest import collection_name_for, ensure_ingested, ingest
from ingestion.vector_db import (
    BuildKnowledge,
    CollectionNotFoundError,
    collection_exists,
    sanitize_collection_name,
)
from retreival.retrieval import Retrival

DOC = """Marshall is a local RAG framework built by Harshal.
It loads documents, chunks them, and embeds the chunks into a Chroma collection.
The retrieval layer supports simple, mmr, mqr and wiki modes.
Long-term memory about the user is stored separately from the document index.
The agent reaches the loaded document only through the search_current_doc tool.
"""


def _ollama_up() -> bool:
    try:
        import urllib.request
        urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=3)
        return True
    except Exception:
        return False


needs_ollama = pytest.mark.skipif(not _ollama_up(), reason="Ollama not reachable on :11434")


@pytest.fixture(scope="module")
def workspace():
    tmp = Path(tempfile.mkdtemp(prefix="marshall_r1_"))
    doc = tmp / "doc.txt"
    doc.write_text(DOC, encoding="utf-8")
    yield {"tmp": tmp, "doc": doc, "db": tmp / "vdb"}
    shutil.rmtree(tmp, ignore_errors=True)


def _count(collection_name: str, db_path: Path) -> int:
    import chromadb
    client = chromadb.PersistentClient(path=str(db_path))
    return client.get_collection(collection_name).count()


# --------------------------------------------------------------------------
# naming — the one string ingestion and retrieval must agree on
# --------------------------------------------------------------------------

def test_collection_name_matches_the_legacy_pipeline_convention():
    """Collections built before the split must still resolve to the same name."""
    path = "test_pdf's/Pipes_2025.pdf"
    assert collection_name_for(path) == sanitize_collection_name(f"VDB for {path}")


def test_retrival_sanitizes_identically_to_ingestion(workspace):
    """Both sides run the name through the same helper, so neither can mangle it alone."""
    raw = f"VDB for {workspace['doc']}"
    ingest(file_path=str(workspace["doc"]), mode_docs="txt",
           persist_directory_path=str(workspace["db"]), collection_name=raw)
    r = Retrival(mode="simple", collection_name=raw, path=str(workspace["db"]))
    assert r.collection_name == sanitize_collection_name(raw)
    assert _count(r.collection_name, workspace["db"]) > 0


# --------------------------------------------------------------------------
# ingestion writes something durable
# --------------------------------------------------------------------------

def test_ingest_builds_a_collection_on_disk(workspace):
    name = ingest(file_path=str(workspace["doc"]), mode_docs="txt",
                  persist_directory_path=str(workspace["db"]))
    assert collection_exists(name, str(workspace["db"]))
    assert _count(name, workspace["db"]) > 0


def test_collection_survives_a_fresh_process(workspace):
    """The real durability claim: a process that never ingested can still query it."""
    name = ingest(file_path=str(workspace["doc"]), mode_docs="txt",
                  persist_directory_path=str(workspace["db"]))
    script = textwrap.dedent(f"""
        import sys, warnings
        warnings.filterwarnings("ignore")
        sys.path.insert(0, {str(REPO)!r})
        from retreival.retrieval import Retrival
        r = Retrival(mode="simple", collection_name={name!r}, path={str(workspace["db"])!r})
        out = r.invoke({{"query": "what retrieval modes are supported?", "k": 2}})
        joined = " ".join(d.page_content for d in out["context"])
        print("HITS", len(out["context"]))
        print("MATCH", "mmr" in joined)
    """)
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-2000:]
    # content, not count: this corpus chunks into fewer than k pieces, and what is being
    # proved is that a process which never ingested read the real rows off disk
    assert "HITS 0" not in proc.stdout, proc.stdout + proc.stderr[-2000:]
    assert "MATCH True" in proc.stdout, proc.stdout + proc.stderr[-2000:]


# --------------------------------------------------------------------------
# retrieval opens what ingestion wrote — with no documents
# --------------------------------------------------------------------------

def test_retrieval_opens_existing_collection_without_documents(workspace):
    name = ingest(file_path=str(workspace["doc"]), mode_docs="txt",
                  persist_directory_path=str(workspace["db"]))
    r = Retrival(mode="simple", collection_name=name, path=str(workspace["db"]))
    assert r.docs is None, "opening an index must not require the corpus"
    result = r.invoke({"query": "how is the user's long-term memory stored?", "k": 2})
    assert result["context"], "opening the collection returned no chunks"
    joined = " ".join(d.page_content for d in result["context"]).lower()
    assert "memory" in joined, f"retrieved chunks are not relevant: {joined!r}"


def test_retrieval_works_after_the_source_document_is_gone():
    """Independence, stated as sharply as it can be: ingestion consumed the source, and
    retrieval never needs it again. The old constructor could not survive this at all."""
    tmp = Path(tempfile.mkdtemp(prefix="marshall_r1_gone_"))
    try:
        doc = tmp / "doc.txt"
        doc.write_text(DOC, encoding="utf-8")
        db = tmp / "vdb"
        name = ingest(file_path=str(doc), mode_docs="txt", persist_directory_path=str(db))
        doc.unlink()

        r = Retrival(mode="simple", collection_name=name, path=str(db))
        joined = " ".join(d.page_content
                          for d in r.invoke({"query": "what is Marshall?", "k": 2})["context"])
        assert "Marshall" in joined
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_opening_does_not_load_or_chunk_anything(workspace, monkeypatch):
    """The cost claim. If the corpus were still read, these would fire."""
    import ingestion.ingest as ingest_module

    name = ingest(file_path=str(workspace["doc"]), mode_docs="txt",
                  persist_directory_path=str(workspace["db"]))

    def _boom(*a, **kw):
        raise AssertionError("a warm start must not touch the loader or the chunker")

    monkeypatch.setattr(ingest_module, "LoadDocs", _boom)
    monkeypatch.setattr(ingest_module, "Chunking", _boom)

    assert ensure_ingested(file_path=str(workspace["doc"]), mode_docs="txt",
                           persist_directory_path=str(workspace["db"])) == name
    r = Retrival(mode="simple", collection_name=name, path=str(workspace["db"]))
    assert r.invoke({"query": "chunks", "k": 1})["context"]


# --------------------------------------------------------------------------
# C1 stays fixed through the new entry point
# --------------------------------------------------------------------------

def test_ingesting_twice_does_not_duplicate(workspace):
    tmp = Path(tempfile.mkdtemp(prefix="marshall_r1_dup_"))
    try:
        doc = tmp / "doc.txt"
        doc.write_text(DOC, encoding="utf-8")
        db = tmp / "vdb"
        name = ingest(file_path=str(doc), mode_docs="txt", persist_directory_path=str(db))
        first = _count(name, db)
        ingest(file_path=str(doc), mode_docs="txt", persist_directory_path=str(db))
        ingest(file_path=str(doc), mode_docs="txt", persist_directory_path=str(db))
        assert first > 0
        assert _count(name, db) == first, "re-ingestion appended a second copy of the corpus"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------
# the explicit failure — never a silently empty index
# --------------------------------------------------------------------------

def test_missing_collection_raises_a_clear_error(workspace):
    with pytest.raises(CollectionNotFoundError) as excinfo:
        Retrival(mode="simple", collection_name="never_ingested_anything",
                 path=str(workspace["db"]))
    message = str(excinfo.value)
    assert "never_ingested_anything" in message, "the error must name the collection"
    assert "ingest" in message.lower(), "the error must say how to fix it"


def test_failed_open_does_not_create_the_collection(workspace):
    """Chroma's constructor is get-or-create; a failed open must not leave a stub behind."""
    with pytest.raises(CollectionNotFoundError):
        BuildKnowledge(persist_directory_path=str(workspace["db"])).invoke(
            {"collection_name": "phantom_collection"})
    assert not collection_exists("phantom_collection", str(workspace["db"]))


def test_no_persist_path_and_no_docs_also_raises():
    """An in-memory store with nothing in it is the same silent-empty failure."""
    with pytest.raises(CollectionNotFoundError):
        Retrival(mode="simple", collection_name="in_memory_nothing", path=None)


def test_wiki_mode_needs_no_collection():
    """The wiki branch never reads the vector store, so it must not demand one."""
    r = Retrival(mode="wiki", collection_name="wiki_needs_no_index", path=None)
    assert r.vector_store is None


# --------------------------------------------------------------------------
# both generations still construct through their existing calling patterns
# --------------------------------------------------------------------------

def test_legacy_pipeline_constructors_still_work(workspace):
    from pipeline import ChatRagPipeline, ChatRagPipelineWithMemory, NRagPipeline

    for cls in (NRagPipeline, ChatRagPipeline, ChatRagPipelineWithMemory):
        obj = cls(
            file_path=str(workspace["doc"]),
            mode_docs="txt",
            mode_retreiver="simple",
            persist_directory_path=str(workspace["db"]),
        )
        assert obj.obj_3.invoke({"query": "what is Marshall?", "k": 2})["context"], cls.__name__


def test_graph_constructors_still_work(workspace):
    from pipeline_lg import AgentGraph, ChatRagGraph, ChatRagWithMemoryGraph

    for cls in (ChatRagGraph, ChatRagWithMemoryGraph, AgentGraph):
        obj = cls(
            file_path=str(workspace["doc"]),
            mode_docs="txt",
            mode_retreiver="simple",
            persist_directory_path=str(workspace["db"]),
        )
        assert obj.retrival.invoke({"query": "what is Marshall?", "k": 2})["context"], cls.__name__


def test_pipelines_ingest_on_a_cold_index(workspace):
    """The constructors must still work end to end when nothing has been ingested yet."""
    tmp = Path(tempfile.mkdtemp(prefix="marshall_r1_cold_"))
    try:
        doc = tmp / "doc.txt"
        doc.write_text(DOC, encoding="utf-8")
        db = tmp / "vdb"
        from pipeline_lg import ChatRagGraph

        graph = ChatRagGraph(file_path=str(doc), mode_docs="txt", mode_retreiver="simple",
                             persist_directory_path=str(db))
        assert collection_exists(collection_name_for(str(doc)), str(db))
        assert graph.retrival.invoke({"query": "what is Marshall?", "k": 2})["context"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_structure_pipeline_still_builds_from_docs(workspace):
    """Structure_pipeline is a placeholder and was left on the old build-in-constructor
    path deliberately — so Retrival(docs=...) must keep working exactly as before."""
    from ingestion.chunking import Chunking
    from ingestion.doc_loader import LoadDocs

    docs = LoadDocs(file_path=str(workspace["doc"]), mode="txt").invoke({})
    chunked = Chunking(chunk_size=500, chunk_overlap=50, mode="recursive").invoke({"docs": docs})
    r = Retrival(docs=chunked, mode="simple", collection_name="legacy_build_path",
                 path=str(workspace["db"]))
    assert r.invoke({"query": "what is Marshall?", "k": 2})["context"]


# --------------------------------------------------------------------------
# end to end — a real answer off an index this process did not build
# --------------------------------------------------------------------------

@needs_ollama
def test_chat_graph_answers_from_a_pre_ingested_index(workspace):
    from pipeline_lg import ChatRagGraph

    ingest(file_path=str(workspace["doc"]), mode_docs="txt",
           persist_directory_path=str(workspace["db"]))
    graph = ChatRagGraph(file_path=str(workspace["doc"]), mode_docs="txt",
                         mode_retreiver="simple", persist_directory_path=str(workspace["db"]))
    answer = graph.invoke({"query": "who built Marshall?"},
                          config={"configurable": {"thread_id": "r1-e2e"}})
    assert answer and answer.strip(), "no answer off the pre-ingested index"


@needs_ollama
def test_legacy_pipeline_answers_from_a_pre_ingested_index(workspace):
    from pipeline import NRagPipeline

    ingest(file_path=str(workspace["doc"]), mode_docs="txt",
           persist_directory_path=str(workspace["db"]))
    pipe = NRagPipeline(file_path=str(workspace["doc"]), mode_docs="txt",
                        mode_retreiver="simple", persist_directory_path=str(workspace["db"]))
    answer = pipe.invoke({"query": "what does Marshall load?"})
    assert answer and answer.strip()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--no-header", "-p", "no:cacheprovider"]))
