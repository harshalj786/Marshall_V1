import hashlib
import re
from pathlib import Path
from typing import Optional
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import Runnable
from langchain_core.documents import Document


class CollectionNotFoundError(RuntimeError):
    """Asked to open a collection for querying, and there is nothing on disk to open.

    The alternative — handing back an empty store — is worse than failing: retrieval
    would succeed, return zero chunks, and the model would answer "I could not find
    the answer in the provided context" for every question, with nothing anywhere
    saying the index was never built.
    """


_EMBEDDING_MODEL = None


def _get_embedding_model():
    #one model per process, mirroring memory/long_term_memory.py:15-23. Two reasons here:
    #construction costs ~3s and ~90MB, and ingestion and retrieval are now separate calls
    #that MUST embed with the same function — sharing one instance makes that structural
    #rather than a convention two call sites have to remember.
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        _EMBEDDING_MODEL = HuggingFaceEmbeddings(
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
        )
    return _EMBEDDING_MODEL


def sanitize_collection_name(name: str, fallback: str = "VDB_collection") -> str:
    #Chroma rejects anything outside [a-zA-Z0-9._-] and anything shorter than 3 chars.
    #Ingestion and retrieval both run their name through here so a source that ingests
    #as one collection can never be queried under a differently-mangled one.
    clean_name = re.sub(r'[^a-zA-Z0-9._-]', '_', name)
    clean_name = clean_name.strip('_.-')
    return clean_name if len(clean_name) >= 3 else fallback


def collection_exists(collection_name: str, persist_directory_path: Optional[str]) -> bool:
    """True when `collection_name` holds queryable rows under `persist_directory_path`.

    Goes to chromadb directly instead of opening a langchain `Chroma`: that constructor
    is get-or-CREATE, so asking it whether a collection exists would create the very
    thing being asked about — and would load MiniLM to do it.

    Counts rather than just checking presence, because a half-finished ingest leaves an
    empty collection behind and skipping ingestion for one of those is the same silent
    failure this module exists to prevent.
    """
    if not persist_directory_path:
        #persist_directory=None is an in-memory store: nothing on disk, and nothing that
        #could survive into a later process. Callers read this as "you must ingest".
        return False
    if not Path(persist_directory_path).exists():
        #PersistentClient would CREATE the directory; a question must not have side effects
        return False
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(persist_directory_path))
        return client.get_collection(collection_name).count() > 0
    except Exception:
        #missing collection, unreadable directory, version skew — all mean "cannot confirm
        #an index is there". False sends the caller to ingestion, which is idempotent, so
        #a wrong answer here costs time and never correctness.
        return False


def _chunk_id(doc: Document) -> str:
    #content+metadata fingerprint, not content alone: a header repeated on every page of a
    #PDF produces identical text on different pages, and those are distinct chunks
    meta = "|".join(f"{key}={doc.metadata[key]}" for key in sorted(doc.metadata))
    return hashlib.sha1(f"{meta}||{doc.page_content}".encode("utf-8")).hexdigest()


class BuildKnowledge(Runnable):
    def __init__(self,docs : Optional[list[Document]] = None ,persist_directory_path : Optional[str] = None):
        #docs must of Document Object
        #docs=None OPENS an existing collection instead of building one — that is what lets
        #a session start against an index ingested by an earlier run without loading,
        #chunking or embedding anything (R1). Supplying docs still builds, as before.
        self.docs = docs
        self.path = persist_directory_path
        self.embedding_model = _get_embedding_model()
    def invoke(self,input,config = None):
        name = input["collection_name"]
        if not self.docs and not collection_exists(name, self.path):
            #fail loudly here rather than one Chroma() call later: opening would CREATE an
            #empty collection, and every future collection_exists() would then answer for
            #a collection this failure invented
            raise CollectionNotFoundError(
                f"no vector collection {name!r} with any content under "
                f"persist_directory={self.path!r}, and no documents were given to build one. "
                "Ingest it first — ingestion.ingest.ingest(file_path=..., "
                "persist_directory_path=...) — or pass docs= to build it here."
            )
        #open (or create) the collection instead of Chroma.from_documents, which APPENDS the
        #whole corpus on every construction: after m runs a k-hit search returns ~k/m distinct
        #chunks. Same lazy-open + hash-id fix already proven in memory/long_term_memory.py:35-46.
        #NOTE the kwarg rename: the constructor takes embedding_function=, from_documents took
        #embedding=. Passing the wrong one yields a store that silently embeds nothing.
        vector_store = Chroma(
            collection_name=name,
            embedding_function=self.embedding_model,
            persist_directory=self.path,
        )
        #vector_store.persist() {no need to persist when we are using persist_directory}
        if not self.docs:
            return vector_store
        ids = [_chunk_id(doc) for doc in self.docs]
        try:
            #ask which ids are already indexed so a warm start costs one metadata read
            #instead of re-embedding the corpus. include=[] keeps the payload to ids only.
            already_indexed = set(vector_store.get(ids=ids, include=[])["ids"])
        except Exception:
            #a failed lookup must not block ingestion — upserting is still correct, just slower
            already_indexed = set()
        #dict, not list: also collapses chunks that are duplicates of each other within one
        #batch, which Chroma would otherwise reject as repeated ids in a single upsert
        pending = {}
        for doc, chunk_id in zip(self.docs, ids):
            if chunk_id not in already_indexed:
                pending[chunk_id] = doc
        if pending:
            vector_store.add_documents(list(pending.values()), ids=list(pending.keys()))
        return vector_store
