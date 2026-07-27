import hashlib
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import Runnable
from langchain_core.documents import Document


def _chunk_id(doc: Document) -> str:
    #content+metadata fingerprint, not content alone: a header repeated on every page of a
    #PDF produces identical text on different pages, and those are distinct chunks
    meta = "|".join(f"{key}={doc.metadata[key]}" for key in sorted(doc.metadata))
    return hashlib.sha1(f"{meta}||{doc.page_content}".encode("utf-8")).hexdigest()


class BuildKnowledge(Runnable):
    def __init__(self,docs : list[Document] ,persist_directory_path : str):
        #docs must of Document Object
        self.docs = docs
        self.path = persist_directory_path
        self.embedding_model =  HuggingFaceEmbeddings(
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
        )
    def invoke(self,input,config = None):
        name = input["collection_name"]
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