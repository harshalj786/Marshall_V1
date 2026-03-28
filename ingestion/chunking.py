from langchain_text_splitters import RecursiveCharacterTextSplitter,Language
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import Runnable
from typing import Optional,Literal
from langchain_core.documents import Document
class Chunking(Runnable):
    def __init__(self,chunk_size,chunk_overlap,mode : Literal["recursive","semantic"] = "recursive",separator : Optional[str] = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.mode = mode
        self.separators = [separator] if separator else None
        if self.mode == "semantic":
         self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        elif self.mode == "recursive":
            self.embedding_model = None
    def invoke(self,input,config = None):
        #The input in this invoke function must be a list of documents
        docs = input if isinstance(input, list) else input.get("docs")
        if not isinstance(docs, list):
         raise ValueError("Input must be a list of Document objects")
        if self.mode == "recursive":
            text_splitter = RecursiveCharacterTextSplitter(
              chunk_size = self.chunk_size,
              chunk_overlap = self.chunk_overlap,
              separators=self.separators
    )
            chunked_docs = text_splitter.split_documents(docs)
            print(f"No of chunked docs = {len(chunked_docs)}")
            return chunked_docs
        elif self.mode == "semantic":
            text_splitter = SemanticChunker(
                embeddings = self.embedding_model,
                breakpoint_threshold_amount=90,
                breakpoint_threshold_type="percentile"
            )
            chunked_docs = text_splitter.split_documents(docs)
            print(f"No of chunked_docs = {len(chunked_docs)}")
            return chunked_docs
        else:
            raise ValueError("Mode should be recursive or semantic only")