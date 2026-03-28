from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import Runnable
from langchain_core.documents import Document
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
        vector_store = Chroma.from_documents(
            documents=self.docs,
            embedding=self.embedding_model,
            collection_name=name,
            persist_directory=self.path,
        )
        #vector_store.persist() {no need to persist when we are using persist_directory}
        return vector_store