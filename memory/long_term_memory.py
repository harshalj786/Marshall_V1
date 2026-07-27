import hashlib
import warnings
from langchain_core.runnables import Runnable
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from typing import Optional
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.messages import HumanMessage,AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

_EMBEDDING_MODEL = None

def _get_embedding_model():
    #the model is stateless for inference but costs ~3s to construct, and the memory gate
    #plus the recall tool rebuild UserMemory often — so build it once per process
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        _EMBEDDING_MODEL = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return _EMBEDDING_MODEL

class UserMemory(Runnable):
    def __init__(self,chat_history : list[str] = None,collection_name = "user_history",persist_directory_path : Optional[str] = None):
        self.chat_history = chat_history or []
        self.collection_name = collection_name
        self.persist_directory_path = persist_directory_path
        self.human_messages = [msg for msg in self.chat_history if isinstance(msg,HumanMessage)]
        self.ai_messages = [msg for msg in self.chat_history if isinstance(msg,AIMessage)]
        self.embedding_model = _get_embedding_model()
        if self.persist_directory_path is None:
            #Chroma treats persist_directory=None as an IN-MEMORY store. Long-term memory then
            #works perfectly for the whole session and is gone at exit, with nothing to show it
            #ever happened — the most misleading failure mode in the system, so say it out loud.
            warnings.warn(
                "UserMemory has no persist_directory_path, so its collection "
                f"'{self.collection_name}' is in-memory only: anything remembered is lost when "
                "this process exits. Pass persist_directory_path=... for durable recall.",
                UserWarning,
                stacklevel=2,
            )
        human_docs = [Document(page_content=msg.content) for msg in self.human_messages]
        #open (or create) the collection lazily so an empty chat_history does not crash,
        #and so an existing persisted collection can be reopened across sessions
        self.vector_store_user_data = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_model,
            persist_directory=self.persist_directory_path
        )
        if human_docs:
            #content-hash ids make re-indexing idempotent (add_texts upserts), so
            #rebuilding on a later turn will not duplicate earlier messages
            self.vector_store_user_data.add_documents(
                human_docs,
                ids=[hashlib.sha1(doc.page_content.encode("utf-8")).hexdigest() for doc in human_docs]
            )
    def invoke(self,input,config=None):
        #here the invoke function compress the human from chat_history and then give it to the llm for personal context
        #no input for this function just call this function to get the compressed user_history
        llm = Ollama(model = "mistral",temperature = 0)
        parser = StrOutputParser()
        template = PromptTemplate(
            template=""" You are an ai assistant that would be given an user's chat_history just compress it and keep the personal things and relations about the user \n 
            chat_history :- {human_messages} , compress 
""",input_variables=["human_messages"]
        )
        chain = template | llm | parser
        user_history = chain.invoke({"human_messages": "\n".join([msg.content for msg in self.human_messages])})
        return user_history
    def retreival(self,query:str=None,k : int = 4):
        #here we use simple retreival
        retreiver = self.vector_store_user_data.as_retriever(search_kwargs = {"k":k})
        retreived_docs = retreiver.invoke(query)
        final_retreived_context = "\n\n".join([docs.page_content for docs in retreived_docs])
        return final_retreived_context