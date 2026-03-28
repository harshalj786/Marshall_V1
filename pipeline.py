from ingestion.doc_loader import LoadDocs
from ingestion.chunking import Chunking
from ingestion.vector_db import BuildKnowledge
from retreival.retrieval import Retrival
from langchain_core.runnables import Runnable
from typing import Optional,Literal
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate,MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage,AIMessage
from langchain_community.llms import Ollama
class NRagPipeline(Runnable):
    def __init__(self,
                 file_path,mode_docs : Optional[str] = "pdf" , 
                 glob_pattern : Optional[str] = None,
                 mode_chunks : Optional[Literal["recursive","semantic"]] = "recursive",
                 persist_directory_path : Optional[str] = None,
                 mode_retreiver : Optional[Literal["mmr","simple","mqr","wiki"]] = "simple"):
        self.file_path = file_path
        self.mode_docs = mode_docs
        self.glob_pattern = glob_pattern
        self.mode_chunks = mode_chunks
        self.directory_path = persist_directory_path
        self.mode_retreiver = mode_retreiver
        if not glob_pattern:
            obj_1 = LoadDocs(file_path=self.file_path,mode=self.mode_docs)
            self.docs = obj_1.invoke({})
        else:
            obj_1 = LoadDocs(file_path=self.file_path,mode=self.mode_docs)
            self.docs = obj_1.invoke({"glob_pattern": self.glob_pattern})
        obj_2 = Chunking(chunk_size=500,chunk_overlap=50,mode=self.mode_chunks)# here arguments are hardcoded
        self.chunked_docs = obj_2.invoke({"docs":self.docs})
        self.obj_3 = Retrival(docs=self.chunked_docs,mode=self.mode_retreiver,collection_name=f"VDB for {self.file_path}",path=self.directory_path)
    def invoke(self,input,config=None):
        #putting k here gives the power for how much context we want to send to the llm
        k : Optional[int] = input.get("k",4)
        lambda_mult : Optional[float] = input.get("lambda_mult",0.5)
        user_query : str = input.get("query",None)
        language : Optional[str] = input.get("language","en") 
        retrival_dict = self.obj_3.invoke({"k":k,"query":user_query,"lambda_mult":lambda_mult})
        retreived_context = "\n\n".join([doc.page_content for doc in retrival_dict["context"]])
        template = PromptTemplate(
            template=""" You are an expert AI assistant.
            Your name is Marshall. You are the AI son of Harshal 
Answer the question based strictly on the provided context.

CONTEXT:
{context}

QUERY:
{query}

GUIDELINES:
- Extract only relevant information from the context.
- Do not include irrelevant details.
- If multiple sources exist, combine them logically.
- If the answer is not found, respond with:
  "I could not find the answer in the provided context."
- Cite parts of the context implicitly (refer naturally, no fake citations).

ANSWER:
""",input_variables=["context","query"])
        parser = StrOutputParser()
        llm = Ollama(model = "mistral",temperature = 0)
        chain = template | llm | parser
        response = chain.invoke({"query": user_query,"context":retreived_context})
        return response
    
class ChatRagPipeline(Runnable):
    def __init__(self,
                 file_path,mode_docs : Optional[str] = "pdf" , 
                 glob_pattern : Optional[str] = None,
                 mode_chunks : Optional[Literal["recursive","semantic"]] = "recursive",
                 persist_directory_path : Optional[str] = None,
                 mode_retreiver : Optional[Literal["mmr","simple","mqr","wiki"]] = "simple"):
        self.file_path = file_path
        self.mode_docs = mode_docs
        self.glob_pattern = glob_pattern
        self.mode_chunks = mode_chunks
        self.directory_path = persist_directory_path
        self.mode_retreiver = mode_retreiver
        if not glob_pattern:
            obj_1 = LoadDocs(file_path=self.file_path,mode=self.mode_docs)
            self.docs = obj_1.invoke({})
        else:
            obj_1 = LoadDocs(file_path=self.file_path,mode=self.mode_docs)
            self.docs = obj_1.invoke({"glob_pattern": self.glob_pattern})
        obj_2 = Chunking(chunk_size=500,chunk_overlap=50,mode=self.mode_chunks)# here arguments are hardcoded
        self.chunked_docs = obj_2.invoke({"docs":self.docs})
        self.obj_3 = Retrival(docs=self.chunked_docs,mode=self.mode_retreiver,collection_name=f"VDB for {self.file_path}",path =self.directory_path)
    def invoke(self,input_dict,config=None):
        chat_history = []
        k : Optional[int] = input_dict.get("k",4)
        lambda_mult : Optional[float] = input_dict.get("lambda_mult",0.5)
        language : Optional[str] = input_dict.get("language","en")
        template = ChatPromptTemplate.from_messages([
            ("system","""You are an expert AI assistant.
Your name is Marshall. You are the AI son of Harshal.

GUIDELINES:
- Extract only relevant information from the context.
- Do not include irrelevant details.
- If multiple sources exist, combine them logically.
- If the answer is not found, respond with:
  "I could not find the answer in the provided context."
- Cite parts of the context implicitly (refer naturally, no fake citations
Context:
{context}
"""),MessagesPlaceholder(variable_name="chat_history_placeholder"),
("human","{query}")
        ])
        llm = Ollama(model="mistral",temperature = 0)
        parser = StrOutputParser()
        chain = template | llm | parser
        response = None
        while(True):
          user_query = input("what is your query?")
          retreived_dict = self.obj_3.invoke({"k":k,"lambda_mult":lambda_mult,"query":user_query})
          retreived_context = "\n\n".join([doc.page_content for doc in retreived_dict["context"]])
          print("Ask a question or quit to exit or history for chat history")
          if user_query.lower() == "quit":
                break
          elif user_query.lower() == "history":
              for chat in chat_history:
                  print(f"{chat}")
              continue
          else:
                response = chain.invoke({"context":retreived_context,"query":user_query,"chat_history_placeholder":chat_history})
                chat_history.append(HumanMessage(content=user_query))
                chat_history.append(AIMessage(content=response))
                print(response)
        return response