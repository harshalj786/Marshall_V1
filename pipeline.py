from ingestion.ingest import ensure_ingested
from retreival.retrieval import Retrival
from langchain_core.runnables import Runnable
from typing import Optional,Literal
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate,MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage,AIMessage
from langchain_community.llms import Ollama
from memory.long_term_memory import UserMemory
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
        #ingestion is a separate step now: this loads, chunks and embeds only when the
        #collection is missing, and otherwise opens what an earlier run wrote (R1)
        self.collection_name = ensure_ingested(
            file_path=self.file_path,
            mode_docs=self.mode_docs,
            glob_pattern=self.glob_pattern,
            mode_chunks=self.mode_chunks,
            persist_directory_path=self.directory_path,
        )
        self.obj_3 = Retrival(mode=self.mode_retreiver,collection_name=self.collection_name,path=self.directory_path)
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
        self.collection_name = ensure_ingested(
            file_path=self.file_path,
            mode_docs=self.mode_docs,
            glob_pattern=self.glob_pattern,
            mode_chunks=self.mode_chunks,
            persist_directory_path=self.directory_path,
        )
        self.obj_3 = Retrival(mode=self.mode_retreiver,collection_name=self.collection_name,path =self.directory_path)
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

class ChatRagPipelineWithMemory(Runnable):
    def __init__(self,
                 file_path,mode_docs : Optional[str] = "pdf" , 
                 glob_pattern : Optional[str] = None,
                 mode_chunks : Optional[Literal["recursive","semantic"]] = "recursive",
                 persist_directory_path : Optional[str] = None,
                 mode_retreiver : Optional[Literal["mmr","simple","mqr","wiki"]] = "simple",
                 user_history_directory_path : Optional[str] = None):
        self.file_path = file_path
        self.mode_docs = mode_docs
        self.glob_pattern = glob_pattern
        self.mode_chunks = mode_chunks
        self.directory_path = persist_directory_path
        self.mode_retreiver = mode_retreiver
        self.user_history_directory_path = user_history_directory_path
        self.chat_history = []
        self.collection_name = ensure_ingested(
            file_path=self.file_path,
            mode_docs=self.mode_docs,
            glob_pattern=self.glob_pattern,
            mode_chunks=self.mode_chunks,
            persist_directory_path=self.directory_path,
        )
        self.obj_3 = Retrival(mode=self.mode_retreiver,collection_name=self.collection_name,path =self.directory_path)
    def invoke(self,input_dict,config=None):
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
        template_with_user_history = ChatPromptTemplate.from_messages([
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
{context},
             Also take a look at the user's personal history :- {user_history} and answer accordingly
"""),MessagesPlaceholder(variable_name="chat_history_placeholder"),
("human","{query}")
        ])
        llm = Ollama(model="mistral",temperature = 0)
        parser = StrOutputParser()
        chain = template | llm | parser
        chain_2 = template_with_user_history | llm | parser
        response = None
        count = 0
        while(True):
          user_query = input("Ask a question or quit to exit or history for chat history")
          retreived_dict = self.obj_3.invoke({"k":k,"lambda_mult":lambda_mult,"query":user_query})
          retreived_context = "\n\n".join([doc.page_content for doc in retreived_dict["context"]])
          if user_query.lower() == "quit":
                break
          elif user_query.lower() == "history":
              for chat in self.chat_history:
                  print(f"{chat}")
              continue
          else:
                count = count + 1 
                if count<5 :
                    response = chain.invoke({"query":user_query,"context":retreived_context,"chat_history_placeholder":self.chat_history})
                    self.chat_history.append(HumanMessage(content=user_query))
                    self.chat_history.append(AIMessage(content=response))
                    print(response)
                elif count>=5:
                 if count%5 ==0:
                    obj_4 = UserMemory(
                        chat_history=self.chat_history,
                        persist_directory_path=self.user_history_directory_path
                    )
                    user_history = obj_4.invoke({})
                 response = chain_2.invoke({"query":user_query,"context":retreived_context,"chat_history_placeholder":self.chat_history,"user_history":user_history})
                 self.chat_history.append(HumanMessage(content=user_query))
                 self.chat_history.append(AIMessage(content=response))
                 if len(self.chat_history) >= 10:
                        self.chat_history.clear()
                 print(f"{response}\n")
        return response
    def MsgHistory(self):
        user_messages = [msg for msg in self.chat_history if isinstance(msg,HumanMessage)]
        ai_messages = [msg for msg in self.chat_history if isinstance(msg,AIMessage)]
        print(f"user_messages : {user_messages} \n ai_messages : {ai_messages}")
    def UserSummary(self):
        obj = UserMemory(
            chat_history=self.chat_history,
            persist_directory_path=self.user_history_directory_path
        )
        user_history = obj.invoke({})
        print(user_history)
        return user_history
