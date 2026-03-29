from langchain_community.retrievers import WikipediaRetriever
from ingestion.vector_db import BuildKnowledge
from langchain_core.runnables import Runnable
from langchain_core.documents import Document
from typing import Literal,Optional
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
import re
class Retrival(Runnable):
    def __init__(self,docs:list[Document],mode:Literal["mmr","simple","mqr","wiki"],collection_name:Optional[str] = "Default",path : Optional[str] = None):
        #mqr stands for multi query retrival
        self.docs = docs
        self.mode = mode
        self.path = path
        clean_name = re.sub(r'[^a-zA-Z0-9._-]', '_', collection_name)
        clean_name = clean_name.strip('_.-')
        if len(clean_name) < 3:          
          clean_name = "VDB_collection"
        self.collection_name = clean_name
        self.llm = Ollama(model="mistral",temperature = 0)#this llm is only for removing ambiguity not for reasoning
        class schema(BaseModel):
            multi_query : list[str] = Field(description="Write 5 more queries similar to the user's query to remove the ambiguity of the original query.")
        self.parser = PydanticOutputParser(pydantic_object=schema)
        builder = BuildKnowledge(docs = self.docs,persist_directory_path=self.path)
        self.vector_store = builder.invoke({"collection_name" : self.collection_name})
    def invoke(self,input,config=None):
        k : int = input.get("k",4)
        lambda_mult : Optional[float] = input.get("lambda_mult",0.5)
        query : str = input.get("query",None)
        language : Optional[str] = input.get("language","en")
        template1 = PromptTemplate(
                template="""You MUST return ONLY valid JSON.
Do not explain anything.
Do not add text before or after JSON.
User query:
{query}
Return exactly this format:
{format_instructions}
""",input_variables=["query"],partial_variables={"format_instructions" : self.parser.get_format_instructions()}
            )
        if self.mode == "mmr":
            retreiver = self.vector_store.as_retriever(
                search_type = "mmr",
                search_kwargs = {"k":k,"lambda_mult":lambda_mult}
            )
            context = retreiver.invoke(query)
            #now join the context----------
            return {
                "query" : query,
                "context" : context,
                "mode" : self.mode
            }
        elif self.mode == "simple":
            retreiver = self.vector_store.as_retriever(
                search_kwargs = {"k":k}
            )
            context = retreiver.invoke(query)
            return {
                "query" : query,
                "context" : context,
                "mode" : self.mode
            }
            #now join the context-----------
        elif self.mode == "mqr":
            chain = template1 | self.llm | self.parser
            response = chain.invoke({"query": query})
            #this response is <class '__main__.Schema'>
            queries = response.multi_query
            context = []
            retreiver = self.vector_store.as_retriever(
                search_kwargs = {"k":k}
            )
            for query_s in queries:
                result = retreiver.invoke(query_s)
                context.extend(result)
            return {
                "query" : query,
                "context" : context,
                "mode" : self.mode
            }
        elif self.mode == "wiki":
            retreiver = WikipediaRetriever(top_k_results=k,lang=language)
            docs = retreiver.invoke(query)
            return {
                "query":query,
                "context" : docs,
                "mode": self.mode
            }