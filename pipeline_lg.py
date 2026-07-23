from typing import TypedDict, Annotated, Optional, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama

from ingestion.doc_loader import LoadDocs
from ingestion.chunking import Chunking
from retreival.retrieval import Retrival
from memory.long_term_memory import UserMemory
from tools import calculator, wikipedia_lookup, web_search, make_search_current_doc


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    query: str
    context: str
    user_history_summary: str
    turn_count: int
    k: int
    lambda_mult: float


BASE_SYSTEM_TEMPLATE = """You are an expert AI assistant.
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
"""

MEMORY_SYSTEM_TEMPLATE = """You are an expert AI assistant.
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
"""


class ChatRagGraph:
    """LangGraph version of ChatRagPipeline.

    Single-turn per .invoke() — session state is held by MemorySaver keyed on thread_id.
    """

    def __init__(self,
                 file_path,
                 mode_docs: Optional[str] = "pdf",
                 glob_pattern: Optional[str] = None,
                 mode_chunks: Optional[Literal["recursive", "semantic"]] = "recursive",
                 persist_directory_path: Optional[str] = None,
                 mode_retreiver: Optional[Literal["mmr", "simple", "mqr", "wiki"]] = "simple"):
        loader = LoadDocs(file_path=file_path, mode=mode_docs)
        docs = loader.invoke({"glob_pattern": glob_pattern} if glob_pattern else {})
        chunker = Chunking(chunk_size=500, chunk_overlap=50, mode=mode_chunks)
        chunked = chunker.invoke({"docs": docs})
        self.retrival = Retrival(
            docs=chunked,
            mode=mode_retreiver,
            collection_name=f"VDB for {file_path}",
            path=persist_directory_path,
        )
        self.llm = Ollama(model="mistral", temperature=0)
        self.parser = StrOutputParser()
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", BASE_SYSTEM_TEMPLATE),
            MessagesPlaceholder(variable_name="messages"),
            ("human", "{query}"),
        ])
        self.checkpointer = MemorySaver()
        self.graph = self._build_graph()

    def _retrieve_node(self, state: ChatState) -> dict:
        result = self.retrival.invoke({
            "k": state.get("k", 4),
            "lambda_mult": state.get("lambda_mult", 0.5),
            "query": state["query"],
        })
        context = "\n\n".join(doc.page_content for doc in result["context"])
        return {"context": context}

    def _generate_node(self, state: ChatState) -> dict:
        chain = self.prompt | self.llm | self.parser
        response = chain.invoke({
            "context": state["context"],
            "query": state["query"],
            "messages": state["messages"],
        })
        return {
            "messages": [HumanMessage(content=state["query"]), AIMessage(content=response)],
        }

    def _build_graph(self):
        g = StateGraph(ChatState)
        g.add_node("retrieve", self._retrieve_node)
        g.add_node("generate", self._generate_node)
        g.add_edge(START, "retrieve")
        g.add_edge("retrieve", "generate")
        g.add_edge("generate", END)
        return g.compile(checkpointer=self.checkpointer)

    def invoke(self, input_dict: dict, config: Optional[dict] = None) -> str:
        if config is None:
            config = {"configurable": {"thread_id": "default"}}
        state_in = {
            "query": input_dict["query"],
            "k": input_dict.get("k", 4),
            "lambda_mult": input_dict.get("lambda_mult", 0.5),
        }
        final = self.graph.invoke(state_in, config=config)
        last_ai = next(
            (m for m in reversed(final["messages"]) if isinstance(m, AIMessage)),
            None,
        )
        return last_ai.content if last_ai else ""

    def run_chat(self, thread_id: str = "default", k: int = 4, lambda_mult: float = 0.5) -> None:
        """Backwards-compat REPL matching pipeline.py:116-131."""
        config = {"configurable": {"thread_id": thread_id}}
        print("Ask a question or quit to exit or history for chat history")
        while True:
            user_query = input("what is your query?")
            if user_query.lower() == "quit":
                break
            if user_query.lower() == "history":
                for m in self.get_history(thread_id):
                    print(m)
                continue
            response = self.invoke(
                {"query": user_query, "k": k, "lambda_mult": lambda_mult},
                config=config,
            )
            print(response)

    def get_history(self, thread_id: str = "default") -> list[BaseMessage]:
        snapshot = self.graph.get_state({"configurable": {"thread_id": thread_id}})
        return snapshot.values.get("messages", []) if snapshot else []


class ChatRagWithMemoryGraph:
    """LangGraph version of ChatRagPipelineWithMemory.

    Adds a conditional memory-gate: every 5 turns from turn 5 onward, the user's
    human-message history is compressed via UserMemory and injected into the prompt.
    Message list is reset once it grows past 10 (matches pipeline.py:229-230).
    """

    def __init__(self,
                 file_path,
                 mode_docs: Optional[str] = "pdf",
                 glob_pattern: Optional[str] = None,
                 mode_chunks: Optional[Literal["recursive", "semantic"]] = "recursive",
                 persist_directory_path: Optional[str] = None,
                 mode_retreiver: Optional[Literal["mmr", "simple", "mqr", "wiki"]] = "simple",
                 user_history_directory_path: Optional[str] = None):
        loader = LoadDocs(file_path=file_path, mode=mode_docs)
        docs = loader.invoke({"glob_pattern": glob_pattern} if glob_pattern else {})
        chunker = Chunking(chunk_size=500, chunk_overlap=50, mode=mode_chunks)
        chunked = chunker.invoke({"docs": docs})
        self.retrival = Retrival(
            docs=chunked,
            mode=mode_retreiver,
            collection_name=f"VDB for {file_path}",
            path=persist_directory_path,
        )
        self.user_history_directory_path = user_history_directory_path
        self.llm = Ollama(model="mistral", temperature=0)
        self.parser = StrOutputParser()
        self.base_prompt = ChatPromptTemplate.from_messages([
            ("system", BASE_SYSTEM_TEMPLATE),
            MessagesPlaceholder(variable_name="messages"),
            ("human", "{query}"),
        ])
        self.memory_prompt = ChatPromptTemplate.from_messages([
            ("system", MEMORY_SYSTEM_TEMPLATE),
            MessagesPlaceholder(variable_name="messages"),
            ("human", "{query}"),
        ])
        self.checkpointer = MemorySaver()
        self.graph = self._build_graph()

    def _retrieve_node(self, state: ChatState) -> dict:
        result = self.retrival.invoke({
            "k": state.get("k", 4),
            "lambda_mult": state.get("lambda_mult", 0.5),
            "query": state["query"],
        })
        context = "\n\n".join(doc.page_content for doc in result["context"])
        return {
            "context": context,
            "turn_count": state.get("turn_count", 0) + 1,
        }

    def _memory_gate(self, state: ChatState) -> str:
        turn = state["turn_count"]
        if turn >= 5 and turn % 5 == 0:
            return "summarize"
        return "generate"

    def _summarize_user_history_node(self, state: ChatState) -> dict:
        memory = UserMemory(
            chat_history=state["messages"],
            persist_directory_path=self.user_history_directory_path,
        )
        summary = memory.invoke({})
        return {"user_history_summary": summary}

    def _generate_node(self, state: ChatState) -> dict:
        summary = state.get("user_history_summary", "")
        if summary:
            chain = self.memory_prompt | self.llm | self.parser
            response = chain.invoke({
                "context": state["context"],
                "query": state["query"],
                "messages": state["messages"],
                "user_history": summary,
            })
        else:
            chain = self.base_prompt | self.llm | self.parser
            response = chain.invoke({
                "context": state["context"],
                "query": state["query"],
                "messages": state["messages"],
            })
        return {
            "messages": [HumanMessage(content=state["query"]), AIMessage(content=response)],
        }

    def _trim_history_node(self, state: ChatState) -> dict:
        # Preserve pipeline.py:229-230 behavior: clear when >=10.
        if len(state["messages"]) >= 10:
            from langchain_core.messages import RemoveMessage
            return {"messages": [RemoveMessage(id=m.id) for m in state["messages"] if m.id]}
        return {}

    def _build_graph(self):
        g = StateGraph(ChatState)
        g.add_node("retrieve", self._retrieve_node)
        g.add_node("summarize_user_history", self._summarize_user_history_node)
        g.add_node("generate", self._generate_node)
        g.add_node("trim_history", self._trim_history_node)
        g.add_edge(START, "retrieve")
        g.add_conditional_edges(
            "retrieve",
            self._memory_gate,
            {"summarize": "summarize_user_history", "generate": "generate"},
        )
        g.add_edge("summarize_user_history", "generate")
        g.add_edge("generate", "trim_history")
        g.add_edge("trim_history", END)
        return g.compile(checkpointer=self.checkpointer)

    def invoke(self, input_dict: dict, config: Optional[dict] = None) -> str:
        if config is None:
            config = {"configurable": {"thread_id": "default"}}
        state_in = {
            "query": input_dict["query"],
            "k": input_dict.get("k", 4),
            "lambda_mult": input_dict.get("lambda_mult", 0.5),
        }
        final = self.graph.invoke(state_in, config=config)
        last_ai = next(
            (m for m in reversed(final["messages"]) if isinstance(m, AIMessage)),
            None,
        )
        return last_ai.content if last_ai else ""

    def run_chat(self, thread_id: str = "default", k: int = 4, lambda_mult: float = 0.5) -> None:
        """Backwards-compat REPL matching pipeline.py:202-231."""
        config = {"configurable": {"thread_id": thread_id}}
        while True:
            user_query = input("Ask a question or quit to exit or history for chat history")
            if user_query.lower() == "quit":
                break
            if user_query.lower() == "history":
                for m in self.get_history(thread_id):
                    print(m)
                continue
            response = self.invoke(
                {"query": user_query, "k": k, "lambda_mult": lambda_mult},
                config=config,
            )
            print(f"{response}\n")

    def get_history(self, thread_id: str = "default") -> list[BaseMessage]:
        snapshot = self.graph.get_state({"configurable": {"thread_id": thread_id}})
        return snapshot.values.get("messages", []) if snapshot else []

    def user_summary(self, thread_id: str = "default") -> str:
        """On-demand user-history compression, replacing pipeline.py:237-244."""
        messages = self.get_history(thread_id)
        memory = UserMemory(
            chat_history=messages,
            persist_directory_path=self.user_history_directory_path,
        )
        summary = memory.invoke({})
        print(summary)
        return summary


AGENT_SYSTEM_PROMPT = """You are Marshall, the AI son of Harshal.

You have four tools:
- `calculator` for arithmetic (e.g. "what is 47 * 82?")
- `search_current_doc` for questions about the specific document you have loaded
- `wikipedia_lookup` for general factual / encyclopedic questions
- `web_search` for current events, news, or anything time-sensitive

Guidelines:
- Prefer `search_current_doc` when the user asks about the loaded document.
- Use `wikipedia_lookup` for stable factual questions Wikipedia would cover.
- Use `web_search` only when the answer depends on recent/current information.
- Use `calculator` only for arithmetic — do not do arithmetic in your head.
- If no tool is needed (greeting, chit-chat, opinion), answer directly.
- After a tool call returns, use its result to write the final answer.
"""


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


class AgentGraph:
    """ReAct-style agent over a document with a fixed toolset.

    Uses ChatOllama (which supports .bind_tools) — leaves the older Ollama class
    in place for the two chat graphs. Loop:
        START -> agent -> tools_condition -> {tools -> agent, END}
    """

    def __init__(self,
                 file_path,
                 mode_docs: Optional[str] = "pdf",
                 glob_pattern: Optional[str] = None,
                 mode_chunks: Optional[Literal["recursive", "semantic"]] = "recursive",
                 persist_directory_path: Optional[str] = None,
                 mode_retreiver: Optional[Literal["mmr", "simple", "mqr", "wiki"]] = "simple",
                 model: str = "llama3.1",
                 extra_tools: Optional[list] = None):
        from langchain_ollama import ChatOllama

        loader = LoadDocs(file_path=file_path, mode=mode_docs)
        docs = loader.invoke({"glob_pattern": glob_pattern} if glob_pattern else {})
        chunker = Chunking(chunk_size=500, chunk_overlap=50, mode=mode_chunks)
        chunked = chunker.invoke({"docs": docs})
        self.retrival = Retrival(
            docs=chunked,
            mode=mode_retreiver,
            collection_name=f"VDB for {file_path}",
            path=persist_directory_path,
        )

        search_current_doc = make_search_current_doc(self.retrival)
        self.tools = [calculator, wikipedia_lookup, web_search, search_current_doc]
        if extra_tools:
            self.tools.extend(extra_tools)

        self.llm = ChatOllama(model=model, temperature=0).bind_tools(self.tools)
        self.checkpointer = MemorySaver()
        self.graph = self._build_graph()

    def _agent_node(self, state: AgentState) -> dict:
        response = self.llm.invoke(state["messages"])
        return {"messages": [response]}

    def _build_graph(self):
        g = StateGraph(AgentState)
        g.add_node("agent", self._agent_node)
        g.add_node("tools", ToolNode(self.tools))
        g.add_edge(START, "agent")
        g.add_conditional_edges("agent", tools_condition)
        g.add_edge("tools", "agent")
        return g.compile(checkpointer=self.checkpointer)

    def invoke(self, query: str, config: Optional[dict] = None) -> str:
        if config is None:
            config = {"configurable": {"thread_id": "default"}}
        existing = self.get_history(config["configurable"]["thread_id"])
        incoming: list[BaseMessage] = []
        if not existing:
            incoming.append(SystemMessage(content=AGENT_SYSTEM_PROMPT))
        incoming.append(HumanMessage(content=query))
        final = self.graph.invoke({"messages": incoming}, config=config)
        last_ai = next(
            (m for m in reversed(final["messages"])
             if isinstance(m, AIMessage) and not getattr(m, "tool_calls", None)),
            None,
        )
        return last_ai.content if last_ai else ""

    def run_chat(self, thread_id: str = "default") -> None:
        config = {"configurable": {"thread_id": thread_id}}
        print("Ask a question or quit to exit or history for chat history")
        while True:
            user_query = input("what is your query?")
            if user_query.lower() == "quit":
                break
            if user_query.lower() == "history":
                for m in self.get_history(thread_id):
                    print(m)
                continue
            response = self.invoke(user_query, config=config)
            print(response)

    def get_history(self, thread_id: str = "default") -> list[BaseMessage]:
        snapshot = self.graph.get_state({"configurable": {"thread_id": thread_id}})
        return snapshot.values.get("messages", []) if snapshot else []
