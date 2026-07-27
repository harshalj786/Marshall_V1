import re
from typing import TypedDict, Annotated, Optional, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig
from langchain_community.llms import Ollama

from ingestion.doc_loader import LoadDocs
from ingestion.chunking import Chunking
from retreival.retrieval import Retrival
from memory.long_term_memory import UserMemory
from tools import (
    calculator,
    wikipedia_lookup,
    web_search,
    make_search_current_doc,
    make_recall_user_memory,
)


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


#: extra bullet + guidance, spliced in only when the recall tool is actually bound.
#: The prompt must never advertise a tool that is not bound — told it has a tool it
#: lacks, llama3.1 writes the call out as prose JSON instead of emitting a tool_call.
_RECALL_TOOL_ENTRY = """
- `recall_user_memory` for what the user has told you about THEMSELVES in earlier
  conversations (their name, family, job, location, preferences, past projects)"""

_RECALL_GUIDELINES = """
- Use `recall_user_memory` ONLY when the user ASKS a question about themselves that you
  cannot answer from the conversation in front of you ("what is my name?", "where do I
  live?", "do you remember my job?"). Never use it for facts about the world or the
  loaded document."""

AGENT_SYSTEM_PROMPT_TEMPLATE = """You are Marshall, the AI son of Harshal.

You have these tools:
- `calculator` for arithmetic (e.g. "what is 47 * 82?")
- `search_current_doc` for questions about the specific document you have loaded
- `wikipedia_lookup` for general factual / encyclopedic questions
- `web_search` for current events, news, or anything time-sensitive{recall_tool_entry}

Guidelines:
- Prefer `search_current_doc` when the user asks about the loaded document.
- Use `wikipedia_lookup` for stable factual questions Wikipedia would cover.
- Use `web_search` only when the answer depends on recent/current information.
- Use `calculator` only for arithmetic — do not do arithmetic in your head.{recall_guidelines}
- A statement is not a question. When the user TELLS you something they are giving you
  information, not requesting it, so call NO tool and simply reply:
    "my name is Harshal"              -> no tool, reply "Nice to meet you, Harshal!"
    "I live in Pune"                  -> no tool, acknowledge it
    "I work on a RAG agent"           -> no tool, respond with interest
    "hi" / "hello there"              -> no tool, greet them back
- If no tool is needed (greeting, chit-chat, opinion), answer directly.
- After a tool call, answer with the actual content you got back.
- Speak to the user purely as Marshall. Your reply must begin directly with the substance
  of your answer and contain nothing about your own internal process or machinery. Write
  every reply as though you had simply known the answer all along.
"""

#: default (recall-enabled) prompt, kept as a module constant for callers/tests
AGENT_SYSTEM_PROMPT = AGENT_SYSTEM_PROMPT_TEMPLATE.format(
    recall_tool_entry=_RECALL_TOOL_ENTRY,
    recall_guidelines=_RECALL_GUIDELINES,
)

PROFILE_BLOCK_TEMPLATE = """

What you already know about this user, compressed from earlier conversations:
{user_history}

Use this to personalise your answers. Do not recite it back unless asked.
"""


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    turn_count: int
    user_history_summary: str


class AgentGraph:
    """ReAct-style agent over a document with a fixed toolset and a long-term memory gate.

    Uses ChatOllama (which supports .bind_tools) — leaves the older Ollama class
    in place for the two chat graphs. Loop:
        START -> count_turn -> {summarize_user_history -> agent, agent}
        agent -> tools_condition -> {tools -> agent, END}

    Memory works in both directions:
      * passive — every `memory_every_n_turns` turns the gate compresses the user's
        human-message history via UserMemory and folds it into the system prompt
        (set memory_every_n_turns=0 to disable).
      * active — the `recall_user_memory` tool lets the agent search the user's own
        past messages on demand, including from a thread that never saw them, as long
        as `user_history_directory_path` is set.
    """

    def __init__(self,
                 file_path,
                 mode_docs: Optional[str] = "pdf",
                 glob_pattern: Optional[str] = None,
                 mode_chunks: Optional[Literal["recursive", "semantic"]] = "recursive",
                 persist_directory_path: Optional[str] = None,
                 mode_retreiver: Optional[Literal["mmr", "simple", "mqr", "wiki"]] = "simple",
                 model: str = "llama3.1",
                 extra_tools: Optional[list] = None,
                 user_history_directory_path: Optional[str] = None,
                 memory_every_n_turns: int = 5,
                 enable_recall_tool: bool = True,
                 user_id: Optional[str] = None):
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

        self.user_history_directory_path = user_history_directory_path
        self.memory_every_n_turns = memory_every_n_turns
        # Long-term memory is scoped to the USER, not the thread: a thread_id identifies one
        # conversation, so per-thread collections would make it no better than short-term
        # state. Default None keeps UserMemory's own "user_history" name, so this graph shares
        # a profile with ChatRagWithMemoryGraph; pass user_id to separate several users.
        raw_name = "user_history" if user_id is None else f"user_history_{user_id}"
        # UserMemory does not sanitize collection_name — mirror retreival/retrieval.py:17-21
        clean_name = re.sub(r'[^a-zA-Z0-9._-]', '_', raw_name).strip('_.-')
        self.user_memory_collection = clean_name if len(clean_name) >= 3 else "user_history"
        # thread_id -> (human_message_count, UserMemory): tracks how many of THAT thread's
        # messages are already indexed into the shared collection. Content-hash ids make the
        # overlapping writes idempotent, so reads still span everything the user ever said.
        self._memory_cache: dict[str, tuple[int, UserMemory]] = {}

        search_current_doc = make_search_current_doc(self.retrival)
        self.tools = [calculator, wikipedia_lookup, web_search, search_current_doc]
        if enable_recall_tool:
            self.tools.append(make_recall_user_memory(self._build_user_memory))
        if extra_tools:
            self.tools.extend(extra_tools)

        # the prompt must describe exactly the tools that are bound, or the model narrates
        # calls to phantom tools in prose instead of emitting real tool_calls
        self.system_prompt = AGENT_SYSTEM_PROMPT_TEMPLATE.format(
            recall_tool_entry=_RECALL_TOOL_ENTRY if enable_recall_tool else "",
            recall_guidelines=_RECALL_GUIDELINES if enable_recall_tool else "",
        )

        # tool list must be final before bind_tools; ToolNode(self.tools) then picks it up
        self.llm = ChatOllama(model=model, temperature=0).bind_tools(self.tools)
        self.checkpointer = MemorySaver()
        self.graph = self._build_graph()

    def _build_user_memory(self, messages: list[BaseMessage],
                           thread_id: str = "default") -> Optional[UserMemory]:
        """Build (or reuse) the UserMemory for one thread. Returns None if nothing to remember.

        Single funnel used by both the memory gate and the `recall_user_memory` tool, so the
        two paths never disagree. Indexes this thread's human messages into the user-scoped
        collection, so a later conversation can recall what was said in an earlier one.
        """
        humans = [m for m in messages if isinstance(m, HumanMessage)]
        # fingerprint, not just a count: two threads at the same message count must never
        # share an entry, or one thread's profile gets compressed from another's history
        fingerprint = (len(humans), humans[-1].content if humans else "")
        cached = self._memory_cache.get(thread_id)
        if cached and cached[0] == fingerprint:
            return cached[1]
        if not humans:
            # nothing of this thread's own to index; fall back to reading the shared store,
            # which may still hold messages from earlier conversations
            if self.user_history_directory_path is None:
                return None
            memory = UserMemory(
                chat_history=[],
                collection_name=self.user_memory_collection,
                persist_directory_path=self.user_history_directory_path,
            )
            self._memory_cache[thread_id] = (fingerprint, memory)
            return memory
        memory = UserMemory(
            chat_history=humans,
            collection_name=self.user_memory_collection,
            persist_directory_path=self.user_history_directory_path,
        )
        self._memory_cache[thread_id] = (fingerprint, memory)
        return memory

    def _count_turn_node(self, state: AgentState) -> dict:
        # Sits between START and `agent` so it runs exactly once per invoke(). Counting inside
        # `agent` would inflate, since the tool loop re-enters `agent` after every tool call.
        return {"turn_count": state.get("turn_count", 0) + 1}

    def _memory_gate(self, state: AgentState) -> str:
        n = self.memory_every_n_turns
        if not n or n <= 0:          # guard before the modulo: n=0 disables, n<0 is nonsense
            return "agent"
        turn = state.get("turn_count", 0)
        # `turn >= n` matches ChatRagWithMemoryGraph:212-216 and stops turn 0 from firing
        return "summarize" if turn >= n and turn % n == 0 else "agent"

    def _summarize_user_history_node(self, state: AgentState, config: RunnableConfig = None) -> dict:
        # must be annotated RunnableConfig — with Optional[dict] langgraph silently injects
        # nothing, thread_id collapses to "default", and threads share a cache entry
        thread_id = ((config or {}).get("configurable") or {}).get("thread_id", "default")
        memory = self._build_user_memory(state["messages"], thread_id)
        if memory is None:
            return {}
        try:
            return {"user_history_summary": memory.invoke({})}
        except Exception:
            # a memory failure must never destroy the user's turn; keep the previous summary
            return {}

    def _with_profile(self, messages: list[BaseMessage], summary: str) -> list[BaseMessage]:
        """Return a NEW message list with the user profile folded into the system prompt.

        Builds a fresh SystemMessage rather than copying the stored one: a copy would keep
        its .id, and if it ever reached a {"messages": [...]} return, add_messages would
        overwrite the stored prompt and the profile would compound on every gate hit.
        """
        block = PROFILE_BLOCK_TEMPLATE.format(user_history=summary)
        if messages and isinstance(messages[0], SystemMessage):
            return [SystemMessage(content=messages[0].content + block)] + list(messages[1:])
        return [SystemMessage(content=self.system_prompt + block)] + list(messages)

    def _agent_node(self, state: AgentState) -> dict:
        messages = state["messages"]
        summary = state.get("user_history_summary", "")
        if summary:
            # composed per call, never written back — so it also applies on post-tool re-entry
            messages = self._with_profile(messages, summary)
        response = self.llm.invoke(messages)
        return {"messages": [response]}

    def _build_graph(self):
        g = StateGraph(AgentState)
        g.add_node("count_turn", self._count_turn_node)
        g.add_node("summarize_user_history", self._summarize_user_history_node)
        g.add_node("agent", self._agent_node)
        g.add_node("tools", ToolNode(self.tools))
        g.add_edge(START, "count_turn")
        g.add_conditional_edges(
            "count_turn",
            self._memory_gate,
            {"summarize": "summarize_user_history", "agent": "agent"},
        )
        g.add_edge("summarize_user_history", "agent")
        # explicit path map: without it every node is declared a possible target, which
        # discards graph-shape validation and draws phantom edges in get_graph()
        g.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
        g.add_edge("tools", "agent")
        return g.compile(checkpointer=self.checkpointer)

    def invoke(self, query: str, config: Optional[dict] = None) -> str:
        if config is None:
            config = {"configurable": {"thread_id": "default"}}
        # cap the ReAct loop: langgraph's default is 10000, so an oscillating tool loop
        # would otherwise grind through ~10000 Ollama calls before erroring
        config = {**config, "recursion_limit": config.get("recursion_limit", 25)}
        existing = self.get_history(config["configurable"]["thread_id"])
        incoming: list[BaseMessage] = []
        if not existing:
            incoming.append(SystemMessage(content=self.system_prompt))
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

    def get_turn_count(self, thread_id: str = "default") -> int:
        snapshot = self.graph.get_state({"configurable": {"thread_id": thread_id}})
        return snapshot.values.get("turn_count", 0) if snapshot else 0

    def get_user_history_summary(self, thread_id: str = "default") -> str:
        snapshot = self.graph.get_state({"configurable": {"thread_id": thread_id}})
        return snapshot.values.get("user_history_summary", "") if snapshot else ""

    def user_summary(self, thread_id: str = "default") -> str:
        """On-demand user-history compression, mirroring ChatRagWithMemoryGraph.user_summary."""
        memory = self._build_user_memory(self.get_history(thread_id), thread_id)
        if memory is None:
            return ""
        summary = memory.invoke({})
        print(summary)
        return summary
