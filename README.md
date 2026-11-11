# Marshall - Advanced RAG Pipeline Framework

**Marshall** is a modular Retrieval-Augmented Generation (RAG) framework built on **LangChain + LangGraph**. It loads documents from many formats, chunks them, stores embeddings in a vector database, and answers questions through either classic RAG pipelines, LangGraph state-machine graphs, or a **ReAct-style agent** with five tools (calculator, doc-search, Wikipedia, web search, and long-term user-memory recall). You can **type** to the agent (`chat_main.py`) or **talk** to it (`voice_main.py`), and in a password-unlocked session it can **check and run Python, C and C++ code** — each run confirmed by you, outside the model.

> *"I am Marshall, the AI son of Harshal."*

---

## Features

- **Multi-format Document Loading**: PDF, scanned PDFs, text files, CSV, web URLs, YouTube transcripts, and folder batch loading
- **Flexible Chunking**: Recursive character splitting or semantic chunking
- **Vector Storage**: ChromaDB with HuggingFace embeddings (`all-MiniLM-L6-v2`)
- **Multiple Retrieval Modes**:
  - `simple` - Basic similarity search
  - `mmr` - Maximal Marginal Relevance (diversity-aware)
  - `mqr` - Multi-Query Retrieval (disambiguation)
  - `wiki` - Wikipedia retrieval
- **LangChain + LangGraph Orchestration**: legacy Runnable-based pipelines alongside newer LangGraph state-machine graphs
- **Pipeline / Graph Modes**:
  - `NRagPipeline` - Single query-response (LangChain)
  - `ChatRagPipeline` - Interactive chat with session history (LangChain)
  - `ChatRagPipelineWithMemory` - Interactive chat with persistent user memory (LangChain)
  - `ChatRagGraph` - LangGraph version of `ChatRagPipeline` with `MemorySaver` checkpointing
  - `ChatRagWithMemoryGraph` - LangGraph version with a 5-turn memory-gate
  - `AgentGraph` - **ReAct-style agent** with tool-calling *and* a long-term memory gate
- **Tool-Calling Layer** (`tools/`): `calculator`, `wikipedia_lookup`, `web_search`, `search_current_doc`, `recall_user_memory`
- **Code Tools** (`tools/code_tools.py`): `check_code` compiles or syntax-checks Python, C and
  C++ without running it; `run_code` runs it in a temp directory with a time limit — admin
  sessions only, and only after you confirm with a terminal `[y/N]` or a spoken "yes"
- **Admin and Guest Sessions** (`admin_session.py`): an Argon2id-hashed admin password asked
  at startup. Guests can have pasted code checked but never run, and leave no memory behind;
  the password never reaches the model
- **Voice** (`voice/`, `voice_main.py`): open mic → silero VAD → faster-whisper → the agent →
  Kokoro TTS, with barge-in (talk over Marshall and he stops) — headphones recommended
- **Dual-Layer Memory on the Agent**: short-term LangGraph state alongside a persistent
  ChromaDB user profile — injected automatically every *N* turns, and searchable on demand
  by the agent itself. Recall works across conversations and across process restarts.
- **Conversation Summarization**: long conversations are compressed into a running summary
  so the context stays bounded

---

## Project Structure

```
Marshall_V1$/
├── chat_main.py          # Type to Marshall: admin password prompt, then the agent's chat loop
├── voice_main.py         # Talk to Marshall: password prompt, model boot, the voice loop, --check
├── admin_session.py      # Admin/guest sessions: Argon2id password, guest counter, [y/N] confirmer
├── pipeline.py           # Legacy LangChain pipelines (NRagPipeline, ChatRagPipeline, ChatRagPipelineWithMemory)
├── pipeline_lg.py        # LangGraph graphs (ChatRagGraph, ChatRagWithMemoryGraph, AgentGraph)
├── warmup.py             # Loads the embedding models ahead of the first turn
├── ingestion/
│   ├── doc_loader.py     # Document loading (PDF, scanned PDF, web, YouTube, CSV, TXT, folder)
│   ├── chunking.py       # Text chunking strategies (recursive, semantic)
│   ├── vector_db.py      # ChromaDB vector store builder
│   ├── ingest.py         # Ingest once, then reopen the existing index
│   └── __init__.py
├── retreival/
│   ├── retrieval.py      # Retrieval strategies (simple, MMR, MQR, wiki)
│   └── __init__.py
├── memory/
│   ├── long_term_memory.py  # UserMemory — persistent user profile (compress + semantic recall)
│   ├── reconciliation.py    # Tool-safe cut points for trimming the message list
│   └── __init__.py
├── tools/
│   ├── tools.py          # Agent tools (calculator, wikipedia_lookup, web_search,
│   │                     #   search_current_doc, recall_user_memory)
│   ├── code_tools.py     # check_code and run_code
│   ├── code_guards.py    # What the code tools may read: languages, paths, includes
│   └── __init__.py
├── voice/
│   ├── orchestrator.py   # LISTENING → THINKING → (CONFIRMING) → SPEAKING loop, barge-in
│   ├── audio_io.py       # Microphone and speaker
│   ├── vad.py            # silero VAD: utterance endpointing and the barge-in guard
│   ├── stt.py            # faster-whisper transcription
│   ├── tts.py            # Kokoro speech synthesis
│   ├── confirm.py        # The spoken yes/no for run_code, matched in code
│   ├── config.py         # VoiceConfig
│   ├── telemetry.py      # Per-turn latency rows (voice_logs/turns.jsonl)
│   └── __init__.py
├── scripts/
│   ├── set_admin_password.py      # Set the admin password (once)
│   ├── check_routing.py           # Does llama3.1 still pick the right tool? (R4's table)
│   └── check_code_tool_routing.py # Do the code tools make ordinary chat worse? (A/B)
├── tests/                # pytest suite; `pytest -m "not needs_ollama"` is what CI runs
├── docs/                 # Architecture review, security policy, designs and plans
├── db/                   # Vector stores, memory, summaries, admin password hash (gitignored)
├── .env                  # API keys (gitignored)
├── .gitignore           # Git ignore configuration
└── README.md            # Project documentation
```

---

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd Marshall_V1$

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies (exact pins; see the header of requirements.txt)
pip install -r requirements.txt -c constraints.txt
```

`constraints.txt` is a **guard, not a bootstrap spec.** It holds `torch==2.6.0`
without the `+cu124` local version identifier, so that PEP 440 treats an already
installed `2.6.0+cu124` as satisfying it — pip leaves the GPU build alone, while any
new dependency that wants a different torch fails loudly instead of quietly pulling
the CPU wheel over it.

On a machine with **no torch installed yet**, that constraint protects nothing: pip
will happily satisfy `==2.6.0` with the CPU wheel from PyPI. Build the GPU
environment first — see NOTE 1 at the bottom of `requirements.txt` for why torch is
deliberately not pinned there and how to install it:

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

Running the test suite additionally needs `pytest`, which is not a runtime dependency
and so is not in `requirements.txt`:

```bash
pip install pytest
pytest                      # the whole suite
pytest -m "not needs_ollama"  # only what runs without a local model server
```

---

## Configuration

Create a `.env` file with your API keys:

```env
OPENAI_API_KEY="your-openai-key"
GEMINI_API_KEY="your-gemini-key"
```

---

## Usage

### Quick start: talk to Marshall

```bash
python chat_main.py                  # type; asks for the admin password first (Enter = guest)
python voice_main.py                 # talk; same password prompt, headphones recommended
```

See [Sessions, the admin password and running code](#sessions-the-admin-password-and-running-code).

### Single Query Mode

```python
from pipeline import NRagPipeline

# Initialize with a PDF
rag = NRagPipeline(
    file_path="path/to/document.pdf",
    mode_docs="pdf",
    mode_chunks="recursive",
    mode_retreiver="mmr"
)

# Query
response = rag.invoke({
    "query": "What is the main topic?",
    "k": 4,
    "lambda_mult": 0.5
})
print(response)
```

### Interactive Chat Mode

```python
from pipeline import ChatRagPipeline

# Initialize
chat_rag = ChatRagPipeline(
    file_path="path/to/document.pdf",
    mode_docs="pdf",
    mode_retreiver="simple"
)

# Start interactive session
chat_rag.invoke({})
# Type 'quit' to exit, 'history' to view chat history
```

### Chat with Persistent Memory

```python
from pipeline import ChatRagPipelineWithMemory

# Initialize with memory persistence
memory_chat = ChatRagPipelineWithMemory(
    file_path="path/to/document.pdf",
    mode_docs="pdf",
    mode_retreiver="simple",
    user_history_directory_path="path/to/memory/storage"
)

# Chat with personalized responses based on user history
memory_chat.invoke({})
```

### LangGraph Chat Mode

```python
from pipeline_lg import ChatRagGraph

graph = ChatRagGraph(
    file_path="path/to/document.pdf",
    mode_docs="pdf",
    mode_retreiver="simple",
)

# thread_id scopes conversation state via MemorySaver
cfg = {"configurable": {"thread_id": "session-1"}}
answer = graph.invoke({"query": "What is this document about?"}, config=cfg)
print(answer)
```

### Agent Mode with Tool-Calling

`AgentGraph` gives Marshall five tools and lets it decide when to use them.

```python
from pipeline_lg import AgentGraph

agent = AgentGraph(
    file_path="path/to/document.pdf",
    mode_docs="pdf",
    mode_retreiver="simple",
    model="llama3.1",     # llama3.1 or qwen2.5 — mistral does not reliably emit tool_calls
)

cfg = {"configurable": {"thread_id": "agent-1"}}

# Arithmetic → routed to the `calculator` tool
print(agent.invoke("what is 47 times 82?", config=cfg))

# About the loaded document → routed to `search_current_doc`
print(agent.invoke("what does the document say about OPVC pipes?", config=cfg))

# Encyclopedic → routed to `wikipedia_lookup`
print(agent.invoke("who invented the transistor?", config=cfg))

# Current events → routed to `web_search`
print(agent.invoke("what are the top news stories today?", config=cfg))
```

**Available tools** (`tools/tools.py`):

| Tool | Purpose |
|------|---------|
| `calculator` | Safe AST-based arithmetic (`+ - * / ** %`) — no `eval`, no builtins |
| `search_current_doc` | Semantic search over the document loaded into `AgentGraph` |
| `wikipedia_lookup` | Top-k Wikipedia articles with retry-on-empty |
| `web_search` | DuckDuckGo web search for time-sensitive queries |
| `recall_user_memory` | Semantic search over what the *user* said in earlier conversations |
| `check_code` | Compile or syntax-check Python, C or C++ **without running it** — bound only with a `session` (below) |
| `run_code` | Run Python, C or C++ after the user confirms **outside the model** — admin sessions only |

### Sessions, the admin password and running code

Set the admin password once (stored as an Argon2id hash in `db/admin/`, never in git):

```bash
python scripts/set_admin_password.py
```

Both entry points ask for it before anything loads — `python chat_main.py` for typing,
`python voice_main.py` for talking. Press Enter instead for a guest session.

| | Guest session | Admin session |
|---|---|---|
| `check_code` | pasted code only; file paths refused; C/C++ may only pull in system headers (no `#embed`, `#line`, quoted or absolute includes) | pasted code or a path to a `.py` `.c` `.h` `.cpp` `.cc` `.cxx` `.hpp` file; `.env*` and the password hash are refused in every session |
| `run_code` | not available | yes — all the code is shown first, then `[y/N]` in the terminal or a spoken "yes" matched in code |
| Memory | nothing kept: no recall, no summary, no long-term store | as before |
| Guest count | recorded (a timestamp per session, nothing said) | Marshall can tell you on request |

The password never reaches the model: it is checked in `admin_session.py` before the agent
is built, and all the agent receives is which session it is. Runs happen in a temporary
directory with no stdin, a 10 s limit and the whole process tree killed on timeout.
`AgentGraph(session=None)` — the default — is exactly the five-tool agent described above.
Design: `docs/superpowers/specs/2026-11-10-code-check-and-run-design.md`;
policy: `docs/security.md` §4.

**Extension hook.** `AgentGraph(agent_hook=(node, router))` inserts one caller-supplied
node on the agent's outgoing edge, so it sees every generation — the intermediate
tool-calling ones as well as the final answer. `node(state, config)` is an ordinary
LangGraph node and may return `{}`; `router(state)` returns one of three strings:

| Verdict | Effect |
|---------|--------|
| `"proceed"` | Carry on exactly as an unhooked graph would (run the requested tools, or finish) |
| `"revise"` | Send the turn back for another generation instead of acting on this one |
| `"stop"` | End the turn here, without running any tool the generation asked for |

Anything else raises rather than silently ending the turn. `state["hook_state"]` is
scratch space for the hook — checkpointed with everything else, never read or written
by Marshall itself. With `agent_hook` unset (the default) nothing is added to the
graph and it is wired exactly as before. Note that a hook costs one extra superstep
per generation, so a hooked caller should raise `recursion_limit` above
`DEFAULT_RECURSION_LIMIT` in its `config` — see `tests/test_agent_hook.py` for a
minimal hook and `tests/test_recursion_budget.py` for the arithmetic.

### Agent Memory Gate

`AgentGraph` remembers the user in two directions at once:

- **Passive** — every `memory_every_n_turns` turns, the user's own messages are compressed
  into a personal profile by `UserMemory` and folded into the agent's system prompt.
- **Active** — the agent can call `recall_user_memory` itself when asked something about
  the user that the visible conversation cannot answer.

```python
agent = AgentGraph(
    file_path="path/to/document.pdf",
    mode_docs="pdf",
    mode_retreiver="simple",
    user_history_directory_path="path/to/memory/storage",  # required for durable recall
    memory_every_n_turns=5,     # 0 disables the passive gate entirely
    enable_recall_tool=True,    # False → 4-tool config, prompt adjusts to match
    user_id=None,               # None → shares one profile with ChatRagWithMemoryGraph
)

# --- conversation one ---
a = {"configurable": {"thread_id": "mon"}}
for msg in ["my name is Harshal",
            "I live in Pune",
            "I work on a RAG agent called Marshall",
            "my favourite topic is structural analysis",
            "I like building pipelines"]:          # the 5th turn trips the gate
    agent.invoke(msg, config=a)

print(agent.get_turn_count("mon"))               # 5
print(agent.get_user_history_summary("mon"))
# Compressed Chat History:
#   * Name: Harshal
#   * Location: Pune, India
#   * Profession: works on a RAG agent named Marshall
#   * Interest: Structural Analysis, Building Pipelines
# ...and from here on this profile is folded into the agent's system prompt

# --- conversation two: a different thread, nothing in its context ---
b = {"configurable": {"thread_id": "tue"}}
print(agent.invoke("what is my name?", config=b))
# → calls recall_user_memory, answers "Harshal" from the persisted store
```

Memory is scoped to the **user**, not the thread — a `thread_id` is one conversation, so a
new thread still recalls what earlier ones were told. Pass `user_id` only if you need to
keep several people's profiles apart.

> **Note on an 8B model.** `llama3.1` occasionally calls `recall_user_memory` on a plain
> statement ("I work on a RAG agent") where no lookup is needed. Answers stay correct — the
> cost is a wasted retrieval. Measured against the 4-tool baseline the delta is about one
> extra call in four conversational turns, and the baseline over-triggers too. Set
> `enable_recall_tool=False` or try `model="qwen2.5"` if this matters for your use case.

### Conversation Summarization

`AgentGraph` compresses its own history instead of letting it grow. Every
`summarize_every_n_turns` turns (default 7), once the message list is longer than
`summary_keep_recent` (default 14), the `summarize_and_trim` node folds everything older
into a running summary and replaces the list with that summary plus the recent tail:

```python
agent = AgentGraph(
    file_path="doc.pdf",
    summarize_every_n_turns=7,          # 0 disables the cycle
    summary_keep_recent=14,             # messages kept verbatim
    summary_persist_path="db/",         # defaults to persist_directory_path
)

print(agent.get_running_summary("mon"))    # the compressed record
print(len(agent.get_history("mon")))       # bounded, not linear in turn count
```

The cut point is never chosen blindly: agent turns emit 3-4 messages each, and cutting
between an `AIMessage`'s `tool_calls` and its `ToolMessage` orphans the pair, which Ollama
rejects. `memory/reconciliation.py:find_safe_cut_point` walks backward to the nearest
boundary that keeps every pair whole — a pure function over a plain message list, so voice
barge-in can reuse it to discard an interrupted tail.

The summary rides in a `HumanMessage` inside `<conversation_summary>` tags, never in the
system message, for the same reason retrieved context does. It is written to
`{summary_persist_path}/conversation_summary.json` keyed by `thread_id` and reloaded when
a new process picks the conversation up; a missing or corrupt file simply starts fresh. If
the summarizing call fails, nothing changes — the turn still answers and the next turn
retries.

### Loading Different Document Types

```python
# YouTube transcript (use video ID)
rag = NRagPipeline(file_path="VIDEO_ID", mode_docs="youtube")

# Web URL
rag = NRagPipeline(file_path="https://example.com", mode_docs="web_url")

# Scanned PDF (better OCR with PDFPlumber)
rag = NRagPipeline(file_path="scanned.pdf", mode_docs="scanned_pdf")

# Text file
rag = NRagPipeline(file_path="document.txt", mode_docs="txt")

# CSV file
rag = NRagPipeline(file_path="data.csv", mode_docs="csv")

# Folder of documents (specify glob pattern)
rag = NRagPipeline(
    file_path="path/to/folder",
    mode_docs="folder",
    glob_pattern="**/*.pdf"  # Required for folder mode
)
```

---

## Retrieval Modes

| Mode | Description |
|------|-------------|
| `simple` | Basic similarity search |
| `mmr` | Maximal Marginal Relevance - balances relevance and diversity |
| `mqr` | Multi-Query Retrieval - generates 5 similar queries to improve recall |
| `wiki` | Retrieves from Wikipedia |

---

## Advanced Features

### Memory Management
Marshall includes a sophisticated memory system that:
- Stores user conversation history persistently
- Creates vector embeddings of user queries for personalized responses
- Integrates past context with current document retrieval

### Configurable Chunking
- **Recursive chunking**: Fixed-size chunks with overlap (default: 500 chars, 50 overlap)
- **Semantic chunking**: Context-aware intelligent text splitting

### Flexible Retrieval
Configure retrieval parameters at query time:
- `k`: Number of documents to retrieve (default: 4)
- `lambda_mult`: MMR diversity parameter (default: 0.5)

---

## Requirements

- Python 3.12 (`requirements.txt` is tested on CPython 3.12.9, Windows)
- `gcc` and `g++` on PATH for checking and running C/C++ code (MinGW-w64 on Windows);
  without them `check_code` still handles Python and reports that no compiler was found
- Ollama running locally
  - `mistral` for the RAG pipelines (LangChain + LangGraph chat graphs), and for
    `UserMemory`'s profile compression — non-tool text generation, which it does well
  - `llama3.1` (or `qwen2.5`) for `AgentGraph` — mistral cannot reliably emit tool-calls
- ChromaDB
- LangChain + LangGraph ecosystem
- Internet connection for HuggingFace model downloads and Wikipedia/web tools

## File Structure Details

### Pipeline Classes (LangChain — `pipeline.py`)
- **NRagPipeline**: Single query-response pipeline for one-off questions
- **ChatRagPipeline**: Interactive chat with session-based memory
- **ChatRagPipelineWithMemory**: Interactive chat with persistent user memory across sessions

### Graph Classes (LangGraph — `pipeline_lg.py`)
- **ChatRagGraph**: `START → retrieve → generate → END` with `MemorySaver` for thread-scoped history
- **ChatRagWithMemoryGraph**: adds a conditional memory-gate that compresses the user's history every 5 turns
- **AgentGraph**: ReAct loop plus a memory gate, using `ChatOllama.bind_tools` + `langgraph.prebuilt.ToolNode`:

  ```text
  START → count_turn → {summarize_and_trim, summarize_user_history, agent}
  summarize_and_trim → {summarize_user_history, agent}
  agent → tools_condition → {tools → agent, END}      # or through agent_hook, if one is set
  ```

  The turn counter deliberately sits *before* `agent` rather than inside it: the ReAct loop
  re-enters `agent` after every tool call, so counting there would make the gate fire on
  tool usage instead of conversation length.

### Document Loading Support
- **PDF**: Regular PDFs using PyPDFLoader
- **Scanned PDF**: OCR-enabled PDFs using PDFPlumberLoader  
- **Web URLs**: Web scraping with WebBaseLoader
- **YouTube**: Transcript extraction via YouTube API
- **Text files**: Plain text documents
- **CSV files**: Structured data loading
- **Folders**: Batch processing with glob patterns

### Memory System
The `UserMemory` class embeds the user's own messages into a separate ChromaDB collection to
provide personalized context in future conversations. It exposes three paths:

- `.index(messages=None)` — the **only** write. Defaults to the object's own `chat_history`
- `.invoke({})` — compresses the user's messages into a short personal profile via `mistral`
- `.retreival(query, k)` — semantic search over those messages, used by `recall_user_memory`

Indexing is **idempotent**: documents are upserted under content-hash ids, so re-running the
memory gate on a growing conversation will not duplicate earlier messages (which would
otherwise shrink effective recall diversity to `k/m` after `m` gate hits). The collection is
opened lazily rather than built from documents, so an empty history is safe and a persisted
collection can be reopened in a later session.

**Constructing a `UserMemory` writes nothing.** It used to: the write was a constructor side
effect, so every reader was also a writer — `recall_user_memory` indexed the user's question
and then searched a collection that now contained it, retrieving the question back at `k=1`.
`AgentGraph` therefore keeps two funnels over one implementation: `_build_user_memory`
(the memory gate, writes) and `_read_user_memory` (the recall tool, does not).

---

## Development

This project uses:
- **LangChain** for the legacy Runnable-based pipelines
- **LangGraph** for the newer state-machine graphs and the ReAct agent (`ToolNode`, `tools_condition`, `MemorySaver`)
- **ChromaDB** for vector storage
- **HuggingFace Transformers** for embeddings
- **Ollama** for local LLM inference
- **Mistral** as the default RAG model; **llama3.1** for the tool-calling agent
- **silero-vad**, **faster-whisper** and **Kokoro** for the voice loop
- **gcc / g++** and **argon2-cffi** for the code tools and the admin password

Run the tests with `pytest -m "not needs_ollama"` (what CI runs), or plain `pytest` with
Ollama up and llama3.1 pulled to include the live tool-routing gates.

---

## Author

**Harshal** - Creator of Marshall

*Marshall represents an advanced RAG implementation with memory capabilities, designed for both simple document Q&A and complex conversational AI applications.*
