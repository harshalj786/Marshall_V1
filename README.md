# Marshall - Advanced RAG Pipeline Framework

**Marshall** is a modular Retrieval-Augmented Generation (RAG) framework built on **LangChain + LangGraph**. It loads documents from many formats, chunks them, stores embeddings in a vector database, and answers questions through either classic RAG pipelines, LangGraph state-machine graphs, or a **ReAct-style agent** with four tools (calculator, doc-search, Wikipedia, web search).

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
  - `AgentGraph` - **ReAct-style agent** with tool-calling (calculator, doc-search, Wikipedia, web)
- **Tool-Calling Layer** (`tools/`): `calculator`, `wikipedia_lookup`, `web_search`, `search_current_doc`

---

## Project Structure

```
Marshall_V1$/
├── pipeline.py           # Legacy LangChain pipelines (NRagPipeline, ChatRagPipeline, ChatRagPipelineWithMemory)
├── pipeline_lg.py        # LangGraph graphs (ChatRagGraph, ChatRagWithMemoryGraph, AgentGraph)
├── ingestion/
│   ├── doc_loader.py     # Document loading (PDF, scanned PDF, web, YouTube, CSV, TXT, folder)
│   ├── chunking.py       # Text chunking strategies (recursive, semantic)
│   ├── vector_db.py      # ChromaDB vector store builder
│   └── __init__.py
├── retreival/
│   ├── retrieval.py      # Retrieval strategies (simple, MMR, MQR, wiki)
│   └── __init__.py
├── memory/
│   ├── long_term_memory.py  # User memory management for persistent chat history
│   └── __init__.py
├── tools/
│   ├── tools.py          # Agent tools (calculator, wikipedia_lookup, web_search, search_current_doc)
│   └── __init__.py
├── db/                   # Vector database persistence (gitignored)
├── labbb.ipynb          # Jupyter notebook for testing/experimentation
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

# Install dependencies
pip install langchain langchain-community langchain-huggingface langchain-experimental
pip install langgraph langgraph-checkpoint langgraph-prebuilt langchain-ollama
pip install chromadb sentence-transformers
pip install pypdf pdfplumber youtube-transcript-api
pip install ollama
# Agent tool-calling extras:
pip install duckduckgo-search ddgs wikipedia
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

`AgentGraph` gives Marshall four tools and lets it decide when to use them.

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

- Python 3.9+
- Ollama running locally
  - `mistral` for the RAG pipelines (LangChain + LangGraph chat graphs)
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
- **AgentGraph**: ReAct loop `START → agent → tools_condition → {tools → agent, END}` using `ChatOllama.bind_tools` + `langgraph.prebuilt.ToolNode`

### Document Loading Support
- **PDF**: Regular PDFs using PyPDFLoader
- **Scanned PDF**: OCR-enabled PDFs using PDFPlumberLoader  
- **Web URLs**: Web scraping with WebBaseLoader
- **YouTube**: Transcript extraction via YouTube API
- **Text files**: Plain text documents
- **CSV files**: Structured data loading
- **Folders**: Batch processing with glob patterns

### Memory System
The `UserMemory` class creates vector embeddings of user queries to provide personalized context in future conversations, stored in a separate ChromaDB collection.

---

## Development

This project uses:
- **LangChain** for the legacy Runnable-based pipelines
- **LangGraph** for the newer state-machine graphs and the ReAct agent (`ToolNode`, `tools_condition`, `MemorySaver`)
- **ChromaDB** for vector storage
- **HuggingFace Transformers** for embeddings
- **Ollama** for local LLM inference
- **Mistral** as the default RAG model; **llama3.1** for the tool-calling agent

---

## Author

**Harshal** - Creator of Marshall

*Marshall represents an advanced RAG implementation with memory capabilities, designed for both simple document Q&A and complex conversational AI applications.*
