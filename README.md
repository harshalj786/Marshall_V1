# Marshall - Advanced RAG Pipeline Framework

**Marshall** is a modular Retrieval-Augmented Generation (RAG) pipeline built with LangChain. It enables you to load documents, chunk them intelligently, store embeddings in a vector database, and query them using an LLM with advanced memory capabilities.

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
- **Three Pipeline Modes**:
  - `NRagPipeline` - Single query-response
  - `ChatRagPipeline` - Interactive chat with session history
  - `ChatRagPipelineWithMemory` - Interactive chat with persistent user memory

---

## Project Structure

```
Marshall_V1$/
├── pipeline.py           # Main RAG pipeline classes (NRagPipeline, ChatRagPipeline, ChatRagPipelineWithMemory)
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
pip install chromadb sentence-transformers
pip install pypdf pdfplumber youtube-transcript-api
pip install ollama
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
- Ollama (with `mistral` model installed)
- ChromaDB
- LangChain ecosystem
- Internet connection for HuggingFace model downloads

## File Structure Details

### Pipeline Classes
- **NRagPipeline**: Single query-response pipeline for one-off questions
- **ChatRagPipeline**: Interactive chat with session-based memory
- **ChatRagPipelineWithMemory**: Interactive chat with persistent user memory across sessions

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
- **LangChain** for pipeline orchestration
- **ChromaDB** for vector storage
- **HuggingFace Transformers** for embeddings
- **Ollama** for local LLM inference
- **Mistral** as the default language model # can be changed anytime 
>>>>>>> 9b9091e (added memory module and updated pipeline)

---

## Author

**Harshal** - Creator of Marshall

*Marshall represents an advanced RAG implementation with memory capabilities, designed for both simple document Q&A and complex conversational AI applications.*
