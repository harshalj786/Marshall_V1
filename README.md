# Marshall - RAG Pipeline Framework

**Marshall** is a modular Retrieval-Augmented Generation (RAG) pipeline built with LangChain. It enables you to load documents, chunk them intelligently, store embeddings in a vector database, and query them using an LLM.

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
- **Two Pipeline Modes**:
  - `NRagPipeline` - Single query-response
  - `ChatRagPipeline` - Interactive chat with history

---

## Project Structure

```
Marshall_V1$/
├── pipeline.py           # Main RAG pipeline classes
├── ingestion/
│   ├── doc_loader.py     # Document loading (PDF, web, YouTube, etc.)
│   ├── chunking.py       # Text chunking strategies
│   └── vector_db.py      # ChromaDB vector store builder
├── retreival/
│   └── retrieval.py      # Retrieval strategies (simple, MMR, MQR, wiki)
├── db/                   # Vector database persistence (gitignored)
└── .env                  # API keys (gitignored)
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

### Loading Different Document Types

```python
# YouTube transcript
rag = NRagPipeline(file_path="VIDEO_ID", mode_docs="youtube")

# Web URL
rag = NRagPipeline(file_path="https://example.com", mode_docs="web_url")

# Folder of PDFs
rag = NRagPipeline(
    file_path="path/to/folder",
    mode_docs="folder",
    glob_pattern="**/*.pdf"
)

# CSV file
rag = NRagPipeline(file_path="data.csv", mode_docs="csv")
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

## Requirements

- Python 3.9+
- Ollama (with `mistral` model installed)
- ChromaDB
- LangChain ecosystem


---

## Author

**Harshal** - Creator of Marshall
