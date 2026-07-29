# Marshall V1 — What I Built

> A snapshot of the current, hardcoded state of Marshall V1 — my own modular RAG framework.
> *"I am Marshall, the AI son of Harshal."*

---

## The Big Picture

Marshall V1 is a **modular Retrieval-Augmented Generation (RAG) pipeline** I built from scratch using LangChain. It takes documents (from many formats), chunks them, embeds them into a local vector database, and answers user questions on top of them — with optional chat history and long-term user memory.

Everything is wired as **`Runnable` classes** so each stage is composable and can be invoked independently.

---

## Architecture at a Glance

```
                ┌─────────────────────┐
                │   Source Document   │
                │  (PDF/URL/YT/CSV/…) │
                └──────────┬──────────┘
                           ▼
              ┌──────────────────────────┐
              │  ingestion/doc_loader.py │   → LoadDocs
              └──────────┬───────────────┘
                         ▼
              ┌──────────────────────────┐
              │  ingestion/chunking.py   │   → Chunking (recursive / semantic)
              └──────────┬───────────────┘
                         ▼
              ┌──────────────────────────┐
              │  ingestion/vector_db.py  │   → BuildKnowledge (ChromaDB)
              └──────────┬───────────────┘
                         ▼
              ┌──────────────────────────┐
              │  retreival/retrieval.py  │   → Retrival (simple / mmr / mqr / wiki)
              └──────────┬───────────────┘
                         ▼
              ┌──────────────────────────┐
              │        pipeline.py       │   → Ollama (mistral) + Prompt + Parser
              └──────────┬───────────────┘
                         ▼
                ┌───────────────────┐
                │   Final Answer    │
                └───────────────────┘

     memory/long_term_memory.py  → UserMemory (persistent per-user history)
```

---

## Modules I Built

### 1. `ingestion/doc_loader.py` — `LoadDocs`
A single class that handles **7 different document sources** via a `mode` flag:

| mode | Loader used | Notes |
|------|-------------|-------|
| `pdf` | `PyPDFLoader` | Standard PDFs |
| `scanned_pdf` | `PDFPlumberLoader` | Better for scanned / OCR-heavy PDFs |
| `web_url` | `WebBaseLoader` | Web page scraping |
| `youtube` | `YouTubeTranscriptApi` | Fetches YouTube transcript by video ID; raises on `TranscriptsDisabled` |
| `txt` | `TextLoader` | Plain text |
| `csv` | `CSVLoader` | Structured CSV rows |
| `folder` | `DirectoryLoader` | Batch load a folder; requires `glob_pattern`; auto-picks loader for `.pdf`/`.txt`/`.csv` |

### 2. `ingestion/chunking.py` — `Chunking`
Handles splitting docs into retrievable chunks in **two modes**:

- **`recursive`** → `RecursiveCharacterTextSplitter` with configurable `chunk_size`, `chunk_overlap`, optional separator.
- **`semantic`** → `SemanticChunker` from `langchain_experimental`, backed by `sentence-transformers/all-MiniLM-L6-v2`, using 90th-percentile breakpoint threshold.

Accepts either a raw list of `Document`s or a dict `{"docs": [...]}`.

### 3. `ingestion/vector_db.py` — `BuildKnowledge`
Wraps **ChromaDB** persistence:

- Embeddings: `HuggingFaceEmbeddings("sentence-transformers/all-MiniLM-L6-v2")`
- Creates a `Chroma` vector store from documents with a configurable `collection_name` and `persist_directory`.
- No manual `.persist()` needed — handled by `persist_directory`.

### 4. `retreival/retrieval.py` — `Retrival`
A unified retriever with **4 strategies** on top of the same Chroma store:

| mode | What it does |
|------|--------------|
| `simple` | Basic similarity search (top-k) |
| `mmr` | Maximal Marginal Relevance — diversity-aware (uses `lambda_mult`) |
| `mqr` | **Multi-Query Retrieval**: uses an Ollama Mistral LLM + `PydanticOutputParser` to generate 5 rewritten queries, retrieves for each, and merges results — kills query ambiguity |
| `wiki` | `WikipediaRetriever` — pulls context straight from Wikipedia (with language support) |

It also **sanitizes collection names** via regex so bad file paths don't break Chroma, and falls back to `"VDB_collection"` if the name is too short.

Returns a standard dict: `{"query", "context", "mode"}`.

### 5. `memory/long_term_memory.py` — `UserMemory`
Persistent per-user memory built on ChromaDB:

- On init, filters `chat_history` into **HumanMessage** vs **AIMessage** lists.
- Embeds only the human messages into a `Chroma` collection (default `"user's_history"`) with a configurable `persist_directory_path`.
- `.invoke()` → runs a Mistral prompt that **compresses** the human-message stream into a personal-context summary of the user (relations, personal facts).
- `.retreival(query, k)` → similarity search over the user's own past messages, returns joined context.

### 6. `pipeline.py` — Four Pipeline Classes

All four wire together **LoadDocs → Chunking → Retrival → Ollama Mistral → StrOutputParser** and expose `.invoke()`.

#### `NRagPipeline`
Single-shot Q&A. Runs one query, returns one answer. Uses a strict "answer only from context" prompt where Marshall introduces himself as *"the AI son of Harshal."*

#### `ChatRagPipeline`
Interactive REPL loop (`while True`): reads user input, retrieves context per turn, and keeps `chat_history` in memory using `HumanMessage` / `AIMessage`. Supports commands `quit` and `history`.

#### `ChatRagPipelineWithMemory`
The main event. Same interactive loop, but:

- Counts turns.
- For turns **1–4** → uses the basic template with just chat history.
- From turn **5 onward** → every 5 turns it fires `UserMemory` to build a compressed user profile, then switches to a **second template** that also injects `{user_history}` into context.
- Clears `chat_history` once it grows past 10 entries to prevent context bloat.
- Exposes helpers: `.MsgHistory()` (prints human/AI splits) and `.UserSummary()` (returns the compressed user profile).

#### `Structure_pipeline`
A **domain-specialized variant** of the memory pipeline, prompt-tuned as a **structural-analysis tutor**. Same 5-turn memory kick-in, but the system prompt restricts scope to trusses/beams/FEM/stress/strain/etc., treats input as a video transcript, and refuses off-topic questions.

---

## What Makes This Non-Trivial

- **Everything is a `Runnable`** — each stage plugs into LangChain's runnable interface, so any module can be chained or swapped standalone.
- **Multi-Query Retrieval with Pydantic-parsed LLM output** — I use a schema-forced JSON output to guarantee I get exactly 5 rewritten queries.
- **Adaptive memory injection** — the pipeline doesn't blindly stuff history in; it waits until turn 5, compresses history into a user profile, and only *then* enriches the prompt.
- **Chat-history rolling window** — cleared every 10 messages to stay inside the LLM context budget.
- **Collection-name sanitization** — regex-cleans file paths so ChromaDB doesn't reject them.

---

## Config / Environment

- `.env` holds `OPENAI_API_KEY`, `GEMINI_API_KEY`, `exchange_rate_api_key`, `claude_api` (gitignored).
- LLM backend: **Ollama** running local **Mistral** at `temperature=0`.
- Embeddings: **HuggingFace `all-MiniLM-L6-v2`** everywhere (retrieval, semantic chunking, user memory).
- Vector store: **ChromaDB**, persisted under `db/`.

---

## Where I Hardcoded Things (self-flagged for V2)

- `Chunking(chunk_size=500, chunk_overlap=50, …)` — literal in every pipeline `__init__`.
- LLM is always `Ollama(model="mistral", temperature=0)` — not configurable per-call.
- Prompt templates are inlined inside `.invoke()` — no external prompt store.
- Memory-injection cadence is a magic number (`count >= 5`, `count % 5 == 0`, clear at `len >= 10`).
- The `tools/` folder exists but `tools.py` is empty — a stub I haven't filled in yet.
- `labbb.ipynb` is my sandbox notebook, not a formal test suite.

---

## What's Not Built Yet

- `tools/tools.py` is empty — no tool-calling / function-calling layer yet.
- No formal test suite.
- No CLI wrapper — everything runs from the notebook or by importing the pipeline classes.
- No streaming responses; all `.invoke()` calls are blocking.
- No API / server layer.

---

## Author

**Harshal** — designed and hand-built Marshall V1 end-to-end.
