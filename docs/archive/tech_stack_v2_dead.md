# Marshall V2 — Technology Stack (`tech_stack.md`)

This document outlines the complete technology stack, libraries, and target infrastructure for the Marshall V2 agentic framework.

---

## 1. Core Orchestration & AI Engine
*   **Agent Framework:** LangGraph (State machine and orchestrator)
*   **Pipeline Abstractions:** LangChain Core (`langchain_core`)
*   **Local LLM Inference:** Ollama (Targeting Mistral / Llama 3)
*   **Embedding Model:** HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`)
*   **Structured Output Validation:** Pydantic

## 2. Memory & Data Storage
*   **Vector Database:** ChromaDB (Local persistence under `db/`)
*   **Agent State Checkpointer:** `SqliteSaver` (SQLite) or `InMemorySaver` (for active Human-in-the-Loop interrupt pauses)
*   **Document Loaders:** `PyPDFLoader`, `PDFPlumberLoader`, `YouTubeTranscriptApi`, `CSVLoader`

## 3. Action Tools & Automation
*   **Browser Agent Engine:** `browser-use`
*   **Web Drivers:** Playwright (Chromium, headful for HITL payments, headless for background tasks)
*   **Dynamic Web Scraper:** `Crawl4AI` (Asynchronous LLM-optimized crawler)
*   **Gmail API Integration:** `google-api-python-client`, `google-auth-oauthlib`, `google-auth-httplib2`

## 4. Target Hardware & Infrastructure
*   **Deployment:** Fully local runtime
*   **Target CPU:** Intel i9
*   **Target GPU:** NVIDIA RTX 4080 (12GB VRAM) dedicated to low-latency local inference and rapid embedding generation.

## 5. Security & Environment Utilities
*   **Environment Configuration:** `python-dotenv` (Loading `.env` variables)
*   **Package Management:** `pip` (Strict version pinning in `requirements.txt`)
*   **Vulnerability Scanning:** `pip-audit` or `safety`