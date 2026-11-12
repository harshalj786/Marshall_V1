# Marshall V2 — System Architecture & Orchestration

## High-Level System Architecture

```text
              ┌─────────────────────────────────────┐
              │          LangGraph State            │
              │ (Messages, Memory, Context Variables)│
              └──────────────────┬──────────────────┘
                                 ▼
              ┌─────────────────────────────────────┐
              │           Ollama Agent              │
              │  (Decides which tool to call next)  │
              └──────────┬─────────────────┬────────┘
                         ▼                 ▼
             ┌───────────────┐     ┌───────────────┐
             │  RAG Engine   │     │   Tools Dir   │
             │ (V1 Pipeline) │     │ (V2 Adapters) │
             └───────────────┘     └───────┬───────┘
                                           │
                       ├── gmail_tool.py (Google API)
                       ├── browser_tool.py (browser-use + Playwright)
                       └── scraper_tool.py (Crawl4AI)