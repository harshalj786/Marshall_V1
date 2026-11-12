# Marshall V2 — Master To-Do List (`todo.md`)

## 0. Tech Debt & Refactoring (From V1)
- [ ] Move `chunk_size` (500) and `chunk_overlap` (50) from literal inits to `config/settings.py`.
- [ ] Make the LLM backend configurable (remove hardcoded `Ollama(model="mistral", temperature=0)` from every file).
- [ ] Extract inline prompt templates from `.invoke()` calls into `docs/prompts.md` or a dedicated prompts registry.
- [ ] Replace magic numbers in memory injection (e.g., `count >= 5`, `len >= 10`) with configurable environment variables.

## 1. Orchestrator Engine Migration (LangGraph)
- [ ] Install LangGraph dependencies: `pip install langgraph langchain_core`.
- [ ] **Define State:** Create `orchestrator/state.py` and define `AgentState` with `add_messages` reducer and `user_summary` field.
- [ ] **Build Nodes:** Create `orchestrator/nodes.py`. Implement `agent_node` (to call Mistral with tools) and `tool_node` (to execute tool calls).
- [ ] **Compile Graph:** Create `orchestrator/agent_graph.py`. Set up nodes, conditional edge routing (`should_continue`), and attach the `InMemorySaver` or `SqliteSaver` checkpointer.

## 2. Implement Human-in-the-Loop (HITL) Safeguards
- [ ] Setup `interrupt()` logic within the LangGraph nodes.
- [ ] Create a CLI or UI prompt in `main.py` that listens for the interrupt, asks the human to authorize/complete a task (e.g., typing "done"), and resumes the graph using the saved `thread_id`.

## 3. Tool 1: Browser Automation & Booking (`tools/browser_tool.py`)
- [ ] Install dependencies: `pip install browser-use playwright` (and run `playwright install`).
- [ ] Wrap `browser-use` in a LangChain `@tool`.
- [ ] Write logic to accept a natural language task and execute it via the Playwright Chromium driver.
- [ ] **HITL Integration:** Add regex URL monitoring. If the browser navigates to a URL containing `/checkout`, `/payment`, or `/confirm`, freeze execution and trigger the LangGraph `interrupt()`.

## 4. Tool 2: Gmail Triage (`tools/gmail_tool.py`)
- [ ] Set up Google Cloud Console project and download OAuth 2.0 `credentials.json`.
- [ ] Implement token generation and caching (`token.json`) using `google-api-python-client`.
- [ ] Define the `EmailClassification` and `EmailTriageBatch` Pydantic schemas.
- [ ] Create the `@tool` to fetch `is:unread` emails, pass them to the LLM parser, automatically remove the unread label from `IGNORABLE` emails, and return a summary of `URGENT`/`ACTION_REQUIRED` items.

## 5. Tool 3: Web & Social Trend Scraper (`tools/scraper_tool.py`)
- [ ] Install dependency: `pip install crawl4ai`.
- [ ] Create the `@tool` that takes a target URL and uses `AsyncWebCrawler` to extract clean Markdown.
- [ ] Pipe the output Markdown through the V1 `Chunking` module.
- [ ] Implement a Map-Reduce prompt chain to summarize the chunks into themes, sentiment, and actionable insights.

## 6. Tool 4: V1 RAG & Memory Adapters (`tools/rag_tool.py`)
- [ ] Wrap the V1 `Retrival` pipeline in a `@tool` so the V2 agent can query local ChromaDB collections on demand.
- [ ] Update `main.py` to trigger the V1 `UserMemory` summarization upon startup, injecting the resulting profile directly into the LangGraph `AgentState["user_summary"]`.

## 7. Security & Deployment Validations
- [ ] Create `.env` file and add it to `.gitignore`. Move all API keys out of the codebase.
- [ ] Ensure `SECURITY.md` protocols are followed (no `os.system` calls in tools, inputs to ChromaDB are sanitized).
- [ ] Run end-to-end testing of the fully cyclic graph in `main.py` via the terminal REPL.