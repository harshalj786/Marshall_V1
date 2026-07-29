# Marshall V2 — Implementation Roadmap & Checklist

## Phase Breakdown

### Phase 1: Core State Machine & Orchestrator Setup
- Set up directory structure (`orchestrator/`, `tools/`, `config/`).
- Create `AgentState` schema in `orchestrator/state.py`.
- Implement `agent_node` and `tool_node` in `orchestrator/nodes.py`.
- Construct `StateGraph` in `orchestrator/agent_graph.py` with `SqliteSaver` checkpointer.

### Phase 2: Human-in-the-Loop (HITL) Safeguard System
- Implement `interrupt()` mechanisms in graph execution nodes.
- Define payment/checkout keyword intercept gates.
- Connect CLI resume logic in `main.py`.

### Phase 3: Gmail Triage Module (`tools/gmail_tool.py`)
- Configure Google OAuth credentials and client library.
- Build email fetch functions for unread inbox messages.
- Implement structured output parser matching `EmailTriageBatch` schema.

### Phase 4: Browser Automation Agent (`tools/browser_tool.py`)
- Integrate `browser-use` with Playwright drivers.
- Implement navigation workflows for target services.
- Connect HITL trigger on checkout screen detection.

### Phase 5: Dynamic Web & Social Trend Scraper (`tools/scraper_tool.py`)
- Setup `Crawl4AI` async crawler engine.
- Build Markdown extraction and cleaning pipeline.
- Implement trend analysis summarization chain.

### Phase 6: Adapter Integration & Final Testing
- Wrap V1 RAG system in `tools/rag_tool.py`.
- Connect V1 `UserMemory` profile loader into agent state initialization.
- End-to-end testing across all tool workflows in `main.py`.

---

## Developer Execution Checklist

- [ ] Setup folder structure and install V2 dependencies (`langgraph`, `browser-use`, `crawl4ai`).
- [ ] Complete `orchestrator/state.py` and `orchestrator/agent_graph.py`.
- [ ] Implement `tools/rag_tool.py` adapter.
- [ ] Implement `tools/gmail_tool.py` with OAuth and Pydantic triage.
- [ ] Implement `tools/scraper_tool.py` using `Crawl4AI`.
- [ ] Implement `tools/browser_tool.py` with Playwright and payment interrupt gates.
- [ ] Integrate long-term memory preloading in `main.py`.
- [ ] Verify execution loops and HITL state pause/resume functionality.