# Marshall V1 — Architecture Review

**Scope of this review:** every Python file in the repository read in full (`pipeline.py` 377 L, `pipeline_lg.py` 617 L, `ingestion/doc_loader.py` 78 L, `ingestion/chunking.py` 42 L, `ingestion/vector_db.py` 21 L, `retreival/retrieval.py` 89 L, `memory/long_term_memory.py` 64 L, `tools/tools.py` 178 L, `tools/__init__.py` 15 L, three empty package `__init__.py` files), plus `README.md`, `WHAT_I_BUILT.md`, all ten `docs/*.md`, `labbb.ipynb`, `.gitignore`, `.claude/settings.local.json`, and the uncommitted diff on `pipeline.py`.

**Status:** living document. Findings are kept after they are resolved, marked `[FIXED]` with the change that closed them, so the history of *why* the code looks the way it does stays readable.

| Finding | Status |
|---|---|
| **C1** — corpus re-appended on every construction | `[FIXED]` — `ingestion/vector_db.py` |
| **C2** — long-term memory silently non-durable | `[FIXED]` — `memory/long_term_memory.py` |
| **C3** — no dependency manifest | `[FIXED]` — `requirements.txt` |
| **H1** — untrusted content in the system message | `[FIXED]` — `pipeline_lg.py` |
| **M11** — `scanned_pdf` mode broken | `[FIXED]` — `requirements.txt` |
| **H2–H6, M1–M10, L1–L14** | open |

All three criticals and the highest-severity high are closed. **H2** (unbounded agent context plus an in-RAM checkpointer) is now the highest-severity open finding, and the one that most directly blocks voice. **H4** remains the cheapest win on the board at roughly ten lines.

**Note on line references:** the H1 fix restructured the prompt constants at the top of `pipeline_lg.py`, so line numbers cited in §1–§7 for anything above `ChatRagGraph` have shifted. Symbol names are still accurate.

**Note on `WHAT_I_BUILT.md`:** it is stale. It claims "`tools/tools.py` is empty — a stub I haven't filled in yet" (line 154, 161) and describes `UserMemory`'s collection as `"user's_history"` (line 101). Both are false as of the current tree — `tools/tools.py` holds five working tools, and the collection default is `"user_history"`. `README.md` is the accurate document.

---

## 1. Architecture Overview

### Overall architecture

Marshall is a **layered, locally-hosted RAG system with two generations of orchestration living side by side**. The bottom three layers (ingestion, vector storage, retrieval) are shared and generation-agnostic. The top layer exists twice: a legacy LangChain `Runnable` chain implementation in `pipeline.py`, and a LangGraph state-machine implementation in `pipeline_lg.py`. Both drive the same Ollama-hosted local models and the same ChromaDB stores.

There is no server, no CLI entry point, no `main.py`, and no dependency manifest. Every consumer is a Python import — from `labbb.ipynb` or from an interactive interpreter. The `while True: input(...)` REPLs inside the pipeline classes *are* the application shell.

### Design philosophy

Four philosophies are visible, and they are not fully consistent with each other:

1. **"Everything is a `Runnable`."** `LoadDocs`, `Chunking`, `BuildKnowledge`, `Retrival`, `UserMemory`, and all four `pipeline.py` classes subclass `langchain_core.runnables.Runnable` and expose a single `invoke(input_dict, config=None)`. Stages are meant to be swappable and independently callable. In practice only `invoke` is overridden — no `ainvoke`, no `InputType`/`OutputType`, no `batch` — so the abstraction buys a uniform call shape and nothing else.
2. **"Mode flags over subclasses."** Rather than `PDFLoader`/`CSVLoader` classes, `LoadDocs` takes `mode: Literal["pdf","scanned_pdf","web_url","txt","youtube","folder","csv"]` and branches. Same for `Chunking` (`recursive`/`semantic`) and `Retrival` (`simple`/`mmr`/`mqr`/`wiki`). Configuration is constructor arguments; behavior is an `if/elif` ladder.
3. **"Local and private by default."** Every LLM call goes to a local Ollama (`mistral` for text, `llama3.1` for tool-calling). Every embedding is local (`sentence-transformers/all-MiniLM-L6-v2`). Every vector store is a local Chroma directory. No network calls except Wikipedia/DuckDuckGo tools and the initial HuggingFace model download. `docs/vision.md` names this explicitly as Pillar 4.
4. **"Empirically-tuned agent prompting."** `pipeline_lg.py:327-364` and `tools/tools.py:146-164` are unusually careful — the prompt is assembled so it describes *exactly* the tools bound, with an explicit comment (`pipeline_lg.py:452-453`) that describing an unbound tool makes llama3.1 narrate a call in prose instead of emitting a real `tool_call`. This layer was clearly built against measured failures, and it is the highest-quality code in the repository.

### Major components and responsibilities

| Component | File | Responsibility |
|---|---|---|
| `LoadDocs` | `ingestion/doc_loader.py:11` | Source → `list[Document]`. Seven modes. Only stateful field is `file_path`/`mode`. |
| `Chunking` | `ingestion/chunking.py:7` | `list[Document]` → chunked `list[Document]`. Recursive or semantic. Loads a MiniLM model only in semantic mode. |
| `BuildKnowledge` | `ingestion/vector_db.py:5` | Chunks → a live `Chroma` vector store. Owns the embedding model and the persist directory. |
| `Retrival` | `retreival/retrieval.py:11` | Owns a `BuildKnowledge`, therefore owns the vector store. Sanitizes the collection name. Exposes four retrieval strategies through one `invoke`. Returns `{"query", "context", "mode"}`. |
| `UserMemory` | `memory/long_term_memory.py:24` | Long-term, user-scoped memory. Two paths: `.invoke({})` compresses the user's human messages into a prose profile via mistral; `.retreival(query, k)` does semantic search over those messages. |
| Tools | `tools/tools.py` | Five agent tools: two static-and-pure (`calculator`), two static-and-networked (`wikipedia_lookup`, `web_search`), two factory-built and closure-bound (`make_search_current_doc`, `make_recall_user_memory`). |
| Legacy pipelines | `pipeline.py` | `NRagPipeline` (one-shot), `ChatRagPipeline` (REPL + session history), `ChatRagPipelineWithMemory` (REPL + 5-turn memory gate), `Structure_pipeline` (domain-specialized copy of the previous, currently broken — see §4). |
| Graphs | `pipeline_lg.py` | `ChatRagGraph`, `ChatRagWithMemoryGraph`, `AgentGraph`. |

### Execution flow from startup to response

**Startup is expensive and entirely synchronous.** Every pipeline and graph constructor does the same four things before returning (`pipeline.py:25-33`, `82-90`, `150-158`, `261-269`; `pipeline_lg.py:79-88`, `182-191`, `419-428`):

1. `LoadDocs(...).invoke({})` — reads and parses the source document.
2. `Chunking(chunk_size=500, chunk_overlap=50, ...).invoke({"docs": docs})` — splits it.
3. `Retrival(docs=chunked, ..., collection_name=f"VDB for {file_path}", path=persist_dir)` — which internally constructs `BuildKnowledge`, loads a fresh MiniLM model (`ingestion/vector_db.py:10-12`), and calls `Chroma.from_documents` (`ingestion/vector_db.py:15-20`), embedding and writing every chunk.
4. For graphs only: build the LLM handle, the prompts, a `MemorySaver`, and compile the `StateGraph`.

There is no cache and no "collection already exists, skip" check. Constructing the object *is* the ingestion.

**Response flow** then depends on which class you built. Three distinct shapes:

- **One-shot** (`NRagPipeline.invoke`, `pipeline.py:34-67`): retrieve → join chunks with `\n\n` → fill a `PromptTemplate` → `template | Ollama(mistral) | StrOutputParser` → return string.
- **REPL** (`ChatRagPipeline.invoke:91-132`, `ChatRagPipelineWithMemory.invoke:159-232`, `Structure_pipeline.invoke:270-378`): the same, wrapped in a `while True` that calls `input()`, maintains a `chat_history` list of `HumanMessage`/`AIMessage`, and handles the literal commands `quit` and `history`.
- **Graph** (`pipeline_lg.py`): one `.invoke()` per turn; conversation state lives in the checkpointer keyed by `thread_id`, not in the object.

### LangGraph workflow

Three compiled graphs, all with `MemorySaver` checkpointers, detailed in §5.

- `ChatRagGraph`: `START → retrieve → generate → END`.
- `ChatRagWithMemoryGraph`: `START → retrieve → (gate) → {summarize_user_history → generate | generate} → trim_history → END`.
- `AgentGraph`: `START → count_turn → (gate) → {summarize_user_history → agent | agent}`, then the ReAct cycle `agent → (tools_condition) → {tools → agent | END}`.

### RAG pipeline

Retrieval is `Retrival.invoke` (`retreival/retrieval.py:28-90`), a four-branch dispatch over one shared Chroma store:

- `simple` (55-64): `as_retriever(search_kwargs={"k": k})`.
- `mmr` (43-54): `search_type="mmr"` with `lambda_mult`.
- `mqr` (66-82): runs a `PromptTemplate | Ollama(mistral) | PydanticOutputParser` chain that forces a JSON list of five rewritten queries, then retrieves top-k for each and `extend`s them into one flat list — **no dedup, no rank fusion**, so the result is up to `5k` documents with duplicates.
- `wiki` (83-90): bypasses the vector store entirely and hits `WikipediaRetriever`.

Every branch returns `{"query", "context", "mode"}` where `context` is a `list[Document]`. Callers universally do `"\n\n".join(doc.page_content for doc in result["context"])` — that exact line appears at `pipeline.py:41`, `119`, `205`, `351`, `pipeline_lg.py:105`, `214`, and in slightly different form at `tools/tools.py:125`.

**The agent's RAG path is structurally different from the chat graphs'.** The chat graphs retrieve unconditionally in a node and stuff `context` into the system prompt. `AgentGraph` has no retrieval node and no `context` field in its state — the document is reachable only through the `search_current_doc` tool, which the model chooses to call. This is the right design and worth preserving.

### Agent workflow

`AgentGraph.__init__` (`pipeline_lg.py:404-462`) is a careful sequence with real ordering constraints, each documented in-line:

1. Ingest and build `Retrival` (419-428).
2. Compute a sanitized, **user-scoped** memory collection name (436-439) — `"user_history"` by default, `f"user_history_{user_id}"` if a `user_id` is passed.
3. Initialize `self._memory_cache: dict[str, tuple[fingerprint, UserMemory]]` (443).
4. Build the tool list: `[calculator, wikipedia_lookup, web_search, search_current_doc]`, conditionally append `recall_user_memory`, then extend with `extra_tools` (445-450).
5. Format the system prompt so it describes exactly the tools that will be bound (454-457).
6. `ChatOllama(model=..., temperature=0).bind_tools(self.tools)` — **must come after step 4** (459-460).
7. Compile the graph with `ToolNode(self.tools)` (546-563).

`AgentGraph.invoke` (565-582) reads the checkpointer to see whether the thread exists; if not, it prepends a `SystemMessage`; appends the `HumanMessage`; caps `recursion_limit` at 25 (default is 10 000, so an oscillating tool loop would otherwise burn ~10 000 Ollama calls); and finally extracts the last `AIMessage` **that has no `tool_calls`** (577-580) so a tool-call stub is never returned as the answer.

### Memory workflow

Memory is **user-scoped, not thread-scoped** — deliberate, and documented at `pipeline_lg.py:432-435`. A `thread_id` is one conversation; per-thread collections would make long-term memory no better than short-term state.

`AgentGraph._build_user_memory` (464-497) is the single funnel used by both the passive gate and the active recall tool, so the two can never disagree. It filters human messages, fingerprints them as `(count, last_content)` — not just a count, so two threads at the same message count can't collide — checks the cache, and otherwise constructs a `UserMemory`.

`UserMemory.__init__` (`memory/long_term_memory.py:25-46`) opens the Chroma collection *lazily* (constructing `Chroma(...)` rather than `Chroma.from_documents(...)`), so an empty history doesn't crash and a persisted collection is reopened rather than rebuilt; then upserts the human messages under SHA-1 content-hash ids, making re-indexing idempotent.

Two directions:
- **Passive:** every `memory_every_n_turns` turns, `_summarize_user_history_node` (512-523) compresses history to a prose profile stored in `state["user_history_summary"]`. `_agent_node` (537-544) folds it into a *freshly constructed* `SystemMessage` per call via `_with_profile` (525-535) and never writes it back — so the profile also applies on post-tool re-entry and cannot compound.
- **Active:** `recall_user_memory` (`tools/tools.py:141-176`) lets the model search the user's past messages on demand, including from a thread that never saw them.

---

## 2. Dependency Graph

### Direct import dependencies (first-party only)

```
pipeline.py
  → ingestion.doc_loader.LoadDocs          (direct, used)
  → ingestion.chunking.Chunking            (direct, used)
  → ingestion.vector_db.BuildKnowledge     (direct, DEAD — imported at line 3, never referenced)
  → retreival.retrieval.Retrival           (direct, used)
  → memory.long_term_memory.UserMemory     (direct, used)

pipeline_lg.py
  → ingestion.doc_loader.LoadDocs          (direct)
  → ingestion.chunking.Chunking            (direct)
  → retreival.retrieval.Retrival           (direct)
  → memory.long_term_memory.UserMemory     (direct)
  → tools (package)                        (direct, 5 symbols, lines 17-23)
  → langchain_ollama.ChatOllama            (direct, DEFERRED — function-local import at line 417)

retreival/retrieval.py
  → ingestion.vector_db.BuildKnowledge     (direct — the only cross-package edge below the top layer)

ingestion/vector_db.py     → (leaf, LangChain only)
ingestion/chunking.py      → (leaf, LangChain only)
ingestion/doc_loader.py    → (leaf, LangChain + youtube_transcript_api)
memory/long_term_memory.py → (leaf, LangChain only)
tools/tools.py             → (leaf, LangChain + langgraph.prebuilt.InjectedState)
tools/__init__.py          → tools.tools (re-export)
```

**No circular dependencies exist.** The import graph is a clean DAG. `ingestion/`, `memory/`, and `tools/` are all leaves; only `retreival/` reaches sideways into `ingestion/`.

### Implicit couplings (not visible in the import graph)

1. **The retrieval result contract.** `Retrival.invoke` returns a bare dict `{"query", "context", "mode"}`. Seven call sites depend on `result["context"]` being a `list[Document]` — `pipeline.py:41,119,205,351`, `pipeline_lg.py:105,214`, `tools/tools.py:122-124`. Nothing enforces this: it is not a `TypedDict`, not a Pydantic model, not documented outside a comment. Changing the key name breaks seven places silently.

2. **`make_search_current_doc(retrival)` duck-types its argument.** `tools/tools.py:107` accepts an untyped `retrival` and only ever calls `.invoke({"query", "k"})`. The tool layer has no import edge to `retreival/` — the dependency is injected. This is good design (it is why `tools/` is a leaf), but it means the contract is enforced only at runtime.

3. **`make_recall_user_memory(build_memory)` is a reverse dependency.** `tools/tools.py:132` receives `AgentGraph._build_user_memory` — a bound method of the caller — as a callback. So `tools/` calls back into `pipeline_lg.py` at runtime while importing nothing from it. Invisible to any static import analysis, and the single hardest coupling to reason about in the codebase.

4. **`InjectedState` / `RunnableConfig` annotations couple `tools/tools.py` to LangGraph's runtime.** `recall_user_memory`'s `state` and `config` parameters (`tools/tools.py:144-145`) are populated by `ToolNode`, not by the model. That tool cannot be invoked outside a LangGraph `ToolNode` without manual argument passing.

5. **The Chroma persist directory is a shared global resource.** Multiple `Retrival` objects, multiple `UserMemory` objects, and multiple processes can address the same `(persist_directory, collection_name)` pair with zero coordination. Nothing in the code owns or locks it.

6. **`UserMemory`'s default collection name is a coordination point.** `ChatRagWithMemoryGraph` never passes `collection_name`, so it gets `"user_history"` (`memory/long_term_memory.py:25`). `AgentGraph` computes `"user_history"` when `user_id is None` (`pipeline_lg.py:436`). The two classes therefore intentionally share a profile — documented in `README.md:230` and `pipeline_lg.py:432-435`, but it is a shared-name convention, not an enforced one.

### Shared globals

- **`memory/long_term_memory.py:12` — `_EMBEDDING_MODEL`.** A module-level singleton behind `_get_embedding_model()` (14-22), added because construction costs ~3 s and the memory gate plus recall tool rebuild `UserMemory` frequently. Correct optimization; the check-then-set is not synchronized (§4).
- **`ingestion/vector_db.py:10-12` has no such singleton.** `BuildKnowledge.__init__` constructs a fresh `HuggingFaceEmbeddings` every time — i.e. once per `Retrival`, i.e. once per pipeline/graph object. Inconsistent with `UserMemory`.

### Tight coupling

- **`Retrival` owns `BuildKnowledge` in its constructor** (`retreival/retrieval.py:26-27`). You cannot construct a `Retrival` over an *existing* collection without re-supplying and re-ingesting `docs`. Retrieval and index-construction are welded together. This is the root cause of the duplicate-write problem in §8.
- **All six top-level classes hard-code their entire ingestion chain in `__init__`.** There is no way to inject a pre-built `Retrival` into `AgentGraph` or any pipeline.
- **The LLM is constructed at the point of use**, not injected: `pipeline.py:64,112,196,342`, `retreival/retrieval.py:22`, `memory/long_term_memory.py:50`, `pipeline_lg.py:89,193,460`. Seven independent construction sites for what is conceptually one configured backend.

### Duplicated logic

| Duplicate | Locations | Notes |
|---|---|---|
| The load→chunk→`Retrival` constructor block | `pipeline.py:25-33, 82-90, 150-158, 261-269`; `pipeline_lg.py:79-88, 182-191, 419-428` | **7 copies**, character-for-character identical apart from variable names. |
| The REPL loop (`while True` / `input` / `quit` / `history`) | `pipeline.py:116-131, 202-231, 348-377`; `pipeline_lg.py:143-159, 294-309, 584-596` | **6 copies.** |
| `get_history(thread_id)` | `pipeline_lg.py:161-163, 311-313, 598-600` | 3 identical copies. |
| "last AIMessage" extraction | `pipeline_lg.py:137-141, 288-292, 577-582` | 3 copies, the third with an extra `tool_calls` filter. |
| The Marshall system prompt | `pipeline.py:96-111, 163-178` (inline) and `pipeline_lg.py:36-48` (constant) | The graph version correctly hoisted it to a module constant; `pipeline.py` did not. |
| Collection-name sanitization regex | `retreival/retrieval.py:17-21` and `pipeline_lg.py:437-439` | The second carries the comment *"UserMemory does not sanitize collection_name — mirror retreival/retrieval.py:17-21"*. A knowingly-copied invariant. |
| `ChatRagPipelineWithMemory` vs `Structure_pipeline` | `pipeline.py:134-244` vs `245-378` | ~95% identical; only the prompt strings differ. 133 lines duplicated to change two prompts. |

### Modules doing too much

- **`pipeline.py`** — four classes, each of which is simultaneously an ingestion orchestrator, a prompt store, an LLM client factory, a conversation-state manager, a memory scheduler, and a terminal UI.
- **`Retrival`** (`retreival/retrieval.py:11-90`) — name sanitization, index construction, four retrieval strategies, *and* it owns a second LLM for `mqr` query rewriting (line 22). Query expansion is a distinct concern from retrieval.
- **`AgentGraph`** (`pipeline_lg.py:387-617`) — 230 lines covering ingestion, memory-collection naming, a memory cache with fingerprinting, tool registration, prompt assembly, graph construction, the public API, and a REPL. It is well-commented and internally coherent, but it is the natural place for the codebase's next split.

---

## 3. Data Flow

Traced concretely for `AgentGraph`, then for the two chat graphs and the legacy pipelines.

### 3a. `AgentGraph` — full trace

**Stage 0 — Startup (once, at construction).**
`AgentGraph.__init__` (`pipeline_lg.py:404`) →
`LoadDocs.__init__/invoke` (`ingestion/doc_loader.py:12`, `19`) → branches on `mode` → e.g. `PyPDFLoader(file_path).load()` (21-24) → `list[Document]` →
`Chunking.invoke` (`ingestion/chunking.py:19`) → `RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)` (25-30) → prints the chunk count (31) → chunked `list[Document]` →
`Retrival.__init__` (`retreival/retrieval.py:12`) → sanitizes `f"VDB for {file_path}"` via regex (17-21) → constructs `BuildKnowledge` (26) → `BuildKnowledge.__init__` loads `HuggingFaceEmbeddings("all-MiniLM-L6-v2")` (`ingestion/vector_db.py:10-12`) → `BuildKnowledge.invoke` calls `Chroma.from_documents(...)` (15-20) → embeds every chunk and writes it to `persist_directory` → returns the live store, assigned to `Retrival.vector_store` (27).
Back in `AgentGraph.__init__`: memory collection name computed (436-439), `_memory_cache` created (443), `make_search_current_doc(self.retrival)` closes over the `Retrival` instance (445), tool list assembled (446-450), system prompt formatted to match (454-457), `ChatOllama(...).bind_tools(self.tools)` (460), `MemorySaver()` (461), `_build_graph()` (462).

**Stage 1 — User input.**
`AgentGraph.invoke(query, config)` (`pipeline_lg.py:565`). Sets `recursion_limit=25` (570). Calls `self.get_history(thread_id)` (571) → `self.graph.get_state(...)` (599) — a checkpointer read. If the thread is new, a `SystemMessage(self.system_prompt)` is prepended (573-574). A `HumanMessage(query)` is appended (575). `self.graph.invoke({"messages": incoming}, config)` (576).

**Stage 2 — Turn counting.**
`_count_turn_node` (499-502): `{"turn_count": state.get("turn_count", 0) + 1}`. It sits between `START` and `agent` precisely so it runs once per user turn — counting inside `agent` would inflate, because the ReAct loop re-enters `agent` after every tool call (comment at 500-501).

**Stage 3 — Memory gate (conditional).**
`_memory_gate` (504-510) reads `turn_count`; guards `n <= 0` *before* the modulo; returns `"summarize"` when `turn >= n and turn % n == 0`, else `"agent"`.

**Stage 3a — Memory write + compress (only when the gate fires).**
`_summarize_user_history_node` (512-523) → extracts `thread_id` from the injected `RunnableConfig` (515) → `_build_user_memory(state["messages"], thread_id)` (516, defined 464-497) → filters `HumanMessage`s (472) → fingerprints `(len, last_content)` (475) → cache hit returns early (476-478) → otherwise `UserMemory(chat_history=humans, collection_name=..., persist_directory_path=...)` (491-495).
Inside `UserMemory.__init__` (`memory/long_term_memory.py:25-46`): human/AI split (29-30) → `_get_embedding_model()` singleton (31) → `Chroma(collection_name=..., embedding_function=..., persist_directory=...)` opened lazily (35-39) → **`add_documents(human_docs, ids=[sha1(content)])` — the vector write happens here, in the constructor** (43-46).
Back in the node: `memory.invoke({})` (`memory/long_term_memory.py:47-59`) → builds a fresh `Ollama(mistral)` (50) → `PromptTemplate | llm | StrOutputParser` over `"\n".join(all human message contents)` (52-58) → prose profile string → wrapped in try/except (`pipeline_lg.py:519-523`) → written to `state["user_history_summary"]`.

**Stage 4 — LLM call.**
`_agent_node` (537-544): if a summary exists, `_with_profile(messages, summary)` (525-535) builds a **new** list whose head is a fresh `SystemMessage` = original system prompt + `PROFILE_BLOCK_TEMPLATE.format(user_history=summary)` (372-378). A fresh message is used rather than a copy so it carries no `.id` — a copied id would let `add_messages` overwrite the stored prompt and compound the profile on every gate hit (comment at 528-530). Then `self.llm.invoke(messages)` (543) → blocking HTTP to Ollama → returns an `AIMessage`, possibly carrying `.tool_calls`. Returned as `{"messages": [response]}` and merged by the `add_messages` reducer.

**Stage 5 — Tool routing.**
`tools_condition` (`pipeline_lg.py:561`) inspects the last message's `.tool_calls`. Explicit path map `{"tools": "tools", END: END}` — without it, LangGraph declares every node a possible target and discards graph-shape validation (comment 559-560).

**Stage 6 — Tool execution.**
`ToolNode(self.tools)` (551) dispatches by name:
- `calculator` (`tools/tools.py:50-61`): `ast.parse(mode="eval")` → `_eval_ast` (36-47) walks a whitelist of 7 binops + 2 unaryops. No `eval`, no names, no calls. Returns `str(result)` or `"calculator error: ..."`.
- `search_current_doc` (`tools/tools.py:114-127`, closure over `retrival`): `retrival.invoke({"query", "k"})` → `Retrival.invoke` simple/mmr/mqr/wiki branch → `result.get("context", [])` → joined `page_content`.
- `wikipedia_lookup` (64-83): `import wikipedia` (70, **outside** the try), `set_user_agent`, up to 2 attempts constructing a `WikipediaRetriever` and calling `.invoke(query)`.
- `web_search` (86-104): `DuckDuckGoSearchAPIWrapper(max_results=k).results(...)`, formats `title / snippet / link` triples.
- `recall_user_memory` (141-176): pulls `messages` from `InjectedState`, `thread_id` from `RunnableConfig`, calls the injected `build_memory` callback → `AgentGraph._build_user_memory` → `UserMemory` (**which writes on construction, even on this read path**) → `memory.retreival(query, k)` (`memory/long_term_memory.py:60-65`) → `as_retriever(search_kwargs={"k":k}).invoke(query)` → joined `page_content`.

Each result becomes a `ToolMessage`; the edge `tools → agent` (562) loops back to Stage 4.

**Stage 7 — Response.**
`END` is reached when the model emits an `AIMessage` with no `tool_calls`. Back in `AgentGraph.invoke` (577-582), the last such message is found and `.content` returned as a plain `str`. `MemorySaver` has written a checkpoint after every super-step along the way.

### 3b. `ChatRagWithMemoryGraph` — trace

`invoke` (279-292) seeds `{"query", "k", "lambda_mult"}` →
`_retrieve_node` (208-218): `Retrival.invoke` → joins chunks → returns `{"context": ..., "turn_count": +1}` — note the counter lives in the *retrieve* node here, not a dedicated one →
`_memory_gate` (220-224): `turn >= 5 and turn % 5 == 0` →
`_summarize_user_history_node` (226-232) if fired — constructs `UserMemory` from **all** `state["messages"]` (not just humans; `UserMemory.__init__` filters internally) and calls `.invoke({})`. **No try/except here**, unlike the agent's version →
`_generate_node` (234-253): picks `memory_prompt` if `user_history_summary` is non-empty, else `base_prompt`; runs `prompt | Ollama(mistral) | StrOutputParser`; returns `{"messages": [HumanMessage(query), AIMessage(response)]}` →
`_trim_history_node` (255-260): if `len(messages) >= 10`, returns `RemoveMessage` for every message carrying an `.id`, wiping the thread's history →
`END`.

### 3c. `ChatRagGraph` — trace

`invoke` (128-141) → `_retrieve_node` (99-106) → `_generate_node` (108-117) → `END`. No memory, no trim, no tools.

### 3d. Legacy `pipeline.py` — trace

`ChatRagPipelineWithMemory.invoke` (159-232) is a single `while True`. Per iteration: `input()` (203) → **`Retrival.invoke` runs immediately (204), before the `quit`/`history` checks at 206-211** → so typing `quit` still performs a retrieval, and under `mqr` still burns an LLM call → `count += 1` (213) → turns 1-4 use `chain` (215), turn ≥ 5 uses `chain_2` with `{user_history}` (226), and on every 5th turn a fresh `UserMemory` is constructed and compressed (220-225) → messages appended (216-217 / 227-228) → `chat_history.clear()` at `len >= 10` (229-230).

---

## 4. Code Quality Review

### Breaks under load, repeated invocation, or concurrency

**A. `Chroma.from_documents` re-appends the entire document on every construction.** `[FIXED]`
`ingestion/vector_db.py:15-20`, reached unconditionally from `retreival/retrieval.py:26-27`. `from_documents` **adds**; it does not upsert or reconcile. With a `persist_directory` set, each process start re-embeds every chunk and appends a second (third, Nth) copy. After *m* runs, a `k=4` similarity search returns roughly `4/m` distinct chunks — effective context collapses while token cost stays flat. The codebase already knew the fix: `UserMemory` (`memory/long_term_memory.py:35-46`) opens lazily and upserts under content-hash ids. `BuildKnowledge` never got the same treatment.

> **Resolution.** `BuildKnowledge.invoke` now opens the collection with `Chroma(...)` and upserts under a `_chunk_id` fingerprint of *metadata + content* — metadata included so a header repeated on every page of a PDF stays as many distinct chunks rather than collapsing into one. Ids already present are skipped via a `get(ids=..., include=[])` probe, so a warm start costs one metadata read instead of re-embedding the corpus; if that probe fails it falls back to a plain upsert, which is slower but still correct. Measured on `test_pdf's/Pipes_2025.pdf` (48 chunks): collection count across three consecutive builds was `[4, 8, 12]` on the four-doc reproduction before the change and is `[48, 48, 48]` after, with the top-4 retrieval identical on builds 1 and 3.

**B. No embedding-model singleton in `BuildKnowledge`.**
`ingestion/vector_db.py:10-12` constructs `HuggingFaceEmbeddings` per instance; `ingestion/chunking.py:14-16` does the same in semantic mode. Only `memory/long_term_memory.py:14-22` caches. Every additional pipeline object loads another full MiniLM into RAM (~90 MB + torch overhead), and pays the ~3 s construction the memory module explicitly calls out.

**C. `AgentGraph._memory_cache` is unbounded and unsynchronized.**
`pipeline_lg.py:443`, written at 489 and 496, read at 476. Keyed by `thread_id`, never evicted — a long-lived process serving many conversations leaks one `UserMemory` (each holding a Chroma client handle) per thread forever. The read-modify-write is not lock-protected, so two threads on the same `thread_id` can each construct a `UserMemory` and double-write.

**D. `_EMBEDDING_MODEL` has a check-then-set race.**
`memory/long_term_memory.py:17-22`. Two threads entering simultaneously both see `None` and both construct. Benign in outcome (the model is stateless) but wastes ~3 s and a model's worth of RAM.

**E. `MemorySaver` is in-process, unbounded, and non-durable.**
`pipeline_lg.py:96`, `205`, `461`. Every checkpoint of every thread stays in RAM for the process lifetime, and all of it is lost on restart. `docs/tech_stack.md` §2 already specifies `SqliteSaver` as the target. Additionally, each graph object gets its *own* `MemorySaver`, so two `AgentGraph` instances over the same document share no history at all.

**F. `AgentGraph` never trims message history.**
There is no `trim_history` node in `_build_graph` (546-563) — a deliberate choice documented at `README.md:267-269`, because cutting between an `AIMessage`'s `tool_calls` and its matching `ToolMessage` orphans the pair and Ollama rejects it. The consequence is real: agent turns emit 3-4 messages each, so prompt size, latency, and cost grow linearly and unboundedly until the model's context window is exceeded.

**G. `UserMemory.invoke` sends the entire human-message history in one prompt.**
`memory/long_term_memory.py:58`. No windowing, no map-reduce, no incremental summarization. At turn 100 the compression prompt contains all 100 user messages. It also constructs a fresh `Ollama` client per call (50).

**H. `mqr` mode is 6 serial network calls per query with no dedup.**
`retreival/retrieval.py:66-82`: one LLM call to generate five queries, then five sequential `retriever.invoke` calls, results flat-`extend`ed. Latency is the sum; the result list has duplicates that then get joined verbatim into the prompt.

**I. Blocking `input()` REPLs inside library classes.**
`pipeline.py:117, 203, 349`; `pipeline_lg.py:148, 298, 588`. These block the calling thread and make the classes unusable from any server, scheduler, or async context.

**J. The memory gate injects a synchronous multi-second LLM call into the request path.**
`pipeline_lg.py:512-523` and `226-232`. Turn 5, 10, 15… each carry an extra full compression round-trip before the user's answer starts. In text this is a visible pause; in voice (§10) it is a dead-air stall.

**K. Network tools have no timeout, backoff, or rate-limit handling.**
`tools/tools.py:73-83` retries Wikipedia exactly twice by reconstructing the retriever, with no delay. `tools/tools.py:91-104` calls DuckDuckGo with no retry at all — DDG rate-limits aggressively under repeated agent use, and the failure surfaces to the model as the string `"web_search error: ..."`, which it may then narrate to the user.

**L. Chat graphs retrieve unconditionally.**
`pipeline_lg.py:99-106` and `208-218` run a vector search for every turn including `"hi"`. Under `mqr` that is 6 model/store calls to answer a greeting.

### Missing error handling in critical paths

| Location | Problem |
|---|---|
| `ingestion/doc_loader.py:19-79` | No `else` branch. An unrecognized `mode` falls off the end and returns `None`. The failure then surfaces in `Chunking.invoke` (`ingestion/chunking.py:22-23`) as `"Input must be a list of Document objects"` — a misleading message that points at the wrong module. |
| `retreival/retrieval.py:28-90` | Same shape: no `else`, returns `None` for an unknown mode. The caller at `pipeline.py:41` then does `retrival_dict["context"]` → `TypeError: 'NoneType' object is not subscriptable`. |
| `retreival/retrieval.py:67-68` | `mqr`'s `PydanticOutputParser` raises `OutputParserException` whenever mistral emits prose around the JSON — unhandled, kills the turn. The prompt (34-42) works hard to prevent this but has no fallback. |
| `retreival/retrieval.py:26-27` | Chroma / embedding / disk failures propagate out of `Retrival.__init__`, i.e. out of every pipeline and graph constructor, leaving a half-built object. |
| `pipeline.py:66, 128, 215, 226` | No try/except around any Ollama call. A dropped connection terminates the REPL loop and discards `chat_history`. |
| `pipeline_lg.py:226-232` | `ChatRagWithMemoryGraph._summarize_user_history_node` has **no** try/except, while the structurally identical `AgentGraph._summarize_user_history_node` (519-523) wraps `memory.invoke` and comments *"a memory failure must never destroy the user's turn"*. The right lesson was learned in one class and not backported to the other. |
| `tools/tools.py:70` | `import wikipedia` sits **outside** the `try` at 74. If the package is missing, `ImportError` escapes the tool's own error handling and propagates through `ToolNode`. |
| `pipeline_lg.py:141, 292, 582` | All three `invoke` methods return `""` when no matching `AIMessage` is found. A silent empty string is indistinguishable from a legitimately empty answer. |
| `memory/long_term_memory.py:58` | With an empty history, `"\n".join([])` is `""` — the model is asked to compress nothing and will invent a profile. |

### Hardcoded values that should be configurable

`docs/todo.md` §0 already lists the first four. Full inventory:

- `chunk_size=500, chunk_overlap=50` — `pipeline.py:31, 88, 156, 267`; `pipeline_lg.py:81, 184, 421`. Each carries the comment `# here arguments are hardcoded`.
- Model `"mistral"`, `temperature=0` — `pipeline.py:64, 112, 196, 342`; `pipeline_lg.py:89, 193`; `retreival/retrieval.py:22`; `memory/long_term_memory.py:50`. Seven sites.
- Embedding model name `"sentence-transformers/all-MiniLM-L6-v2"` — `ingestion/vector_db.py:11`, `ingestion/chunking.py:15`, `memory/long_term_memory.py:20`.
- Memory cadence and trim threshold — `pipeline.py:214, 219, 220, 229`; `pipeline_lg.py:222, 257`. `AgentGraph` correctly parameterizes this as `memory_every_n_turns` (414); the other four call sites do not.
- `recursion_limit=25` — `pipeline_lg.py:570`.
- `k=4`, `lambda_mult=0.5` defaults — roughly eight sites across `pipeline.py`, `pipeline_lg.py`, `tools/tools.py:115`.
- Collection name `f"VDB for {file_path}"` — `pipeline.py:33, 90, 158, 269`; `pipeline_lg.py:86, 189, 426`.
- `thread_id` fallback `"default"` — ten sites.
- Semantic-chunk breakpoint percentile `90` — `ingestion/chunking.py:36`.
- Wikipedia user-agent string — `tools/tools.py:71`.
- Prompt templates: hoisted to module constants in `pipeline_lg.py:36-63, 330-378` (good) but still inline inside `.invoke()` in `pipeline.py:42-62, 96-111, 163-195, 274-341` (bad).
- `.env` holds four keys (`OPENAI_API_KEY`, `GEMINI_API_KEY`, `exchange_rate_api_key`, `claude_api`) and **no code anywhere reads it** — there is no `dotenv`, `os.environ`, or `os.getenv` reference in the entire repository.

### Dead code and unused imports

- `pipeline.py:3` — `BuildKnowledge` imported, never used.
- `pipeline.py:39, 95, 162, 273` — `language` is read from the input dict in all four pipelines and never used. Worse, it is never forwarded to `Retrival.invoke`, whose `wiki` branch (`retreival/retrieval.py:84`) is the only consumer — so `mode_retreiver="wiki"` can never be non-English through any pipeline.
- `ingestion/doc_loader.py:4` — `SeleniumURLLoader` unused. `:9` — `BaseModel, Field` unused.
- `ingestion/chunking.py:1` — `Language` unused. `:6` — `Document` unused.
- `pipeline.py:92` — `ChatRagPipeline.chat_history` is a *local* variable, so history is per-`invoke` rather than per-object; `ChatRagPipelineWithMemory` correctly makes it `self.chat_history` (149). Inconsistent between two adjacent classes.
- `docs/coding_standards.md` and `docs/decisions.md` are 0 bytes.
- `db/user_history_test/` is a stale test artifact on disk (gitignored, harmless, but noise).

**`Structure_pipeline` (`pipeline.py:245-378`) is broken.** Its first template declares the variables `{transcript}` and `MessagesPlaceholder(variable_name="Chat_history")` (302, 309), but `chain.invoke` at 361 passes `query`, `context`, and `chat_history_placeholder`. Neither name matches, so turns 1-4 raise a missing-input-variable error. (Turns ≥ 5 use `template_with_user_history`, whose variables *do* match.) I am flagging this only — the class is a known placeholder and this review makes no changes.

### Inconsistent patterns across similar modules

1. **Two different `HuggingFaceEmbeddings`.** `ingestion/vector_db.py:1` imports from `langchain_huggingface.embeddings`; `ingestion/chunking.py:3` and `memory/long_term_memory.py:6` import from the deprecated `langchain_community.embeddings`. Same model, two class objects, two loads.
2. **Two different Ollama clients.** Deprecated `langchain_community.llms.Ollama` in `pipeline.py`, `pipeline_lg.py:11`, `retreival/retrieval.py:6`, `memory/long_term_memory.py:7`; modern `langchain_ollama.ChatOllama` only at `pipeline_lg.py:417`, function-local, because it is the one that supports `.bind_tools`. The notebook output in `labbb.ipynb` already shows the `LangChainDeprecationWarning` firing.
3. **`Runnable` subclassing is inconsistent.** All four `pipeline.py` classes and all five lower-layer classes subclass `Runnable`; none of the three `pipeline_lg.py` graph classes do. The two generations are not interchangeable in a LangChain chain.
4. **Error-handling discipline diverges between the two memory-gate implementations** (see the table above).
5. **`__init__.py` conventions differ.** `ingestion/`, `retreival/`, `memory/` are 0-byte; `tools/__init__.py` re-exports five symbols with `__all__`.
6. **`print()` is used as an output channel** in library code: `ingestion/chunking.py:31, 40`; `pipeline_lg.py:323, 616`; plus every REPL.
7. **Typo'd names are now public API:** the package `retreival/` and the method `UserMemory.retreival` (`memory/long_term_memory.py:60`, called from `tools/tools.py:171`). Both are load-bearing.

### Opportunities for meaningful abstraction

These are the three that pay for themselves; the rest of the codebase does *not* need more indirection.

1. **A `KnowledgeBase` / index-lifecycle object** that separates "build or open an index" from "query an index" — this is the one abstraction that fixes the duplicate-write bug (A) rather than papering over it, and it simultaneously removes the 7-way duplicated constructor block.
2. **A single config/settings object** (`docs/todo.md` §0 already calls for `config/settings.py`) carrying chunk sizes, model names, embedding name, k/lambda defaults, memory cadence, and paths. This collapses ~40 hardcoded literals into one place.
3. **A shared `BaseRagGraph`** holding `get_history`, `run_chat`, the last-`AIMessage` extraction, and checkpointer wiring — removing 3-6 copies each of four methods in `pipeline_lg.py`.

---

## 5. LangGraph Review

### `ChatRagGraph` (`pipeline_lg.py:66-163`)

**Nodes**

| Node | Function | Responsibility | State written |
|---|---|---|---|
| `retrieve` | `_retrieve_node` (99-106) | Calls `Retrival.invoke` with `k`/`lambda_mult`/`query` from state; joins `page_content` with `\n\n`. | `context` |
| `generate` | `_generate_node` (108-117) | `self.prompt \| self.llm \| self.parser` over `context`, `query`, `messages`. | `messages` (appends `HumanMessage` + `AIMessage`) |

**Edges:** `START → retrieve → generate → END` (123-125). Purely linear; no conditionals.

**Note:** `_generate_node` appends the `HumanMessage` *after* generation, alongside the answer. The prompt itself renders the current query through its trailing `("human", "{query}")` slot (94), so the query is never double-counted within a turn.

### `ChatRagWithMemoryGraph` (`pipeline_lg.py:166-324`)

**Nodes**

| Node | Function | Responsibility | State written |
|---|---|---|---|
| `retrieve` | `_retrieve_node` (208-218) | Retrieval **and** turn counting. | `context`, `turn_count` |
| `summarize_user_history` | `_summarize_user_history_node` (226-232) | Builds a `UserMemory` from all `state["messages"]` and compresses it. Unguarded. | `user_history_summary` |
| `generate` | `_generate_node` (234-253) | Selects `memory_prompt` when a summary exists, else `base_prompt`. | `messages` |
| `trim_history` | `_trim_history_node` (255-260) | Emits `RemoveMessage` for every message with an `.id` once `len >= 10`. | `messages` (deletions) |

**Edges:** `START → retrieve` (268); conditional `retrieve → {summarize_user_history | generate}` via `_memory_gate` with an explicit path map (269-273); `summarize_user_history → generate` (274); `generate → trim_history` (275); `trim_history → END` (276).

**Routing logic:** `_memory_gate` (220-224) returns `"summarize"` when `turn >= 5 and turn % 5 == 0`. The `turn >= 5` clause exists to stop turn 0 from firing. It reads `state["turn_count"]` by subscript rather than `.get` — safe only because `retrieve` always writes it first.

**Two behavioral notes.** First, `user_history_summary` is *sticky*: once written it is never cleared, so from turn 5 onward the memory prompt is used on every turn, with a summary that is up to four turns stale. Second, `trim_history` wiping all messages at `len >= 10` interacts badly with the every-5 gate — see §6.

### `AgentGraph` (`pipeline_lg.py:387-617`)

**Nodes**

| Node | Function | Responsibility | State written |
|---|---|---|---|
| `count_turn` | `_count_turn_node` (499-502) | Increments `turn_count`. Deliberately its own node so it runs exactly once per `invoke`, not once per ReAct iteration (comment 500-501). | `turn_count` |
| `summarize_user_history` | `_summarize_user_history_node` (512-523) | Extracts `thread_id` from `RunnableConfig`, funnels through `_build_user_memory`, compresses. Wrapped in try/except. | `user_history_summary` |
| `agent` | `_agent_node` (537-544) | Composes the profile-augmented message list per call, invokes the tool-bound `ChatOllama`. | `messages` |
| `tools` | `ToolNode(self.tools)` (551) | Executes tool calls, emits `ToolMessage`s. | `messages` |

**Edges:** `START → count_turn` (552); conditional `count_turn → {summarize_user_history | agent}` via `_memory_gate` (553-557); `summarize_user_history → agent` (558); conditional `agent → {tools | END}` via `tools_condition` with an explicit path map (561); `tools → agent` (562).

**Two edge-logic details are load-bearing and correct:**
- The explicit path map on `tools_condition` (561). Without it LangGraph treats every node as a possible target, discarding graph-shape validation and drawing phantom edges in `get_graph()` (comment 559-560).
- `_memory_gate` (504-510) guards `if not n or n <= 0` **before** the modulo, so `memory_every_n_turns=0` disables the gate rather than raising `ZeroDivisionError`.

### State schema

**`ChatState`** (`pipeline_lg.py:26-33`)

| Field | Reducer | Carries |
|---|---|---|
| `messages` | `add_messages` | Full conversation; append-with-id-merge; honors `RemoveMessage`. |
| `query` | overwrite | Current user query — a per-turn *input* stored in durable state. |
| `context` | overwrite | Joined retrieved chunks — per-turn scratch, but checkpointed and persisted. |
| `user_history_summary` | overwrite | Compressed profile. Sticky once set. |
| `turn_count` | overwrite | User turns so far. |
| `k`, `lambda_mult` | overwrite | Retrieval params — configuration stored in state. |

**`AgentState`** (`pipeline_lg.py:381-385`) carries only `messages`, `turn_count`, `user_history_summary`. **It has no `context` field** — the agent reaches the document exclusively through `search_current_doc`. This is the cleaner of the two schemas: nothing per-turn is persisted, so there is no stale scratch data to invalidate.

`ChatState` mixes three categories that should be separated: durable conversation (`messages`, `turn_count`, `user_history_summary`), per-turn scratch (`query`, `context`), and configuration (`k`, `lambda_mult`). Only the first belongs in a checkpointed state; the third belongs in `config["configurable"]`.

### Checkpoint behavior

- A fresh `MemorySaver()` per graph object — `pipeline_lg.py:96`, `205`, `461`. Consequences: **(a)** nothing survives process restart; **(b)** two graph objects over the same document share nothing; **(c)** every checkpoint of every thread is retained in RAM for the process lifetime, with no TTL or cap.
- Checkpoints are written after every super-step, so a turn with three tool calls writes many snapshots, all retained.
- `graph.get_state` is called on every `invoke` in `AgentGraph` (571) plus in `get_history` (598-600), `get_turn_count` (602-604), and `get_user_history_summary` (606-608). Cheap against `MemorySaver`; each becomes a real I/O round-trip under `SqliteSaver`.
- **No `interrupt()` anywhere in the repository** — confirmed by grep. `docs/roadmap.md` Phase 2 and `docs/vision.md` both make native Human-in-the-Loop a headline V2 feature; none of the scaffolding exists yet.
- `docs/tech_stack.md` §2 specifies `SqliteSaver`; the code uses `MemorySaver`. A known, documented gap.

### How tool execution integrates with the graph

The integration is textbook-correct and the ordering constraints are real:

1. `self.tools` must be complete before `bind_tools` (`pipeline_lg.py:446-450` then `460`) — the comment at 459 says so explicitly.
2. `ToolNode(self.tools)` (551) reads the same list, so the executor and the model's schema can never drift.
3. `tools_condition` (561) is LangGraph's prebuilt predicate over the last `AIMessage.tool_calls`.
4. `tools → agent` (562) closes the ReAct loop; `recursion_limit=25` (570) bounds it.
5. `recall_user_memory` uses `Annotated[dict, InjectedState]` and `RunnableConfig` (`tools/tools.py:144-145`) so `ToolNode` populates `state` and `config` while hiding both from the model's tool schema — the LLM only ever sees `query` and `k`.
6. `AgentGraph.invoke` (577-580) filters for an `AIMessage` **without** `tool_calls`, so a tool-call stub is never returned as the user-visible answer.
7. The system prompt is generated from the same conditional as the tool list (454-457), enforcing the invariant that the prompt describes exactly the bound tools.

**One gap:** `extra_tools` (412, 449) appends tools to the bound list but never touches the prompt. A tool passed that way is bound and executable but undescribed — which is precisely the failure mode the comments at 328-329 and 452-453 warn about, in reverse.

### How memory reads and writes interact with graph state

- **Write path.** The Chroma write is a **side effect of `UserMemory.__init__`** (`memory/long_term_memory.py:40-46`), not of `.invoke()`. So *constructing* a `UserMemory` persists data. Both the passive gate (`pipeline_lg.py:516`) and the active recall tool (`tools/tools.py:168`) route through `_build_user_memory`, which means **the read path writes too**. The `_memory_cache` fingerprint (475) makes this a no-op when nothing has changed, and content-hash ids (`memory/long_term_memory.py:45`) make repeats idempotent — so it is safe, but it is surprising.
- **Passive read.** `state["user_history_summary"]` → `_agent_node` (539) → `_with_profile` (525-535) builds a **new** `SystemMessage` per call and never returns it in a `{"messages": ...}` update. That is the crucial detail: a copied message would keep its `.id`, `add_messages` would overwrite the stored system prompt, and the profile block would compound on every gate hit. Composing per-call also means the profile survives post-tool re-entry into `agent` (comment 541).
- **Active read.** `recall_user_memory` → `build_memory(...)` → `UserMemory.retreival(query, k)` (`memory/long_term_memory.py:60-65`). Its result enters state as a `ToolMessage` like any other tool output — it does not touch `user_history_summary`.
- **Scope boundary.** Graph state is thread-scoped (via `thread_id`); the Chroma memory collection is user-scoped (`pipeline_lg.py:436-439`). This is what lets thread `"tue"` answer *"what is my name?"* from what thread `"mon"` was told, as demonstrated in `README.md:251-255`.

---

## 6. Memory Review

### Design and data model

Two layers, per `docs/memory.md`:

- **Short-term working memory** — LangGraph `AgentState`/`ChatState`, thread-scoped, in `MemorySaver`.
- **Long-term user memory** — a ChromaDB collection of the user's own past messages, user-scoped, on disk.

The long-term data model is deliberately minimal and **carries no metadata at all**. `memory/long_term_memory.py:32`:

```python
human_docs = [Document(page_content=msg.content) for msg in self.human_messages]
```

No timestamp, no `thread_id`, no `user_id`, no turn index, no source. Only `HumanMessage`s are stored — `self.ai_messages` (30) is computed and then never used anywhere in the codebase. Document ids are `sha1(page_content)` (45).

The consequences follow directly from the absent metadata:
- **No expiry or decay.** Nothing can be aged out, because nothing records when it was said.
- **No filtered retrieval.** You cannot scope a search to one conversation, one time range, or one user within a collection.
- **No ordering.** "What did I tell you most recently?" is unanswerable.
- **No attribution.** A recalled snippet cannot be traced back to its conversation.
- **Cross-identity collision.** Content-hash ids mean two different users who both type `"my name is Harshal"` produce one row. Separation exists only via the `user_id`-derived *collection name* (`pipeline_lg.py:436`), not within a collection.
- **No update or correction path.** If the user says `"I moved to Mumbai"`, the old `"I live in Pune"` row stays forever with equal weight. Memory is append-only with no supersession.

### Persistent storage mechanism

`Chroma(collection_name=..., embedding_function=..., persist_directory=...)` opened lazily at `memory/long_term_memory.py:35-39`, then `add_documents(..., ids=[sha1(...)])` at 43-46. The two comments there (33-34, 41-42) capture the reasoning precisely: lazy open so an empty history doesn't crash and a persisted collection reopens across sessions; content-hash ids so `add_documents` upserts and re-indexing a growing conversation doesn't duplicate earlier messages.

`AgentGraph` sanitizes the collection name before passing it (`pipeline_lg.py:437-439`), because — as its comment says — `UserMemory` itself does no sanitization and Chroma rejects several characters. `ChatRagWithMemoryGraph` passes no `collection_name` at all and relies on the `"user_history"` default (`memory/long_term_memory.py:25`).

### How memories are written, retrieved, and updated

- **Written:** in `UserMemory.__init__` (40-46), as a constructor side effect, from whatever `chat_history` is handed in. `AgentGraph._build_user_memory` (472) pre-filters to human messages; `ChatRagWithMemoryGraph` (228) passes all messages and lets `UserMemory` filter internally (29).
- **Retrieved, two ways:**
  - *Compression* — `UserMemory.invoke({})` (47-59): a single mistral call over `"\n".join(all human contents)` producing a prose profile. Used by the passive gate.
  - *Semantic search* — `UserMemory.retreival(query, k)` (60-65): plain top-k similarity over the collection, joined with `\n\n`. Used by `recall_user_memory`.
- **Updated:** never, in the sense of revision. New messages are appended; identical ones are deduplicated by hash; nothing is edited or deleted. There is no purge routine, despite `docs/security.md` §5 calling for `python main.py --clear-memory`.

### Summarization / compression

One prompt, `memory/long_term_memory.py:52-56`:

> *"You are an ai assistant that would be given an user's chat_history just compress it and keep the personal things and relations about the user"*

Characteristics: **unbounded input** (every human message ever, in one prompt); **non-incremental** (recomputed from scratch on every gate hit, never a summary-of-summaries); **unstructured output** (free prose, no schema — contrast `Retrival`'s `mqr` branch, which does use a `PydanticOutputParser`); and **not persisted** — the summary exists only in `state["user_history_summary"]`, so with `MemorySaver` it dies on restart while the raw messages survive in Chroma. Every restart re-pays the full compression cost.

The `AgentGraph` short-circuit is the `_memory_cache` fingerprint (`pipeline_lg.py:475-478`), which avoids rebuilding `UserMemory` when nothing changed — but it caches the *object*, not the *summary*, so `memory.invoke({})` still runs a full LLM call on every gate hit.

### Limitations and failure modes

1. **Silent non-durability when `persist_directory_path=None`.** `[FIXED]` `memory/long_term_memory.py:38` passed `None` straight to Chroma, producing an **in-memory** store. Both `ChatRagWithMemoryGraph` (181) and `AgentGraph` (413) default `user_history_directory_path` to `None`. Memory then appeared to work perfectly within a session and vanished on exit, with no warning. `AgentGraph` guards this **only in the empty-humans branch** (482-483) — the main path (491-495) constructed a non-persistent `UserMemory` without complaint. This was the single most misleading behavior in the system.

   > **Resolution.** `UserMemory.__init__` now raises a `UserWarning` naming the collection and stating that anything remembered is lost at process exit. It warns rather than raises because both graphs default the path to `None`, so raising would break every current caller including `labbb.ipynb`; and the defaults were left as `None` rather than pointed at a directory, because writing user data to disk unbidden is the worse failure. `stacklevel=2` puts the warning on the caller's line, and Python's default filter shows it once per process rather than on every gate hit. The behaviour is unchanged — it is now merely audible.

2. **The V1 clear-at-10 / gate-at-5 interaction destroys most of the profile.** In `pipeline.py:159-232`, trace the turn counter:
   - Turns 1-4: history grows to 8 messages.
   - Turn 5: gate fires and compresses those 8 messages; then 2 more are appended → 10 → **`chat_history.clear()`** (229-230).
   - Turns 6-9: history rebuilds to 8 messages; `user_history` is the stale turn-5 summary.
   - Turn 10: gate fires over only turns 6-9; append → 10 → cleared again.

   The profile is therefore always derived from at most the last four turns, and everything before the previous clear is permanently invisible to compression. `ChatRagWithMemoryGraph` inherits the same pattern via `_trim_history_node` (255-260) — though the Chroma write in `UserMemory.__init__` does capture the messages before each clear, so the *searchable* memory is intact even when the *compressed* one is not. Note this affects only the two chat pipelines; `AgentGraph` never trims (§4-F) and so does not suffer it.

3. **`ChatRagWithMemoryGraph` never persists any memory unless the caller supplies a path** and, unlike `AgentGraph`, does not sanitize the collection name — a caller-supplied path plus the default name is the only configuration it accepts.

4. **Unguarded LLM failure in `ChatRagWithMemoryGraph`** (226-232): if Ollama is down on turn 5, the entire turn raises. `AgentGraph` (519-523) handles this correctly.

5. **Empty-history hallucination.** `"\n".join([])` is `""` (58); the model is asked to compress nothing and will confabulate a profile, which is then injected into the system prompt as fact.

6. **`user_history_summary` is sticky and stale.** Once set it is never cleared or invalidated; between gate hits the agent personalizes against a summary up to `n-1` turns old.

7. **Memory is written on the read path** (§5), which is safe today only because of the fingerprint cache and hash ids.

8. **No PII controls.** Raw user utterances are stored verbatim, unencrypted, with no masking — against `docs/security.md` §5 (encryption at rest) and §7 (PII masking).

---

## 7. Tool System Review

### How tools are defined

Two categories, both in `tools/tools.py`:

**Static tools** — module-level `@tool`-decorated functions whose schema is inferred from type hints and whose description is the docstring:
- `calculator(expression: str) -> str` (50-61). AST-whitelist evaluation via `_eval_ast` (36-47) over 7 binops (21-29) and 2 unaryops (30-33). No `eval`, no name resolution, no calls, no imports — compliant with `docs/security.md` §4's prohibition on `eval`/`exec`.
- `wikipedia_lookup(query: str, k: int = 3) -> str` (64-83).
- `web_search(query: str, k: int = 5) -> str` (86-104).

**Factory tools** — closures built at agent-init because they must capture live agent state:
- `make_search_current_doc(retrival) -> BaseTool` (107-129) closes over the `Retrival` instance.
- `make_recall_user_memory(build_memory) -> BaseTool` (132-178) closes over a callback, and additionally uses `Annotated[dict, InjectedState]` and `RunnableConfig` (144-145) for runtime-injected, model-invisible parameters.

**The docstrings are the interface.** `recall_user_memory`'s docstring (146-164) is 19 lines of behavioral specification — when to call, four explicit *do not* cases, and what to do with the result. This is not documentation; it is prompt engineering, and it is load-bearing on an 8B model. Any edit to it changes routing behavior.

**Metadata:** none beyond name/description/args-schema. No categories, no cost or latency hints, no permission or side-effect flags, no versioning, no timeouts. All five tools return `str` and swallow their own exceptions into `"<toolname> error: {e}"` strings (the sole exception being `import wikipedia` at line 70, which sits outside the try).

### How tools are registered

**Manual, in three files.** `tools/tools.py` defines → `tools/__init__.py` re-exports (two lists: the import at 1-7 and `__all__` at 9-15) → `pipeline_lg.py:17-23` imports → `pipeline_lg.py:446-450` assembles:

```python
search_current_doc = make_search_current_doc(self.retrival)
self.tools = [calculator, wikipedia_lookup, web_search, search_current_doc]
if enable_recall_tool:
    self.tools.append(make_recall_user_memory(self._build_user_memory))
if extra_tools:
    self.tools.extend(extra_tools)
```

There is no registry, no decorator-based discovery, no plugin scan. The list is a literal.

**The prompt is a fourth registration surface.** `AGENT_SYSTEM_PROMPT_TEMPLATE` (340-364) enumerates the tools in prose, and `_RECALL_TOOL_ENTRY`/`_RECALL_GUIDELINES` (330-338) are spliced in only when that tool is actually bound (454-457). The invariant — *the prompt must describe exactly the bound tools* — is enforced by hand at one site, with the comment at 452-453 explaining why it matters: told it has a tool it lacks, llama3.1 writes the call out as prose JSON instead of emitting a real `tool_call`.

### How tools are invoked

Exclusively through LangGraph. `ChatOllama(...).bind_tools(self.tools)` (460) gives the model the schemas; `ToolNode(self.tools)` (551) executes; `tools_condition` (561) routes; `tools → agent` (562) loops results back. No direct calls, no manual dispatch, no fallback path. `pipeline.py` has no tool layer at all.

Tool *results* are always `ToolMessage`s carrying a string. Errors are strings too, so the model sees `"web_search error: Ratelimit"` as ordinary tool output and may narrate it — mitigated by the prompt rule at 361-363 forbidding Marshall from mentioning its own machinery.

### Adding a new tool today — files to touch

| # | File | Edit |
|---|---|---|
| 1 | `tools/tools.py` | Define the `@tool` function (or factory). |
| 2 | `tools/__init__.py` | Add to the import list **and** to `__all__` (two edits). |
| 3 | `pipeline_lg.py:17-23` | Add to the import. |
| 4 | `pipeline_lg.py:446-450` | Append to `self.tools` — and add an `enable_*` constructor flag if optional. |
| 5 | `pipeline_lg.py:340-364` | Add a prose bullet to the system prompt; if optional, add a `_XXX_TOOL_ENTRY` constant + a `.format()` placeholder + a conditional at 454-457, mirroring the recall pattern. |
| 6 | `README.md:203-211` | Add a row to the tools table. |

**Three files, six-plus edit sites, and one of them (5) is a correctness requirement rather than documentation.**

### Friction points for extending the tool system

1. **The prompt-and-binding invariant is manually maintained.** Nothing detects drift. A tool bound but undescribed is exactly the failure the code warns about — and `extra_tools` (412, 449) produces that state by construction. Anything passed via `extra_tools` is bound, callable, and invisible to the prompt.
2. **Optional tools require a bespoke splice pattern.** The `_RECALL_TOOL_ENTRY`/`_RECALL_GUIDELINES`/`{recall_tool_entry}`/`{recall_guidelines}` mechanism is four coordinated pieces per optional tool. It does not generalize — a second optional tool means a second quadruple.
3. **`AGENT_SYSTEM_PROMPT` (367-370) is a module constant computed at import time with recall enabled.** It is exported "for callers/tests" but is *not* what a `enable_recall_tool=False` agent uses (which computes its own at 454-457). A test asserting against the constant tests the wrong prompt.
4. **Factory tools cannot be registered statically** — they need agent state, so they can only be constructed inside `AgentGraph.__init__`, which is why the tool list can't move to `tools/`.
5. **No shared tool infrastructure.** Each tool reimplements its own try/except-to-string, its own retry (`wikipedia_lookup` retries twice; `web_search` doesn't), its own result formatting. No timeouts anywhere, no output-size cap — `wikipedia_lookup` with `k=3` can return tens of thousands of characters straight into the context.
6. **`recall_user_memory` is not unit-testable in isolation** — `InjectedState` means it only works inside a `ToolNode`.
7. **Only `AgentGraph` has tools.** The two chat graphs and all four `pipeline.py` classes cannot use them.
8. **The tool count itself has a measured cost.** `README.md:261-265` records that adding the fifth tool costs roughly one spurious call per four conversational turns on llama3.1. Tool-set growth is not free on an 8B model, and the codebase is the rare one that has actually measured it.

---

## 8. Technical Debt

### CRITICAL

**C1. Every construction re-embeds and re-appends the whole document.** `[FIXED]`
`ingestion/vector_db.py:15-20` (`Chroma.from_documents`), reached from `retreival/retrieval.py:26-27`, reached from all seven top-level constructors.
*Risk:* with a persist directory, the collection grew by one full copy per process start. Retrieval degraded toward returning k near-identical chunks; disk and embedding time grew without bound. Silent — nothing errored, answers just got worse.
*Resolved by* lazy open + metadata-and-content hash ids + a warm-start skip probe in `BuildKnowledge.invoke`. See §4-A for the measurement. **Note for anyone with an existing `db/`:** collections built before this change still contain their accumulated duplicates — the fix stops new ones, it does not retroactively clean old ones. Delete and rebuild any collection that predates it.

**C2. Long-term memory is silently non-durable by default.** `[FIXED]`
`memory/long_term_memory.py:35-39`; defaults at `pipeline_lg.py:181` and `413`.
*Risk:* `persist_directory=None` yields an in-memory Chroma. Memory worked flawlessly in a session and was gone on exit. A user testing the feature saw it work; a user relying on it lost everything.
*Resolved by* a `UserWarning` at `UserMemory.__init__`. See §6-1. The ephemeral behaviour itself is unchanged and still the default — this makes it visible, it does not make it durable. Callers who want durable recall must still pass `user_history_directory_path`.

**C3. No dependency manifest.** `[FIXED]`
No `requirements.txt`, `pyproject.toml`, `setup.py`, `Pipfile`, or lock file anywhere. Install was unpinned `pip install` lines at `README.md:77-83`.
*Risk:* the environment was unreproducible. This repository already had a LangGraph version crisis — `.claude/settings.local.json` records hand-pinning through `langgraph==0.3.34` and then `1.1.2`, plus `langgraph-checkpoint`, `langgraph-prebuilt`, `langgraph-sdk`, and `langchain-core` bounds. That knowledge lived only in a permissions allowlist. Directly contradicted `docs/security.md` §6 ("explicitly pin exact versions") and `docs/tech_stack.md` §5.

> **Resolution.** `requirements.txt` pins 21 packages with `==`, verified against the running environment and by a clean `pip install --dry-run` (exit 0, no conflicts, nothing new to install) on CPython 3.12.9.
>
> The scope decision matters: the file pins **the entire langchain/langgraph family including its transitive members** — `langgraph-checkpoint`, `langgraph-sdk`, `ormsgpack` — none of which the code imports directly. A direct-imports-only manifest would not have prevented the original crisis, because that crisis *was* those packages drifting against each other and against `langchain-core`. Everything outside that family is left to resolve normally.
>
> **`torch` is deliberately not pinned.** The working machine runs `torch==2.6.0+cu124`; that local version identifier does not exist on PyPI, so pinning it would make `pip install -r` fail outright, while pinning plain `torch==2.6.0` would resolve to the CPU wheel and silently replace a working GPU install. It arrives transitively via `sentence-transformers`, so leaving it unpinned is the only option safe in both directions. The exact CUDA build and its `--index-url` are documented in the file.

### HIGH

**H1. Untrusted content is concatenated into the system prompt.** `[FIXED in pipeline_lg.py; still present in pipeline.py]`
`pipeline_lg.py:36-48, 50-63` interpolated retrieved `{context}` into the **system** message; `_with_profile` (532-535) concatenated the LLM-generated user profile onto it. `pipeline.py:42-62, 96-111, 163-195, 274-341` do the same.
*Risk:* `docs/security.md` §4 states the rule this violates verbatim — *"Pass untrusted input exclusively inside designated variable slots within isolated system prompts—never concatenate raw web data into system instructions."* `LoadDocs` ingests web pages, YouTube transcripts, and arbitrary PDFs. Any of them can carry instructions that arrive with system-message authority. This becomes acute the moment `docs/roadmap.md` Phases 3-5 land Gmail, browser, and scraper tools with real side effects.

> **Resolution (LangGraph generation).** The system message is now instructions only. Retrieved material and the user profile travel in the **human** turn, fenced in `<retrieved_context>` and `<user_profile>` tags, and a new `_UNTRUSTED_DATA_RULE` — which does sit in the trusted position — declares tagged blocks to be data that cannot alter the rules. The chat prompts gained `BASE_HUMAN_TEMPLATE` / `MEMORY_HUMAN_TEMPLATE` for this; no node signature changed, since both slots are filled from the same invoke dict.
>
> `AgentGraph` needed less: its tool results were already correctly positioned as `ToolMessage`s, so the profile in `_with_profile` was its only exposed surface. That is now fenced, and the agent prompt gained a matching data rule.
>
> Verified by a live injection attempt, not by inspection alone: a document instructing the model to reply only `ZZPWNEDZZ` was ingested and queried, and the model returned the document's actual content instead. Structural assertions confirm the context appears in the human turn and never in the system message, and that the fenced block does not accumulate into stored history — `_generate_node` still records the bare query.
>
> **Still open in `pipeline.py`.** All four legacy classes retain the original pattern. Left untouched deliberately: that file holds uncommitted work in progress and `Structure_pipeline` is a known placeholder. Fix it there if the LangChain generation is kept — see R9 on deciding its fate first.

**H2. Unbounded context growth in `AgentGraph` plus an unbounded in-RAM checkpointer.**
No trim node in `_build_graph` (546-563), by design (`README.md:267-269`); `MemorySaver` at 461.
*Risk:* prompt size, latency, and token cost grow linearly per turn until the context window is exceeded and the turn hard-fails. A long-running process additionally retains every checkpoint of every thread in RAM.

**H3. Two critical-path functions return `None` on an unknown mode.**
`ingestion/doc_loader.py:19-79` and `retreival/retrieval.py:28-90` both fall off the end with no `else`.
*Risk:* the error surfaces one or two layers away as a misleading `ValueError` from `Chunking` or a `TypeError` at `pipeline.py:41`. Debugging cost is high and the true cause is never named.

**H4. Divergent error handling between the two memory gates.**
`pipeline_lg.py:226-232` (unguarded) vs `512-523` (guarded, with the comment *"a memory failure must never destroy the user's turn"*).
*Risk:* an Ollama hiccup on turn 5 of `ChatRagWithMemoryGraph` raises out of `invoke` and loses the user's turn. The correct pattern exists ten lines of scrolling away.

**H5. `Structure_pipeline` is broken and is a 133-line copy.**
`pipeline.py:245-378`; variable mismatch between the template at 302/309 and the invocation at 361.
*Risk:* turns 1-4 raise. It is also ~95% duplicated from `ChatRagPipelineWithMemory`, so any fix to the memory logic must be applied twice. (Known placeholder — flagged, not touched.)

**H6. `UserMemory` writes to Chroma from its constructor.**
`memory/long_term_memory.py:40-46`.
*Risk:* every code path that merely *reads* memory also writes it, including the `recall_user_memory` tool. Safe today only because of the `_memory_cache` fingerprint and content-hash ids — both of which are `AgentGraph`'s doing, not `UserMemory`'s. Any new caller that constructs a `UserMemory` inherits the write.

### MEDIUM

**M1. The ingestion block is duplicated seven times.** `pipeline.py:25-33, 82-90, 150-158, 261-269`; `pipeline_lg.py:79-88, 182-191, 419-428`. Any ingestion change is a seven-site edit.

**M2. Two `HuggingFaceEmbeddings` classes and no singleton in `BuildKnowledge`.** `ingestion/vector_db.py:1` vs `ingestion/chunking.py:3` / `memory/long_term_memory.py:6`; construction at `ingestion/vector_db.py:10-12`. Duplicate model loads; one import path is deprecated.

**M3. Deprecated `langchain_community.llms.Ollama` in four files.** `pipeline.py`, `pipeline_lg.py:11`, `retreival/retrieval.py:6`, `memory/long_term_memory.py:7`. The deprecation warning is already visible in `labbb.ipynb`, and the notice says removal in LangChain 1.0.0 — which the project has effectively already moved to (`langchain-core>=1.2.13` per the settings allowlist).

**M4. ~40 hardcoded literals across 9 files, no config module.** Enumerated in §4. `docs/todo.md` §0 already tracks the top four.

**M5. `_memory_cache` is unbounded and unsynchronized.** `pipeline_lg.py:443, 476-496`.

**M6. Collection names are derived from absolute file paths.** `f"VDB for {file_path}"` at seven sites, sanitized by a lossy regex (`retreival/retrieval.py:17-21`). Moving the document orphans its collection; two different paths can collapse to the same name after sanitization.

**M7. `mqr` returns duplicates with no dedup or rank fusion.** `retreival/retrieval.py:75-77`. Duplicate chunks are joined into the prompt verbatim, wasting context.

**M8. Chat graphs retrieve unconditionally.** `pipeline_lg.py:99-106, 208-218`. A greeting costs a full vector search — six model/store calls under `mqr`.

**M9. `import wikipedia` sits outside its try block.** `tools/tools.py:70` vs the try at 74. `ImportError` escapes the tool's error contract.

**M10. No output-size cap or timeout on any tool.** `tools/tools.py` throughout. `wikipedia_lookup(k=3)` can inject tens of thousands of characters into the context in one `ToolMessage`.

**M11. `mode_docs="scanned_pdf"` is broken in the working environment.** `[FIXED]` `ingestion/doc_loader.py:33-38`. `pdfplumber` was not installed, so `PDFPlumberLoader` imported fine (langchain defers the underlying import) and then raised `ImportError: pdfplumber package not found` at `.load()`. Verified by running it, not inferred from the import list. `README.md:280` and `WHAT_I_BUILT.md:63` both advertise the mode as working. *Discovered while pinning dependencies.*

> **Resolution.** `pdfplumber==0.11.10` installed and pinned; `pip check` still clean. The mode is verified working — it loads all 24 pages of `test_pdf's/Pipes_2025.pdf`. No code change was needed: the loader was always correct, the dependency was simply absent.

### LOW

**L1.** Unused imports: `pipeline.py:3` (`BuildKnowledge`), `ingestion/doc_loader.py:4` (`SeleniumURLLoader`), `:9` (`BaseModel`, `Field`), `ingestion/chunking.py:1` (`Language`), `:6` (`Document`).
**L2.** Dead `language` parameter at `pipeline.py:39, 95, 162, 273`, never forwarded — `wiki` mode is permanently English through any pipeline.
**L3.** Typo'd public names: the `retreival/` package and `UserMemory.retreival` (`memory/long_term_memory.py:60`, called at `tools/tools.py:171`). The cost of renaming rises with every new caller.
**L4.** `ChatRagPipeline.chat_history` is a local (`pipeline.py:92`) while its sibling uses `self.` (149).
**L5.** `print()` as an output channel in library code: `ingestion/chunking.py:31, 40`; `pipeline_lg.py:323, 616`.
**L6.** `__init__.py` conventions differ — three empty, one populated.
**L7.** `docs/coding_standards.md` and `docs/decisions.md` are 0 bytes.
**L8.** `.env` holds four API keys that no code reads; no `dotenv`/`os.environ` reference exists.
**L9.** Graph classes are plain classes; pipeline classes subclass `Runnable`. The two generations aren't interchangeable in a chain.
**L10.** `AGENT_SYSTEM_PROMPT` (`pipeline_lg.py:367-370`) is exported for tests but isn't the prompt a `enable_recall_tool=False` agent uses.
**L11.** `WHAT_I_BUILT.md` is stale (claims `tools/tools.py` is empty; wrong collection name).
**L12.** `.claude/settings.local.json` allowlists `Bash(pip install *)`, `Bash(git push *)`, and `Bash(git commit *)` unbounded.
**L13.** Stale artifact `db/user_history_test/` on disk.
**L14.** `README.md:83` tells you to `pip install duckduckgo-search ddgs`. Only `ddgs` is used — the installed `langchain-community` does `from ddgs import DDGS`, so `duckduckgo-search` is vestigial. Omitted from `requirements.txt` rather than pinned as dead surface.

---

## 9. Refactoring Recommendations

All preserve current behavior. **Not implemented — recommendations only.** Ordered by value-to-risk.

### R1. Split index construction from index querying — *partially done*

*Problem:* C1, M1, M6, and the tight `Retrival`→`BuildKnowledge` coupling all originate in one decision — `Retrival.__init__` (`retreival/retrieval.py:26-27`) builds the index as a side effect of constructing a querier.
*Done:* the C1 half — `BuildKnowledge.invoke` now opens-or-creates and upserts by hash id, so re-ingestion is idempotent and a warm start skips embedding. A single batched `get(ids=..., include=[])` turned out to be enough; the manifest file suggested below was not needed.
*Still open:* the structural half. `Retrival` still *requires* `docs` and still builds its index inside its constructor, so you cannot open an existing collection without re-supplying and re-chunking the source. That is what keeps M1 (the 7× duplicated ingestion block) and the startup cost in §10 alive.
*Change:* let `Retrival` accept either `docs` (build) or an existing store handle (open), keeping the current signature working so no caller changes.
*Trade-off:* ids are derived from chunk metadata + text, so changing `chunk_size`/`chunk_overlap` produces entirely new ids and the old chunks linger alongside the new ones. Re-chunking with different parameters still needs a manual collection reset.

### R2. Introduce `config/settings.py`

*Problem:* M4 — ~40 literals across 9 files; changing the model name means editing seven sites.
*Change:* one dataclass (or Pydantic settings object) holding chunk size/overlap, model names, embedding name, k/lambda defaults, memory cadence, trim threshold, recursion limit, and paths — env-overridable via `python-dotenv`, which `docs/tech_stack.md` §5 already specifies and which nothing currently imports. Keep every existing constructor argument as an explicit override that wins over the default.
*Trade-off:* a settings object is a new global-ish dependency reachable from every module; without discipline it becomes a junk drawer. Mitigate by keeping it a flat frozen dataclass with no behavior.

### R3. Extract a `BaseRagGraph`

*Problem:* M1 plus the 3-6× duplication of `get_history`, `run_chat`, last-`AIMessage` extraction, and checkpointer wiring in `pipeline_lg.py`.
*Change:* a base class owning ingestion (delegating to R1), the checkpointer, `get_history`, `run_chat`, and `_last_answer`. The three graph classes then contribute only their nodes and edges.
*Trade-off:* inheritance across three classes whose graph shapes genuinely differ — `AgentGraph` has no `context` in state and no retrieve node. Keep the base strictly to plumbing; do **not** try to unify `ChatState` and `AgentState`, which are correctly different.

### R4. Move retrieved context and the user profile out of the system message — *done for the graphs*

*Problem:* H1 — prompt injection from ingested documents, against the project's own `docs/security.md` §4.
*Done:* implemented for all three LangGraph classes, as described under H1 in §8. Answer quality was re-checked on the real PDF across a two-turn exchange — answers stayed grounded and the follow-up still resolved against history.
*Still open:* the same pattern in all four `pipeline.py` classes.
*Trade-off, as realised:* the chat graphs use mistral and were re-verified live. **The agent's tool routing was not**, because only `mistral:latest` is pulled locally and `AgentGraph` needs `llama3.1`. Its prompt gained a data-rule bullet, and this codebase's agent behaviour is empirically tuned to an 8B model (`README.md:261-265`). The prompt/binding invariant and both `enable_recall_tool` branches were asserted structurally, but **routing behaviour should be re-checked against llama3.1 before relying on it** — a longer prompt is exactly the kind of change that shifts an 8B model's tool selection.

### R5. Give `UserMemory` a durability guard and a read-only mode — *partially done*

*Problem:* C2 (silent in-memory fallback) and H6 (constructor writes on the read path).
*Done:* the C2 half — the durability guard, as a `UserWarning`.
*Still open:* H6. The Chroma write is still a side effect of `UserMemory.__init__`, so `recall_user_memory` still writes while reading.
*Change:* separate the write from construction with an explicit `.index(messages)` method, and have `make_recall_user_memory`'s callback request a read-only handle.
*Trade-off:* changes `UserMemory`'s public contract, and `AgentGraph._build_user_memory` currently *relies* on the constructor write to index a new thread's messages before recall. That call site must be updated in the same change.

### R6. Add metadata to memory documents

*Problem:* §6 — no timestamp, thread, user, or turn index means no expiry, no filtering, no ordering, no correction.
*Change:* attach `{"timestamp", "thread_id", "user_id", "turn"}` to each `Document` at `memory/long_term_memory.py:32`, and extend `.retreival` with an optional metadata filter. Consider including `thread_id` in the id hash so identical text from different conversations stays distinct.
*Trade-off:* changing the id scheme breaks dedup against already-stored rows — existing memory collections would need a rebuild or a migration. Metadata also enlarges every row slightly. The unlock (expiry, scoping, supersession) is worth it, but it should land before the memory store gets large.

### R7. Make the tool registry declarative

*Problem:* §7 — three files and six edit sites per tool, and a prompt/binding invariant maintained by hand that `extra_tools` silently violates.
*Change:* a small registry where each entry pairs a tool (or factory) with its prompt bullet and optional usage guideline. `AgentGraph` then builds `self.tools` and the system prompt from one source, making drift structurally impossible and giving `extra_tools` a way to contribute its description.
*Trade-off:* this is genuine new indirection for a five-tool system, and it would be over-engineering today were it not for `docs/roadmap.md` Phases 3-5 adding three more tools with real side effects. Recommend building it *when* the sixth tool lands, not before.

### R8. Adopt a durable checkpointer

*Problem:* H2 and §5 — `MemorySaver` is in-RAM, per-object, and non-durable. `docs/tech_stack.md` §2 already specifies `SqliteSaver`.
*Change:* make the checkpointer a constructor parameter defaulting to `MemorySaver`, so `SqliteSaver` (or `AsyncSqliteSaver`) can be injected.
*Trade-off:* `SqliteSaver` requires context-manager lifecycle management, which the current constructor-owns-everything pattern doesn't accommodate. Every `get_state` call (four sites in `AgentGraph`) becomes real I/O; `AgentGraph.invoke:571` does one per turn.

### R9. Delete or fix `Structure_pipeline`; deduplicate `pipeline.py`

*Problem:* H5 — broken and 95% duplicated.
*Change:* the four pipeline classes differ only in prompts and a memory flag. One class parameterized by a prompt pair plus `enable_memory` would replace all four.
*Trade-off:* `pipeline.py` is the legacy generation and `Structure_pipeline` is explicitly a placeholder. Investment here may be wasted if the LangChain generation is retired outright — which is the more likely trajectory given `pipeline_lg.py` already covers every non-`Structure_pipeline` case. **Recommend deciding the fate of `pipeline.py` before spending effort on it.**

### R10. Standardize on `langchain_ollama` and one embeddings import

*Problem:* M2, M3 — deprecated classes in four files, two `HuggingFaceEmbeddings` import paths, duplicate model loads.
*Change:* `OllamaLLM`/`ChatOllama` from `langchain_ollama` everywhere; `HuggingFaceEmbeddings` from `langchain_huggingface` everywhere; hoist the singleton from `memory/long_term_memory.py:14-22` into a shared module used by `BuildKnowledge` and `Chunking` too.
*Trade-off:* mostly mechanical, but `OllamaLLM` is not a drop-in for `Ollama` in every prompt-chain shape, and `retreival/retrieval.py:22`'s `mqr` chain feeds a `PydanticOutputParser` that is sensitive to output formatting. Verify `mqr` end-to-end after the swap. Should land together with C3 (pinned dependencies).

---

## 10. Voice Readiness Assessment

The blunt finding: **there is not a single `async`, `await`, `ainvoke`, `astream`, or `interrupt(` token anywhere in the repository** (verified by grep across all `.py` files). Marshall is entirely synchronous, request-response, and terminal-bound.

That said, the *graph topology* is not the obstacle. The obstacle is the layer directly above and below it.

### What supports voice today

1. **The compiled graphs already expose async and streaming APIs for free.** `CompiledStateGraph` provides `.ainvoke()`, `.astream()`, `.astream_events()`, and `.stream(stream_mode="messages")` regardless of whether the node functions are sync. The graph shape needs no change to stream.
2. **Thread-scoped checkpointing already exists.** `MemorySaver` keyed on `thread_id` (`pipeline_lg.py:96, 205, 461`) means resumable, addressable sessions are already a solved primitive — the durability swap (R8) is the only gap.
3. **`AgentState` has no per-turn scratch state.** `pipeline_lg.py:381-385` carries only `messages`, `turn_count`, `user_history_summary`. On a barge-in or cancel there is no half-computed `context` to invalidate — a genuinely important property that `ChatState` (which stores `query` and `context`, `pipeline_lg.py:26-33`) does not have.
4. **Turn counting already lives in a dedicated node.** `_count_turn_node` (499-502) is exactly the hook an event-driven scheduler needs to fire per-utterance side effects.
5. **Tools are proper `BaseTool`s.** Adding async support means giving each an `async def` coroutine; `ToolNode` will prefer it automatically. No dispatch changes needed.
6. **The ReAct loop is bounded.** `recursion_limit=25` (570) means a runaway tool loop terminates in seconds rather than minutes — important when the user is waiting in real time.
7. **The system prompt already forbids process narration.** Lines 361-363 instruct Marshall to answer as though it had simply known — exactly right for speech, where "let me search for that" is dead air.
8. **`docs/tech_stack.md` §3 already commits to async infrastructure** (`Crawl4AI`'s `AsyncWebCrawler`, Playwright), so the async direction is already the project's stated intent.

### What blocks each capability

**Real-time streaming input/output — blocked.**
- Every node calls a blocking client: `self.llm.invoke` (`pipeline_lg.py:543`), `chain.invoke` (110, 237, 245), `Ollama`/`ChatOllama` sync constructors (89, 193, 460).
- The public API discards streaming by construction: `AgentGraph.invoke` (565-582) runs the graph to completion and returns a `str`. `ChatRagGraph.invoke` (128-141) and `ChatRagWithMemoryGraph.invoke` (279-292) do the same. Every caller is request-response by contract.
- There is no input side at all. `run_chat` (143-159, 294-309, 584-596) uses blocking `input()`; `pipeline.py:117, 203, 349` likewise. A voice loop needs a push-based utterance source, not a pull-based prompt.
- `print()` is the output channel (`pipeline_lg.py:323, 616`; `ingestion/chunking.py:31, 40`). There is no sink to redirect to a TTS stream.
- Tool results are complete strings only (`tools/tools.py` throughout) — no partial or progressive output, so a slow `web_search` produces silence then a wall of text.

**Async execution — blocked, but shallowly.**
- No async anywhere. LangGraph will run sync nodes in a threadpool under `.ainvoke()`, so you get *concurrency* almost for free — but not *cancellation*, because a blocked HTTP call in a worker thread cannot be interrupted.
- The `Runnable` base gives `ainvoke` by delegation to `invoke` in an executor. It works and it blocks a thread. Not a solution, a trap.
- Every network call is sync with no timeout: Ollama (543), Chroma (`retreival/retrieval.py:48, 59, 76`; `memory/long_term_memory.py:63`), Wikipedia (`tools/tools.py:76`), DuckDuckGo (`tools/tools.py:93`).
- **Shared mutable state on a single graph object blocks concurrent sessions:** `self._memory_cache` (`pipeline_lg.py:443`) is written without synchronization at 489 and 496; `self.tools` and `self.llm` are shared; `_EMBEDDING_MODEL` (`memory/long_term_memory.py:12`) has a check-then-set race. One `AgentGraph` serving several simultaneous voice sessions will interleave these.

**Event-driven state transitions — partially supported.**
- The graph *is* a state machine, and conditional edges (`pipeline_lg.py:553-557, 561`) already route on state. That is the right substrate.
- But every transition is driven by a completed function return. There is no way to inject an external event (VAD silence detected, user started speaking, timeout elapsed) into a running graph.
- `ChatState` stores per-turn inputs (`query`, `k`, `lambda_mult`) in durable state — in an event-driven model these become racy: a second utterance overwrites `query` mid-flight.
- Turn boundaries are implicit (one `invoke` = one turn). Voice needs explicit utterance boundaries with barge-in semantics.

**Interrupt / cancel mid-generation — blocked entirely.**
- No `interrupt()` anywhere, despite `docs/roadmap.md` Phase 2 and `docs/vision.md` both naming native HITL as a headline feature.
- Blocking sync calls can't be cancelled; killing the thread leaves Chroma and Ollama clients in an unknown state.
- Cancelling mid-ReAct-loop leaves `messages` with an `AIMessage` carrying `tool_calls` and no matching `ToolMessage` — the exact orphaned-pair state that `README.md:267-269` records Ollama rejecting. **Cancellation therefore corrupts the thread**, and this is the single hardest voice problem in the current design. Any cancel path must either complete or synthesize the pending `ToolMessage`s.
- `MemorySaver` writes a checkpoint per super-step, so a mid-turn kill leaves a checkpoint in exactly that broken state, and it persists into the next turn.

**Continuous conversation loops — blocked.**
- The only loops that exist are `while True: input(...)` (six sites).
- **Startup latency is fatal to a voice session:** the constructor synchronously loads, chunks, embeds, and writes the entire document (`pipeline_lg.py:419-428`) plus a MiniLM model (`ingestion/vector_db.py:10-12`) before returning. Nothing can be said until that finishes. The C1 fix materially improves the *warm* case — a restart against an existing collection now skips embedding entirely and pays only a metadata read — but the cold case, the model load, and the load+chunk work ahead of it are all untouched, so this remains the first thing to fix for voice.
- **The memory gate injects a multi-second stall into turns 5, 10, 15…** `_summarize_user_history_node` (512-523) runs a full LLM compression synchronously in the request path, and `UserMemory.invoke` (`memory/long_term_memory.py:47-59`) sends the entire history in one prompt so the stall grows with conversation length. In text this is a pause; in voice it is dead air at a fixed, predictable cadence.
- **`AgentGraph`'s unbounded history** (§4-F) means time-to-first-token grows monotonically through a long conversation — the exact opposite of what voice needs.
- Nothing measures latency. There is no instrumentation for time-to-first-token, per-node duration, or tool latency, so there is no baseline to regress against.

### Recommended sequencing for voice (not implemented)

1. Fix startup cost first (R1) — nothing else matters if the session can't begin.
2. Make the memory gate non-blocking (background task or incremental summarization) — the fixed-cadence stall is the most audible defect.
3. Add async node variants and async tool coroutines, then switch the public API to `astream`.
4. Add a durable checkpointer (R8) before adding cancellation, so a corrupted mid-turn state is recoverable.
5. Solve the orphaned-`ToolMessage` problem explicitly — a cancel path that reconciles pending tool calls — before exposing barge-in.
6. Make `AgentGraph` per-session, or make `_memory_cache` thread-safe, before serving concurrent sessions.
7. Only then add `interrupt()`/HITL per `docs/roadmap.md` Phase 2.

---

## Closing assessment

Marshall V1 is a **well-layered system with a clean acyclic dependency graph, one genuinely excellent module, and three structural problems that will compound if not addressed before V2.**

The agent layer (`pipeline_lg.py:327-617` and `tools/tools.py`) is materially better than the rest of the codebase. It is empirically tuned, its comments record *why* rather than *what*, and several of its decisions — turn counting in its own node, the explicit `tools_condition` path map, per-call profile composition that cannot compound, the fingerprinted memory cache, guarding `n <= 0` before the modulo, capping `recursion_limit`, and keeping the prompt exactly in sync with the bound tool list — are the kind that are normally learned only after a production incident.

The three problems identified as blocking further work:

1. ~~**C1** — index construction is welded to querying, so every start re-appends the corpus.~~ `[FIXED]` The re-append is gone and warm starts now skip embedding. The *structural* half — `Retrival` still cannot open a collection without re-supplying `docs` — remains open as R1 and still drives M1 and the voice startup cost.
2. ~~**C2** — long-term memory silently defaults to non-durable.~~ `[FIXED]` It is now loud rather than silent. It is still ephemeral by default; callers must pass a path.
3. ~~**C3** — no dependency manifest, in a project that has already survived one LangGraph version crisis whose resolution is recorded only in a permissions allowlist.~~ `[FIXED]` 21 packages pinned and verified. The known-good baseline now exists in the repository rather than in shell history.

**All three criticals are closed, and so is H1** — untrusted document content no longer reaches the system message in the LangGraph generation, verified by a live injection attempt rather than by inspection. That mattered for sequencing: `docs/roadmap.md` Phases 3-5 add Gmail, browser, and scraper tools with real side effects, driven by a model whose system prompt any ingested PDF or web page could previously write to. Fixing it before those land was much cheaper than after.

The highest-severity open finding is now **H2** — unbounded message growth in `AgentGraph` combined with an in-RAM, per-object `MemorySaver`. It is also the one that most directly blocks voice: time-to-first-token grows monotonically through a conversation, which is the opposite of what real-time interaction needs, and nothing survives a restart. **H4** remains the cheapest item on the board — ten lines to backport `AgentGraph`'s error-handling discipline to `ChatRagWithMemoryGraph`, whose correct pattern already sits a few hundred lines away in the same file.

**One caveat carried forward from the H1 fix:** `AgentGraph`'s tool routing has not been re-verified against `llama3.1`, which is not pulled locally. See R4.

**One operational note carried over from C1:** the fix prevents new duplicates but does not remove ones already on disk. Any collection under `db/` built before this change should be deleted and rebuilt, or it will keep serving the copies it already accumulated.

For voice specifically: the graph topology is sound and needs no redesign. The work is entirely in the layer above (async, streaming, an event-driven input source, cancellation semantics) and the layer below (startup cost, blocking clients, per-session isolation). The single hardest problem is not async — it is that cancelling mid-ReAct-loop orphans a `tool_calls`/`ToolMessage` pair in a way Ollama rejects, which means barge-in requires a reconciliation protocol that does not exist yet.

---

*Original review conducted read-only against commit `c100c2e`. Subsequently updated in place as C1 and C2 were fixed; every other finding is as first written and unverified against later changes.*
