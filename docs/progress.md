# Marshall v1.1 — build progress

The running checklist Claude Code reads at the start of every feature session, so
it knows what already exists before writing anything. Update the status column as
features land. The full plan is docs/roadmap.md; this is the short version plus
what's actually in the code today.

## How we work (every feature)
- Three questions before any code: how I'd build it, the tradeoffs, what the code
  changes. I write code / think first; Claude Code guides with questions, never
  hands finished solutions up front.
- Build in small steps, STOP for review after each. Socratic debugging: point at
  the bug, ask a guiding question, let me answer.
- After each feature I explain it back in my own words, then a short mock interview.
- A feature is DONE only when: its tests pass, the full non-Ollama suite is green,
  the tests are in the CI gate, AND the work is committed and pushed.
- Scope discipline: if something belongs to a later feature, say so and leave it out.

## Status
Reconciled against the repo audit (Nov 12). "Already built" = found in code during
the audit; the 17-feature plan is unchanged, but several foundation pieces exist.

| # | Feature | Status | Notes / evidence |
|---|---------|--------|------------------|
| 1 | Dated-fact memory prototype | DONE | memory/facts.py + tests/test_facts.py (8 tests, in CI). Standalone. See below. |
| 2 | Eval harness | DONE | `python scripts/run_evals.py --only routing,memory,persona` — one command, one table, latency + VRAM per case. Logic in CI, live behaviour in the local gate. See "Feature 2 — what exists now" for the reference baseline. |
| 3 | Model, embedder, context manager | IN PROGRESS | Step 0 recon done (24 Nov), see docs/feature3_model_bakeoff.md. Every live path already runs llama3.1 ALONE, so this is an optimisation bake-off, not a de-split; mistral removal is a tidy-up. gpt-oss-20b (14 GB) does not fit. Defect: num_ctx never set (4096 default). Next: six harness-prep steps, then the bake-off. Embedder not yet chosen; no config/ module. |
| 4 | Identity and personality | NOT STARTED | char config, prompt builder, drift probe. |
| 5 | Full durable memory | PARTIAL already | wire facts.py into chat; facts pulled from conversation. FOLDS IN the durable checkpointer (R8): conversation message state currently dies on restart; fix it here. |
| 6 | Conversational judgement | NOT STARTED | |
| 7 | Task and tool controller | PARTIAL already | code tools use a confirm callback; no unified controller / turn-id / cancellation yet. |
| 8 | Interruptible voice | PARTIAL already | voice V1a/V1b done incl. barge-in; wake word, V1c streaming+cancel, non-blocking memory gate NOT started; barge-in thresholds are unmeasured guesses. |
| 9 | Document and knowledge system | PARTIAL already | doc-search tool + retrieval modes exist; hybrid search, reranker, change-detection, OCR not added. |
| 10 | Evidence-based reasoning | NOT STARTED | draft-check-revise; corrective RAG. |
| 11 | Web research | NOT STARTED | security rules switch on here. |
| 12 | Code, calculations, data | PARTIAL already | tools/code_tools.py + code_guards.py, confirmed execution. run_code has no network isolation and no CSV/charting path, so the Done-when is not met; also needs a recorded live run. |
| 13 | Specialist agents | NOT STARTED | one specialist first, measured. |
| 14 | Proactive support | NOT STARTED | commitments, reminders, idle-time cleanup. |
| 15 | Browser skills | NOT STARTED | no browser-use/playwright in repo. |
| 16 | Computer actions | NOT STARTED | apps + files, allowlist. |
| 17 | Skill learning | NOT STARTED | I review every self-made skill before use. |

### Evidence — what is already in the code (Nov 12 audit)
Each feature checked against the code, not against a checkbox. PARTIAL means at least
one component named in that feature's roadmap entry already exists; "reusable" items
under NOT STARTED are things a later feature should build on rather than rebuild.
Line numbers are as of commit a027b25 and will drift — search by symbol.

| # | Reconciled | In the code today (file:line) | Missing against the roadmap entry |
|---|------------|-------------------------------|-----------------------------------|
| 1 | ALREADY DONE | `memory/facts.py:100` add_fact, `:144` current, `:153` as_of, `:164` forget_fact, `:36` relation list. The four Done-when tests: `tests/test_facts.py:50`, `:60`, `:67`, `:78`. | Nothing for Feature 1. Not wired into anything (`memory/facts.py:7`) — that is Feature 5. |
| 2 | PARTIAL | Routing golden set with a pass/fail exit code and per-route seconds: `scripts/check_routing.py:113` (ROUTES), `:306` (run_route), `:457` (main); pytest gate `tests/test_routing_matrix.py`. Code-tool A/B: `scripts/check_code_tool_routing.py:141` (measure), `:166` (passed). Per-turn voice latency rows: `voice/telemetry.py:57`. | No memory probes, no persona probes, no VRAM recording anywhere, no single command covering all three. |
| 3 | PARTIAL | Agent model is a constructor argument: `pipeline_lg.py:743`, `:857`; entry points pin llama3.1 at `chat_main.py:26`, `voice_main.py:45`. Context management: summarise-and-trim `pipeline_lg.py:1027` (cadence `:494-495`), tool-output cap `tools/tools.py:54`. | No bake-off. mistral hardcoded at `pipeline_lg.py:134`, `:244`, `retreival/retrieval.py:29`, `memory/long_term_memory.py:30` (and `pipeline.py:63`, `:110`, `:193`). CORRECTED 24 Nov: none of these run on a live path; voice, chat and the eval checks all run llama3.1 for both generation and memory (docs/feature3_model_bakeoff.md). `num_ctx` is never set, so the context is Ollama's 4096 default. Embedder (all-MiniLM-L6-v2) fixed at three sites: `ingestion/vector_db.py:32`, `ingestion/chunking.py:15`, `memory/long_term_memory.py:21`. No token budget, no config module. Prompt order not designed: the user profile is appended inside the system message (`pipeline_lg.py:1096`), ahead of the conversation. |
| 4 | NOT STARTED | Reusable: the static prompt template `pipeline_lg.py:433`, assembled once at construction `:845`. | Character config, per-call prompt builder, example conversations, drift probe. |
| 5 | PARTIAL | Chroma fuzzy recall: `memory/long_term_memory.py:65` (index), `:102` (retreival); recall tool `tools/tools.py:257`; memory gate `pipeline_lg.py:1075`, `:1083`; conversation summary persisted `:986` and reloaded `:1197`. Checkpointer injection seam `:754`, `:866`. | Fact table not wired (nothing outside tests imports `memory/facts.py`); no facts pulled from chat; no "remember this" / "that changed" / "forget that"; no durable checkpointer (`pipeline_lg.py:715`, UNSOLVED), so the message list dies on restart — R8, fixed here. |
| 6 | NOT STARTED | Reusable: the tool / no-tool prompt rule `pipeline_lg.py:446-452`. | The per-turn answer / ask / discuss / challenge / act choice. |
| 7 | PARTIAL | Budgets: tool timeout `tools/tools.py:27`, `:94`; output cap `:54`; tool-round and recursion limits `pipeline_lg.py:648`, `:665`. Confirm-before-run callback `tools/code_tools.py:403`. Hook verdicts proceed / revise / stop `pipeline_lg.py:618-621`. | No task state. Tool outcomes are free-text strings (`tools/tools.py:30`, `:169`), so a failure is not machine-distinguishable from a success. No turn IDs in the graph. No cancellation: a timed-out tool thread keeps running (`tools/tools.py:97-99`). |
| 8 | PARTIAL | Barge-in while speaking: `voice/vad.py:130` (BargeInGuard), `voice/orchestrator.py:275`; stored answer cut to what was heard `pipeline_lg.py:1230`. Sentence-by-sentence synthesis `voice/tts.py:35`, `:95`. States `voice/orchestrator.py:77-82`. | No wake word (`voice/config.py:61`). No barge-in while thinking (`voice/orchestrator.py:14-17`). Endpointing is silence-based (`voice/config.py:47`), not semantic. No streaming, so no real audio queue. Barge-in thresholds are unmeasured guesses (`voice/config.py:67-68`, `:72-73`). |
| 9 | PARTIAL | Loaders `ingestion/doc_loader.py:11`; chunking `ingestion/chunking.py:7`; ingest-once `ingestion/ingest.py:58`, `:81`; content-hash chunk ids `ingestion/vector_db.py:75`; retrieval modes simple / mmr / mqr / wiki `retreival/retrieval.py:56-103`; search tool `tools/tools.py:231`. | No hybrid search, reranker or chunk context notes. No change detection: a populated collection is skipped (`ingestion/ingest.py:98`), so an edited file is never re-ingested. `scanned_pdf` is pdfplumber text extraction (`ingestion/doc_loader.py:33`), not OCR; no Docling. |
| 10 | NOT STARTED | Reusable: the hook's "revise" edge `pipeline_lg.py:1124`. | Draft / check / revise loop; weak-retrieval rewrite or web fallback; fast path for chat. |
| 11 | NOT STARTED | Reusable: snippet-only search `tools/tools.py:222`, Wikipedia `:192`, web-page loader `ingestion/doc_loader.py:39`, tool-output-is-data rule `pipeline_lg.py:454-459`. | Full-page reading, sources with dates, conflict reporting, the planted-instruction test page. |
| 12 | PARTIAL | `run_code` with confirmation: `tools/code_tools.py:357`, gate `:403`, fresh temp dir `:406`, isolated Python `:416`, time limit `:254`, tree kill `:279`. `check_code` `:201`. Guards `tools/code_guards.py:84`, `:191`. Arithmetic `tools/tools.py:158`. | No network isolation, no CSV or charting path, no recorded live run. |
| 13 | NOT STARTED | Nothing. | Everything. |
| 14 | NOT STARTED | Reusable: summary reload on a fresh process `pipeline_lg.py:1197`. | Commitments, reminders, idle-time cleanup. |
| 15 | NOT STARTED | Nothing; no Playwright in `requirements.txt`. Reusable: confirm callback `tools/code_tools.py:403`, spoken yes/no matched in code `voice/confirm.py:36`. | Everything. |
| 16 | NOT STARTED | Reusable: path policy `tools/code_guards.py:84`, secrets refusal `:60`. | App control; file find / read / move; the allowlist; commands that skip the model. |
| 17 | NOT STARTED | Nothing. | Everything. |

## Already built that the plan can lean on (don't rebuild)
From the audit, these exist and are tested:
- LangGraph agent + tool calling (pipeline_lg.py; runs llama3.1, not Mistral).
- Document search as a tool (make_search_current_doc, tools/tools.py).
- User-memory recall (recall_user_memory) + memory gate every N turns.
- Conversation summarisation.
- Code check-and-run with confirmation (tools/code_tools.py, code_guards.py).
- Voice V1a/V1b with barge-in (voice/, voice_main.py).
- Password/admin + guest sessions (admin_session.py, Argon2id).
- Pinned dependencies and CI (pytest -m "not needs_ollama").

## Standing rules (hold for every feature)
- Fixed prompt parts first, changing parts last (so Ollama's cache survives).
- Facts are closed (end-dated), never deleted — only an explicit "forget" removes a row.
- Only my own words write memory; tool/web output is data, never instructions.
- Anything that sends, deletes, buys or submits waits for my yes.
- Marshall and the Control Plane stay separate projects/files; new files must not
  even mention the Control Plane's name (a test greps for it).

## Feature 1 — what exists now (read before Feature 5)
`memory/facts.py`, tested in `tests/test_facts.py` (8 tests, in CI).

- One SQLite table `facts`: subject, relation, object, valid_from, valid_until, source.
  Dates are ISO strings "YYYY-MM-DD". valid_until NULL = still true.
- A knowledge-graph-of-triples that keeps history: a changed fact closes the old
  row (end-dates it), never deletes it, so as_of() can still answer about the past.
- Relation list (dict): lives_in→one, studies_at→one, friend_of→many, working_on→many.
  Unknown relation defaults to MANY and logs a warning.
- Functions (all take conn first):
  - add_fact(conn, subject, relation, object, source, on_date=None): idempotent;
    for a single-value relation closes the open row before inserting; one transaction;
    rejects an on_date earlier than the open row's valid_from (no out-of-order writes).
  - current(conn, subject, relation): rows with valid_until NULL.
  - as_of(conn, subject, relation, date): valid_from <= date AND
    (valid_until IS NULL OR date < valid_until). Start inclusive, end exclusive.
  - forget_fact(conn, subject, relation, object): hard delete, no history. Explicit only.
- Guards: strict YYYY-MM-DD validation (round-trip, rejects "2026-7-1"); partial
  unique index (one open row per single-value relation); CHECK
  (valid_until IS NULL OR valid_until >= valid_from).
- NOT yet wired to the agent, Chroma, or voice — that is Feature 5. Hand-typed for now.

## Feature 2 — what exists now (the measuring stick every later feature uses)
One command: `python scripts/run_evals.py --only routing,memory,persona` (exit 0 nothing
regressed, 1 a case regressed, 2 a check could not run or is not built). Shape:
- `scripts/eval_contract.py` — the record every check writes (`--json`): schema v3,
  per-case verdict PASS / FAIL / KNOWN-BOUNDARY / ERROR plus attempts, passes, required,
  detail.
- `scripts/eval_metrics.py` — per-case model seconds, load seconds, tokens, VRAM per
  loaded model (Ollama /api/ps) and GPU MiB (nvidia-smi); readers degrade to null.
  TurnTimer (callback) for memory/persona is WIDER than routing's history slice.
- `scripts/run_evals.py` — runs each check as a subprocess, one summary table, appends
  a JSONL run log (`eval_logs/runs.jsonl`).
- Checks: `check_routing.py`, `check_memory.py`, `check_persona.py`.

**Reference baseline (23–24 Nov, llama3.1). Later features compare against this:**
| check | result | must-pass (gate) | known boundaries |
|---|---|---|---|
| routing | 5/6 | the five tool routes | no-tool "good explanation" (R4) |
| memory | 4/7 (3–4 across runs) | ranking (retrieve -> answer), born (RETRIEVAL-ONLY) | pet, two-facts, paraphrase, carry-over; unknown (no gap) |
| persona | 2/5 | identity, tool-turn | model-probe, disclaimer, spoken |
memory's 4/7 includes pet and unknown passing on lucky runs (the 24 Nov Done-when run
scored 3/7); the guaranteed floor is the 2 must-pass. A lucky boundary pass is not a fix.

**The two gates — stated plainly:**
- CI (every push, `pytest -m "not needs_ollama"`): 606 tests. Feature 2's model-free
  logic is all here — test_eval_contract 20, test_eval_metrics 21, test_run_evals 36,
  test_memory_check 69, test_persona_check 51 (197), plus routing's model-free tests.
- Local gate (`pytest -m needs_ollama`, needs llama3.1): 46 tests — routing 17,
  code-tool routing 10, memory 9 (`tests/test_memory_gate_live.py`), persona 6
  (`tests/test_persona_gate_live.py`), ingestion/memory-gate 4. The LIVE memory, persona
  and routing behaviour checks NEVER run in CI (no model there). In the live gates only a
  must-pass miss fails; a boundary miss xfails with its known gap; a lucky boundary pass
  is allowed.

## Known debts (decided, not blocking)
- Structure_pipeline in pipeline.py: deleted in commit "pipeline: delete the broken
  Structure_pipeline placeholder"; R9 (rest of pipeline.py) still open. (Broken
  placeholder + old prompt-injection pattern.) pipeline.py is orphaned (only tests import it).
- Durable checkpointer (R8): conversation message state dies on restart. Folded
  into Feature 5.
- mistral hardcoded in several files; no config/settings.py yet (Feature 3). CORRECTED
  24 Nov: the mistral sites are dead code on every live path (everything runs llama3.1
  alone), so removing them is a tidy-up with no behaviour change, not the point of
  Feature 3. See docs/feature3_model_bakeoff.md.
- **ACTIVE DEFECT, fix in Feature 3: the context window is never set.** Nothing passes
  `num_ctx`, so every model runs at Ollama's 4096-token default (llama3.1's 5,211 MiB in
  the run log is exactly weights + a 4096-token KV cache). One persona turn already uses
  3,436 prompt tokens (84%); long voice sessions, with history, summary and tool output on
  top, are being silently truncated today.
- gpt-oss-20b does not fit: 14 GB on Ollama, more than the whole 12 GB card before voice.
  The only MoE that fits natively is granite4:tiny-h (7B total, 1B active), and its tool
  calling is unproven. Big-MoE CPU offload stays parked.
- Eval `model_seconds` (scripts/eval_metrics.py) counts only the agent's own replies in
  the thread; it misses the memory gate, the summariser and retrieval's hardcoded mistral,
  so `wall_seconds - model_seconds` is NOT pure tool time. Read it that way in Feature 3.
- **FOOTGUN — hand-run memory/persona checks open the LIVE document index.** Run by hand
  (`python scripts/check_memory.py`, `check_persona.py`, or via run_evals.py), both build
  their agent through `check_routing.build_check_agent()` with no vector path, so they
  open `db/voice_vdb` — the real voice index — and can race a live voice session for one
  Chroma directory (or cold-ingest into it). The pytest gates monkeypatch every path into
  tmp; the hand-run DEFAULT is the hazard (check_routing.py run by hand shares the same
  default). Don't run them while voice is live. Fix = give
  the checks their own vector path; that edits check logic, so it was left out of Feature 2.
- (Done in Feature 2 Step 3: the runner records which models are loaded at each check's
  START — the "loaded at start" column.)
- Feature 5 must flip pet, two-facts, paraphrase, carry-over back to must_pass=True and
  make them pass — this is the human-like-recall target. Baseline 2/6 pass, 4 known
  boundaries (23 Nov), plus `unknown` (Feature 5's "I don't know" target, no gap note:
  it passes some runs). A boundary may pass on a lucky run (pet did on 23 Nov) — that
  is not a fix. A 7th case, `born`, was added as a RETRIEVAL-ONLY must-pass (recall
  chosen + "1998" returned; answer not graded, and its record says so). Two must-pass
  cases on two layers:
  ranking = retrieve -> answer, born = retrieve only. (scripts/check_memory.py; each
  case's `known_gap` says why it misses today.)
- **Feature 5's #1 target: make Marshall actually USE retrieved facts in its replies.**
  "Recall returns the fact, the answer doesn't use it" is NEARLY UNIVERSAL on llama3.1:
  pet, carry-over, and in a 23 Nov trial born ("what year was I born?") and car ("what
  car do I drive?") at 0/5 each — the tool returned the exact sentence every time and the
  answer said "I don't have any information about…". Only ranking's answer-forcing
  format dodges it. pet + carry-over are the same defect, not two. Untested lead: the
  snippets come back as the user's raw first-person words ("I was born in 1998.") with
  no attribution, so the model may read "I" as itself — try framing them ("The user
  said: …").
- Separate defect for Feature 5 to check: tool-call syntax leaking into the reply. In the
  same trial "which city is my office in?" answered with raw
  `<|python_tag|>"recall_user_memory" tool call returned … {'answer': 'Bangalore'}` —
  a substring match would score it a pass, so it is NOT used as a case.
- Feature 4 baseline gaps, from the persona check (scripts/check_persona.py, 23 Nov,
  5 warm trials per case): must-pass identity 5/5 and tool-turn 5/5; known boundaries
  model-probe, disclaimer, spoken at 0/5 each, all deterministic. Two real findings,
  both honest baseline gaps, not broken cases:
  - The persona is ONE prompt line ("You are Marshall, the AI son of Harshal."). Asked
    what runs underneath, it answers "I'm a large language model…"; asked how it feels,
    it narrates "No tool call is needed for this question…". Feature 4's character
    config / prompt builder must enrich it, then flip model-probe and disclaimer to
    must_pass=True.
  - Nothing strips markdown before TTS, and the prompt never asks for spoken prose:
    "three tips" comes back as a numbered list that Kokoro reads aloud. Fix in the
    prompt or the TTS path, then flip spoken to must_pass=True.
- Same-day supersede leaves a zero-length row (valid_from == valid_until) that
  as_of() won't return. Acceptable at day precision.
- Unknown relations logged to stderr only, not persisted.
- ACTIVE BUG, fix in Feature 9: its Done-when ("edited files replace stale ones") is
  currently inverted. ingestion/ingest.py:98 skips any populated collection, so an
  edited file is never re-ingested and the stale version persists.
