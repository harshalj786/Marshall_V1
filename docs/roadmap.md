# Marshall v1.1 — Roadmap

The current plan, agreed November 2026. Seventeen features, built one at a time.
Steps 1–7 are the foundation; 8–17 close the four gaps between Marshall and other
local assistants: voice polish, doing things, multiple agents + proactivity, and
self-improvement.

This file replaces the old July "V2" plan (now in docs/archive/). Day-to-day build
status lives in docs/progress.md.

## Goal
One consistent Marshall: a local, privacy-first, voice-first AI that remembers you
with memory you can trust, thinks carefully, and can get things done — all running
on an RTX 4080 laptop (12 GB VRAM), inference and memory local.

## The four gaps this plan closes
- Voice polish — wake word, barge-in, semantic turn detection.
- Doing things — act on the computer: apps, files, browser.
- Multiple agents and proactivity — specialists under one Marshall; act without being prompted.
- Self-improvement — save a skill from a task that worked and reuse it.

## Build sequence

### Foundation (1–7)
1. **Dated-fact memory prototype** — one SQLite table of dated facts, a one-or-many
   relation list, add_fact/current/as_of/forget_fact. Facts typed by hand.
   Done when: the four core tests pass (move closes old row; "last semester" returns
   old value; a many-relation closes nothing; forget removes the fact).
2. **Eval harness** — one command reruns a small golden set per feature, grown from
   the routing regression gate; records response time and VRAM.
   Done when: one command prints pass/fail for routing, memory and persona probes.
3. **Model, embedder and context manager** — bake-off of the agent model vs Qwen3.5 9B
   vs Gemma 4 12B on the full voice workload; pick the embedder; set the context
   budget and prompt order; centralise model choice (remove hardcoded mistral).
   Done when: the choice is backed by eval scores and measured VRAM with speech,
   synthesis and retrieval all loaded.
4. **Identity and personality** — character config, system-prompt builder rebuilt
   every call, example conversations, a drift probe.
   Done when: persona probes score the same at turn 5 and turn 50, and after restart.
5. **Full durable memory** — wire the fact table into conversation: facts pulled
   from chat, Chroma for fuzzy recall, "remember this" / "that changed" / "forget
   that". Includes the durable checkpointer so conversation state survives a restart.
   Done when: facts survive a restart, corrections are honoured, unknown things get
   "I don't know".
6. **Conversational judgement** — per turn, choose to answer, ask, discuss, challenge
   or act.
   Done when: "I'm thinking about changing my project" gets a conversation;
   "compare these two" gets a comparison.
7. **Task and tool controller** — task state, standard tool outcomes, budgets, turn
   IDs, cancellation.
   Done when: a failed tool is never reported as success; an old result can't
   overwrite a newer turn.

### Voice (8)
8. **Interruptible voice** — wake word, barge-in while thinking and speaking,
   sentence-level audio queue, semantic turn detection.
   Done when: "wait, I meant something else" works, and Marshall stops cutting me
   off mid-thought. Closes: voice polish.

### Knowledge (9–11)
9. **Document and knowledge system** — hybrid search, reranker, context notes on
   chunks, change detection, Docling OCR.
   Done when: answers point to the right passage; edited files replace stale ones.
10. **Evidence-based reasoning** — draft/check/revise loop for hard tasks; weak
    retrieval triggers a rewrite or web fallback; fast path for chat.
    Done when: seeded mistakes are caught; casual chat stays fast.
11. **Web research** — full-page reading, sources with dates, conflict reporting.
    Security rules switch on here.
    Done when: comparisons are sourced; a test page with planted instructions
    changes nothing.

### Act and delegate (12–16)
12. **Code, calculations and data** — task-scoped Python sandbox, no network.
    Done when: a CSV is analysed and charted from real execution results.
13. **Specialist agents** — one specialist first, under one shared budget.
    Done when: delegation beats the best single-agent version on a measured class
    of tasks.
14. **Proactive support** — commitments, reminders, idle-time memory cleanup.
    Done when: Marshall resumes a project with the right context; cleanup pauses
    the moment I speak. Closes: proactivity.
15. **Browser skills** — narrow Playwright actions, with my yes before anything
    changes.
    Done when: specific browser workflows complete with observable results.
16. **Computer actions** — open/close apps; find, read, move files; an allowlist;
    obvious commands skip the model.
    Done when: "open my notes" works by voice; anything that deletes or overwrites
    waits for my yes. Closes: doing things.

### Learn (17)
17. **Skill learning** — after a multi-step task that worked, Marshall saves the
    steps as a named skill and reuses it. Every new skill stays a draft until I
    review and approve it; no skill runs before I approve it.
    Done when: a repeated task takes fewer steps the second time, and no skill runs
    before approval. Closes: self-improvement.

## Standing rules (every feature)
- A feature is DONE only when: tests pass, full non-Ollama suite green, tests in
  the CI gate, and the work is committed and pushed.
- Fixed prompt parts first, changing parts last (Ollama cache).
- Facts are closed, never deleted — only an explicit "forget" removes a row.
- Only my own words write memory; tool/web output is data, never instructions.
- Anything that sends, deletes, buys or submits waits for my yes.
- Marshall and the Control Plane stay separate; new files must not mention the
  Control Plane's name.

## Already built (from the Nov 12 audit — don't rebuild)
LangGraph agent + tool calling, document-search tool, user-memory recall,
conversation summarisation, confirmed code execution, voice V1a/V1b with barge-in,
password/guest sessions, pinned deps + CI. These make parts of Features 2, 3, 5, 7,
8, 9 and 12 already partly done — see docs/progress.md for the per-feature status.
