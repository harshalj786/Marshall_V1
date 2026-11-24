# Feature 3 — model bake-off: recon findings and design

Step 0 recon, 24 Nov 2026, against commit 07e7aa4. Nothing was pulled or changed to
produce it. This is the reference for every later Feature 3 step; the build status
lives in docs/progress.md. Line numbers drift; search by symbol.

## What Feature 3 actually is (corrected)

The plan assumed Marshall runs two models, mistral for RAG and llama3.1 for the agent,
and that Feature 3 would collapse them into one. **That split is not live.** Every live
path already runs llama3.1 alone:

- `voice_main.py` and `chat_main.py` pass the same model as both `model` and
  `memory_model` (`VOICE_MODEL`, `CHAT_MODEL`).
- All three eval checks build their agent through `check_routing.build_check_agent()`,
  which imports `voice_main.VOICE_MODEL` for both.
- Every caller uses retriever mode `"simple"`, so the mistral inside `Retrival` is built
  but never called.

So **Feature 3 is an optimisation**: is there a better single model than llama3.1 for a
12 GB card with voice loaded? Removing mistral is a tidy-up that changes no live
behaviour. It is not the point of the feature.

mistral was seen resident during routing only because the ingestion tests ran just
before (`scripts/run_evals.py` docstring), not because the agent used it.

## Where mistral is hardcoded (none of it runs on a live path)

| Site | Job | Runs live? |
|---|---|---|
| `pipeline_lg.py:134` | `ChatRagGraph` answer generation | No. Only tests build it. |
| `pipeline_lg.py:244` | `ChatRagWithMemoryGraph` answer generation | No. Tests only. |
| `retreival/retrieval.py:29` | Query rewriting, `mode="mqr"` only | Built in every `Retrival`, never called: no caller uses `mqr`. Building it does not connect. |
| `memory/long_term_memory.py:30` | `DEFAULT_MEMORY_MODEL = "mistral"`, the profile-compression default | Every live caller overrides it with llama3.1. `voice_main._ensure_durable_memory` builds a `UserMemory` with the default but never invokes it. |
| `pipeline.py:63`, `:110`, `:193` | Three old RAG pipeline classes | No. The file is orphaned (R9). |

Also mention mistral without running it: the `requirements.txt` pull instructions,
`tests/test_voice_prep.py` (asserts the default is `"mistral"`), and the end-to-end
test in `tests/test_memory_gate_resilience.py`.

**How the model choice flows.** It is a constructor argument. It reaches exactly one
`ChatOllama(...)` call in `AgentGraph.__init__`, which serves the agent and the
summariser; `memory_model` passes through to `UserMemory`. llama3.1 is named at:
`AgentGraph`'s `model=` default, `voice_main.VOICE_MODEL`, `chat_main.CHAT_MODEL`, and
the `_has_llama31()` helpers in the four live test gates (`test_routing_matrix`,
`test_code_tool_routing`, `test_memory_gate_live`, `test_persona_gate_live`). There is
no config module.

## Current defect: the context window is never set

Nothing passes `num_ctx`, so every model runs at Ollama's 4096-token default. Evidence:
llama3.1 reports 5,211 MiB in `eval_logs/runs.jsonl`, which is 4.9 GB of weights plus
exactly a 4096-token fp16 KV cache (128 KiB per token).

The system prompt plus tool schemas is about 1,950 tokens on its own, and the largest
single-turn prompt in the logs is **3,436 tokens (persona), 84% of the window**. A real
voice session adds history, the conversation summary and tool output on top, so long
sessions are being silently truncated today. Fix in Feature 3, and pin the same value
for every bake-off candidate.

## VRAM budget with voice loaded

| Component | Device | VRAM | Source |
|---|---|---|---|
| Desktop with nothing of ours loaded (wallpaper engine, browsers, VS Code, overlays) | GPU | 2,776 MiB on 24 Nov; varies | measured, `nvidia-smi` |
| llama3.1 Q4_K_M at 4096 context | GPU | 5,211 MiB | measured, run log |
| MiniLM embedder plus torch's CUDA setup, in the agent's process | GPU (torch is cu124, so sentence-transformers picks CUDA) | about 0.4–0.5 GB | inferred from the 8,456 MiB peak in the run log |
| faster-whisper large-v3-turbo, `int8_float16` | GPU, with its own CUDA setup | about 1–1.5 GB | **estimate, unmeasured** |
| Kokoro-82M | GPU (KPipeline uses CUDA when available) | about 0.3–0.5 GB | **estimate, unmeasured** |
| Silero VAD | CPU (`load_silero_vad()` default) | 0 | read in code |

Budget: 12,282 − about 2.8 desktop − about 0.5 embedder − about 1.5–2 voice ≈
**7–7.5 GB for the model, KV cache included**. The voice figures are estimates; the
ballast step exists to measure them.

Two blind spots in the harness as of Feature 2:

- `eval_metrics.ollama_vram()` records only `size_vram`, not the model's full `size`.
  When VRAM runs short, Ollama does not fail; it quietly puts layers on the CPU. The
  harness cannot see that today.
- voice_main loads the audio models first and the LLM last. That is exactly the order
  in which Ollama sees less free memory and spills. On Windows the driver can also page
  GPU memory into system RAM instead of failing, so "fits" has to mean "no latency
  jump", not only "no crash".

## Candidates (ollama.com, fetched 24 Nov)

| Model | Q4 size | Tools | Context | Long-chat memory | Fits about 7 GB with voice? |
|---|---|---|---|---|---|
| llama3.1:8b (baseline) | 4.9 GB (5.2 loaded at 4K) | yes | 128K | GQA, 128 KiB/token: about +4 GB at 32K | yes, measured |
| qwen3.5:9b | 6.6 GB q4_K_M (includes a vision encoder) | yes, plus thinking | 256K | hybrid Gated DeltaNet plus some full attention: small KV | borderline, measure |
| qwen3:8b | 5.2 GB | yes, plus thinking | 40K | GQA, about 144 KiB/token (from the HF config, not the Ollama page) | likely |
| gemma4:12b (named in the roadmap) | 7.2 GB qat – 8.0 GB q4_K_M | yes, plus thinking | 256K | sliding window: small KV | probably not |
| granite4:tiny-h (7B total, 1B active MoE) | 4.2 GB | yes | 1M | Mamba-2 hybrid: very small KV | yes, easily |
| gpt-oss:20b | **14 GB** | yes; thinking can only be set to low, not off | 128K | MoE, MXFP4 | **no, not even on its own** |

**gpt-oss-20b does not fit.** At 14 GB it exceeds the whole 12 GB card before voice
loads, so it would need CPU offload, which is parked. The MoE research finding holds
(MoE saves speed, not memory), but on 12 GB the only small MoE that fits natively is
**granite4:tiny-h**, and its 1B-active tool calling is unproven.

Other notes: the current granite4 library has no 8B dense model (it jumps from 3b to the
7b-a1b MoE). Gemma 4's lineup is E2B, E4B, 12B, 26B-A4B and 31B.

## Ollama vs llama.cpp

Ollama 0.24.0 covers the whole bake-off: every candidate is in its library, and pull,
unload, `/api/ps`, `num_ctx`, `think` and KV-cache quantisation (`OLLAMA_KV_CACHE_TYPE`,
server-wide) are all available. llama.cpp would only be needed to place a big MoE's
experts on the CPU per tensor, which is parked. No stack migration.

## Fairness traps (all fixed by the prep steps before any benchmark)

1. **Model swap.** The only way today is editing `voice_main.VOICE_MODEL`, which also
   changes live voice.
2. **Thinking defaults.** Qwen3, Qwen3.5, Gemma 4 and gpt-oss can think, and nothing
   sets `reasoning`/`think`. Thinking text in the reply breaks the persona and routing
   checks and inflates latency. Hold it off for every candidate; gpt-oss cannot turn it
   off.
3. **Context size.** Pin `num_ctx` to the same value for everyone; nothing passes it.
4. **Leftover models.** With `keep_alive="30m"` the previous candidate stays in VRAM,
   corrupting VRAM readings and possibly pushing the next one onto the CPU. Unload all
   models and warm the new one before each candidate.
5. **Timing scopes differ.** Routing's `model_seconds` counts only the agent's replies;
   memory and persona use `TurnTimer`, which also counts the gate, the summariser and
   retrieval. Compare each check across models, never across checks.
6. **Leak detectors written for llama3.1.** The persona check's `IDENTITY_LEAKS` has no
   Qwen, Alibaba, Gemma, Google, Granite or IBM. Routing's typed-tool-call detector has
   llama's `<function=` pattern; Qwen's `<tool_call>{"name":…}` is caught by the generic
   `"name"` pattern, but gpt-oss's `to=functions.x` is not.
7. **Labels calibrated on llama3.1.** Must-pass flags and known-gap notes came from
   llama3.1's behaviour. Compare raw passes per case over the same trial counts, not
   verdicts alone. Memory already moves 3–4/7 between llama3.1 runs.

Plus two measurement gaps: offload is invisible (`size` not read), and `gpu_used_mib`
includes desktop apps whose usage varies, so record an idle reading first. And the
footgun from Feature 2 still applies: hand-run checks open `db/voice_vdb`, so the voice
side of a "fits with voice" run must not be a live voice session.

## Bake-off design (approved 24 Nov)

**Configs**
1. llama3.1:8b alone, the real baseline.
2. qwen3.5:9b, the main challenger. qwen3:8b is the fallback if 3.5 fails the fit gate.
3. granite4:tiny-h, the small MoE.
4. gemma4:12b, fit check only.

**Held constant:** same commit, golden cases, trial counts, system prompt and tools;
MiniLM embedder; temperature 0; a pinned `num_ctx` (proposed 8192, since 4096 is already
84% used in one turn); thinking off; same check order; every model unloaded and the
candidate warmed before measuring; the voice ballast loaded; the idle desktop reading
recorded first.

**Criteria, in order**
1. **Fit (must pass).** The model is 100% on the GPU (`size_vram == size`) at the pinned
   context with the ballast loaded; peak GPU use stays at least 1 GB below 12,282 MiB;
   a Whisper pass and a Kokoro pass still work after the model loads.
2. **Required cases (must pass).** Every current must-pass case passes (routing's five
   tool routes, memory's ranking and born, persona's identity and tool-turn), and no
   tool call shows up as reply text.
3. **Score.** Total passes over the 18 cases, from per-case pass rates. A challenger has
   to beat llama3.1 by more than its own run-to-run swing: 2 or more extra cases passing
   reliably.
4. **Speed.** Median warm `model_seconds` per check, plus routing's p50/p95 wall time. A
   voice turn may not get worse than about 1.25× llama3.1.
5. **Long-chat tie-break.** VRAM at 8K vs 32K context.

**What "fits with voice" looks like.** A ballast process loads voice_main's real audio
models and the embedder, runs one Whisper and one Kokoro pass so their memory pools
grow, then holds. `run_evals` runs while it is alive, loading the LLM last to match boot
order. Recorded: idle, ballast only, LLM loaded, peak during checks; each model's `size`
vs `size_vram`; Whisper and Kokoro re-run afterwards.

## Prep steps, in order, each its own reviewed step

No model is benchmarked until all six are in.

1. Model override (flag or env) passed from `run_evals` into the checks.
2. `num_ctx` and `reasoning` passed through `AgentGraph` into `ChatOllama`.
3. The harness reads `size` vs `size_vram` from `/api/ps` and takes the idle reading.
4. Widen the leak lists: Qwen/Alibaba, Gemma/Google, Granite/IBM, and gpt-oss's
   `to=functions.x`.
5. The ballast script: real Whisper, Kokoro and embedder, one pass each, then hold.
6. Unload all models between candidates.

Centralising model choice (a config module) and deleting mistral are separate tidy-up
steps, ordered before or after the decision.
