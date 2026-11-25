"""Per-case measurements for eval checks: model time from the turn, VRAM after it.

A check calls `case_metrics(turn_messages)` once per case, AFTER the case's wall clock
has stopped, and merges the result into its `CaseResult` (scripts/eval_contract.py).
Nothing here can raise into a check or touch its verdict: every reading is guarded on
its own and degrades to None, the same rule as voice/telemetry.py.

READ THIS BEFORE USING model_seconds (e.g. in the Feature 3 bake-off)
---------------------------------------------------------------------
`model_seconds` is the sum of Ollama's `total_duration` over the AIMessages IN THE
TURN'S MESSAGE LIST. It therefore counts only the agent's own replies (the tool-choosing
reply and the final answer). It MISSES every LLM call whose reply never lands in the
thread:

  * the memory gate (pipeline_lg.py, every N turns, on `memory_model`),
  * the conversation summariser,
  * the hardcoded mistral in retreival/retrieval.py (mqr query rewriting).

So `wall_seconds - model_seconds` is NOT pure tool time. It is tools + embeddings +
graph overhead + HTTP + those hidden LLM calls. On a one-turn routing thread the gate
and summariser never fire, so there the remainder is close to tool time; on a
multi-turn check it is not. Comparing models on model_seconds alone would hide
whatever the hidden calls cost.

`load_seconds` is raw. A warm model still reports ~0.09 s of load_duration, so judge
"cold" with eval_contract.is_cold_load, never with `> 0`.

TWO WAYS TO TIME A TURN — and they measure different things
-----------------------------------------------------------
  * `turn_timing(messages)` / `case_metrics(messages)`: reads the replies left in the
    thread. Routing uses it. Narrow (agent replies only) and it needs a slice of the
    history, which a summariser rewrite can invalidate on a long thread.
  * `TurnTimer` (a LangChain callback) / `timer_metrics(timer)`: collects every LLM
    response AS IT IS PRODUCED during the turn. The memory check uses it. Wider: it also
    catches the memory gate, the summariser and any nested model call (see its docstring).

Same field names, different scope. Do not compare one check's model_seconds with
another's unless both were timed the same way.
"""
import json
import subprocess
import threading
import urllib.request
from dataclasses import dataclass
from typing import Optional

from langchain_core.callbacks import BaseCallbackHandler

OLLAMA_PS_URL = "http://127.0.0.1:11434/api/ps"
NVIDIA_SMI = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
NVIDIA_SMI_TOTAL = ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"]

_NS = 1_000_000_000
_MIB = 1024 * 1024


@dataclass
class TurnTiming:
    model_seconds: Optional[float] = None
    load_seconds: Optional[float] = None
    prompt_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


def turn_timing(messages) -> TurnTiming:
    """Sum Ollama's timings over the AI messages of ONE turn. Pure; never raises.

    All-or-null: if any AI message lacks the metadata, every field is None. A partial
    sum would undercount and read as a fast model. A turn with no AI message at all
    (an ERROR case) is also all None.
    """
    try:
        replies = [m for m in messages or [] if getattr(m, "type", None) == "ai"]
        return _sum_metadata([getattr(reply, "response_metadata", None) for reply in replies])
    except Exception:
        return TurnTiming()


def _sum_metadata(metas) -> TurnTiming:
    """Sum Ollama timing dicts. The ONE place the all-or-null rule lives.

    Empty -> all None (nothing was measured). Any entry missing, not a dict, or lacking
    total_duration -> all None (a partial sum would undercount and read as a fast model).
    """
    try:
        metas = list(metas or [])
        if not metas:
            return TurnTiming()
        total = load = prompt = output = 0
        for meta in metas:
            if not isinstance(meta, dict) or meta.get("total_duration") is None:
                return TurnTiming()
            total += int(meta["total_duration"])
            load += int(meta.get("load_duration") or 0)
            prompt += int(meta.get("prompt_eval_count") or 0)
            output += int(meta.get("eval_count") or 0)
        return TurnTiming(model_seconds=total / _NS, load_seconds=load / _NS,
                          prompt_tokens=prompt, output_tokens=output)
    except Exception:
        return TurnTiming()


class TurnTimer(BaseCallbackHandler):
    """Collect each LLM response's Ollama timing as it is produced. Never raises.

        timer = TurnTimer()
        timer.reset()                                   # before each timed turn
        agent.invoke(q, config={"configurable": {...}, "callbacks": [timer]})
        timer.timing()                                  # -> TurnTiming for that turn

    Why a callback and not a history slice: on a multi-turn thread the summariser replaces
    the message list (RemoveMessage(REMOVE_ALL) + summary + tail), so "len(history) before
    the turn" can point past the end or at the wrong messages afterwards. The callback
    never looks at the list, so a rewrite cannot corrupt it.

    SCOPE — WIDER THAN ROUTING'S model_seconds. Callbacks passed in the invoke config
    propagate to nested runnables even when the nested `.invoke()` passes no config
    (langchain_core's child-config context; pinned by tests/test_eval_metrics.py). So this
    counts EVERY model call made during the turn: the agent's replies, plus the memory gate
    (UserMemory compression on turns 5, 10, ...), the conversation summariser, and the
    retriever's mistral if search_current_doc runs. Routing's model_seconds
    (`turn_timing`) counts only the agent's replies. **A memory-check model_seconds and a
    routing model_seconds are not the same measurement: do not compare them naively in
    Feature 3.** For a single-turn thread where neither the gate nor the summariser fires,
    and no document search runs, the two coincide.

    All-or-null, like `turn_timing`: one response without timing (or an errored call)
    nulls the whole turn. Thread-safe: ToolNode may run tools, and so nested model calls,
    on worker threads.
    """

    def __init__(self):
        super().__init__()
        self._lock = threading.Lock()
        self._metas: list = []

    def reset(self) -> None:
        with self._lock:
            self._metas = []

    @property
    def call_count(self) -> int:
        """How many model responses were seen since the last reset."""
        with self._lock:
            return len(self._metas)

    def on_llm_end(self, response, **kwargs) -> None:
        try:
            found = []
            for generations in getattr(response, "generations", None) or []:
                for generation in generations:
                    message = getattr(generation, "message", None)
                    meta = getattr(message, "response_metadata", None)
                    if not meta:
                        #: completion-style models put it here instead
                        meta = getattr(generation, "generation_info", None)
                    found.append(dict(meta) if isinstance(meta, dict) else None)
            if not found:
                found = [None]   # a response with no generation is an unmeasured call
        except Exception:
            found = [None]
        with self._lock:
            self._metas.extend(found)

    def on_llm_error(self, error, **kwargs) -> None:
        #: a call that failed has no timing; recording None nulls the turn rather than
        #: letting the remaining calls read as the whole cost
        with self._lock:
            self._metas.append(None)

    def timing(self) -> TurnTiming:
        with self._lock:
            metas = list(self._metas)
        return _sum_metadata(metas)


def ollama_models(url: Optional[str] = None, timeout: float = 2.0) -> Optional[dict]:
    """Every loaded model as {name: {"size": MiB, "vram": MiB, "ctx": int | None}}.

    None = could not read; {} = read fine, nothing loaded.

    `size` is the model's whole footprint and `vram` the part on the GPU. When VRAM runs
    short Ollama does NOT fail: it quietly puts layers on the CPU, and the only trace is
    `size > vram` (see `offload_mib`). Reading `vram` alone — all Feature 2 did — makes a
    half-offloaded model look like a small one that fits. `ctx` is the context Ollama
    actually loaded, which catches a reload at a context nobody asked for.
    """
    try:
        #: resolved at call time, not as a default argument, so OLLAMA_PS_URL is the one
        #: place to repoint it (a default would freeze the value at import)
        with urllib.request.urlopen(url or OLLAMA_PS_URL, timeout=timeout) as response:
            data = json.load(response)
        #: `size` per model, not per reply: an entry without it (an older server) keeps
        #: its vram reading and simply has no offload verdict, rather than losing all three
        return {str(entry["name"]): {
                    "size": None if entry.get("size") is None
                    else round(int(entry["size"]) / _MIB),
                    "vram": round(int(entry["size_vram"]) / _MIB),
                    "ctx": entry.get("context_length")}
                for entry in data.get("models", [])}
    except Exception:
        return None


def ollama_vram(url: Optional[str] = None, timeout: float = 2.0) -> Optional[dict]:
    """{loaded model: size_vram in MiB}. None = could not read; {} = nothing loaded."""
    models = ollama_models(url, timeout)
    return None if models is None else {name: m["vram"] for name, m in models.items()}


def offload_mib(size_mib: Optional[dict], vram_mib: Optional[dict]) -> dict:
    """{model: MiB on the CPU} for every model whose whole size is NOT on the GPU. Pure.

    Derived by readers, never stored (like the wall - model remainder). A model missing
    from either map is skipped rather than guessed at. Empty = nothing spilled, which is
    the only reading the bake-off's fit gate accepts.
    """
    out = {}
    for name, size in (size_mib or {}).items():
        vram = (vram_mib or {}).get(name)
        if size is not None and vram is not None and size > vram:
            out[name] = size - vram
    return out


def gpu_used_mib(timeout: float = 5.0) -> Optional[int]:
    """Whole-GPU memory in use (first GPU), from nvidia-smi. None if unavailable.

    Covers everything on the card — Ollama, Whisper, Kokoro, embeddings, other apps —
    which is why it is recorded beside, not instead of, the per-model /api/ps numbers.
    """
    try:
        done = subprocess.run(NVIDIA_SMI, capture_output=True, text=True,
                              timeout=timeout, check=False)
        if done.returncode != 0:
            return None
        return int(done.stdout.strip().splitlines()[0].strip())
    except Exception:
        return None


def gpu_total_mib(timeout: float = 5.0) -> Optional[int]:
    """The first GPU's total memory from nvidia-smi (12,282 MiB here). None if unavailable."""
    try:
        done = subprocess.run(NVIDIA_SMI_TOTAL, capture_output=True, text=True,
                              timeout=timeout, check=False)
        if done.returncode != 0:
            return None
        return int(done.stdout.strip().splitlines()[0].strip())
    except Exception:
        return None


def case_metrics(turn_messages) -> dict:
    """The v2 + v5 CaseResult fields for one case, as keyword arguments. Never raises.

    Call after the case's timer has stopped: the readings (~0.2 s for nvidia-smi) must
    not land inside wall_seconds.
    """
    return _with_vram(turn_timing(turn_messages))


def timer_metrics(timer: Optional[TurnTimer]) -> dict:
    """The same eight fields, timed by a TurnTimer instead of the history. Never raises.

    See TurnTimer's docstring: this model_seconds is WIDER than case_metrics'.
    """
    try:
        timing = timer.timing() if timer is not None else TurnTiming()
    except Exception:
        timing = TurnTiming()
    return _with_vram(timing)


def _with_vram(timing: TurnTiming) -> dict:
    #: ONE /api/ps read for all three model fields, so size, vram and ctx describe the
    #: same instant (two reads could straddle a reload and disagree)
    models = ollama_models()
    return {
        "model_seconds": timing.model_seconds,
        "load_seconds": timing.load_seconds,
        "prompt_tokens": timing.prompt_tokens,
        "output_tokens": timing.output_tokens,
        "vram_models": None if models is None else {n: m["vram"] for n, m in models.items()},
        "vram_gpu_mib": gpu_used_mib(),
        "model_size_mib": None if models is None else {n: m["size"] for n, m in models.items()},
        "model_ctx": None if models is None else {n: m["ctx"] for n, m in models.items()},
    }
