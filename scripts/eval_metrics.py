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
"""
import json
import subprocess
import urllib.request
from dataclasses import dataclass
from typing import Optional

OLLAMA_PS_URL = "http://127.0.0.1:11434/api/ps"
NVIDIA_SMI = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]

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
        if not replies:
            return TurnTiming()
        total = load = prompt = output = 0
        for reply in replies:
            meta = getattr(reply, "response_metadata", None) or {}
            if meta.get("total_duration") is None:
                return TurnTiming()
            total += int(meta["total_duration"])
            load += int(meta.get("load_duration") or 0)
            prompt += int(meta.get("prompt_eval_count") or 0)
            output += int(meta.get("eval_count") or 0)
        return TurnTiming(model_seconds=total / _NS, load_seconds=load / _NS,
                          prompt_tokens=prompt, output_tokens=output)
    except Exception:
        return TurnTiming()


def ollama_vram(url: Optional[str] = None, timeout: float = 2.0) -> Optional[dict]:
    """{loaded model: size_vram in MiB}. None = could not read; {} = nothing loaded."""
    try:
        #: resolved at call time, not as a default argument, so OLLAMA_PS_URL is the one
        #: place to repoint it (a default would freeze the value at import)
        with urllib.request.urlopen(url or OLLAMA_PS_URL, timeout=timeout) as response:
            data = json.load(response)
        return {str(entry["name"]): round(int(entry["size_vram"]) / _MIB)
                for entry in data.get("models", [])}
    except Exception:
        return None


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


def case_metrics(turn_messages) -> dict:
    """The v2 CaseResult fields for one case, as keyword arguments. Never raises.

    Call after the case's timer has stopped: the readings (~0.2 s for nvidia-smi) must
    not land inside wall_seconds.
    """
    timing = turn_timing(turn_messages)
    return {
        "model_seconds": timing.model_seconds,
        "load_seconds": timing.load_seconds,
        "prompt_tokens": timing.prompt_tokens,
        "output_tokens": timing.output_tokens,
        "vram_models": ollama_vram(),
        "vram_gpu_mib": gpu_used_mib(),
    }
