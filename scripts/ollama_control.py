"""Put Ollama in a known state before a measured run: nothing loaded, then one model warm.

Feature 3 prep step 6. `scripts/eval_metrics.py` only READS Ollama; this module ACTS on
it, which is why it is separate. Used by `run_evals.py --clean` (and `--with-voice`,
which implies it). A plain `run_evals` never calls it, so its documented behaviour —
record what an earlier run left loaded, never sanitise it — is unchanged.

Why each piece exists:

* **unload_all** — with `keep_alive="30m"` the previous candidate stays in VRAM. It
  corrupts the next candidate's VRAM numbers and can push it partly onto the CPU. An
  unload request returns at once while the model leaves in the background (a stale
  `/api/ps` right after `keep_alive:0` was seen on 24 Nov), so this POLLS until `/api/ps`
  is empty, and reports what is still there if it never empties.
* **warm** — loads the model ONCE, at exactly the `num_ctx` the checks will send. Ollama
  reloads a model whenever a request's num_ctx differs from the loaded one, so a warm at
  the wrong context is a reload in the first case; no warm at all puts the cold load
  (19.5 s on 24 Nov) inside the first case's wall time. Same idea as voice_main.warm_llm.

Every function takes its HTTP and clock dependencies as arguments so CI can drive them
with fakes; the defaults talk to the real server.
"""
import json
import time
import urllib.request
from typing import Callable, Optional

from scripts.eval_metrics import ollama_models

OLLAMA_URL = "http://127.0.0.1:11434"

#: how long unload_all waits for /api/ps to empty. An 8B model leaves in well under a
#: second; a minute only matters when something is wrong, and then the run must not start.
UNLOAD_TIMEOUT_S = 60.0
UNLOAD_POLL_S = 0.5

#: generous: a cold 8–9B load from disk took ~8 s here, a larger candidate takes longer
WARM_TIMEOUT_S = 300.0


def post_json(path: str, body: dict, timeout: float = WARM_TIMEOUT_S) -> dict:
    """POST to Ollama and return the (last line of the) JSON reply. Raises on failure."""
    request = urllib.request.Request(OLLAMA_URL + path, json.dumps(body).encode("utf-8"),
                                     {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        text = response.read().decode("utf-8").strip()
    return json.loads(text.splitlines()[-1]) if text else {}


def unload_all(timeout_s: float = UNLOAD_TIMEOUT_S, poll_s: float = UNLOAD_POLL_S,
               read: Callable[[], Optional[dict]] = ollama_models,
               post: Callable[..., dict] = post_json,
               sleep: Callable[[float], None] = time.sleep,
               clock: Callable[[], float] = time.monotonic) -> Optional[dict]:
    """Ask every loaded model to leave, then wait until /api/ps is empty.

    Returns {} when nothing is loaded any more; {name: info} for whatever was still
    loaded at the timeout; None when /api/ps could not be read at all (Ollama down) —
    in which case nothing is known and the caller must not treat the state as clean.
    """
    models = read()
    if models is None:
        return None
    for name in models:
        try:
            post("/api/generate", {"model": name, "keep_alive": 0}, timeout=30)
        except Exception:
            pass          # the poll below is the judge, not this request
    deadline = clock() + timeout_s
    while True:
        models = read()
        if models == {}:
            return {}
        if clock() >= deadline:
            return models
        sleep(poll_s)


def warm_request(model: str, num_ctx: int, keep_alive) -> dict:
    """The one request that loads `model` exactly as the checks will use it. Pure.

    num_ctx and keep_alive match the checks' ChatOllama, so their first request finds the
    model already loaded at the right context. think:false matches CHECK_REASONING (a
    no-op on a model without thinking; never true, which such a model rejects with 400).
    num_predict=1: the answer is irrelevant, only the load is wanted.
    """
    return {"model": model, "messages": [{"role": "user", "content": "hi"}],
            "stream": False, "think": False, "keep_alive": keep_alive,
            "options": {"num_ctx": num_ctx, "temperature": 0, "num_predict": 1}}


def warm(model: str, num_ctx: int, keep_alive,
         post: Callable[..., dict] = post_json,
         clock: Callable[[], float] = time.perf_counter) -> dict:
    """Load `model` at `num_ctx`. Never raises: a failed warm is reported, and the checks
    then report the same problem (model not pulled, Ollama down) with their own exit 2.

    Returns {"seconds", "load_seconds", "error"}; `load_seconds` is Ollama's own figure.
    """
    started = clock()
    try:
        reply = post("/api/chat", warm_request(model, num_ctx, keep_alive),
                     timeout=WARM_TIMEOUT_S)
        load_ns = reply.get("load_duration")
        return {"seconds": clock() - started,
                "load_seconds": None if load_ns is None else load_ns / 1e9,
                "error": None}
    except Exception as exc:
        return {"seconds": clock() - started, "load_seconds": None,
                "error": f"{type(exc).__name__}: {exc}"}
