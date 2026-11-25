"""Hold the voice models in VRAM while the eval checks run: the "fits with voice" test.

    python scripts/voice_ballast.py            # load, one round trip, hold until Enter
    python scripts/voice_ballast.py --serve    # what run_evals --with-voice launches

Feature 3 prep step 5. A live voice session holds faster-whisper, Kokoro and Silero on
top of the LLM. The eval checks build only the agent, so on their own they measure a
model that fits WITHOUT voice. This process loads voice_main's real audio models from
the same VoiceConfig, so a candidate is measured against the memory voice actually takes.

**What it loads, and what it deliberately does not:**
  * faster-whisper large-v3-turbo (int8_float16, its own CUDA context), Kokoro-82M (on
    torch's CUDA), Silero VAD (CPU) — exactly voice_main.boot()'s audio models.
  * NOT the MiniLM embedder: the check process already loads it, with torch's CUDA
    context, because the agent builds the embeddings. A second copy here would fail
    models on memory that does not exist in a real session.
  * NOT the agent or any document index, so it can never race a Chroma directory.

**The one place it still over-counts:** in real voice, Kokoro and the embedder share ONE
process and so one torch CUDA context; here the ballast and the check each have one. The
"torch CUDA context" stage reports that cost, so a verdict can say how conservative it is.

**Why a round trip and not just a load:** CTranslate2 and torch both grow their memory
pools on first use. A freshly loaded model under-reports what it holds mid-conversation,
so the ballast synthesises CHECK_PHRASE with Kokoro and transcribes it with Whisper —
voice_main --check's round trip without the speakers — before it reports ready.

Protocol (--serve): one JSON object per line on stdout, each prefixed `BALLAST `, so a
library writing its own output to stdout can never be mistaken for a message.
    -> ready  {stages: [...], warmup: {...}, gpu_mib}   (after the warm-up round trip)
    <- pass   -> pass {ok, tts_ms, stt_ms, round_trip_ms, overlap, transcript, gpu_mib, error}
    <- quit   -> bye
stdin closing is a quit, so the ballast can never outlive the runner that started it.
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable, Optional

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.eval_metrics import gpu_used_mib  # noqa: E402

PREFIX = "BALLAST "

#: a round trip that loses more than this much of the phrase is a broken voice stack,
#: the same bar voice_main --check uses
MIN_OVERLAP = 0.8


def emit(out, event: str, **data) -> None:
    out.write(PREFIX + json.dumps({"event": event, **data}) + "\n")
    out.flush()


def parse_message(line: str) -> Optional[dict]:
    """The message on a protocol line, or None for any other output. Pure."""
    if not line.startswith(PREFIX):
        return None
    try:
        message = json.loads(line[len(PREFIX):])
    except ValueError:
        return None
    return message if isinstance(message, dict) and "event" in message else None


class RealVoice:
    """voice_main's audio models, built from the same VoiceConfig. Needs a GPU."""

    def __init__(self, config=None):
        from voice.config import VoiceConfig
        self.config = config or VoiceConfig()
        self.vad = self.stt = self.tts = None

    def stages(self) -> list:
        """(name, loader) in voice_main.boot()'s order. The CUDA context comes first so
        its cost is reported as its own line rather than hidden inside Kokoro's."""
        return [
            ("torch CUDA context", self._torch_cuda),
            ("silero-vad", self._load_vad),
            ("faster-whisper", self._load_stt),
            ("kokoro", self._load_tts),
        ]

    def _torch_cuda(self):
        import torch
        if torch.cuda.is_available():
            torch.zeros(1, device="cuda")      # forces the context into existence
            return "cuda"
        return "cpu (no CUDA: Kokoro will run on the CPU)"

    def _load_vad(self):
        from voice.vad import SileroVad
        self.vad = SileroVad(self.config)
        return "cpu"

    def _load_stt(self):
        from voice.stt import Transcriber
        self.stt = Transcriber(self.config)
        fell = " (FELL BACK)" if self.stt.fell_back else ""
        return f"{self.stt.device}/{self.stt.compute_type}{fell}"

    def _load_tts(self):
        from voice.tts import build_engine
        self.tts = build_engine(self.config)
        return self.tts.engine_name

    def round_trip(self) -> dict:
        """Kokoro synthesises CHECK_PHRASE, Whisper transcribes it back, Silero scores a
        frame. The same models doing the same work as a voice turn, minus the speakers."""
        import numpy as np
        from voice.audio_io import resample_linear
        from voice.config import VAD_FRAME_SAMPLES
        from voice_main import CHECK_PHRASE, token_overlap

        started = time.perf_counter()
        audio = self.tts.synthesize(CHECK_PHRASE)
        tts_ms = (time.perf_counter() - started) * 1000
        if len(audio) == 0:
            return {"ok": False, "tts_ms": tts_ms, "error": "Kokoro produced no audio"}
        speech = resample_linear(audio, self.config.tts_sample_rate, self.config.sample_rate)
        self.vad.speech_prob(np.asarray(speech[:VAD_FRAME_SAMPLES], dtype=np.float32))
        started = time.perf_counter()
        transcript = self.stt.transcribe(speech)
        stt_ms = (time.perf_counter() - started) * 1000
        overlap = token_overlap(CHECK_PHRASE, transcript)
        return {"ok": overlap >= MIN_OVERLAP, "tts_ms": tts_ms, "stt_ms": stt_ms,
                "round_trip_ms": tts_ms + stt_ms, "overlap": overlap,
                "transcript": transcript,
                "error": None if overlap >= MIN_OVERLAP else
                f"round trip kept only {overlap:.0%} of the phrase (need {MIN_OVERLAP:.0%})"}


def load(voice, gpu: Callable[[], Optional[int]] = gpu_used_mib, log=None) -> list:
    """Run each stage, recording seconds and the whole-GPU reading after it."""
    stages = []
    before = gpu()
    for name, loader in voice.stages():
        started = time.perf_counter()
        device = loader()
        after = gpu()
        stages.append({"stage": name, "device": device,
                       "seconds": round(time.perf_counter() - started, 2), "gpu_mib": after,
                       "delta_mib": None if before is None or after is None else after - before})
        if log:
            log(f"  {name:<20} {stages[-1]['seconds']:6.2f}s  {device:<28} "
                f"+{stages[-1]['delta_mib'] if stages[-1]['delta_mib'] is not None else '?'} MiB")
        before = after
    return stages


def one_pass(voice, gpu: Callable[[], Optional[int]] = gpu_used_mib) -> dict:
    """One round trip that never raises: an exception is a failed pass, not a crash."""
    try:
        result = voice.round_trip()
    except Exception as exc:
        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    result["gpu_mib"] = gpu()
    return result


def warm_up(voice, stages: list, gpu: Callable[[], Optional[int]] = gpu_used_mib,
            log=None) -> dict:
    """One throwaway round trip, recorded as its own stage. Returns the pass result.

    The FIRST round trip is not a steady-state one: on 25 Nov it took 2,402 ms against
    417 ms for the next, Kokoro's first synthesis alone 1.8 s. Used as the reference for
    the FIT slowdown gate it would let a later pass be ~5.8x slower than a warm one and
    still "pass 1.5x". It also grows both libraries' memory pools (+419 MiB that day),
    which is exactly the memory a live session holds — so it belongs before `ready`.
    """
    before = stages[-1]["gpu_mib"] if stages else gpu()
    started = time.perf_counter()
    result = one_pass(voice, gpu)
    after = result.get("gpu_mib")
    stages.append({"stage": "warm-up round trip", "device": "ok" if result["ok"] else "FAILED",
                   "seconds": round(time.perf_counter() - started, 2), "gpu_mib": after,
                   "delta_mib": None if before is None or after is None else after - before})
    if log:
        log(f"  {'warm-up round trip':<20} {stages[-1]['seconds']:6.2f}s  "
            f"{stages[-1]['device']:<28} +{stages[-1]['delta_mib'] if stages[-1]['delta_mib'] is not None else '?'} MiB")
    return result


def serve(voice, stdin, out, gpu: Callable[[], Optional[int]] = gpu_used_mib) -> int:
    """The --serve loop. Exit 0 on quit or stdin EOF; 2 if loading failed (and says why)."""
    log = lambda text: print(text, file=sys.stderr)  # noqa: E731
    try:
        stages = load(voice, gpu, log=log)
    except Exception as exc:
        emit(out, "error", error=f"loading failed: {type(exc).__name__}: {exc}")
        return 2
    warmup = warm_up(voice, stages, gpu, log=log)
    #: a failed warm-up is reported, not fatal: the next pass (the FIT reference) will
    #: fail too, and the verdict then says so with the reason
    emit(out, "ready", stages=stages, warmup=warmup, gpu_mib=gpu())
    for line in stdin:
        command = line.strip().lower()
        if command == "pass":
            emit(out, "pass", **one_pass(voice, gpu))
        elif command == "quit":
            break
        elif command:
            emit(out, "error", error=f"unknown command {command!r}")
    emit(out, "bye")
    return 0


def main(argv=None, voice=None) -> int:
    parser = argparse.ArgumentParser(
        prog="scripts/voice_ballast.py",
        description="Hold voice_main's audio models (Whisper, Kokoro, Silero) in VRAM.")
    parser.add_argument("--serve", action="store_true",
                        help="speak the line protocol on stdin/stdout (run_evals --with-voice)")
    args = parser.parse_args(argv)
    voice = voice or RealVoice()

    if args.serve:
        #: protocol lines go to the REAL stdout; anything Python code prints goes to stderr,
        #: so library chatter cannot interleave with a half-written message
        protocol, sys.stdout = sys.stdout, sys.stderr
        try:
            return serve(voice, sys.stdin, protocol)
        finally:
            sys.stdout = protocol

    print("loading the voice models (no embedder, no agent: see the module docstring)")
    print(f"  {'idle':<20} {'':>7}  {'':<28} {gpu_used_mib()} MiB in use")
    stages = load(voice, log=print)
    warm_up(voice, stages, log=print)
    result = one_pass(voice)
    print(f"  round trip           {result.get('round_trip_ms', 0):6.0f}ms  "
          f"overlap {result.get('overlap', 0):.0%}  {'ok' if result['ok'] else result['error']}")
    print(f"  holding at {result['gpu_mib']} MiB — press Enter to release")
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        pass
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
