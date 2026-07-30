"""Talk to Marshall. V1a — open mic, one utterance at a time.

    python voice_main.py --check                 # verify the stack, exit
    python voice_main.py --list-devices          # audio devices, exit
    python voice_main.py                         # talk to it
    python voice_main.py --file path/to/doc.pdf  # over a different document

Mic -> VAD endpoint -> Whisper -> the existing synchronous `AgentGraph.invoke` ->
Kokoro -> speakers -> back to listening. No wake word, no barge-in, no streaming.

**Boot order matters and is not incidental.** Embeddings are warmed first, because
`docs/architecture_review.md` §10 measures the MiniLM load at ~4.8 s and says it is
"still synchronous and still ahead of the first word". Then the agent is built with
one model for both generation and memory compression, pinned resident, so no turn
pays a model swap. Then the audio models. Every second of it is boot time, where
nobody is waiting on a reply.
"""
import argparse
import asyncio
import logging
import sys
import time
import warnings
from pathlib import Path

#: default corpus — the same PDF R4's routing verification was measured against
DEFAULT_DOCUMENT = "test_pdf's/Pipes_2025.pdf"

#: One model, held resident, for BOTH generation and the memory gate. Two separate
#: reasons, and they compound: llama3.1 is the only locally-pulled model that reliably
#: emits tool_calls (mistral narrates them as prose), and Ollama serves one model per
#: GPU at a time — so a gate asking for mistral on turns 5, 10, 15… would force an
#: unload + reload into the request path on top of finding J's compression stall.
VOICE_MODEL = "llama3.1"

#: long enough that no realistic pause between turns drops the model, short enough that
#: quitting eventually gives the GPU back. "-1" (forever) would strand 4.9 GB after exit.
KEEP_ALIVE = "30m"

#: durable by default, and separately from the text pipelines' scratch directories.
#: The C2 failure — long-term memory silently in-memory, working perfectly all session
#: and gone at exit — is the one `docs/architecture_review.md` calls "the most
#: misleading failure mode in the system". `_ensure_durable_memory` proves it is closed
#: at boot rather than trusting these strings.
VECTOR_DB_PATH = "db/voice_vdb"
USER_HISTORY_PATH = "db/voice_user_history"
SUMMARY_PATH = "db/voice_summaries"

CHECK_PHRASE = "The quick brown fox jumps over the lazy dog."


# ==========================================================================
# boot
# ==========================================================================

def _ensure_durable_memory(path: str) -> None:
    """Prove long-term memory will persist, before a single word is spoken.

    Creates the directory and constructs a `UserMemory` against it with warnings
    captured, asserting the C2 `UserWarning` does not fire. Cheap, and it turns "we
    passed a path so it must be durable" into something actually checked — the whole
    point of C2 is that the broken version looks identical until the process exits.
    """
    from memory.long_term_memory import UserMemory

    Path(path).mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        UserMemory(chat_history=[], collection_name="user_history",
                   persist_directory_path=path)
    durability_warnings = [
        str(w.message) for w in caught
        if issubclass(w.category, UserWarning) and "in-memory only" in str(w.message)
    ]
    if durability_warnings:
        raise RuntimeError(
            "long-term memory would not persist (C2): " + durability_warnings[0])


def build_agent(file_path: str, quiet: bool = False):
    """Construct the AgentGraph the voice loop talks to.

    Zero changes to `pipeline_lg` beyond Part A — every difference from a text session
    is a constructor argument.
    """
    from pipeline_lg import AgentGraph

    _ensure_durable_memory(USER_HISTORY_PATH)
    return AgentGraph(
        file_path=file_path,
        mode_docs="pdf" if str(file_path).lower().endswith(".pdf") else "txt",
        persist_directory_path=VECTOR_DB_PATH,
        model=VOICE_MODEL,
        #: same model as generation — see VOICE_MODEL
        memory_model=VOICE_MODEL,
        keep_alive=KEEP_ALIVE,
        user_history_directory_path=USER_HISTORY_PATH,
        #: H2's summary reload gives cross-session continuity for free: `invoke` reloads
        #: this thread's stored summary whenever the in-RAM history is empty, and the
        #: thread_id is fixed, so tomorrow's session opens knowing what today's covered.
        summary_persist_path=SUMMARY_PATH,
    )


def _timed(label: str, build, report) -> tuple:
    started = time.perf_counter()
    result = build()
    elapsed = time.perf_counter() - started
    report(f"  {label:<22} {elapsed:6.2f}s")
    return result, elapsed


def boot(config, file_path: str, report=print) -> dict:
    """Load everything, in the order that keeps cost off the first turn."""
    from warmup import warm_embeddings
    from voice.audio_io import MicrophoneStream, Speaker
    from voice.stt import Transcriber
    from voice.tts import build_engine
    from voice.vad import SileroVad

    report("booting Marshall (voice)")
    total = time.perf_counter()

    embed_started = time.perf_counter()
    embed_timings = warm_embeddings()
    #: the wrapper total exceeds the two model builds because this is also where the
    #: langchain/torch import graph is paid for — first import, once per process. Both
    #: numbers are shown rather than one, because they are cut by different things:
    #: the models by warming, the imports only by importing less.
    report(f"  {'embeddings (MiniLM x2)':<22} {time.perf_counter() - embed_started:6.2f}s"
           f"   [models {embed_timings['ingestion']:.2f}s + {embed_timings['memory']:.2f}s, "
           f"rest is first-import]")

    agent, _ = _timed("agent + document", lambda: build_agent(file_path), report)
    vad, _ = _timed("silero-vad", lambda: SileroVad(config), report)
    stt, _ = _timed("faster-whisper", lambda: Transcriber(config), report)
    tts, _ = _timed("kokoro", lambda: build_engine(config), report)

    if getattr(stt, "fell_back", False):
        report(f"  ! STT fell back to {stt.device}/{stt.compute_type} — see the warning above")
    else:
        report(f"    STT on {stt.device}/{stt.compute_type}")
    report(f"  booted in {time.perf_counter() - total:.2f}s")

    return {
        "agent": agent,
        "vad": vad,
        "stt": stt,
        "tts": tts,
        "mic": MicrophoneStream(config),
        "speaker": Speaker(config),
    }


# ==========================================================================
# --check
# ==========================================================================

def run_check(config, file_path: str) -> int:
    """Load every model, print each load time, then TTS -> speakers -> STT round trip.

    Exercises both audio models end to end without a microphone, so a failure here is
    unambiguous: if `--check` passes and a live session does not, the problem is the
    mic or the room, not the stack.
    """
    from voice.audio_io import list_devices, Speaker, resample_linear

    print("=" * 72)
    print("AUDIO DEVICES")
    print("=" * 72)
    print(list_devices())
    print()

    print("=" * 72)
    print("MODEL LOAD")
    print("=" * 72)
    parts = boot(config, file_path)
    print()

    print("=" * 72)
    print("ROUND TRIP  (synthesize -> play -> transcribe)")
    print("=" * 72)
    print(f"  engine     {parts['tts'].engine_name} / {config.tts_voice}")
    print(f"  phrase     {CHECK_PHRASE!r}")

    started = time.perf_counter()
    audio = parts["tts"].synthesize(CHECK_PHRASE)
    synth_ms = (time.perf_counter() - started) * 1000
    if len(audio) == 0:
        print("  FAILED     TTS produced no audio")
        return 1
    print(f"  synthesis  {synth_ms:.0f}ms -> {len(audio)/config.tts_sample_rate:.2f}s "
          f"of audio @{config.tts_sample_rate}Hz")

    started = time.perf_counter()
    Speaker(config).play(audio)
    print(f"  playback   {(time.perf_counter()-started)*1000:.0f}ms")

    #: Kokoro is 24 kHz, Whisper assumes 16 kHz — the live path never resamples, but
    #: feeding TTS output back into STT does
    started = time.perf_counter()
    transcript = parts["stt"].transcribe(
        resample_linear(audio, config.tts_sample_rate, config.sample_rate))
    stt_ms = (time.perf_counter() - started) * 1000
    print(f"  transcribe {stt_ms:.0f}ms -> {transcript!r}")

    overlap = token_overlap(CHECK_PHRASE, transcript)
    print(f"  overlap    {overlap:.0%}")
    print(f"  round trip {synth_ms + stt_ms:.0f}ms (synthesis + transcription)")
    print()

    ok = overlap >= 0.8
    print("=" * 72)
    if ok:
        print("CHECK PASSED - the voice stack is wired end to end.")
    else:
        print(f"CHECK FAILED - round trip lost too much: {overlap:.0%} token overlap "
              "(need 80%).")
    print("=" * 72)
    return 0 if ok else 1


def token_overlap(expected: str, actual: str) -> float:
    """Fraction of `expected`'s words present in `actual`, case- and punctuation-blind.

    A word-set measure rather than string equality: a correct transcription may still
    differ in casing, a trailing period, or "dog" vs "dog." — none of which say
    anything about whether the audio survived the round trip.
    """
    import re
    tokenize = lambda text: set(re.findall(r"[a-z0-9']+", (text or "").lower()))
    wanted = tokenize(expected)
    if not wanted:
        return 0.0
    return len(wanted & tokenize(actual)) / len(wanted)


# ==========================================================================
# live session
# ==========================================================================

def _make_reporter():
    """Terminal output for the running loop. The orchestrator stays UI-free."""
    def report(event: str, payload: dict) -> None:
        if event == "state":
            symbol = {"listening": "[..]", "thinking": "[**]",
                      "speaking": "[<<]", "stopped": "[--]"}.get(payload["state"], "[??]")
            print(f"\r{symbol} {payload['state']:<10}", end="", flush=True)
        elif event == "ready":
            print(f"\nlistening on thread {payload['thread_id']!r}. Ctrl+C to stop.\n")
        elif event == "transcript" and payload.get("text"):
            print(f"\r  you   ({payload['ms']:.0f}ms)  {payload['text']}")
        elif event == "response" and payload.get("text"):
            print(f"  M     ({payload['ms']:.0f}ms)  {payload['text']}")
        elif event == "discarded":
            print(f"\r  ...   discarded ({payload['reason']})")
        elif event == "error":
            print(f"\r  !     turn {payload['turn_id']} failed: {payload['error']}")
    return report


async def run_voice(config, file_path: str) -> int:
    from voice.orchestrator import VoiceOrchestrator, run_session

    parts = boot(config, file_path)
    orchestrator = VoiceOrchestrator(
        agent=parts["agent"], mic=parts["mic"].start(), vad=parts["vad"],
        stt=parts["stt"], tts=parts["tts"], speaker=parts["speaker"],
        config=config, on_event=_make_reporter(),
    )
    await run_session(orchestrator)
    print(f"\nstopped after {orchestrator.turn_id} turn(s). "
          f"timings in {config.telemetry_path}")
    return 0


# ==========================================================================
# cli
# ==========================================================================

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="voice_main.py",
        description="Talk to Marshall (V1a: open mic, no wake word, no barge-in).")
    parser.add_argument("--file", default=DEFAULT_DOCUMENT,
                        help=f"document to load (default: {DEFAULT_DOCUMENT})")
    parser.add_argument("--check", action="store_true",
                        help="load every model, run a TTS->STT round trip, exit")
    parser.add_argument("--list-devices", action="store_true",
                        help="print audio devices and exit")
    parser.add_argument("--input-device", type=int, default=None,
                        help="input device index (see --list-devices)")
    parser.add_argument("--output-device", type=int, default=None,
                        help="output device index (see --list-devices)")
    parser.add_argument("--thread", default="voice-main",
                        help="conversation thread_id; fixed across runs so H2's "
                             "summary reload carries continuity (default: voice-main)")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    from voice.audio_io import list_devices
    from voice.config import VoiceConfig

    if args.list_devices:
        print(list_devices())
        return 0

    config = VoiceConfig(
        input_device=args.input_device,
        output_device=args.output_device,
        thread_id=args.thread,
    )
    config.validate()

    if not Path(args.file).exists():
        print(f"no such document: {args.file}", file=sys.stderr)
        return 2

    if args.check:
        return run_check(config, args.file)

    try:
        return asyncio.run(run_voice(config, args.file))
    except KeyboardInterrupt:
        #: Ctrl+C during BOOT, before the loop exists to catch it. Clean exit, no spray.
        print("\ninterrupted during startup.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
