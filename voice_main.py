"""Talk to Marshall. V1b — open mic, one utterance at a time, interruptible.

    python voice_main.py --check                 # verify the stack, exit
    python voice_main.py --list-devices          # audio devices, exit
    python voice_main.py                         # talk to it
    python voice_main.py --file path/to/doc.pdf  # over a different document
    python voice_main.py --barge-in off          # V1a behaviour, for speakers

Mic -> VAD endpoint -> Whisper -> the existing synchronous `AgentGraph.invoke` ->
Kokoro -> speakers -> back to listening. No wake word, no streaming. V1b adds barge-in
while Marshall is SPEAKING: talk over him and he stops, and what you said becomes your
next turn. **That mode assumes headphones** — there is no echo cancellation in this
stack, so through speakers his own voice interrupts him. `--barge-in off` is the escape.

**Boot order matters and is not incidental.** Embeddings are warmed first, because
`docs/architecture_review.md` §10 measures the MiniLM load at ~4.8 s and says it is
"still synchronous and still ahead of the first word". Then the agent is built with
one model for both generation and memory compression, pinned resident, so no turn
pays a model swap. Then the audio models. Then one throwaway turn to pull the LLM into
VRAM. Every second of it is boot time, where nobody is waiting on a reply.

**Before any of that, the admin password** (hidden; Enter alone for a guest). An admin
session can have Marshall run code: he asks "Is that okay?" out loud and only a clear
spoken yes — matched in code, never by the model — runs it. A guest session can have code
checked but not run, and keeps no memory. `--check` asks for no password.
"""
import argparse
import asyncio
import logging
import sys
import time
import warnings
from pathlib import Path

from admin_session import prompt_for_session

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

#: One throwaway turn at boot, to pull llama3.1 into VRAM. Without it the FIRST answer of
#: every session pays Ollama's cold model load on top of its own generation — the last
#: multi-second stall V1a left in the request path, and the one the user actually feels
#: because it lands on the first thing they ever say.
#:
#: The thread id is deliberately NOT the session's: a warm-up is not conversation, and on
#: `voice-main` it would sit in history as turn one of every session and be carried into
#: H2's persisted summary forever. The prompt is one the agent's own system prompt names
#: as needing no tool ("hi" -> greet back), so warming cannot fire a network lookup.
WARMUP_THREAD_ID = "voice-warmup"
WARMUP_PROMPT = "hi"


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


def build_agent(file_path: str, session=None, run_confirm=None, quiet: bool = False):
    """Construct the AgentGraph the voice loop talks to.

    Zero changes to `pipeline_lg` beyond Part A — every difference from a text session
    is a constructor argument. `session=None` is the agent `--check` and
    scripts/check_routing.py measure, unchanged.
    """
    from pipeline_lg import AgentGraph

    #: a guest keeps nothing (AgentGraph drops the store), so proving the store durable
    #: would prove something about a path the session never touches
    if session is None or session.is_admin:
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
        session=session,
        run_confirm=run_confirm,
    )


def warm_llm(agent, report=print) -> "float | None":
    """Make the generation model resident before the first real turn. Never fatal.

    Returns the seconds it took, or None if it did not complete. A failure here is
    reported and stepped over on purpose: if Ollama is not up, the session has a larger
    problem than a cold model, and it should surface on a real turn — with the real
    error — rather than as a refusal to boot.
    """
    started = time.perf_counter()
    try:
        agent.invoke(WARMUP_PROMPT,
                     config={"configurable": {"thread_id": WARMUP_THREAD_ID}})
    except Exception as exc:
        report(f"  ! llm warm-up failed ({type(exc).__name__}: {exc})"
               " — the first turn will pay the model load")
        return None
    elapsed = time.perf_counter() - started
    report(f"  {'llm warm-up':<22} {elapsed:6.2f}s")
    return elapsed


def _timed(label: str, build, report) -> tuple:
    started = time.perf_counter()
    result = build()
    elapsed = time.perf_counter() - started
    report(f"  {label:<22} {elapsed:6.2f}s")
    return result, elapsed


def boot(config, file_path: str, report=print, session=None, run_confirm=None) -> dict:
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

    agent, _ = _timed("agent + document",
                      lambda: build_agent(file_path, session=session, run_confirm=run_confirm),
                      report)
    vad, _ = _timed("silero-vad", lambda: SileroVad(config), report)
    stt, _ = _timed("faster-whisper", lambda: Transcriber(config), report)
    tts, _ = _timed("kokoro", lambda: build_engine(config), report)

    #: last, once everything else has taken the VRAM it needs, so the model that has to
    #: STAY resident is the one loaded against what is actually left
    warm_llm(agent, report)

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

def check_microphone(config, seconds: float = 1.5) -> tuple:
    """Open the real capture path and confirm audio is flowing. Returns (ok, summary).

    Nobody has to speak: this asks whether the device opens at 16 kHz mono, whether
    frames arrive at the expected rate, and whether they carry any signal at all. That
    is the difference between "the stack works" and "the stack works on this machine
    with this microphone" — and it is the half a synthesized round trip cannot cover.
    """
    import numpy as np
    from voice.audio_io import MicrophoneStream

    mic = MicrophoneStream(config)
    try:
        mic.start()
    except Exception as exc:
        return False, f"could not open the input device: {type(exc).__name__}: {exc}"
    try:
        frames = []
        deadline = time.perf_counter() + seconds
        while time.perf_counter() < deadline:
            frame = mic.get(0.1)
            if frame is not None:
                frames.append(frame)
    finally:
        mic.stop()

    if not frames:
        return False, "the device opened but delivered no frames"
    audio = np.concatenate(frames)
    rms, peak = float(np.sqrt((audio ** 2).mean())), float(np.abs(audio).max())
    expected = seconds * config.sample_rate / config.blocksize
    summary = (f"{len(frames)} frames in {len(audio)/config.sample_rate:.2f}s "
               f"(expected ~{expected:.0f}), rms={rms:.4f} peak={peak:.4f}, "
               f"overflows={mic.overflow_count}")
    if peak == 0.0:
        #: opens and delivers digital silence — a muted device or the wrong input
        return False, summary + "  -> SILENT: muted, or the wrong input selected"
    if rms < 1e-4:
        #: not a failure. A quiet room is quiet, and failing the check for it would
        #: teach the user to ignore the check.
        return True, summary + "  -> very quiet (fine if the room is silent)"
    return True, summary + "  -> LIVE"


def run_check(config, file_path: str) -> int:
    """Load every model, print each load time, probe the mic, then TTS -> STT round trip.

    Exercises both audio models end to end without anyone speaking, so a failure here is
    unambiguous: if `--check` passes and a live session does not, the problem is the
    room or the thresholds, not the stack or the devices.
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
    print("MICROPHONE  (no need to speak)")
    print("=" * 72)
    import sounddevice as sd
    chosen = ("system default" if config.input_device is None
              else f"[{config.input_device}] {sd.query_devices(config.input_device)['name']}")
    print(f"  device     {chosen}")
    mic_ok, mic_summary = check_microphone(config)
    print(f"  capture    {mic_summary}")
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

    round_trip_ok = overlap >= 0.8
    print("=" * 72)
    if round_trip_ok and mic_ok:
        print("CHECK PASSED - the voice stack is wired end to end.")
    else:
        if not round_trip_ok:
            print(f"CHECK FAILED - round trip lost too much: {overlap:.0%} token "
                  "overlap (need 80%).")
        if not mic_ok:
            #: reported separately because the fix is different: the round trip is the
            #: models, the microphone is the device selection
            print("CHECK FAILED - the microphone did not produce usable audio. "
                  "Pick another with --list-devices then "
                  '--input-device "<name>".')
    print("=" * 72)
    return 0 if (round_trip_ok and mic_ok) else 1


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
            symbol = {"listening": "[..]", "thinking": "[**]", "confirming": "[?!]",
                      "speaking": "[<<]", "stopped": "[--]"}.get(payload["state"], "[??]")
            print(f"\r{symbol} {payload['state']:<10}", end="", flush=True)
        elif event == "ready":
            print(f"\nlistening on thread {payload['thread_id']!r} "
                  f"(barge-in: {payload['barge_in']}). Ctrl+C to stop.\n")
        elif event == "transcript" and payload.get("text"):
            print(f"\r  you   ({payload['ms']:.0f}ms)  {payload['text']}")
        elif event == "response" and payload.get("text"):
            print(f"  M     ({payload['ms']:.0f}ms)  {payload['text']}")
        elif event == "barge_in":
            #: printed the moment playback stops, because this is the line the guard gets
            #: tuned from: too many of these means the thresholds are too eager
            print(f"\r  ><    interrupted — silent in {payload['ms']:.0f}ms")
        elif event == "truncated":
            print(f"  ...   kept {payload['sentences_played']} of "
                  f"{payload['sentences_total']} sentence(s) in history")
        elif event == "discarded":
            print(f"\r  ...   discarded ({payload['reason']})")
        elif event == "error":
            print(f"\r  !     turn {payload['turn_id']} failed: {payload['error']}")
        elif event == "confirm_request":
            #: everything that would run, escaped — the same rule as terminal_confirm
            from admin_session import escape_for_terminal
            print(f"\r  ?     Marshall wants to run this {payload['language']} "
                  f"({escape_for_terminal(payload['origin'])}):")
            for line in payload["source"].split("\n"):
                print(f"        {escape_for_terminal(line)}")
        elif event == "confirm":
            #: the run_code answer as Whisper heard it, so a wrong "no" is visible
            verdict = "yes" if payload["accepted"] else "no"
            print(f"\r  ?     run it? heard {payload['transcript']!r} -> {verdict}")
    return report


async def run_voice(config, file_path: str, session=None) -> int:
    import voice.orchestrator as orchestrator_module
    from voice.confirm import VoiceConfirmer

    #: built empty and bound below: the agent (whose run_code closes over this) is built
    #: inside boot(), before the orchestrator that owns the mic and speaker exists
    confirmer = VoiceConfirmer()
    parts = boot(config, file_path, session=session, run_confirm=confirmer)
    orchestrator = orchestrator_module.VoiceOrchestrator(
        agent=parts["agent"], mic=parts["mic"].start(), vad=parts["vad"],
        stt=parts["stt"], tts=parts["tts"], speaker=parts["speaker"],
        config=config, on_event=_make_reporter(),
    )
    confirmer.orchestrator = orchestrator
    await orchestrator_module.run_session(orchestrator)
    print(f"\nstopped after {orchestrator.turn_id} turn(s). "
          f"timings in {config.telemetry_path}")
    return 0


# ==========================================================================
# cli
# ==========================================================================

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="voice_main.py",
        description="Talk to Marshall (V1b: open mic, no wake word, barge-in while "
                    "he is speaking).")
    parser.add_argument("--file", default=DEFAULT_DOCUMENT,
                        help=f"document to load (default: {DEFAULT_DOCUMENT})")
    parser.add_argument("--check", action="store_true",
                        help="load every model, run a TTS->STT round trip, exit")
    parser.add_argument("--list-devices", action="store_true",
                        help="print audio devices and exit")
    #: str, not int: a name substring survives PortAudio renumbering devices when
    #: something connects, and picks the host API that can actually do 16 kHz
    parser.add_argument("--input-device", default=None,
                        help='input device index or name substring, e.g. 15 or "Realtek"')
    parser.add_argument("--output-device", default=None,
                        help='output device index or name substring, e.g. 20 or "Realtek"')
    #: the escape hatch for a speaker user. There is no echo cancellation in the stack,
    #: so on speakers Marshall's own voice reaches the mic and trips the interrupt guard;
    #: "off" is V1a's behaviour and is better than fighting it.
    parser.add_argument("--barge-in", choices=("vad", "off"), default="vad",
                        help="interrupt Marshall while he speaks. 'vad' needs headphones "
                             "(no echo cancellation); 'off' disables it (default: vad)")
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

    #: resolved to concrete indices at startup, so an unusable device fails HERE with a
    #: message naming the fix — rather than as a raw PortAudioError from inside the
    #: audio callback once the models are already loaded
    from voice.audio_io import resolve_device, DeviceNotUsableError
    from voice.config import VAD_SAMPLE_RATE
    try:
        input_device = resolve_device(args.input_device, VAD_SAMPLE_RATE,
                                      channels=1, want_input=True)
        output_device = resolve_device(args.output_device, VoiceConfig().tts_sample_rate,
                                       channels=1, want_input=False)
    except DeviceNotUsableError as exc:
        print(f"audio device error: {exc}", file=sys.stderr)
        return 2

    config = VoiceConfig(
        input_device=input_device,
        output_device=output_device,
        thread_id=args.thread,
        barge_in_mode=args.barge_in,
    )
    config.validate()

    if not Path(args.file).exists():
        print(f"no such document: {args.file}", file=sys.stderr)
        return 2

    if args.check:
        #: no password: --check builds today's session-less agent and never takes a turn
        return run_check(config, args.file)

    #: before boot, so the password is typed while nothing is loading over it
    session = prompt_for_session()
    try:
        return asyncio.run(run_voice(config, args.file, session=session))
    except KeyboardInterrupt:
        #: Ctrl+C during BOOT, before the loop exists to catch it. Clean exit, no spray.
        print("\ninterrupted during startup.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
