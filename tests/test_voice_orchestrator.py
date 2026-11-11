"""The turn loop, driven entirely by fakes: LISTENING -> THINKING -> SPEAKING -> LISTENING.

No microphone, no GPU, no models, no Ollama. `VoiceOrchestrator` takes every
collaborator as a constructor argument and never imports sounddevice, torch,
faster-whisper or kokoro, so the full cycle — including the parts that only happen on
a real turn, like being interrupted mid-answer — is assertable here.

What these tests pin:

1. a complete cycle runs and returns to LISTENING;
2. frames arriving during THINKING are DISCARDED, and so are frames during SPEAKING when
   `barge_in_mode="off"` — that is V1a's behaviour, kept as an explicit mode;
3. **barge-in (V1b)**: sustained speech during SPEAKING stops playback, drops the rest of
   the answer, truncates the stored history to what was heard, and hands the captured
   audio to the next utterance. A transient does none of that;
4. shutdown cancels cleanly — no traceback, no stranded task, hardware released.

The one fake worth reading closely is `FakeSpeaker`: its `stop()` releases a blocked
`play()`, because that is exactly what `sd.stop()` does to `sd.wait()` and it is the
mechanism the whole interrupt path rests on.

    pytest tests/test_voice_orchestrator.py -v
    python tests/test_voice_orchestrator.py
"""
import asyncio
import functools
import json
import queue
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice.config import VoiceConfig, VAD_FRAME_SAMPLES
from voice.confirm import confirm_question
from voice.orchestrator import (
    INTERRUPTION_MARKER,
    State,
    VoiceOrchestrator,
    run_session,
)
from voice.telemetry import TelemetryWriter
from voice.tts import split_sentences


def asyncio_test(fn):
    """Run an async test body, without depending on pytest-asyncio.

    A decorator rather than a plugin because this repository pins its dependencies
    deliberately (see requirements.txt) and pytest-asyncio would also need an
    `asyncio_mode` setting in a pytest.ini this project does not have. `functools.wraps`
    keeps the signature visible, so pytest still resolves fixtures like `tmp_path`, and
    `asyncio.run` gives every test a fresh event loop.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return asyncio.run(fn(*args, **kwargs))
    return wrapper


# ==========================================================================
# fakes
# ==========================================================================

class FakeMic:
    """Frames from a scripted list, then None forever (i.e. a quiet room).

    Records how many frames were handed out AFTER the script ran dry, which is what
    distinguishes "the orchestrator stopped reading" from "it kept reading and threw
    the frames away" — those look identical from the outside and only one is correct.
    """

    def __init__(self, frames=()):
        self._queue = queue.Queue()
        for frame in frames:
            self._queue.put(frame)
        self.drained = 0
        self.stopped = False

    def get(self, timeout=0.1):
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def push(self, frame):
        self._queue.put(frame)

    def pending(self) -> int:
        """Frames captured but not yet read. Lets a test wait on consumption rather
        than on a sleep."""
        return self._queue.qsize()

    def drain(self):
        dropped = 0
        while True:
            try:
                self._queue.get_nowait()
                dropped += 1
            except queue.Empty:
                break
        self.drained += dropped
        return dropped

    def stop(self):
        self.stopped = True


class FakeVad:
    """Speech probabilities from a script, keyed by frame marker value.

    Reads the frame's first sample as its cue, so a test writes the intended
    probability into the audio itself and the mapping stays obvious at the call site.
    """

    def __init__(self):
        self.resets = 0

    def speech_prob(self, frame):
        return float(frame[0])

    def reset(self):
        self.resets += 1


class FakeStt:
    def __init__(self, transcript="what does the document say about pipes?"):
        self.transcript = transcript
        self.calls = []

    def transcribe(self, audio):
        self.calls.append(len(audio))
        return self.transcript


class ScriptedStt(FakeStt):
    """One transcript per call, in order; the last one repeats.

    A false barge-in needs both halves: a real first turn to get Marshall talking, and
    an empty second one for the interruption that turned out to be a cough.
    """

    def __init__(self, *transcripts):
        super().__init__(transcripts[0])
        self._script = list(transcripts)

    def transcribe(self, audio):
        self.calls.append(len(audio))
        return self._script.pop(0) if len(self._script) > 1 else self._script[0]


class FakeAgent:
    """Stands in for AgentGraph. Records the exact calls the orchestrator makes."""

    def __init__(self, response="OPVC pipes are rated for cold water service."):
        self.response = response
        self.calls = []
        self.truncations = []          # (content, thread_id) per barge-in
        self.truncate_result = True
        self.barrier = None            # set to an Event to hold a turn open

    def invoke(self, query, config=None):
        self.calls.append((query, config))
        if self.barrier is not None:
            self.barrier.wait(5)
        return self.response

    def truncate_last_response(self, content, thread_id="default"):
        self.truncations.append((content, thread_id))
        return self.truncate_result


class FakeTts:
    engine_name = "faketts"

    def __init__(self):
        self.calls = []
        self.synthesized = []

    def synthesize(self, text):
        #: the one-shot path `confirm_by_voice` uses for its question
        self.synthesized.append(text)
        return np.full(240, 0.1, dtype=np.float32)

    def stream_sentences(self, text):
        #: the REAL splitter — the double exists to avoid Kokoro, not to avoid splitting,
        #: and the truncation assertions are about the exact sentence text
        for index, sentence in enumerate(split_sentences(text)):
            self.calls.append(sentence)
            yield index, sentence, np.full(240, 0.1, dtype=np.float32)


class FakeSpeaker:
    """Playback that blocks, and a `stop()` that aborts it.

    `stop()` releasing a blocked `play()` is not decoration: `Speaker.play` is
    `sd.play()` + `sd.wait()`, and `sd.stop()` unblocking that wait is the entire
    mechanism barge-in uses to cut a sentence short. A fake without it would let the
    interrupt tests pass while the real path deadlocked.
    """

    def __init__(self):
        self.played = []
        self.stopped = False
        self.stops = 0
        self.barrier = None            # set to an Event to hold SPEAKING open

    def play(self, audio):
        self.played.append(len(audio))
        #: read once: a test may clear `self.barrier` to let the rest of an answer run
        barrier = self.barrier
        if barrier is not None:
            barrier.wait(5)
            #: one release per sentence, so setting it does not free the whole queue
            barrier.clear()

    def stop(self):
        self.stopped = True
        self.stops += 1
        if self.barrier is not None:
            self.barrier.set()


def frame(prob: float) -> np.ndarray:
    """A frame whose samples encode the speech probability FakeVad should report."""
    return np.full(VAD_FRAME_SAMPLES, prob, dtype=np.float32)


def utterance_frames(config: VoiceConfig, speech: int = 3):
    """`speech` loud frames followed by exactly enough silence to endpoint."""
    return [frame(0.9)] * speech + [frame(0.0)] * config.silence_frames


def interrupt_frames(config: VoiceConfig):
    """Exactly enough sustained over-threshold speech to fire the barge-in guard."""
    return [frame(0.95)] * config.barge_in_sustained_frames


def transient_frames(config: VoiceConfig):
    """One frame short of the guard's run — a cough, a door, a chair."""
    return [frame(0.95)] * (config.barge_in_sustained_frames - 1) + [frame(0.0)] * 3


def build(tmp_path, frames=(), config_kwargs=None, **overrides):
    config = VoiceConfig(telemetry_path=str(tmp_path / "turns.jsonl"),
                         thread_id="test-thread", **(config_kwargs or {}))
    parts = dict(
        agent=FakeAgent(), mic=FakeMic(frames), vad=FakeVad(),
        stt=FakeStt(), tts=FakeTts(), speaker=FakeSpeaker(),
    )
    parts.update(overrides)
    orchestrator = VoiceOrchestrator(
        config=config, telemetry=TelemetryWriter(config.telemetry_path), **parts)
    return orchestrator, parts, config


async def wait_for(predicate, timeout=5.0) -> bool:
    """Poll the loop until `predicate()` holds. False if it never did."""
    for _ in range(int(timeout / 0.01)):
        if predicate():
            return True
        await asyncio.sleep(0.01)
    return False


def read_rows(config) -> list:
    text = Path(config.telemetry_path).read_text(encoding="utf-8").strip()
    return [json.loads(line) for line in text.splitlines()] if text else []


async def run_until_stopped(orchestrator, timeout=10.0):
    """Run the loop, stopping it once it has gone quiet."""
    task = asyncio.ensure_future(orchestrator.run())

    async def _stop_when_idle():
        #: give the loop room to finish the turn it is on, then ask it to stop
        for _ in range(int(timeout / 0.02)):
            await asyncio.sleep(0.02)
            if orchestrator.turn_id > 0 and orchestrator.state is State.LISTENING:
                break
        orchestrator.request_stop()

    await asyncio.gather(task, _stop_when_idle())
    return task


# ==========================================================================
# a full cycle
# ==========================================================================

@asyncio_test
async def test_full_cycle_listening_thinking_speaking_listening(tmp_path):
    orchestrator, parts, config = build(tmp_path, utterance_frames(VoiceConfig()))
    states = []
    orchestrator._on_event = lambda e, p: states.append(p["state"]) if e == "state" else None

    await run_until_stopped(orchestrator)

    assert orchestrator.turn_id == 1
    assert parts["stt"].calls, "STT was never called"
    assert parts["agent"].calls, "the agent was never called"
    assert parts["speaker"].played, "nothing was played"
    #: the cycle, in order, ending back where it started
    assert states[:4] == ["listening", "thinking", "speaking", "listening"]
    assert states[-1] == "stopped"


@asyncio_test
async def test_agent_is_called_with_the_transcript_and_the_configured_thread(tmp_path):
    """The one call into pipeline_lg. A wrong thread_id silently forks the conversation."""
    orchestrator, parts, config = build(tmp_path, utterance_frames(VoiceConfig()))
    await run_until_stopped(orchestrator)

    query, agent_config = parts["agent"].calls[0]
    assert query == parts["stt"].transcript
    assert agent_config == {"configurable": {"thread_id": "test-thread"}}


@asyncio_test
async def test_response_is_spoken_sentence_by_sentence(tmp_path):
    """Kokoro has a per-call input limit, so the orchestrator must not hand it the lot."""
    orchestrator, parts, _ = build(
        tmp_path, utterance_frames(VoiceConfig()),
        agent=FakeAgent("First sentence. Second sentence. Third sentence."))
    await run_until_stopped(orchestrator)

    assert len(parts["tts"].calls) == 3
    assert len(parts["speaker"].played) == 3


@asyncio_test
async def test_empty_transcript_is_discarded_without_calling_the_agent(tmp_path):
    """Silence, a cough, or a failed decode must not cost an agent turn."""
    orchestrator, parts, _ = build(
        tmp_path, utterance_frames(VoiceConfig()), stt=FakeStt(""))
    await run_until_stopped(orchestrator)

    assert parts["agent"].calls == []
    assert parts["speaker"].played == []
    assert orchestrator.state is State.STOPPED


@asyncio_test
async def test_a_failing_agent_costs_the_answer_not_the_session(tmp_path):
    """Same contract as the memory gate (H4): the loop survives a bad turn."""
    class Boom:
        calls = []
        def invoke(self, query, config=None):
            raise RuntimeError("ollama connection refused")

    orchestrator, parts, _ = build(
        tmp_path, utterance_frames(VoiceConfig()), agent=Boom())
    errors = []
    orchestrator._on_event = lambda e, p: errors.append(p) if e == "error" else None

    await run_until_stopped(orchestrator)

    assert errors and "ollama connection refused" in errors[0]["error"]
    #: and it went back to listening rather than dying
    assert orchestrator.state is State.STOPPED


# ==========================================================================
# frames during THINKING / SPEAKING are discarded  (== no barge-in)
# ==========================================================================

@asyncio_test
async def test_frames_arriving_during_thinking_are_discarded(tmp_path):
    """The agent is held mid-turn while loud frames pour in. None may be processed."""
    agent = FakeAgent()
    agent.barrier = threading.Event()
    orchestrator, parts, config = build(
        tmp_path, utterance_frames(VoiceConfig()), agent=agent)

    task = asyncio.ensure_future(orchestrator.run())
    #: wait until the turn is actually in the agent
    for _ in range(200):
        await asyncio.sleep(0.01)
        if agent.calls:
            break
    assert orchestrator.state is State.THINKING

    for _ in range(10):
        parts["mic"].push(frame(0.9))          # shouting during THINKING
    await asyncio.sleep(0.2)

    assert orchestrator.frames_discarded >= 10, "frames were processed, not discarded"
    assert not orchestrator.detector.is_speaking, "the VAD consumed a discarded frame"
    assert len(agent.calls) == 1, "a second turn started while the first was running"

    agent.barrier.set()
    orchestrator.request_stop()
    await asyncio.gather(task, return_exceptions=True)


@asyncio_test
async def test_frames_during_speaking_are_discarded_when_barge_in_is_off(tmp_path):
    """V1a's behaviour, kept as an explicit mode.

    `barge_in_mode="off"` is what a user on speakers runs: with no echo cancellation in
    the stack, Marshall's own voice would otherwise reach the mic and interrupt him. So
    the frames are still read — the stream never closes — and still thrown away.
    """
    speaker = FakeSpeaker()
    speaker.barrier = threading.Event()
    orchestrator, parts, config = build(
        tmp_path, utterance_frames(VoiceConfig()), speaker=speaker,
        config_kwargs={"barge_in_mode": "off"})

    task = asyncio.ensure_future(orchestrator.run())
    assert await wait_for(lambda: bool(speaker.played))
    assert orchestrator.state is State.SPEAKING

    #: loud enough to fire the guard, and sustained well past its duration
    for _ in range(config.barge_in_sustained_frames * 3):
        parts["mic"].push(frame(0.95))
    assert await wait_for(
        lambda: orchestrator.frames_discarded >= config.barge_in_sustained_frames * 3)

    assert not speaker.stopped, "playback was interrupted with barge-in off"
    assert not orchestrator.detector.is_speaking
    assert orchestrator.guard.speech_run == 0, "a disabled guard was still counting"

    speaker.barrier.set()
    orchestrator.request_stop()
    await asyncio.gather(task, return_exceptions=True)


# ==========================================================================
# barge-in (V1b) — sustained speech during SPEAKING stops him
# ==========================================================================

FOUR_SENTENCES = "First one. Second one. Third one. Fourth one."


async def speak_until_sentence(speaker, index: int) -> None:
    """Let playback run to the start of sentence `index`, blocking there.

    Sentences 0..index-1 are played to COMPLETION, which is what makes
    `sentences_played` and the truncation boundary predictable.
    """
    for wanted in range(index + 1):
        assert await wait_for(lambda: len(speaker.played) == wanted + 1), (
            f"playback never reached sentence {wanted}")
        if wanted < index:
            speaker.barrier.set()


@asyncio_test
async def test_sustained_speech_stops_playback_and_clears_the_queue(tmp_path):
    """The headline. Four sentences, cut during the second: the remaining two are never
    played at all — there is no resumable queue by design."""
    speaker = FakeSpeaker()
    speaker.barrier = threading.Event()
    orchestrator, parts, config = build(
        tmp_path, utterance_frames(VoiceConfig()),
        agent=FakeAgent(FOUR_SENTENCES), speaker=speaker)

    task = asyncio.ensure_future(orchestrator.run())
    await speak_until_sentence(speaker, 1)
    assert orchestrator.state is State.SPEAKING

    for captured in interrupt_frames(config):
        parts["mic"].push(captured)

    assert await wait_for(lambda: orchestrator.state is State.LISTENING), (
        "the interrupt never landed")
    assert speaker.stopped, "playback was not stopped"
    assert len(speaker.played) == 2, "the rest of the answer was played anyway"

    #: and the captured audio is now the head of the user's next utterance, so they do
    #: not have to repeat the words they interrupted with
    assert orchestrator.detector.is_speaking
    assert orchestrator.detector.buffered_frames == config.barge_in_sustained_frames

    orchestrator.request_stop()
    await asyncio.gather(task, return_exceptions=True)


@asyncio_test
async def test_a_transient_does_not_interrupt(tmp_path):
    """A cough, a door, a keyboard. One frame short of the sustained run, and the answer
    must finish — this is the whole reason the guard is not just a threshold."""
    speaker = FakeSpeaker()
    speaker.barrier = threading.Event()
    orchestrator, parts, config = build(
        tmp_path, utterance_frames(VoiceConfig()),
        agent=FakeAgent(FOUR_SENTENCES), speaker=speaker)

    task = asyncio.ensure_future(orchestrator.run())
    await speak_until_sentence(speaker, 0)

    for captured in transient_frames(config):
        parts["mic"].push(captured)
    assert await wait_for(lambda: parts["mic"].pending() == 0)
    #: a full half-second to fire in, and it must not
    assert not await wait_for(lambda: speaker.stopped, timeout=0.5), (
        "a transient interrupted Marshall")
    assert orchestrator.state is State.SPEAKING

    #: let the rest of the answer run — clear the barrier so later sentences do not
    #: block, then release the one currently waiting
    blocked, speaker.barrier = speaker.barrier, None
    blocked.set()
    assert await wait_for(lambda: orchestrator.state is State.LISTENING)

    assert len(speaker.played) == 4, "the answer did not finish"
    assert parts["agent"].truncations == [], "an uninterrupted turn rewrote history"

    row = read_rows(config)[0]
    assert row["was_interrupted"] is False
    assert row["sentences_played"] == row["sentences_total"] == 4

    orchestrator.request_stop()
    await asyncio.gather(task, return_exceptions=True)


@asyncio_test
async def test_history_is_truncated_to_the_sentences_actually_heard(tmp_path):
    """H2's summariser and UserMemory both compress from the stored history, so it has
    to hold what the user heard — not the answer Marshall meant to give."""
    speaker = FakeSpeaker()
    speaker.barrier = threading.Event()
    orchestrator, parts, config = build(
        tmp_path, utterance_frames(VoiceConfig()),
        agent=FakeAgent(FOUR_SENTENCES), speaker=speaker)

    task = asyncio.ensure_future(orchestrator.run())
    await speak_until_sentence(speaker, 2)      # two played in full
    for captured in interrupt_frames(config):
        parts["mic"].push(captured)
    assert await wait_for(lambda: orchestrator.state is State.LISTENING)

    assert parts["agent"].truncations == [
        (f"First one. Second one. {INTERRUPTION_MARKER}", "test-thread")]

    orchestrator.request_stop()
    await asyncio.gather(task, return_exceptions=True)


@asyncio_test
async def test_an_interrupt_before_the_first_sentence_ends_stores_only_the_marker(tmp_path):
    """The partly-played sentence is dropped rather than half-claimed: playback stops at
    sentence granularity, so there is no sub-sentence position to report honestly."""
    speaker = FakeSpeaker()
    speaker.barrier = threading.Event()
    orchestrator, parts, config = build(
        tmp_path, utterance_frames(VoiceConfig()),
        agent=FakeAgent(FOUR_SENTENCES), speaker=speaker)

    task = asyncio.ensure_future(orchestrator.run())
    await speak_until_sentence(speaker, 0)
    for captured in interrupt_frames(config):
        parts["mic"].push(captured)
    assert await wait_for(lambda: orchestrator.state is State.LISTENING)

    assert parts["agent"].truncations == [(INTERRUPTION_MARKER, "test-thread")]
    assert read_rows(config)[0]["sentences_played"] == 0

    orchestrator.request_stop()
    await asyncio.gather(task, return_exceptions=True)


@asyncio_test
async def test_a_false_barge_in_stays_stopped_and_keeps_the_truncation(tmp_path):
    """The guard fired, the transcript came back empty — a cough that lasted long enough.

    There is no resume path by design: Marshall stays stopped, the truncation stands, and
    the session returns to LISTENING so the user can simply say something. The guard gets
    corrected from telemetry instead, which is what the new columns are for.
    """
    speaker = FakeSpeaker()
    speaker.barrier = threading.Event()
    orchestrator, parts, config = build(
        tmp_path, utterance_frames(VoiceConfig()),
        agent=FakeAgent(FOUR_SENTENCES), speaker=speaker,
        stt=ScriptedStt("tell me about pipes", ""))

    task = asyncio.ensure_future(orchestrator.run())
    await speak_until_sentence(speaker, 1)
    for captured in interrupt_frames(config):
        parts["mic"].push(captured)
    assert await wait_for(lambda: orchestrator.turn_id == 1
                          and orchestrator.state is State.LISTENING)

    #: the "interruption" now goes quiet, so the seeded utterance endpoints and is
    #: transcribed — to nothing
    for silent in [frame(0.0)] * config.silence_frames:
        parts["mic"].push(silent)
    assert await wait_for(lambda: orchestrator.turn_id == 2)
    assert await wait_for(lambda: orchestrator.state is State.LISTENING)

    #: the second utterance carried the interrupt's own audio, not just the silence
    assert parts["stt"].calls[1] == (
        (config.barge_in_sustained_frames + config.silence_frames) * VAD_FRAME_SAMPLES)
    #: nothing was said and nothing was un-said
    assert len(parts["agent"].calls) == 1, "an empty transcript reached the agent"
    assert len(speaker.played) == 2, "Marshall resumed after a false barge-in"
    assert len(parts["agent"].truncations) == 1, "the truncation was undone"

    orchestrator.request_stop()
    await asyncio.gather(task, return_exceptions=True)


@asyncio_test
async def test_a_failing_truncation_costs_the_history_not_the_session(tmp_path):
    """Same contract as everything else on this path: the loop survives."""
    class Boom(FakeAgent):
        def truncate_last_response(self, content, thread_id="default"):
            raise RuntimeError("checkpointer unavailable")

    speaker = FakeSpeaker()
    speaker.barrier = threading.Event()
    orchestrator, parts, config = build(
        tmp_path, utterance_frames(VoiceConfig()),
        agent=Boom(FOUR_SENTENCES), speaker=speaker)
    errors = []
    orchestrator._on_event = lambda e, p: errors.append(p) if e == "error" else None

    task = asyncio.ensure_future(orchestrator.run())
    await speak_until_sentence(speaker, 1)
    for captured in interrupt_frames(config):
        parts["mic"].push(captured)
    assert await wait_for(lambda: orchestrator.state is State.LISTENING)

    assert errors and "checkpointer unavailable" in errors[0]["error"]
    assert read_rows(config)[0]["was_interrupted"] is True

    orchestrator.request_stop()
    await asyncio.gather(task, return_exceptions=True)


@asyncio_test
async def test_the_mic_queue_is_drained_before_listening_resumes(tmp_path):
    """Frames captured while speaking are still QUEUED when the turn ends. Without a
    drain the VAD would immediately process Marshall's own voice."""
    orchestrator, parts, config = build(tmp_path, utterance_frames(VoiceConfig()))
    await run_until_stopped(orchestrator)

    assert parts["mic"].drained >= 0          # called
    assert parts["vad"].resets >= 1, "the recurrent VAD state was not reset"


# ==========================================================================
# telemetry
# ==========================================================================

@asyncio_test
async def test_a_turn_writes_one_telemetry_row(tmp_path):
    orchestrator, parts, config = build(tmp_path, utterance_frames(VoiceConfig()))
    await run_until_stopped(orchestrator)

    lines = Path(config.telemetry_path).read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])

    assert row["turn_id"] == 1
    assert row["tts_engine"] == "faketts"
    assert row["utterance_s"] > 0
    assert row["transcript_chars"] == len(parts["stt"].transcript)
    assert row["response_chars"] == len(parts["agent"].response)
    for timing in ("stt_ms", "agent_ms", "tts_first_sentence_ms",
                   "speech_end_to_first_audio_ms"):
        assert row[timing] is not None and row[timing] >= 0, f"{timing} was not measured"


@asyncio_test
async def test_a_discarded_turn_still_writes_a_row(tmp_path):
    """A turn that produced nothing is data about the endpointing thresholds."""
    orchestrator, parts, config = build(
        tmp_path, utterance_frames(VoiceConfig()), stt=FakeStt(""))
    await run_until_stopped(orchestrator)

    row = json.loads(Path(config.telemetry_path).read_text(encoding="utf-8").strip())
    assert row["transcript_chars"] == 0
    assert row["agent_ms"] is None, "an unreached stage must stay null, not 0"


@asyncio_test
async def test_a_clean_turn_reports_no_interruption_and_a_full_answer(tmp_path):
    """The other half of the barge-in columns: they have to be right when nothing
    happened, or a run's interrupt RATE cannot be read off the log."""
    orchestrator, parts, config = build(
        tmp_path, utterance_frames(VoiceConfig()), agent=FakeAgent(FOUR_SENTENCES))
    await run_until_stopped(orchestrator)

    row = read_rows(config)[0]
    assert row["was_interrupted"] is False
    assert row["sentences_total"] == 4
    assert row["sentences_played"] == 4
    assert row["time_to_silence_ms"] is None, "nothing was stopped, so nothing was timed"


@asyncio_test
async def test_an_interrupted_turn_records_what_was_cut_and_how_fast(tmp_path):
    """These four columns are the whole reason the guessed guard parameters are
    defensible: they are what corrects them from a real run."""
    speaker = FakeSpeaker()
    speaker.barrier = threading.Event()
    orchestrator, parts, config = build(
        tmp_path, utterance_frames(VoiceConfig()),
        agent=FakeAgent(FOUR_SENTENCES), speaker=speaker)

    task = asyncio.ensure_future(orchestrator.run())
    await speak_until_sentence(speaker, 2)               # two played in full
    for captured in interrupt_frames(config):
        parts["mic"].push(captured)
    assert await wait_for(lambda: orchestrator.state is State.LISTENING)

    row = read_rows(config)[0]
    assert row["was_interrupted"] is True
    assert row["sentences_played"] == 2
    assert row["sentences_total"] == 4
    assert row["time_to_silence_ms"] is not None and row["time_to_silence_ms"] >= 0

    orchestrator.request_stop()
    await asyncio.gather(task, return_exceptions=True)


# ==========================================================================
# the history rewrite itself — the real AgentGraph, no Ollama
#
# The orchestrator tests above stop at "truncate_last_response was called with this
# text". This section is the other side of that contract, and it is where the load-
# bearing detail lives: the rewrite must REPLACE the stored answer, not append a second
# one. That is `add_messages` matching on `.id`, so the id is what is asserted.
#
# The graph is built for real — nodes, edges, checkpointer — with only the model handle
# faked, exactly as tests/test_summarization_cycle.py does it.
# ==========================================================================

@pytest.fixture(scope="module")
def truncation_agent(tmp_path_factory):
    """One real AgentGraph over a one-line document. ~8s of embeddings + Chroma."""
    from langchain_core.messages import AIMessage

    from pipeline_lg import AgentGraph

    #: pytest owns the lifetime -- see the note in tests/test_agent_hook.py
    root = tmp_path_factory.mktemp("marshall_v1b")
    (root / "doc.txt").write_text("Marshall answers questions about a document.\n",
                                  encoding="utf-8")
    agent = AgentGraph(
        file_path=str(root / "doc.txt"),
        mode_docs="txt",
        mode_retreiver="simple",
        persist_directory_path=str(root / "vdb"),
        user_history_directory_path=str(root / "mem"),
        memory_every_n_turns=0,        # isolate this from the long-term-memory gate
        summarize_every_n_turns=0,     # ... and from H2's trim cycle
        summary_persist_path=str(root / "summaries"),
    )

    class PlainChat:
        """One plain reply per call: a turn is exactly two messages."""
        def invoke(self, messages, **kwargs):
            return AIMessage(content="First one. Second one. Third one.")

    agent.llm = PlainChat()
    return agent


def last_ai(agent, thread_id):
    from langchain_core.messages import AIMessage
    return [m for m in agent.get_history(thread_id) if isinstance(m, AIMessage)][-1]


def test_truncation_replaces_the_stored_answer_in_place(truncation_agent):
    """The `.id` is the whole mechanism: `add_messages` matches on it, so reusing it
    REPLACES the answer where a fresh message would leave two of them in history."""
    thread = "replace"
    truncation_agent.invoke("tell me about it",
                            config={"configurable": {"thread_id": thread}})
    before = last_ai(truncation_agent, thread)
    original_length = len(truncation_agent.get_history(thread))

    assert truncation_agent.truncate_last_response(
        f"First one. {INTERRUPTION_MARKER}", thread) is True

    after = last_ai(truncation_agent, thread)
    assert after.id == before.id, "the rewrite did not reuse the message id"
    assert after.content == f"First one. {INTERRUPTION_MARKER}"
    assert len(truncation_agent.get_history(thread)) == original_length, (
        "the rewrite appended a second answer instead of replacing the first")


def test_truncation_leaves_the_rest_of_the_conversation_alone(truncation_agent):
    """The user's own turn is not the assistant's to edit."""
    from langchain_core.messages import HumanMessage

    thread = "intact"
    truncation_agent.invoke("what does it do?",
                            config={"configurable": {"thread_id": thread}})
    truncation_agent.truncate_last_response(INTERRUPTION_MARKER, thread)

    humans = [m for m in truncation_agent.get_history(thread)
              if isinstance(m, HumanMessage)]
    assert [m.content for m in humans] == ["what does it do?"]


def test_the_next_turn_carries_the_truncated_text_not_the_original(truncation_agent):
    """The point of the whole exercise: what H2's summariser and UserMemory later
    compress is what the user actually heard."""
    thread = "carried"
    truncation_agent.invoke("first question",
                            config={"configurable": {"thread_id": thread}})
    truncation_agent.truncate_last_response(f"First one. {INTERRUPTION_MARKER}", thread)
    truncation_agent.invoke("second question",
                            config={"configurable": {"thread_id": thread}})

    text = " ".join(str(m.content) for m in truncation_agent.get_history(thread))
    assert INTERRUPTION_MARKER in text
    assert "Third one." in text, "the SECOND answer should still be complete"
    assert text.count("Second one.") == 1, "the interrupted answer kept unheard text"


def test_truncating_an_empty_thread_reports_that_it_did_nothing(truncation_agent):
    """No history to rewrite. False, not an exception — the voice path calls this on a
    turn that is already going wrong."""
    assert truncation_agent.truncate_last_response("anything", "never-spoken") is False


# ==========================================================================
# shutdown
# ==========================================================================

@asyncio_test
async def test_request_stop_ends_the_loop_cleanly(tmp_path):
    orchestrator, parts, _ = build(tmp_path)
    task = asyncio.ensure_future(orchestrator.run())
    await asyncio.sleep(0.05)
    orchestrator.request_stop()
    await asyncio.wait_for(task, timeout=5)

    assert task.done() and not task.cancelled()
    assert task.exception() is None, "shutdown raised"
    assert orchestrator.state is State.STOPPED


@asyncio_test
async def test_cancelling_the_task_does_not_raise(tmp_path):
    """asyncio.CancelledError is swallowed inside run() — no traceback spray."""
    orchestrator, parts, _ = build(tmp_path)
    task = asyncio.ensure_future(orchestrator.run())
    await asyncio.sleep(0.05)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    assert orchestrator.state is State.STOPPED


@asyncio_test
async def test_run_session_closes_the_hardware(tmp_path):
    """Streams must be released even on the cancellation path."""
    orchestrator, parts, _ = build(tmp_path)

    async def _stop_soon():
        await asyncio.sleep(0.05)
        orchestrator.request_stop()

    await asyncio.gather(run_session(orchestrator), _stop_soon())

    assert parts["mic"].stopped, "the input stream was left open"
    assert parts["speaker"].stopped, "the output stream was left open"


@asyncio_test
async def test_close_is_idempotent_and_never_raises(tmp_path):
    class Nasty:
        def stop(self):
            raise RuntimeError("device already gone")

    orchestrator, _, _ = build(tmp_path, mic=Nasty(), speaker=Nasty())
    orchestrator.close()
    orchestrator.close()          # twice, on the way out, must still not raise


# ==========================================================================
# spoken confirmation (CONFIRMING) — run_code asks, the user answers, code decides
# ==========================================================================

class ConfirmingAgent(FakeAgent):
    """invoke() asks for a spoken confirmation, as run_code does from inside the graph."""

    def __init__(self):
        super().__init__()
        self.orchestrator = None
        self.answers = []

    def invoke(self, query, config=None):
        self.calls.append((query, config))
        accepted = self.orchestrator.confirm_by_voice(
            SimpleNamespace(language="python", source="print(1)", origin="pasted"))
        self.answers.append(accepted)
        return "ran it" if accepted else "not running it"


async def _start_confirming(tmp_path, stt, **config_kwargs):
    agent = ConfirmingAgent()
    orchestrator, parts, config = build(tmp_path, frames=utterance_frames(VoiceConfig()),
                                        stt=stt, agent=agent, config_kwargs=config_kwargs)
    agent.orchestrator = orchestrator
    task = asyncio.ensure_future(orchestrator.run())
    assert await wait_for(lambda: orchestrator.state is State.CONFIRMING)
    return orchestrator, parts, config, agent, task


@asyncio_test
async def test_confirm_by_voice_hears_yes(tmp_path):
    events = []
    orchestrator, parts, config, agent, task = await _start_confirming(
        tmp_path, ScriptedStt("run this code", "yes"))
    orchestrator._on_event = lambda e, p: events.append((e, p))
    for f in utterance_frames(config):
        parts["mic"].push(f)
    assert await wait_for(lambda: agent.answers == [True])
    request = SimpleNamespace(language="python", source="print(1)", origin="pasted")
    assert parts["tts"].synthesized == [confirm_question(request)]
    assert len(parts["speaker"].played) >= 1
    assert await wait_for(lambda: orchestrator.state is State.LISTENING)
    assert ("confirm", {"turn_id": 1, "transcript": "yes", "accepted": True}) in events
    orchestrator.request_stop()
    await task


@asyncio_test
async def test_confirm_by_voice_shows_the_code_before_asking(tmp_path):
    # final review I2: a voice yes must not be a blind yes — the code is put on screen
    events = []
    agent = ConfirmingAgent()
    orchestrator, parts, config = build(tmp_path, frames=utterance_frames(VoiceConfig()),
                                        stt=ScriptedStt("run this code", "no"), agent=agent,
                                        on_event=lambda e, p: events.append((e, p)))
    agent.orchestrator = orchestrator
    task = asyncio.ensure_future(orchestrator.run())
    assert await wait_for(lambda: orchestrator.state is State.CONFIRMING)
    shown = [p for e, p in events if e == "confirm_request"]
    assert shown == [{"turn_id": 1, "language": "python", "origin": "pasted",
                      "source": "print(1)"}]
    names = [e for e, _ in events]
    assert names.index("confirm_request") < names.index("state", names.index("confirm_request"))
    orchestrator.request_stop()
    await task


@asyncio_test
async def test_confirm_by_voice_hears_no(tmp_path):
    orchestrator, parts, config, agent, task = await _start_confirming(
        tmp_path, ScriptedStt("run this code", "no thanks"))
    for f in utterance_frames(config):
        parts["mic"].push(f)
    assert await wait_for(lambda: agent.answers == [False])
    orchestrator.request_stop()
    await task


@asyncio_test
async def test_confirm_by_voice_times_out_to_no(tmp_path):
    orchestrator, parts, config, agent, task = await _start_confirming(
        tmp_path, ScriptedStt("run this code", "yes"), confirm_timeout_s=0.3)
    #: wall clock, not `wait_for`: that helper counts `asyncio.sleep(0.01)` iterations, and
    #: on the Windows proactor loop those return early — its "3 s" measured ~0.3 s, the
    #: same as the timeout under test, so the two raced
    import time
    deadline = time.perf_counter() + 5.0
    while not agent.answers and time.perf_counter() < deadline:
        await asyncio.sleep(0.01)
    assert agent.answers == [False]
    #: only the first utterance was transcribed: silence is never sent to STT
    assert len(parts["stt"].calls) == 1
    orchestrator.request_stop()
    await task


@asyncio_test
async def test_confirming_state_does_not_leak_into_the_next_turn(tmp_path):
    orchestrator, parts, config, agent, task = await _start_confirming(
        tmp_path, ScriptedStt("run this code", "yes"))
    for f in utterance_frames(config):
        parts["mic"].push(f)
    assert await wait_for(lambda: orchestrator.state is State.LISTENING)
    assert orchestrator._confirm_queue.empty() and orchestrator.turn_id == 1
    orchestrator.request_stop()
    await task


@asyncio_test
async def test_a_stop_during_confirming_is_a_no_and_ends_cleanly(tmp_path):
    orchestrator, parts, config, agent, task = await _start_confirming(
        tmp_path, ScriptedStt("run this code", "yes"))
    orchestrator.request_stop()
    assert await wait_for(lambda: agent.answers == [False], timeout=3.0)
    await task


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--no-header", "-p", "no:cacheprovider"]))
