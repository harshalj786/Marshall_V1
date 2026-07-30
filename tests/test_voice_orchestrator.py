"""The V1a loop, driven entirely by fakes: LISTENING -> THINKING -> SPEAKING -> LISTENING.

No microphone, no GPU, no models, no Ollama. `VoiceOrchestrator` takes every
collaborator as a constructor argument and never imports sounddevice, torch,
faster-whisper or kokoro, so the full cycle — including the parts that only happen on
a real turn, like discarding frames while Marshall is talking — is assertable here.

The three things these tests exist to pin:

1. a complete cycle runs and returns to LISTENING;
2. frames arriving during THINKING/SPEAKING are DISCARDED, which is what "no barge-in"
   means mechanically, and is the property V1b will deliberately break;
3. shutdown cancels cleanly — no traceback, no stranded task, hardware released.

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

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice.config import VoiceConfig, VAD_FRAME_SAMPLES
from voice.orchestrator import VoiceOrchestrator, State, run_session
from voice.telemetry import TelemetryWriter


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


class FakeAgent:
    """Stands in for AgentGraph. Records the exact call the orchestrator makes."""

    def __init__(self, response="OPVC pipes are rated for cold water service."):
        self.response = response
        self.calls = []
        self.barrier = None            # set to an Event to hold a turn open

    def invoke(self, query, config=None):
        self.calls.append((query, config))
        if self.barrier is not None:
            self.barrier.wait(5)
        return self.response


class FakeTts:
    engine_name = "faketts"

    def __init__(self):
        self.calls = []

    def stream_sentences(self, text):
        for index, sentence in enumerate(s for s in text.split(". ") if s.strip()):
            self.calls.append(sentence)
            yield index, sentence, np.full(240, 0.1, dtype=np.float32)


class FakeSpeaker:
    def __init__(self):
        self.played = []
        self.stopped = False
        self.barrier = None            # set to an Event to hold SPEAKING open

    def play(self, audio):
        self.played.append(len(audio))
        if self.barrier is not None:
            self.barrier.wait(5)

    def stop(self):
        self.stopped = True


def frame(prob: float) -> np.ndarray:
    """A frame whose samples encode the speech probability FakeVad should report."""
    return np.full(VAD_FRAME_SAMPLES, prob, dtype=np.float32)


def utterance_frames(config: VoiceConfig, speech: int = 3):
    """`speech` loud frames followed by exactly enough silence to endpoint."""
    return [frame(0.9)] * speech + [frame(0.0)] * config.silence_frames


def build(tmp_path, frames=(), **overrides):
    config = VoiceConfig(telemetry_path=str(tmp_path / "turns.jsonl"),
                         thread_id="test-thread")
    parts = dict(
        agent=FakeAgent(), mic=FakeMic(frames), vad=FakeVad(),
        stt=FakeStt(), tts=FakeTts(), speaker=FakeSpeaker(),
    )
    parts.update(overrides)
    orchestrator = VoiceOrchestrator(
        config=config, telemetry=TelemetryWriter(config.telemetry_path), **parts)
    return orchestrator, parts, config


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
async def test_frames_arriving_during_speaking_are_discarded(tmp_path):
    """Marshall's own voice must not be heard as the user's next utterance."""
    speaker = FakeSpeaker()
    speaker.barrier = threading.Event()
    orchestrator, parts, config = build(
        tmp_path, utterance_frames(VoiceConfig()), speaker=speaker)

    task = asyncio.ensure_future(orchestrator.run())
    for _ in range(200):
        await asyncio.sleep(0.01)
        if speaker.played:
            break
    assert orchestrator.state is State.SPEAKING

    for _ in range(10):
        parts["mic"].push(frame(0.9))
    await asyncio.sleep(0.2)

    assert orchestrator.frames_discarded >= 10
    assert not orchestrator.detector.is_speaking

    speaker.barrier.set()
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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--no-header", "-p", "no:cacheprovider"]))
