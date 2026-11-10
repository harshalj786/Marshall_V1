"""The conversation loop: LISTENING -> THINKING -> SPEAKING -> LISTENING.

One utterance at a time, strictly sequential. V1b adds exactly one way out of that
sequence: **while Marshall is SPEAKING, sustained speech from the user stops him.**

Why only SPEAKING. `docs/architecture_review.md` §10 is explicit that cancelling
mid-turn orphans an `AIMessage.tool_calls` from its `ToolMessage` in a way Ollama
rejects, and calls that "the single hardest voice problem in the current design". That
risk lives entirely in THINKING, which is where the ReAct loop runs — and THINKING is
also the phase not worth interrupting (§10 measures warm agent turns at 1.3-8.4 s, most
far shorter). SPEAKING carries none of it: `agent.invoke` is synchronous, so by the time
the first word is spoken the answer is a COMPLETE `AIMessage` with no pending tool calls.
Truncating it is a plain content edit, which is why nothing here calls
`memory/reconciliation.py:find_safe_cut_point` — that primitive is for the streaming case
(V1c), where an interrupt really can land between a tool call and its result.

Frames arriving during THINKING are still read and thrown away, exactly as in V1a.
Frames arriving during SPEAKING now go to `BargeInGuard` instead.

**Why asyncio at all, when every stage blocks.** Not for throughput — for a loop that
stays answerable while a stage runs. `agent.invoke` is a synchronous multi-second call
into LangGraph; run inline it would also freeze frame draining and signal handling.
In an executor, the loop keeps turning, Ctrl+C is noticed immediately, and shutdown is
a task cancellation rather than a kill. §10's warning that `ainvoke` on sync nodes
"works and it blocks a thread — not a solution, a trap" is about pretending the graph
is async. This does not pretend: the graph is sync, it is called in a thread, and the
loop above it is honest about that.

Every collaborator is injected. The orchestrator never imports sounddevice, torch,
faster-whisper or kokoro, so the whole cycle is testable with fakes and no hardware.

**CONFIRMING is the one place a turn listens while it is still THINKING.** `run_code`
must not run anything until the user says yes, and the model must not be the one saying
it. So from inside `agent.invoke` — on the executor thread — `confirm_by_voice` speaks the
question, flips the state to CONFIRMING, and waits; the reader routes frames to a second
endpointer while in that state; the answer is transcribed and judged by
`voice/confirm.py:parse_yes_no`, plain string matching, never the model. Silence, a
shutdown or anything unclear is a no.
"""
import asyncio
import logging
import queue
import time
from enum import Enum
from typing import Optional

from voice.config import VoiceConfig
from voice.confirm import CONFIRM_SPOKEN, parse_yes_no
from voice.telemetry import TelemetryWriter, TurnTelemetry, Stopwatch
from voice.vad import BargeInGuard, UtteranceDetector

logger = logging.getLogger(__name__)

#: appended to an interrupted answer in the stored history. Present so the record says
#: WHY it stops mid-thought: without it a later turn reads a truncated answer as
#: Marshall's complete one, and so do H2's summariser and UserMemory.
INTERRUPTION_MARKER = "[interrupted by user]"


def truncate_to_spoken(sentences, sentences_played: int) -> str:
    """The answer as far as the user actually heard it, plus the marker.

    Sentence granularity, and only sentences played to COMPLETION. Playback is stopped
    per sentence (`sd.stop()`), so the one cut mid-word has no reportable position — and
    between claiming a sentence the user half-heard and dropping one they fully heard,
    dropping is the safe error: the transcript then under-states what Marshall said,
    which is recoverable, rather than asserting he said something he did not.

    `sentences_played == 0` leaves the marker alone, which is the honest record of an
    interruption that landed before the first sentence finished.
    """
    heard = [text.strip() for text in list(sentences)[:max(0, sentences_played)]
             if text and text.strip()]
    return " ".join(heard + [INTERRUPTION_MARKER])


class State(str, Enum):
    LISTENING = "listening"
    THINKING = "thinking"
    #: inside THINKING: `run_code` asked "is that okay?" and the mic is listening for it
    CONFIRMING = "confirming"
    SPEAKING = "speaking"
    STOPPED = "stopped"


class VoiceOrchestrator:
    """Drives one voice session.

    Collaborators are duck-typed rather than subclassed:
        mic       .get(timeout) -> frame | None ; .drain() ; .stop()
        vad       .speech_prob(frame) -> float ; .reset()
        stt       .transcribe(audio) -> str
        tts       .engine_name ; .stream_sentences(text) -> (i, sentence, audio)
                  .synthesize(text) -> audio                     (confirmation only)
        speaker   .play(audio) ; .stop()          .stop() must abort a blocked .play()
        agent     .invoke(query, config=...) -> str
                  .truncate_last_response(content, thread_id) -> bool   (barge-in only)
    """

    def __init__(self, agent, mic, vad, stt, tts, speaker,
                 config: Optional[VoiceConfig] = None,
                 telemetry: Optional[TelemetryWriter] = None,
                 on_event=None):
        self.config = config or VoiceConfig()
        self.agent = agent
        self.mic = mic
        self.vad = vad
        self.stt = stt
        self.tts = tts
        self.speaker = speaker
        self.telemetry = telemetry or TelemetryWriter(self.config.telemetry_path)
        self.detector = UtteranceDetector(self.config)
        self.guard = BargeInGuard(self.config)
        #: reporting hook, so the CLI can print without the orchestrator importing a UI
        self._on_event = on_event or (lambda event, payload: None)

        self.state = State.LISTENING
        self.turn_id = 0
        self._stopping = asyncio.Event()
        self.frames_discarded = 0
        #: completed utterances, reader -> turn handler. Bounded at one in practice:
        #: `_turn_pending` stops the reader producing another until this one is done.
        self._utterances: asyncio.Queue = asyncio.Queue()
        #: closes the window between "the reader emitted an utterance" and "the handler
        #: set THINKING". Without it the reader keeps feeding the VAD for those few
        #: milliseconds and can queue a second utterance behind the first — which is
        #: barge-in, arriving by accident, in the release that is supposed to not have it.
        self._turn_pending = False
        #: barge-in state for the turn in flight. All three are touched only from the
        #: event loop — the reader sets them, `_speak` reads them, `_back_to_listening`
        #: clears them — so no lock is involved and none is needed.
        self._interrupted = False
        #: the user's own words, from the guard's window onward. Handed to the detector
        #: as the head of their next utterance so nobody has to repeat themselves.
        self._interrupt_frames: list = []
        self._time_to_silence_ms: Optional[float] = None
        #: CONFIRMING's own endpointer and hand-off. A separate detector so the answer can
        #: never be mistaken for, or merged into, the next turn's utterance; a thread-safe
        #: queue because the waiter is the agent's executor thread, not the event loop.
        self._confirm_detector = UtteranceDetector(self.config)
        self._confirm_queue: queue.Queue = queue.Queue()

    # --- reporting -------------------------------------------------------------

    def _emit(self, event: str, **payload) -> None:
        try:
            self._on_event(event, payload)
        except Exception:
            #: a reporting hook must never be able to end a session
            logger.debug("on_event hook raised", exc_info=True)

    def _set_state(self, state: State) -> None:
        self.state = state
        self._emit("state", state=state.value)

    # --- the loop --------------------------------------------------------------

    async def run(self) -> None:
        """Listen until stopped. Returns cleanly; never raises on Ctrl+C.

        Two concurrent halves: `_read_frames` drains the microphone without pause, and
        this loop handles one utterance at a time. They are separate because the mic
        must be read CONTINUOUSLY — a turn can take many seconds, and a reader that
        stopped for the duration would let those seconds of audio pile up in the queue
        to be dealt with afterwards, rather than being dropped as they arrive.
        """
        loop = asyncio.get_running_loop()
        self._set_state(State.LISTENING)
        self._emit("ready", thread_id=self.config.thread_id,
                   barge_in=self.config.barge_in_mode)
        reader = asyncio.ensure_future(self._read_frames(loop))
        try:
            while not self._stopping.is_set():
                try:
                    #: the timeout is the shutdown check interval — without it, a silent
                    #: room would park here forever and never notice `request_stop`
                    utterance = await asyncio.wait_for(self._utterances.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue
                await self._handle_utterance(utterance, loop)
        except asyncio.CancelledError:
            #: expected on shutdown — the caller cancelled us, so unwind quietly
            pass
        finally:
            reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)
            self._set_state(State.STOPPED)

    async def _read_frames(self, loop) -> None:
        """Continuously drain the microphone and route each frame by state.

            LISTENING   -> the endpointer, which may finish an utterance
            SPEAKING    -> the barge-in guard (V1b), or discarded when it is off
            THINKING    -> discarded

        Frames are never simply un-captured: the input stream stays open for the whole
        session so the device is not reopened per turn (see `MicrophoneStream`), and what
        changes between states is only where the frames go.
        """
        while not self._stopping.is_set():
            #: the executor hop is what keeps this cancellable — mic.get() blocks, but
            #: only for `timeout`, so cancellation is noticed within one frame
            frame = await loop.run_in_executor(None, self.mic.get, 0.1)
            if frame is None:
                continue
            #: before the `_turn_pending` check, which is set for the whole turn and
            #: would otherwise swallow SPEAKING along with everything else
            if self.state is State.SPEAKING and self.config.barge_in_enabled:
                await self._watch_for_barge_in(frame, loop)
                continue
            #: also before the `_turn_pending` check: CONFIRMING happens mid-turn
            if self.state is State.CONFIRMING:
                answer = await loop.run_in_executor(None, self._feed_confirm, frame)
                if answer is not None:
                    self._confirm_queue.put_nowait(answer)
                continue
            if self.state is not State.LISTENING or self._turn_pending:
                self.frames_discarded += 1
                continue
            utterance = await loop.run_in_executor(None, self._feed, frame)
            if utterance is not None:
                self._turn_pending = True
                self._utterances.put_nowait(utterance)

    def _feed(self, frame):
        """VAD inference plus the state machine. Runs in the executor: silero is a
        torch forward pass, small but not free, and it has no business on the loop."""
        return self.detector.process(frame, self.vad.speech_prob(frame))

    def _feed_guard(self, frame):
        """The same inference, into the interrupt guard. Same executor, same reason."""
        return self.guard.process(frame, self.vad.speech_prob(frame))

    def _feed_confirm(self, frame):
        """The same inference, into CONFIRMING's endpointer."""
        return self._confirm_detector.process(frame, self.vad.speech_prob(frame))

    def confirm_by_voice(self, request) -> bool:
        """Ask "is that okay?" out loud and return whether the user clearly said yes.

        SYNC, and called on the agent's executor thread from inside `agent.invoke` — via
        `voice.confirm.VoiceConfirmer`, which `run_code` closes over. Blocking here is the
        point: the tool must not return until the user has answered. The event loop keeps
        running meanwhile, so the reader is what hears the answer.
        """
        self.speaker.play(self.tts.synthesize(CONFIRM_SPOKEN))
        #: Marshall's own question is in the queue; the answer must start after it
        self.mic.drain()
        self._confirm_detector.reset()
        while not self._confirm_queue.empty():
            self._confirm_queue.get_nowait()
        self._set_state(State.CONFIRMING)
        utterance = None
        try:
            deadline = time.perf_counter() + self.config.confirm_timeout_s
            while utterance is None and not self._stopping.is_set():
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    utterance = self._confirm_queue.get(timeout=min(0.1, remaining))
                except queue.Empty:
                    continue
        finally:
            #: back to THINKING whatever happened: the tool call is still in flight
            self._set_state(State.THINKING)
        transcript = self.stt.transcribe(utterance) if utterance is not None else ""
        accepted = parse_yes_no(transcript)
        self._emit("confirm", turn_id=self.turn_id, transcript=transcript, accepted=accepted)
        return accepted

    async def _watch_for_barge_in(self, frame, loop) -> None:
        """One frame captured while Marshall is talking.

        Two jobs, in sequence. Before the cut, run the guard. After it, KEEP COLLECTING:
        the user is still mid-sentence and those frames are the rest of what they are
        saying, so they go on the same pile the guard's window started.
        """
        if self._interrupted:
            self._interrupt_frames.append(frame)
            return

        captured = await loop.run_in_executor(None, self._feed_guard, frame)
        if captured is None:
            return

        fired = time.perf_counter()
        #: set BEFORE stopping playback, so the `_speak` loop sees it the moment its
        #: blocked `play()` returns and plays no further sentence
        self._interrupted = True
        self._interrupt_frames = captured

        #: `speaker.stop()` blocks while PortAudio aborts the stream (~197 ms measured),
        #: so it goes to the executor rather than onto the loop: this reader has to keep
        #: draining the mic straight through the cut, or the words the user is speaking
        #: at exactly that moment are the ones that get lost.
        await loop.run_in_executor(None, self.speaker.stop)
        self._time_to_silence_ms = (time.perf_counter() - fired) * 1000
        self._emit("barge_in", turn_id=self.turn_id, ms=self._time_to_silence_ms,
                   frames=len(captured))

    async def _handle_utterance(self, utterance, loop) -> None:
        """One full turn. Every stage failure is contained: the session goes back to
        LISTENING no matter what, because a voice loop that exits on a bad turn is a
        voice loop that exits."""
        #: the clock the user actually feels starts the moment they stop talking
        speech_ended = time.perf_counter()
        self.turn_id += 1
        #: a turn starts uninterrupted. `_back_to_listening` already cleared these; doing
        #: it again here is what makes that ordering non-load-bearing.
        self._interrupted = False
        self._interrupt_frames = []
        self._time_to_silence_ms = None
        self.guard.reset()
        utterance_s = len(utterance) / self.config.sample_rate
        telemetry = TurnTelemetry(
            turn_id=self.turn_id,
            utterance_s=utterance_s,
            tts_engine=getattr(self.tts, "engine_name", "unknown"),
        )
        self._emit("utterance", turn_id=self.turn_id, seconds=utterance_s)

        try:
            self._set_state(State.THINKING)

            with Stopwatch() as watch:
                transcript = await loop.run_in_executor(None, self.stt.transcribe, utterance)
            telemetry.stt_ms = watch.ms
            telemetry.transcript_chars = len(transcript or "")
            self._emit("transcript", turn_id=self.turn_id, text=transcript,
                       ms=watch.ms)

            if not transcript or not transcript.strip():
                #: silence, a cough, or a failed decode. Row still written — a turn
                #: that produced nothing is data about the endpointing thresholds.
                self._emit("discarded", turn_id=self.turn_id, reason="empty transcript")
                return

            with Stopwatch() as watch:
                response = await loop.run_in_executor(None, self._invoke_agent, transcript)
            telemetry.agent_ms = watch.ms
            telemetry.response_chars = len(response or "")
            self._emit("response", turn_id=self.turn_id, text=response, ms=watch.ms)

            if not response or not response.strip():
                self._emit("discarded", turn_id=self.turn_id, reason="empty response")
                return

            self._set_state(State.SPEAKING)
            sentences = await self._speak(response, telemetry, speech_ended, loop)

            if self._interrupted:
                await loop.run_in_executor(
                    None, self._truncate_interrupted_answer, sentences, telemetry)

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            #: same contract as the memory gate (H4) and the summariser (H2): a failed
            #: turn costs the answer, never the session
            logger.exception("turn %d failed", self.turn_id)
            self._emit("error", turn_id=self.turn_id, error=f"{type(exc).__name__}: {exc}")
        finally:
            interrupted = self._interrupted
            telemetry.was_interrupted = interrupted
            telemetry.time_to_silence_ms = self._time_to_silence_ms
            self.telemetry.write(telemetry)
            self._back_to_listening(interrupted=interrupted)

    def _invoke_agent(self, transcript: str) -> str:
        """The one call into `pipeline_lg`. Sync, unchanged, unaware of any of this."""
        return self.agent.invoke(
            transcript, config={"configurable": {"thread_id": self.config.thread_id}})

    async def _speak(self, response, telemetry, speech_ended, loop) -> list:
        """Synthesise sentence by sentence and play each in order. Returns the sentences.

        The list comes back so an interrupted turn can be truncated to the part that was
        actually heard; `telemetry.sentences_played` counts only sentences played to
        COMPLETION. Stopping mid-sentence drops that sentence entirely — there is no
        queue held for a resume, because V1b has no resume path: the user cut Marshall
        off, and the answer to that is to listen, not to pick the sentence back up.
        """
        sentences = await loop.run_in_executor(
            None, lambda: list(self.tts.stream_sentences(response)))
        telemetry.sentences_total = len(sentences)
        telemetry.sentences_played = 0
        for index, sentence, audio in sentences:
            if self._stopping.is_set():
                #: mid-answer shutdown: stop talking rather than finishing the speech
                break
            if self._interrupted:
                #: the queue is "cleared" by never playing the rest of it
                break
            if index == 0:
                #: the two headline numbers, both fixed at the first audible sample
                telemetry.tts_first_sentence_ms = (
                    time.perf_counter() - speech_ended) * 1000 - (
                        (telemetry.stt_ms or 0) + (telemetry.agent_ms or 0))
                telemetry.speech_end_to_first_audio_ms = (
                    time.perf_counter() - speech_ended) * 1000
            self._emit("speaking", turn_id=self.turn_id, index=index, text=sentence)
            await loop.run_in_executor(None, self.speaker.play, audio)
            if self._interrupted:
                #: `play` returned because `speaker.stop()` aborted it, not because the
                #: sentence finished — so it is not claimed as played
                break
            telemetry.sentences_played = index + 1
        return [sentence for _index, sentence, _audio in sentences]

    def _truncate_interrupted_answer(self, sentences, telemetry) -> None:
        """Rewrite the stored answer down to what the user actually heard.

        This is the half of barge-in that outlives the turn. H2's summariser and
        UserMemory both compress from the agent's history, so an untruncated answer
        teaches both of them a conversation that did not happen — and the next turn's
        "what you just said" would refer to sentences nobody ever heard.

        Runs in the executor because `update_state` writes a checkpoint. Failure is
        logged and swallowed, same contract as everything else on this path: a history
        that is too long is a worse answer, a raised exception here is a lost session.
        """
        played, total = telemetry.sentences_played or 0, telemetry.sentences_total or 0
        if total and played >= total:
            #: the guard fired on the tail of the last sentence. Everything was heard,
            #: so there is nothing to remove and no marker to justify.
            return
        content = truncate_to_spoken(sentences, played)
        try:
            rewritten = self.agent.truncate_last_response(content, self.config.thread_id)
        except Exception as exc:
            logger.exception("could not truncate the interrupted answer")
            self._emit("error", turn_id=self.turn_id,
                       error=f"truncation failed: {type(exc).__name__}: {exc}")
            return
        self._emit("truncated", turn_id=self.turn_id, sentences_played=played,
                   sentences_total=total, stored=bool(rewritten))

    def _back_to_listening(self, interrupted: bool = False) -> None:
        """Reset everything that could carry Marshall's own voice into the next turn.

        A barge-in inverts two of these steps, because on that path the microphone is not
        holding Marshall's voice — it is holding the user's, mid-sentence:

        * the detector is SEEDED with the guard's capture rather than left empty, so the
          words that did the interrupting are the head of the next utterance;
        * the mic queue is NOT drained, because what is queued is the rest of that same
          sentence, and the recurrent VAD state is left alone for the same reason —
          resetting it mid-word costs the next few probabilities for nothing.

        That is exactly where V1b's headphone requirement is load-bearing: through
        speakers, that undrained queue would also hold the tail of Marshall's own answer.
        """
        self.detector.reset()
        self.guard.reset()
        if interrupted:
            self.detector.seed(self._interrupt_frames)
        else:
            try:
                self.vad.reset()
            except Exception:
                logger.debug("VAD reset failed", exc_info=True)
            dropped = self.mic.drain()
            if dropped:
                self.frames_discarded += dropped
        self._interrupted = False
        self._interrupt_frames = []
        self._time_to_silence_ms = None
        if not self._stopping.is_set():
            self._set_state(State.LISTENING)
        #: last, and only after the state is back to LISTENING — the reader gates on
        #: this, so clearing it earlier would reopen the window it exists to close
        self._turn_pending = False

    # --- shutdown --------------------------------------------------------------

    def request_stop(self) -> None:
        """Ask the loop to finish. Safe from a signal handler and safe to call twice."""
        self._stopping.set()

    def close(self) -> None:
        """Release hardware. Idempotent, and never raises — this runs on the way out."""
        for closer in (self.speaker.stop, self.mic.stop):
            try:
                closer()
            except Exception:
                logger.debug("shutdown step failed", exc_info=True)


async def run_session(orchestrator: VoiceOrchestrator) -> None:
    """Run until Ctrl+C, then shut down without a traceback.

    `KeyboardInterrupt` is caught here rather than installing a signal handler: on
    Windows, `loop.add_signal_handler` raises NotImplementedError, and asyncio already
    surfaces Ctrl+C as KeyboardInterrupt out of the awaited task.
    """
    task = asyncio.ensure_future(orchestrator.run())
    try:
        await task
    except (KeyboardInterrupt, asyncio.CancelledError):
        orchestrator.request_stop()
        task.cancel()
        #: gather rather than await: the task is already cancelled, and this is the
        #: form that will not re-raise the CancelledError we just caused
        await asyncio.gather(task, return_exceptions=True)
    finally:
        orchestrator.request_stop()
        orchestrator.close()
