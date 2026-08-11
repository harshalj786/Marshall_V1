"""Voice activity detection and utterance endpointing.

Split deliberately in two:

* `UtteranceDetector` and `BargeInGuard` are the STATE MACHINES. They consume
  (frame, speech_probability) pairs — one deciding when an utterance has ended, the
  other whether the user is talking over Marshall — and neither loads a model, touches
  an audio device, or imports torch. So the pre-roll, the start trigger, the endpoint
  and the interrupt guard can all be tested against hand-written probability sequences,
  deterministically and in milliseconds.
* `SileroVad` is the MODEL. It turns a frame into one probability and holds the
  recurrent state that makes streaming work. One instance feeds both state machines,
  including through SPEAKING, so its recurrent state never sees a gap.

That is the same split `memory/reconciliation.py` uses for `find_safe_cut_point` — the
decision is a pure function over data, and it is the half that is worth testing.
"""
from collections import deque
from typing import Optional

import numpy as np

from voice.config import VoiceConfig, VAD_SAMPLE_RATE


class UtteranceDetector:
    """Turns a stream of frames into discrete utterances.

    Two states: waiting for speech, and inside speech. The transitions are

        waiting -> speaking     one frame at or above the threshold
        speaking -> waiting     `silence_frames` consecutive sub-threshold frames,
                                or `max_utterance_frames` reached

    On the way out it returns the utterance INCLUDING the pre-roll and including the
    trailing silence. The trailing silence is not trimmed on purpose: Whisper uses it
    to decide the final word has ended, and clipping it truncates the last syllable.
    """

    def __init__(self, config: VoiceConfig):
        self.config = config
        #: bounded ring buffer, so audio recorded while nobody is talking costs a
        #: fixed 500 ms of memory rather than growing until someone speaks
        self._pre_roll: deque = deque(maxlen=config.pre_roll_frames)
        self._frames: list = []
        self._speaking = False
        self._silence_run = 0

    # --- introspection, for tests and for the orchestrator's logging ------------

    @property
    def is_speaking(self) -> bool:
        return self._speaking

    @property
    def buffered_frames(self) -> int:
        return len(self._frames)

    def reset(self) -> None:
        """Drop all state. Called when leaving LISTENING, so half an utterance
        captured before Marshall started talking is not glued onto the next one."""
        self._pre_roll.clear()
        self._frames = []
        self._speaking = False
        self._silence_run = 0

    def seed(self, frames) -> None:
        """Start an utterance already in progress, from frames captured elsewhere.

        This is how a barge-in hands the interruption over (V1b). The frames that fired
        `BargeInGuard` are the user's actual first words — they were spoken over the top
        of Marshall, before the orchestrator was back in LISTENING — so the detector
        resumes mid-utterance with them in the buffer rather than starting empty. Without
        it, everything said up to the moment playback stopped is lost and the user has to
        repeat their own interruption, which is a worse experience than not interrupting.

        Endpointing from here is completely ordinary: `silence_frames` of quiet ends the
        utterance exactly as it would have if it had started in LISTENING.
        """
        self.reset()
        frames = list(frames)
        if not frames:
            return
        self._frames = frames
        self._speaking = True

    # --- the state machine -----------------------------------------------------

    def process(self, frame: np.ndarray, speech_prob: float) -> Optional[np.ndarray]:
        """Feed one frame. Returns the finished utterance, or None if still listening."""
        is_speech = speech_prob >= self.config.vad_threshold

        if not self._speaking:
            if not is_speech:
                #: still quiet — keep the frame only as pre-roll
                self._pre_roll.append(frame)
                return None
            #: speech starts. The pre-roll goes in FRONT of this frame: silero needs a
            #: few frames of a word before it crosses the threshold, so the true start
            #: of the utterance is already behind us.
            self._speaking = True
            self._silence_run = 0
            self._frames = list(self._pre_roll) + [frame]
            self._pre_roll.clear()
            return None

        # inside an utterance
        self._frames.append(frame)
        if is_speech:
            #: a single speech frame resets the run: people pause mid-sentence, and
            #: counting silence cumulatively would end the utterance on the third comma
            self._silence_run = 0
        else:
            self._silence_run += 1
            if self._silence_run >= self.config.silence_frames:
                return self._finish()

        if len(self._frames) >= self.config.max_utterance_frames:
            #: a stuck mic must not grow this without bound. Cutting mid-sentence is a
            #: bad answer; an unbounded buffer handed to Whisper is a dead session.
            return self._finish()
        return None

    def _finish(self) -> np.ndarray:
        utterance = np.concatenate(self._frames) if self._frames else np.zeros(0, np.float32)
        self.reset()
        return utterance


class BargeInGuard:
    """Decides when speech ARRIVING WHILE MARSHALL TALKS should cut him off (V1b).

    Deliberately not `UtteranceDetector` with different numbers. The two answer opposite
    questions: the detector asks "has this utterance finished?" and is tuned to be
    generous at the start (one frame over the threshold begins one), while this asks
    "is someone really talking over me?" and must be hard to convince — the cost of a
    mistake is Marshall silenced mid-sentence by a cough, a door, or a keyboard.

    Hence two guards rather than one, both from config:

        `barge_in_vad_threshold`     how loud/speech-like a frame has to be, ABOVE the
                                     listening threshold
        `barge_in_sustained_frames`  how many of those in an unbroken run

    A single sub-threshold frame resets the run to zero. That is the strict reading on
    purpose: a transient is short, and speech that survives ~300 ms of continuous
    detection is not a transient.

    On firing it returns the captured audio — the ring buffer holds `pre_roll_frames`
    from BEFORE the run plus the run itself, so the words that triggered the interrupt
    become the head of the user's next utterance instead of being thrown away. Both
    numbers are unmeasured starting guesses; `voice/config.py` says so, and the telemetry
    exists to correct them.
    """

    def __init__(self, config: VoiceConfig):
        self.config = config
        #: pre-roll + the guard window, so a fired guard hands back everything from
        #: shortly before the user started talking through to the frame that fired it
        self._window: deque = deque(
            maxlen=config.pre_roll_frames + config.barge_in_sustained_frames)
        self._run = 0

    @property
    def enabled(self) -> bool:
        return self.config.barge_in_enabled

    @property
    def speech_run(self) -> int:
        """Consecutive frames currently over the threshold. For tests and logging."""
        return self._run

    def reset(self) -> None:
        self._window.clear()
        self._run = 0

    def process(self, frame: np.ndarray, speech_prob: float) -> Optional[list]:
        """Feed one frame captured during SPEAKING.

        Returns the captured frames when the interrupt fires, otherwise None. Returns
        None immediately and keeps no state at all when `barge_in_mode` is "off" — that
        mode is V1a, and V1a does not look at these frames.
        """
        if not self.enabled:
            return None

        self._window.append(frame)
        if speech_prob < self.config.barge_in_vad_threshold:
            self._run = 0
            return None

        self._run += 1
        if self._run < self.config.barge_in_sustained_frames:
            return None
        captured = list(self._window)
        #: fires once. The orchestrator owns everything after this frame — it keeps
        #: collecting into the same utterance while playback drains — so leaving the run
        #: armed would only let a second fire race the first one's cut.
        self.reset()
        return captured


class SileroVad:
    """Streaming speech probability, one 512-sample frame at a time.

    The model is recurrent: `reset_states()` between utterances matters, or the tail
    of the last one biases the start of the next.
    """

    def __init__(self, config: Optional[VoiceConfig] = None):
        from silero_vad import load_silero_vad
        self.config = config or VoiceConfig()
        self._model = load_silero_vad()

    def speech_prob(self, frame: np.ndarray) -> float:
        import torch
        #: silero wants float32 in [-1, 1]; sounddevice already hands us that dtype,
        #: but a copy from a ring buffer can be non-contiguous and torch rejects those
        tensor = torch.from_numpy(np.ascontiguousarray(frame, dtype=np.float32))
        return float(self._model(tensor, VAD_SAMPLE_RATE).item())

    def reset(self) -> None:
        self._model.reset_states()
