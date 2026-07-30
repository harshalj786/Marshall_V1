"""Voice activity detection and utterance endpointing.

Split deliberately in two:

* `UtteranceDetector` is the STATE MACHINE. It consumes (frame, speech_probability)
  pairs and returns a complete utterance when one ends. It never loads a model, never
  touches an audio device, and never imports torch — so the pre-roll, the start
  trigger and the endpoint can all be tested against hand-written probability
  sequences, deterministically and in milliseconds.
* `SileroVad` is the MODEL. It turns a frame into one probability and holds the
  recurrent state that makes streaming work.

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
