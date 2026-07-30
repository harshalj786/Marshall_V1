"""Speech to text via faster-whisper (`large-v3-turbo`).

CUDA with `int8_float16` is the target; CPU `int8` is the fallback. The fallback is
not decoration — CTranslate2 needs cuDNN 9, which is not a pip dependency of anything
here and arrives only with a CUDA torch build (requirements.txt note 7). On a machine
without it, `WhisperModel(device="cuda")` raises at construction, and a voice assistant
that refuses to boot because it is slow is worse than a slow voice assistant.
"""
import logging
import time
from typing import Optional

import numpy as np

from voice.config import VoiceConfig

logger = logging.getLogger(__name__)


class Transcriber:
    """Loads on construction, so the model cost lands at boot rather than on turn one."""

    def __init__(self, config: Optional[VoiceConfig] = None):
        from faster_whisper import WhisperModel

        self.config = config or VoiceConfig()
        #: what actually loaded, not what was asked for — the telemetry and `--check`
        #: both report this, so a silent CPU fallback cannot be mistaken for a slow GPU
        self.device = self.config.stt_device
        self.compute_type = self.config.stt_compute_type
        self.fell_back = False

        started = time.perf_counter()
        try:
            self._model = WhisperModel(
                self.config.stt_model,
                device=self.config.stt_device,
                compute_type=self.config.stt_compute_type,
            )
        except Exception as exc:
            #: bare Exception on purpose: CTranslate2 surfaces a missing cuDNN as a
            #: RuntimeError, a missing GPU as a ValueError, and a driver mismatch as
            #: an OSError. Enumerating them would just be a list to get wrong later.
            logger.warning(
                "STT could not initialise on %s/%s (%s: %s) — falling back to cpu/%s. "
                "Transcription will be several times slower. This is usually cuDNN 9 "
                "missing for CTranslate2; see requirements.txt note 7.",
                self.config.stt_device, self.config.stt_compute_type,
                type(exc).__name__, exc, self.config.stt_cpu_compute_type)
            self.device = "cpu"
            self.compute_type = self.config.stt_cpu_compute_type
            self.fell_back = True
            self._model = WhisperModel(
                self.config.stt_model,
                device="cpu",
                compute_type=self.config.stt_cpu_compute_type,
            )
        self.load_seconds = time.perf_counter() - started

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe one buffered utterance. Returns "" for anything unusable.

        Callers treat "" as "say nothing and go back to listening" — the empty string
        is the discard signal, so silence, a cough, and a failed decode all take the
        same path rather than each needing their own.
        """
        if audio is None or len(audio) == 0:
            return ""
        duration = len(audio) / self.config.sample_rate
        if duration < self.config.min_utterance_s:
            #: too short to be speech. Whisper hallucinates confidently on sub-second
            #: noise — "Thank you." and "you" are its favourites — and a hallucinated
            #: transcript costs a full agent turn plus a spoken answer to nothing.
            return ""
        segments, _info = self._model.transcribe(
            np.ascontiguousarray(audio, dtype=np.float32),
            language=self.config.stt_language,
            #: the generator is lazy; joining it here is what actually runs inference
        )
        return " ".join(segment.text.strip() for segment in segments).strip()
