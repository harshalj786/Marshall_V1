"""Every tunable in the voice path, in one dataclass.

`docs/architecture_review.md` §4 lists "hardcoded values that should be configurable"
as a standing finding across the older modules — model names at seven sites, chunk
sizes at seven more. The voice layer starts on the other side of that: one frozen
dataclass, every number named, and the numbers that were guessed labelled as guesses.

Nothing here reads the environment or a file. R2 (a real `config/settings.py`) is
still open, and when it lands this is what it should populate.
"""
from dataclasses import dataclass
from typing import Optional


#: Silero is trained on 16 kHz and accepts EXACTLY 512-sample frames there (256 at
#: 8 kHz). This is not a tunable — the model rejects anything else — so it is a
#: module constant rather than a field, and `VoiceConfig.validate()` enforces it.
VAD_FRAME_SAMPLES = 512
VAD_SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class VoiceConfig:
    """Tunables for one voice session. Frozen: a mid-session change would desync
    the ring buffer sizing from the frames already in it."""

    # --- capture ---------------------------------------------------------------
    sample_rate: int = VAD_SAMPLE_RATE
    #: one VAD frame per callback. Larger blocks would cost latency at the endpoint
    #: (the tail is only detected on a frame boundary) and buy nothing.
    blocksize: int = VAD_FRAME_SAMPLES
    channels: int = 1
    input_device: Optional[int] = None
    output_device: Optional[int] = None

    # --- endpointing -----------------------------------------------------------
    #: Silero speech probability above which a frame counts as speech. 0.5 is the
    #: model's own documented default and has not been tuned against this mic.
    vad_threshold: float = 0.5
    #: audio kept from BEFORE speech was detected. Silero needs a few frames of a
    #: word before it crosses the threshold, so without this the utterance starts
    #: mid-consonant and Whisper drops the first word. 500 ms is generous on purpose:
    #: it is pure memory (16 kB) and clipping the first word is the expensive failure.
    pre_roll_ms: int = 500
    #: trailing silence that ends an utterance. The floor is set by natural
    #: mid-sentence pauses — below ~400 ms Marshall interrupts people who breathe.
    silence_ms: int = 600
    #: anything shorter is a cough, a keystroke, or a door. Whisper will happily
    #: hallucinate a sentence out of it, which is worse than dropping it.
    min_utterance_s: float = 0.3
    #: hard stop on a single utterance, so a stuck-open mic or a running tap cannot
    #: grow the buffer without bound and then hand Whisper a 20-minute clip.
    max_utterance_s: float = 30.0

    # --- speech to text --------------------------------------------------------
    stt_model: str = "large-v3-turbo"
    stt_device: str = "cuda"
    stt_compute_type: str = "int8_float16"
    #: used only when CUDA init raises — see voice/stt.py. int8 rather than
    #: int8_float16 because float16 on CPU is emulated and slower than plain int8.
    stt_cpu_compute_type: str = "int8"
    #: pinned rather than auto-detected: detection costs an extra pass and this
    #: system's prompts, tools and documents are all English anyway.
    stt_language: str = "en"

    # --- text to speech --------------------------------------------------------
    tts_voice: str = "af_heart"
    #: Kokoro's native output rate. Resampling to match the mic would cost quality
    #: for nothing — playback opens its own stream at this rate.
    tts_sample_rate: int = 24_000
    #: Kokoro's language code for American English.
    tts_lang_code: str = "a"
    #: Kokoro has a per-call input limit and degrades before reaching it, so
    #: `voice/tts.py` splits on sentences and this bounds what a "sentence" may be
    #: when the text has no punctuation at all.
    tts_max_chars_per_chunk: int = 400

    # --- telemetry -------------------------------------------------------------
    telemetry_path: str = "voice_logs/turns.jsonl"

    # --- agent -----------------------------------------------------------------
    thread_id: str = "voice-main"

    def validate(self) -> None:
        """Fail loudly at boot rather than subtly at the first utterance.

        Every check here is something that otherwise surfaces as bad audio or a
        confusing exception several layers away.
        """
        if self.sample_rate != VAD_SAMPLE_RATE:
            raise ValueError(
                f"silero-vad requires {VAD_SAMPLE_RATE} Hz, got {self.sample_rate}")
        if self.blocksize != VAD_FRAME_SAMPLES:
            raise ValueError(
                f"silero-vad requires exactly {VAD_FRAME_SAMPLES}-sample frames at "
                f"{VAD_SAMPLE_RATE} Hz, got {self.blocksize}")
        if self.channels != 1:
            raise ValueError(f"the VAD and Whisper both want mono, got {self.channels}")
        if not 0.0 < self.vad_threshold < 1.0:
            raise ValueError(f"vad_threshold must be in (0, 1), got {self.vad_threshold}")
        if self.pre_roll_ms < 0:
            raise ValueError(f"pre_roll_ms must be >= 0, got {self.pre_roll_ms}")
        if self.silence_ms <= 0:
            raise ValueError(f"silence_ms must be > 0, got {self.silence_ms}")
        if self.min_utterance_s < 0:
            raise ValueError(f"min_utterance_s must be >= 0, got {self.min_utterance_s}")
        if self.max_utterance_s <= self.min_utterance_s:
            raise ValueError(
                f"max_utterance_s ({self.max_utterance_s}) must exceed "
                f"min_utterance_s ({self.min_utterance_s})")

    # --- derived ---------------------------------------------------------------
    # frames, not milliseconds, because that is the unit the VAD loop counts in.

    @property
    def frame_duration_s(self) -> float:
        return self.blocksize / self.sample_rate

    @property
    def pre_roll_frames(self) -> int:
        return max(1, round(self.pre_roll_ms / 1000 / self.frame_duration_s))

    @property
    def silence_frames(self) -> int:
        return max(1, round(self.silence_ms / 1000 / self.frame_duration_s))

    @property
    def max_utterance_frames(self) -> int:
        return max(1, round(self.max_utterance_s / self.frame_duration_s))
