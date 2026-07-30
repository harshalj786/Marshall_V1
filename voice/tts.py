"""Text to speech via Kokoro (voice `af_heart`, 24 kHz).

**On espeak-ng.** Every Kokoro guide says it is a system dependency needing an MSI and
a PATH entry on Windows. It is not, here: `espeakng-loader` ships the DLL and the
language data inside a wheel and `phonemizer-fork` loads them from there, so the whole
install is `pip install -r requirements.txt` — no admin prompt, nothing interactive.
That is why the piper-tts fallback contemplated for V1a does not exist: it was
contingent on espeak-ng being uninstallable non-interactively, and it turned out to be
installable by pip. `engine_name` is still reported through the telemetry, because
"which engine actually landed" is a fact worth recording even when there is one.

**Why sentence-by-sentence.** Kokoro has a per-call input limit and its quality falls
off well before that limit. Splitting also buys the number V1b will care about most:
`tts_first_sentence_ms`, the point at which playback COULD start. V1a still plays
sequentially and waits for each sentence, but the measurement is already there.
"""
import re
import time
from typing import Iterator, List, Optional

import numpy as np

from voice.config import VoiceConfig

#: split on the whitespace AFTER . ! ? — optionally after a closing quote or bracket.
#: Two lookbehinds rather than one pattern with `["\')\]]*` inside the separator: that
#: form consumes the closing quote, so `He said "hi." Then left.` loses the `"` off the
#: first chunk. Only whitespace may be consumed here; everything else belongs to a
#: sentence. Deliberately not a real tokenizer — this runs on every turn between the
#: model and the speaker, and spacy's is import-heavy work to make "Dr. Smith"
#: marginally better. A wrong split costs a slightly odd pause, nothing more.
_SENTENCE_END = re.compile(r'(?<=[.!?])\s+|(?<=[.!?]["\'\)\]])\s+')


def split_sentences(text: str, max_chars: int = 400) -> List[str]:
    """Split into speakable chunks, none longer than `max_chars`.

    The length cap is the load-bearing half. Sentence punctuation is a suggestion the
    model may simply not follow — a numbered list, a code-ish answer, or one long
    breathless clause all arrive as a single "sentence" — and handing Kokoro an
    unbounded string is the failure this function exists to prevent.
    """
    if not text or not text.strip():
        return []
    chunks: List[str] = []
    for sentence in _SENTENCE_END.split(text.strip()):
        sentence = sentence.strip()
        if not sentence:
            continue
        while len(sentence) > max_chars:
            #: prefer a word boundary; fall back to a hard cut so a single
            #: max_chars-long token cannot loop forever
            cut = sentence.rfind(" ", 0, max_chars)
            if cut <= 0:
                cut = max_chars
            chunks.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if sentence:
            chunks.append(sentence)
    return chunks


class KokoroEngine:
    """Kokoro-82M. Loads on construction so the cost lands at boot."""

    engine_name = "kokoro"

    def __init__(self, config: Optional[VoiceConfig] = None):
        from kokoro import KPipeline

        self.config = config or VoiceConfig()
        started = time.perf_counter()
        #: repo_id passed explicitly — omitting it works but prints a warning on every
        #: construction, and a boot that looks alarming gets treated as broken
        self._pipeline = KPipeline(lang_code=self.config.tts_lang_code,
                                   repo_id="hexgrad/Kokoro-82M")
        self.load_seconds = time.perf_counter() - started

    @property
    def sample_rate(self) -> int:
        return self.config.tts_sample_rate

    def synthesize(self, text: str) -> np.ndarray:
        """One chunk of text to one float32 waveform. Empty array for empty input."""
        if not text or not text.strip():
            return np.zeros(0, dtype=np.float32)
        pieces = []
        for _graphemes, _phonemes, audio in self._pipeline(text, voice=self.config.tts_voice):
            #: Kokoro yields torch tensors; the speaker and the tests both want numpy
            pieces.append(audio.numpy() if hasattr(audio, "numpy") else np.asarray(audio))
        if not pieces:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(pieces).astype(np.float32)

    def stream_sentences(self, text: str) -> Iterator[tuple]:
        """Yield `(index, sentence, audio)` per sentence, synthesising lazily.

        Lazy on purpose: the caller can start playing sentence 0 while 1 is still being
        synthesised. V1a does not — it plays sequentially — but this is the seam V1d
        needs, and shaping it now costs nothing.
        """
        for index, sentence in enumerate(
                split_sentences(text, self.config.tts_max_chars_per_chunk)):
            yield index, sentence, self.synthesize(sentence)


def build_engine(config: Optional[VoiceConfig] = None):
    """Return the TTS engine for this machine.

    One engine today. This exists so the orchestrator depends on `engine_name` +
    `synthesize` + `stream_sentences` rather than on Kokoro specifically, which is
    what makes the telemetry's `tts_engine` field meaningful and what a second
    backend would slot into.
    """
    return KokoroEngine(config)
