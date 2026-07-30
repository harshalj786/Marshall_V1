"""Microphone capture and speaker playback over sounddevice.

The rule that shapes this module: **the audio callback does exactly one thing —
push a frame onto a queue.** PortAudio calls it from a real-time thread with a hard
deadline (32 ms at our block size); anything slower than that deadline — a VAD
inference, a log line, a lock, an allocation — produces a dropout the user hears as
a glitch and the VAD sees as a false endpoint. Every consumer therefore runs on the
other side of a `queue.Queue`, which is the one structure that is safe to `put` into
from a real-time thread and safe to `get` from anywhere else.
"""
import queue
from typing import Optional

import numpy as np


def resample_linear(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Resample by linear interpolation.

    Needed because the two ends of the pipeline disagree: Kokoro emits 24 kHz and
    Whisper assumes 16 kHz, so anything that feeds TTS output back into STT — the
    `--check` round trip, the integration test — has to convert. Linear interpolation
    is crude (no anti-aliasing filter, so it images above Nyquist) but it is enough for
    a speech round trip and it avoids adding scipy to the dependency list for one call.

    Not on the live path: the mic captures at 16 kHz natively and playback opens its
    own stream at 24 kHz, so neither direction resamples during a real turn.
    """
    if from_rate == to_rate or len(audio) == 0:
        return np.asarray(audio, dtype=np.float32)
    target_length = int(round(len(audio) * to_rate / from_rate))
    if target_length <= 0:
        return np.zeros(0, dtype=np.float32)
    source_positions = np.arange(len(audio), dtype=np.float64)
    target_positions = np.linspace(0, len(audio) - 1, target_length, dtype=np.float64)
    return np.interp(target_positions, source_positions,
                     np.asarray(audio, dtype=np.float64)).astype(np.float32)


def list_devices() -> str:
    """Human-readable device table, for `--list-devices` and `--check`."""
    import sounddevice as sd
    lines = []
    default_in, default_out = sd.default.device
    for index, device in enumerate(sd.query_devices()):
        if device["max_input_channels"] == 0 and device["max_output_channels"] == 0:
            continue
        host = sd.query_hostapis(device["hostapi"])["name"]
        marks = "".join((
            "<in " if index == default_in else "",
            "<out" if index == default_out else "",
        ))
        lines.append(
            f"  [{index:>3}] in={device['max_input_channels']} "
            f"out={device['max_output_channels']} "
            f"{device['default_samplerate']:>6.0f}Hz  {device['name'][:40]:<40} "
            f"({host}) {marks}")
    return "\n".join(lines)


class MicrophoneStream:
    """Continuous 16 kHz mono capture. Frames land on `.queue` as float32 arrays.

    Deliberately NOT stopped while Marshall is speaking. V1a has no barge-in, so those
    frames are discarded by the orchestrator rather than never captured — but keeping
    the stream open means the device is not reopened on every turn, which on Windows
    costs hundreds of milliseconds and occasionally fails outright when another
    process grabbed it in between.
    """

    def __init__(self, config, on_overflow=None):
        self.config = config
        self.queue: queue.Queue = queue.Queue()
        self._stream = None
        self._on_overflow = on_overflow
        self.overflow_count = 0

    def _callback(self, indata, frames, time_info, status) -> None:
        if status:
            #: counted, not logged: printing from the real-time thread is exactly the
            #: kind of work that causes the next overflow
            self.overflow_count += 1
        #: copy() because PortAudio reuses `indata`'s buffer on the next callback —
        #: without it every queued frame silently becomes the most recent one
        self.queue.put_nowait(indata[:, 0].copy())

    def start(self) -> "MicrophoneStream":
        import sounddevice as sd
        self._stream = sd.InputStream(
            samplerate=self.config.sample_rate,
            blocksize=self.config.blocksize,
            channels=self.config.channels,
            dtype="float32",
            device=self.config.input_device,
            callback=self._callback,
        )
        self._stream.start()
        return self

    def get(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """Next frame, or None if none arrived within `timeout`.

        The timeout is what keeps the async loop cancellable: this call is made in an
        executor thread, and a blocking `get()` with no timeout would strand that
        thread past shutdown.
        """
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain(self) -> int:
        """Discard everything queued. Returns how many frames were dropped.

        Called on re-entering LISTENING: without it, the frames captured while
        Marshall was talking are still queued, and the VAD would process his own
        speech as the user's next utterance the moment he stopped.
        """
        dropped = 0
        while True:
            try:
                self.queue.get_nowait()
                dropped += 1
            except queue.Empty:
                return dropped

    def stop(self) -> None:
        if self._stream is not None:
            #: stop before close, and swallow — a device yanked mid-session (a USB
            #: headset unplugged) raises here, and shutdown must still complete
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def __enter__(self) -> "MicrophoneStream":
        return self.start()

    def __exit__(self, *exc_info) -> bool:
        self.stop()
        return False


class Speaker:
    """Blocking playback of float32 mono audio at the TTS rate.

    Blocking is correct for V1a: SPEAKING is a state the orchestrator waits out, and
    the call is made in an executor so the event loop stays responsive. V1b's barge-in
    is what needs this to become interruptible, and it will need a callback-driven
    output stream to do it — not this.
    """

    def __init__(self, config):
        self.config = config

    def play(self, audio: np.ndarray) -> None:
        import sounddevice as sd
        if audio is None or len(audio) == 0:
            return
        sd.play(np.ascontiguousarray(audio, dtype=np.float32),
                samplerate=self.config.tts_sample_rate,
                device=self.config.output_device)
        sd.wait()

    def stop(self) -> None:
        import sounddevice as sd
        try:
            sd.stop()
        except Exception:
            pass
