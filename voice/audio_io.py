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
from typing import Optional, Union

import numpy as np


#: preference among host APIs when several expose the SAME microphone, best first.
#: This is a real-time capture path, so latency is the tiebreak: WASAPI is the modern
#: low-latency API, DirectSound is middling, MME is the oldest and adds the most, and
#: WDM-KS bypasses the mixer so it usually cannot be shared with another application.
#: Lower rank wins; anything unlisted sorts last.
_HOST_API_RANK = {"windows wasapi": 0, "windows directsound": 1, "mme": 2, "windows wdm-ks": 3}


def _host_rank(sd, device: dict) -> int:
    return _HOST_API_RANK.get(sd.query_hostapis(device["hostapi"])["name"].lower(), 9)


class DeviceNotUsableError(RuntimeError):
    """A named device exists but cannot run at the rate the pipeline needs.

    Its own class because the raw PortAudio failure — `Invalid sample rate
    [PaErrorCode -9997]` — names the symptom and not one thing you can act on. The
    message this carries names the cause and the fix.
    """


def resolve_device(spec: Optional[Union[int, str]],
                   samplerate: int,
                   channels: int = 1,
                   want_input: bool = True) -> Optional[int]:
    """Turn a device spec — index, name substring, or None — into a device index.

    **Why names are supported at all.** PortAudio indices are positional and shift
    whenever a device appears or disappears: connecting one Bluetooth controller
    renumbers everything after it. An index written down today can silently address a
    different microphone tomorrow, and the failure is a session that hears nothing
    rather than an error. A substring survives that.

    **Why a match is then TESTED rather than returned.** The same physical microphone
    is enumerated once per host API, and they do not have the same capabilities: the
    WASAPI view of a 48 kHz Realtek input cannot open at 16 kHz, while the MME and
    DirectSound views of that identical microphone resample in software and can. Name
    matching alone would pick whichever came first and fail roughly half the time, so
    every candidate is checked against the settings actually required and the first
    usable one wins.

    Among usable candidates the lowest-latency host API wins (see `_HOST_API_RANK`).

    None means "system default", which is what the old behaviour was and still is.
    """
    import sounddevice as sd

    if spec is None:
        return None

    check = sd.check_input_settings if want_input else sd.check_output_settings
    kind = "input" if want_input else "output"

    def _usable(index: int) -> bool:
        try:
            check(device=index, samplerate=samplerate, channels=channels, dtype="float32")
            return True
        except Exception:
            return False

    #: an explicit index is honoured as given — the caller named one device, so
    #: silently substituting another would be worse than failing
    if isinstance(spec, int) or (isinstance(spec, str) and spec.strip().lstrip("-").isdigit()):
        index = int(spec)
        if not _usable(index):
            name = sd.query_devices(index)["name"]
            raise DeviceNotUsableError(
                f"{kind} device [{index}] {name!r} cannot run at {samplerate} Hz / "
                f"{channels}ch. The same device is usually listed again under a host API "
                f"that resamples (DirectSound or MME). Select it by name instead of index: "
                f"--{kind}-device \"{name.split('(')[0].strip()}\"")
        return index

    needle = spec.strip().lower()
    matches, rejected = [], []
    for index, device in enumerate(sd.query_devices()):
        slots = device["max_input_channels"] if want_input else device["max_output_channels"]
        if slots <= 0 or needle not in device["name"].lower():
            continue
        (matches if _usable(index) else rejected).append((index, device["name"]))

    if matches:
        #: stable sort on (host rank, index): among the several host-API views of one
        #: microphone take the lowest-latency one, and break remaining ties by
        #: enumeration order so the choice is deterministic run to run
        return min(matches, key=lambda m: (_host_rank(sd, sd.query_devices(m[0])), m[0]))[0]
    if rejected:
        raise DeviceNotUsableError(
            f"{len(rejected)} {kind} device(s) match {spec!r} but none can run at "
            f"{samplerate} Hz / {channels}ch: "
            + ", ".join(f"[{i}] {n}" for i, n in rejected)
            + ". Every host API for this device refused the rate.")
    raise DeviceNotUsableError(
        f"no {kind} device matches {spec!r}. Run --list-devices to see what is available.")


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

    Deliberately NOT stopped while Marshall is speaking. V1a discarded those frames in
    the orchestrator rather than never capturing them, because reopening the device every
    turn costs hundreds of milliseconds on Windows and occasionally fails outright when
    another process grabbed it in between. V1b is what that decision bought: barge-in
    needs those exact frames, and they were already arriving.
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
    """Blocking playback of float32 mono audio at the TTS rate, abortable from a thread.

    Blocking stays correct through V1b, and the callback-driven output stream V1a
    expected barge-in to need turned out not to be needed: `sd.stop()` aborts the stream
    from any thread and releases the `sd.wait()` blocking `play`, so ONE sentence is the
    unit of interruption. That is granular enough — the whole cut measures ~197 ms of
    drain plus the guard's ~300 ms, inside human interruption norms — and it avoids
    holding playback position in a real-time callback for the sake of stopping a fraction
    of a second sooner. Sub-sentence position is V1c's problem, with streaming.

    The one contract this places on the caller: `play` runs in an executor and `stop` is
    called from the event loop, so the two really do run concurrently.
    """

    def __init__(self, config):
        self.config = config

    def play(self, audio: np.ndarray) -> None:
        """Play to completion — or until `stop()` cuts it short."""
        import sounddevice as sd
        if audio is None or len(audio) == 0:
            return
        sd.play(np.ascontiguousarray(audio, dtype=np.float32),
                samplerate=self.config.tts_sample_rate,
                device=self.config.output_device)
        sd.wait()

    def stop(self) -> None:
        """Abort playback now. Safe to call while `play` is blocked in another thread —
        which is the whole barge-in mechanism — and safe to call when nothing is playing.
        """
        import sounddevice as sd
        try:
            sd.stop()
        except Exception:
            pass
