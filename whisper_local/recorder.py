"""Audio recording with sounddevice. All audio stays in memory as numpy arrays."""

import logging
import threading

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


class AudioRecorder:
    """Records audio from the microphone into memory.

    Uses sounddevice's InputStream in callback mode. Audio is accumulated
    as a list of numpy arrays and never written to disk.
    """

    def __init__(self, sample_rate: int = 16000, device_index: int | None = None) -> None:
        self._sample_rate = sample_rate
        self._device_index = device_index
        self._lock = threading.Lock()
        self._frames: list[np.ndarray] = []
        self._stream: sd.InputStream | None = None
        self._is_recording = False

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    def start(self) -> None:
        """Start recording audio from the microphone."""
        if self._is_recording:
            return

        with self._lock:
            self._frames.clear()

        self._stream = sd.InputStream(
            samplerate=self._sample_rate,
            channels=1,
            dtype="float32",
            device=self._device_index,
            callback=self._audio_callback,
        )
        self._stream.start()
        self._is_recording = True
        logger.debug("Recording started (device=%s, rate=%d)", self._device_index, self._sample_rate)

    def stop(self) -> np.ndarray:
        """Stop recording and return the captured audio as a single array.

        Returns:
            numpy array of shape (n_samples,) with float32 audio at the configured sample rate.
            Returns an empty array if nothing was recorded.
        """
        if not self._is_recording or self._stream is None:
            return np.array([], dtype=np.float32)

        self._stream.stop()
        self._stream.close()
        self._stream = None
        self._is_recording = False

        with self._lock:
            if not self._frames:
                return np.array([], dtype=np.float32)
            audio = np.concatenate(self._frames, axis=0).flatten()
            self._frames.clear()

        logger.debug("Recording stopped: %.2f seconds of audio", len(audio) / self._sample_rate)
        return audio

    def get_chunk(self) -> np.ndarray:
        """Get the current audio buffer and clear it. Used for continuous mode.

        Returns:
            numpy array of accumulated audio since last call.
            Returns an empty array if no audio accumulated.
        """
        with self._lock:
            if not self._frames:
                return np.array([], dtype=np.float32)
            audio = np.concatenate(self._frames, axis=0).flatten()
            self._frames.clear()
        return audio

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        """sounddevice callback — accumulates audio frames."""
        if status:
            logger.warning("Audio callback status: %s", status)
        with self._lock:
            self._frames.append(indata.copy())

    @staticmethod
    def list_devices() -> list[dict]:
        """List available audio input devices.

        Returns:
            List of dicts with 'index', 'name', and 'channels' for each input device.
        """
        devices = sd.query_devices()
        inputs = []
        for i, dev in enumerate(devices):
            if dev["max_input_channels"] > 0:
                inputs.append({
                    "index": i,
                    "name": dev["name"],
                    "channels": dev["max_input_channels"],
                    "sample_rate": dev["default_samplerate"],
                })
        return inputs

    @staticmethod
    def get_default_device_name() -> str:
        """Get the name of the default input device."""
        try:
            device_info = sd.query_devices(kind="input")
            return device_info["name"]
        except Exception:
            return "Unknown"
