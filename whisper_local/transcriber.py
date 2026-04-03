"""Whisper transcription wrapper using faster-whisper."""

import logging
import time
from typing import NamedTuple

import numpy as np

logger = logging.getLogger(__name__)


class TranscriptionResult(NamedTuple):
    """Result of a transcription operation."""

    text: str
    duration_seconds: float
    detected_language: str


class WhisperTranscriber:
    """Wraps faster-whisper for local speech-to-text.

    The model is loaded once at initialization and reused for all transcriptions.
    """

    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str = "es",
    ) -> None:
        from faster_whisper import WhisperModel

        logger.info(
            "Loading Whisper model '%s' (device=%s, compute_type=%s)...",
            model_size, device, compute_type,
        )
        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self._language = language if language != "auto" else None
        self._model_size = model_size
        logger.info("Model loaded successfully")

    @property
    def model_size(self) -> str:
        return self._model_size

    def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        """Transcribe a numpy audio array to text.

        Args:
            audio: float32 numpy array of audio at 16kHz mono.

        Returns:
            TranscriptionResult with text, processing duration, and detected language.
        """
        start_time = time.perf_counter()

        segments, info = self._model.transcribe(
            audio,
            language=self._language,
            beam_size=5,
            vad_filter=True,
        )

        text_parts = [segment.text for segment in segments]
        text = " ".join(text_parts).strip()

        duration = time.perf_counter() - start_time
        detected_lang = info.language if info.language else (self._language or "unknown")

        logger.debug(
            "Transcription completed: %d chars in %.2fs (lang=%s)",
            len(text), duration, detected_lang,
        )

        return TranscriptionResult(
            text=text,
            duration_seconds=duration,
            detected_language=detected_lang,
        )
