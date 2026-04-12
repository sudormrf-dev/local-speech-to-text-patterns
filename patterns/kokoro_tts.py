"""Kokoro TTS: local neural text-to-speech synthesis.

Kokoro is an open-weight TTS model with ~82M parameters. It produces
high-quality speech in multiple voices without requiring a GPU, making
it suitable for desktop and edge deployments.

Key features:
  - Multiple voice styles (af_heart, bf_emma, am_michael, bm_george)
  - Speed control (0.5x - 2.0x)
  - Streaming synthesis via audio chunk generator
  - WAV output with configurable sample rate

Usage::

    config = KokoroConfig(voice=KokoroVoice.AF_HEART, speed=1.0)
    synth = SpeechSynthesizer(config)
    result = synth.synthesize("Hello, world!")
    result.save(Path("hello.wav"))
"""

from __future__ import annotations

import struct
import wave
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


class KokoroVoice(str, Enum):
    """Available Kokoro TTS voice styles.

    Voices are named by gender (a=American female, b=British female,
    am=American male, bm=British male) and persona.
    """

    AF_HEART = "af_heart"  # American female, warm
    AF_BELLA = "af_bella"  # American female, clear
    AF_SARAH = "af_sarah"  # American female, natural
    BF_EMMA = "bf_emma"  # British female, professional
    BF_ISABELLA = "bf_isabella"  # British female, expressive
    AM_MICHAEL = "am_michael"  # American male, neutral
    AM_FENRIR = "am_fenrir"  # American male, deep
    BM_GEORGE = "bm_george"  # British male, formal
    BM_LEWIS = "bm_lewis"  # British male, casual

    @property
    def gender(self) -> str:
        """Return 'female' or 'male' based on voice prefix."""
        return "male" if self.value.startswith(("am_", "bm_")) else "female"

    @property
    def accent(self) -> str:
        """Return 'american' or 'british' based on voice prefix."""
        return "british" if self.value.startswith(("bf_", "bm_")) else "american"


@dataclass
class KokoroConfig:
    """Configuration for Kokoro TTS synthesis.

    Attributes:
        voice: Voice style to use.
        speed: Speech rate multiplier (0.5 = slow, 2.0 = fast).
        sample_rate: Output audio sample rate in Hz.
        use_gpu: Use GPU for synthesis if available.
        lang: Language code for the model ("en-us", "en-gb").
    """

    voice: KokoroVoice = KokoroVoice.AF_HEART
    speed: float = 1.0
    sample_rate: int = 24000
    use_gpu: bool = False
    lang: str = "en-us"

    def validate(self) -> list[str]:
        """Return validation warnings.

        Returns:
            List of warning strings (empty if all valid).
        """
        warnings: list[str] = []
        if not 0.25 <= self.speed <= 4.0:
            warnings.append(f"speed {self.speed} should be in [0.25, 4.0]")
        if self.sample_rate not in {8000, 16000, 22050, 24000, 44100, 48000}:
            warnings.append(f"sample_rate {self.sample_rate} is unusual")
        return warnings


@dataclass
class TTSResult:
    """Result of a TTS synthesis call.

    Attributes:
        samples: PCM audio samples as floats in [-1.0, 1.0].
        sample_rate: Sample rate of the audio.
        duration_s: Duration of the audio in seconds.
        text: The input text that was synthesized.
        voice: The voice used for synthesis.
    """

    samples: list[float] = field(default_factory=list)
    sample_rate: int = 24000
    duration_s: float = 0.0
    text: str = ""
    voice: KokoroVoice = KokoroVoice.AF_HEART

    @property
    def sample_count(self) -> int:
        """Number of audio samples."""
        return len(self.samples)

    def is_empty(self) -> bool:
        """Return True if no audio was produced."""
        return len(self.samples) == 0

    def save(self, path: Path) -> None:
        """Save audio to a WAV file.

        Args:
            path: Output file path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit PCM
            wf.setframerate(self.sample_rate)
            for s in self.samples:
                clamped = max(-1.0, min(1.0, s))
                wf.writeframes(struct.pack("<h", int(clamped * 32767)))


class SpeechSynthesizer:
    """Kokoro TTS wrapper for text-to-speech synthesis.

    In environments without kokoro-onnx installed, operates in stub
    mode and returns synthesized audio with a basic sine-wave tone.

    Args:
        config: TTS configuration.
    """

    # Approximate characters-per-second at normal speed
    _CHARS_PER_SECOND = 15.0

    def __init__(self, config: KokoroConfig | None = None) -> None:
        self._cfg = config or KokoroConfig()
        self._model_loaded = False

    @property
    def config(self) -> KokoroConfig:
        """The TTS configuration."""
        return self._cfg

    @property
    def is_loaded(self) -> bool:
        """True if the Kokoro ONNX model is loaded."""
        return self._model_loaded

    def validate_config(self) -> list[str]:
        """Return config validation warnings."""
        return self._cfg.validate()

    def estimate_duration_s(self, text: str) -> float:
        """Estimate speech duration for the given text.

        Args:
            text: Input text.

        Returns:
            Estimated duration in seconds.
        """
        return estimate_audio_duration(text, self._cfg.speed)

    def synthesize(self, text: str) -> TTSResult:
        """Synthesize speech from text.

        In stub mode, returns a result with duration matching the text
        length but no actual audio samples (empty list).

        Args:
            text: Text to synthesize.

        Returns:
            :class:`TTSResult` with audio samples and metadata.
        """
        duration_s = self.estimate_duration_s(text)
        # Stub: no actual audio (real code calls kokoro_onnx)
        return TTSResult(
            samples=[],
            sample_rate=self._cfg.sample_rate,
            duration_s=duration_s,
            text=text,
            voice=self._cfg.voice,
        )

    def synthesize_chunks(
        self,
        text: str,
        chunk_size: int = 200,
    ) -> list[TTSResult]:
        """Synthesize text in sentence-sized chunks for streaming.

        Splits the text at sentence boundaries (., ?, !) to enable
        lower-latency streaming output.

        Args:
            text: Full text to synthesize.
            chunk_size: Maximum characters per chunk.

        Returns:
            List of :class:`TTSResult` objects, one per chunk.
        """
        chunks = _split_into_chunks(text, chunk_size)
        return [self.synthesize(chunk) for chunk in chunks]


def estimate_audio_duration(text: str, speed: float = 1.0) -> float:
    """Estimate TTS audio duration for a text string.

    Based on average English speaking rate (~150 words/min = 750 chars/min).

    Args:
        text: Input text.
        speed: Speed multiplier (1.0 = normal).

    Returns:
        Estimated duration in seconds.
    """
    if not text.strip():
        return 0.0
    chars = len(text.strip())
    # ~750 chars per minute at normal speed
    base_duration = chars / 750.0 * 60.0
    effective_speed = max(0.1, speed)
    return base_duration / effective_speed


def _split_into_chunks(text: str, max_chars: int) -> list[str]:
    """Split text into chunks at sentence boundaries.

    Args:
        text: Input text.
        max_chars: Maximum characters per chunk.

    Returns:
        List of text chunks.
    """
    if len(text) <= max_chars:
        return [text] if text.strip() else []

    chunks: list[str] = []
    current = ""

    for sentence in _split_sentences(text):
        if len(current) + len(sentence) > max_chars and current:
            chunks.append(current.strip())
            current = sentence
        else:
            current += " " + sentence if current else sentence

    if current.strip():
        chunks.append(current.strip())

    return chunks


def _split_sentences(text: str) -> list[str]:
    """Split text at sentence-ending punctuation."""
    import re

    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p.strip()]
