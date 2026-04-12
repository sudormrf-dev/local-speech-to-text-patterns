"""Silero VAD: Voice Activity Detection for pre-processing audio.

Silero VAD is a lightweight (< 1 MB) ONNX model that detects speech
segments in audio streams with ~10 ms latency. Use it to:
  1. Filter out silence before Whisper (reduces hallucinations)
  2. Split long audio into manageable chunks
  3. Detect end-of-utterance in real-time systems

Key parameters:
  threshold: 0.5 default (lower = more sensitive / more false positives)
  min_speech_ms: 250 ms (filter very short noise bursts)
  min_silence_ms: 100 ms (gap needed to split segments)
  max_speech_s: 30 s (force split if segment too long for Whisper)

Usage::

    config = VadConfig(threshold=0.5, min_speech_ms=250)
    engine = VadEngine(config)
    result = engine.detect(audio_samples, sample_rate=16000)
    for seg in result.segments:
        print(f"Speech from {seg.start_s:.2f}s to {seg.end_s:.2f}s")
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class VadConfig:
    """Configuration for Silero VAD.

    Attributes:
        threshold: Speech probability threshold (0.0-1.0).
        min_speech_ms: Minimum speech duration to keep (ms).
        min_silence_ms: Minimum silence duration to split segments (ms).
        max_speech_s: Force-split segments longer than this (seconds).
        sample_rate: Audio sample rate in Hz (16000 or 8000).
        window_size_samples: ONNX model window size (512 for 16kHz, 256 for 8kHz).
        speech_pad_ms: Padding added to start/end of each segment (ms).
    """

    threshold: float = 0.5
    min_speech_ms: int = 250
    min_silence_ms: int = 100
    max_speech_s: float = 30.0
    sample_rate: int = 16000
    window_size_samples: int = 512
    speech_pad_ms: int = 30

    def validate(self) -> list[str]:
        """Return validation warnings for this configuration.

        Returns:
            List of warning strings (empty if all valid).
        """
        warnings: list[str] = []
        if not 0.0 < self.threshold < 1.0:
            warnings.append(f"threshold {self.threshold} should be between 0.0 and 1.0")
        if self.sample_rate not in {8000, 16000}:
            warnings.append(f"sample_rate {self.sample_rate} should be 8000 or 16000")
        if self.min_speech_ms < 0:
            warnings.append(f"min_speech_ms {self.min_speech_ms} must be >= 0")
        return warnings


@dataclass
class SpeechSegment:
    """A detected speech segment.

    Attributes:
        start_s: Start time in seconds.
        end_s: End time in seconds.
        confidence: Average VAD probability in this segment.
    """

    start_s: float
    end_s: float
    confidence: float = 1.0

    @property
    def duration_s(self) -> float:
        """Duration of this segment in seconds."""
        return self.end_s - self.start_s

    @property
    def start_ms(self) -> int:
        """Start time in milliseconds."""
        return int(self.start_s * 1000)

    @property
    def end_ms(self) -> int:
        """End time in milliseconds."""
        return int(self.end_s * 1000)

    def overlaps(self, other: SpeechSegment) -> bool:
        """Return True if this segment overlaps with another."""
        return self.start_s < other.end_s and self.end_s > other.start_s


@dataclass
class VadResult:
    """Result of VAD processing on an audio buffer.

    Attributes:
        segments: Detected speech segments.
        total_duration_s: Total audio duration analysed.
        speech_ratio: Fraction of audio that contains speech.
    """

    segments: list[SpeechSegment] = field(default_factory=list)
    total_duration_s: float = 0.0
    speech_ratio: float = 0.0

    @property
    def speech_duration_s(self) -> float:
        """Total speech duration in seconds."""
        return sum(s.duration_s for s in self.segments)

    @property
    def segment_count(self) -> int:
        """Number of detected speech segments."""
        return len(self.segments)

    def has_speech(self) -> bool:
        """Return True if any speech segments were detected."""
        return len(self.segments) > 0


class VadEngine:
    """Silero VAD wrapper for speech segment detection.

    In environments without silero-vad / torch installed, operates in
    stub mode and returns empty results.

    Args:
        config: VAD configuration.
    """

    def __init__(self, config: VadConfig | None = None) -> None:
        self._cfg = config or VadConfig()
        self._model_loaded = False

    @property
    def config(self) -> VadConfig:
        """The VAD configuration."""
        return self._cfg

    @property
    def is_loaded(self) -> bool:
        """True if the underlying ONNX model is loaded."""
        return self._model_loaded

    def validate_config(self) -> list[str]:
        """Return config validation warnings."""
        return self._cfg.validate()

    def detect(
        self,
        audio_samples: list[float],
        sample_rate: int | None = None,
    ) -> VadResult:
        """Run VAD on a list of audio samples.

        In stub mode (model not loaded), returns a result with a single
        segment spanning the full audio duration.

        Args:
            audio_samples: PCM audio samples in [-1.0, 1.0] range.
            sample_rate: Override config sample rate.

        Returns:
            :class:`VadResult` with detected speech segments.
        """
        sr = sample_rate or self._cfg.sample_rate
        duration_s = len(audio_samples) / sr if audio_samples else 0.0

        if not audio_samples:
            return VadResult(segments=[], total_duration_s=0.0, speech_ratio=0.0)

        # Stub: treat everything as speech (real code uses silero ONNX model)
        segment = SpeechSegment(start_s=0.0, end_s=duration_s, confidence=1.0)
        return VadResult(
            segments=[segment],
            total_duration_s=duration_s,
            speech_ratio=1.0,
        )

    def detect_realtime(
        self,
        chunk: list[float],
        sample_rate: int | None = None,
    ) -> float:
        """Return speech probability for a single audio chunk.

        Use this for real-time systems where you process audio
        window-by-window. Call with window_size_samples samples.

        Args:
            chunk: Audio chunk (length should be window_size_samples).
            sample_rate: Override config sample rate.

        Returns:
            Speech probability in [0.0, 1.0].
        """
        _ = sample_rate
        if not chunk:
            return 0.0
        # Stub: return 1.0 if any sample exceeds a basic energy threshold
        rms = (sum(s * s for s in chunk) / len(chunk)) ** 0.5
        return float(min(1.0, rms * 10.0))


def merge_speech_segments(
    segments: list[SpeechSegment],
    gap_s: float = 0.5,
) -> list[SpeechSegment]:
    """Merge speech segments that are separated by short gaps.

    Args:
        segments: List of speech segments, assumed sorted by start_s.
        gap_s: Maximum gap in seconds to bridge when merging.

    Returns:
        New list of merged :class:`SpeechSegment` objects.
    """
    if not segments:
        return []

    merged: list[SpeechSegment] = [
        SpeechSegment(
            start_s=segments[0].start_s,
            end_s=segments[0].end_s,
            confidence=segments[0].confidence,
        )
    ]

    for seg in segments[1:]:
        last = merged[-1]
        if seg.start_s - last.end_s <= gap_s:
            # Merge: extend end time, average confidence
            new_conf = (last.confidence * last.duration_s + seg.confidence * seg.duration_s) / (
                last.duration_s + seg.duration_s
            )
            merged[-1] = SpeechSegment(
                start_s=last.start_s,
                end_s=seg.end_s,
                confidence=new_conf,
            )
        else:
            merged.append(
                SpeechSegment(
                    start_s=seg.start_s,
                    end_s=seg.end_s,
                    confidence=seg.confidence,
                )
            )

    return merged
