"""Local speech-to-text patterns: Whisper CTranslate2, Silero VAD, Kokoro TTS."""

from .kokoro_tts import (
    KokoroConfig,
    KokoroVoice,
    SpeechSynthesizer,
    TTSResult,
    estimate_audio_duration,
)
from .silero_vad import (
    SpeechSegment,
    VadConfig,
    VadEngine,
    VadResult,
    merge_speech_segments,
)
from .whisper_ctranslate2 import (
    ComputeType,
    TranscribeConfig,
    TranscribeResult,
    WhisperModel,
    WhisperSegment,
    format_srt,
    format_vtt,
)

__all__ = [
    "ComputeType",
    "KokoroConfig",
    "KokoroVoice",
    "SpeechSegment",
    "SpeechSynthesizer",
    "TTSResult",
    "TranscribeConfig",
    "TranscribeResult",
    "VadConfig",
    "VadEngine",
    "VadResult",
    "WhisperModel",
    "WhisperSegment",
    "estimate_audio_duration",
    "format_srt",
    "format_vtt",
    "merge_speech_segments",
]
