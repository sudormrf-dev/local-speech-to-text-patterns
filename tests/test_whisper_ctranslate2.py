"""Tests for whisper_ctranslate2.py."""

from __future__ import annotations

import pytest

from patterns.whisper_ctranslate2 import (
    ComputeType,
    TranscribeConfig,
    TranscribeResult,
    WhisperModel,
    WhisperSegment,
    format_srt,
    format_vtt,
)


class TestComputeType:
    def test_for_device_cpu(self):
        assert ComputeType.for_device("cpu") == ComputeType.INT8

    def test_for_device_cuda(self):
        assert ComputeType.for_device("cuda") == ComputeType.FLOAT16


class TestWhisperSegment:
    def test_duration(self):
        seg = WhisperSegment(start=1.0, end=3.5, text="hello")
        assert abs(seg.duration - 2.5) < 1e-9

    def test_is_reliable_true(self):
        seg = WhisperSegment(start=0, end=1, text="hi", avg_logprob=-0.3, no_speech_prob=0.1)
        assert seg.is_reliable is True

    def test_is_reliable_false_low_logprob(self):
        seg = WhisperSegment(start=0, end=1, text="hi", avg_logprob=-1.5, no_speech_prob=0.1)
        assert seg.is_reliable is False

    def test_is_reliable_false_high_no_speech(self):
        seg = WhisperSegment(start=0, end=1, text="hi", avg_logprob=-0.3, no_speech_prob=0.8)
        assert seg.is_reliable is False


class TestTranscribeResult:
    def test_word_count(self):
        r = TranscribeResult(text="hello world foo")
        assert r.word_count == 3

    def test_word_count_empty(self):
        r = TranscribeResult(text="")
        assert r.word_count == 0

    def test_filter_reliable_removes_low_confidence(self):
        segs = [
            WhisperSegment(start=0, end=1, text="good", avg_logprob=-0.5),
            WhisperSegment(start=1, end=2, text="bad", avg_logprob=-2.0),
        ]
        result = TranscribeResult(text="good bad", segments=segs)
        filtered = result.filter_reliable(logprob_threshold=-1.0)
        assert len(filtered.segments) == 1
        assert "good" in filtered.text

    def test_filter_reliable_keeps_all_if_threshold_low(self):
        segs = [
            WhisperSegment(start=0, end=1, text="a", avg_logprob=-0.5),
            WhisperSegment(start=1, end=2, text="b", avg_logprob=-0.8),
        ]
        result = TranscribeResult(text="a b", segments=segs)
        filtered = result.filter_reliable(logprob_threshold=-1.0)
        assert len(filtered.segments) == 2


class TestTranscribeConfig:
    def test_defaults(self):
        cfg = TranscribeConfig()
        assert cfg.beam_size == 5
        assert cfg.vad_filter is True
        assert cfg.temperature == 0.0

    def test_to_kwargs_includes_beam_size(self):
        cfg = TranscribeConfig(beam_size=3)
        kwargs = cfg.to_kwargs()
        assert kwargs["beam_size"] == 3

    def test_to_kwargs_language_absent_when_none(self):
        cfg = TranscribeConfig(language=None)
        kwargs = cfg.to_kwargs()
        assert "language" not in kwargs

    def test_to_kwargs_language_present(self):
        cfg = TranscribeConfig(language="fr")
        kwargs = cfg.to_kwargs()
        assert kwargs["language"] == "fr"

    def test_to_kwargs_vad_parameters_when_vad_filter(self):
        cfg = TranscribeConfig(vad_filter=True, vad_min_silence_ms=300)
        kwargs = cfg.to_kwargs()
        assert "vad_parameters" in kwargs
        assert kwargs["vad_parameters"]["min_silence_duration_ms"] == 300

    def test_to_kwargs_no_vad_parameters_when_disabled(self):
        cfg = TranscribeConfig(vad_filter=False)
        kwargs = cfg.to_kwargs()
        assert "vad_parameters" not in kwargs

    def test_to_kwargs_max_new_tokens_absent_when_none(self):
        cfg = TranscribeConfig(max_new_tokens=None)
        kwargs = cfg.to_kwargs()
        assert "max_new_tokens" not in kwargs

    def test_to_kwargs_max_new_tokens_present(self):
        cfg = TranscribeConfig(max_new_tokens=448)
        kwargs = cfg.to_kwargs()
        assert kwargs["max_new_tokens"] == 448


class TestWhisperModel:
    def test_validate_model_size_valid(self):
        m = WhisperModel(model_size="base")
        assert m.validate_model_size() is True

    def test_validate_model_size_invalid(self):
        m = WhisperModel(model_size="giga")
        assert m.validate_model_size() is False

    def test_not_loaded_initially(self):
        m = WhisperModel()
        assert m.is_loaded is False

    def test_estimated_vram_int8_less_than_float32(self):
        m_f32 = WhisperModel(model_size="base", compute_type=ComputeType.FLOAT32)
        m_int8 = WhisperModel(model_size="base", compute_type=ComputeType.INT8)
        assert m_int8.estimated_vram_mb() < m_f32.estimated_vram_mb()

    def test_estimated_vram_large_more_than_tiny(self):
        m_tiny = WhisperModel(model_size="tiny", compute_type=ComputeType.INT8)
        m_large = WhisperModel(model_size="large-v3", compute_type=ComputeType.INT8)
        assert m_large.estimated_vram_mb() > m_tiny.estimated_vram_mb()

    def test_transcribe_missing_file_raises(self, tmp_path):

        m = WhisperModel()
        with pytest.raises(FileNotFoundError):
            m.transcribe(tmp_path / "nonexistent.wav")

    def test_transcribe_existing_file_returns_result(self, tmp_path):

        audio = tmp_path / "test.wav"
        audio.write_bytes(b"fake audio")
        m = WhisperModel()
        result = m.transcribe(audio)
        assert isinstance(result, TranscribeResult)


class TestFormatSrt:
    def test_empty_returns_empty(self):
        assert format_srt([]) == ""

    def test_single_segment(self):
        segs = [WhisperSegment(start=0.0, end=1.5, text="Hello world")]
        srt = format_srt(segs)
        assert "1\n" in srt
        assert "00:00:00,000 --> 00:00:01,500" in srt
        assert "Hello world" in srt

    def test_multiple_segments_numbered(self):
        segs = [
            WhisperSegment(start=0.0, end=1.0, text="one"),
            WhisperSegment(start=1.0, end=2.0, text="two"),
        ]
        srt = format_srt(segs)
        assert "1\n" in srt
        assert "2\n" in srt


class TestFormatVtt:
    def test_starts_with_webvtt(self):
        vtt = format_vtt([])
        assert vtt.startswith("WEBVTT")

    def test_single_segment(self):
        segs = [WhisperSegment(start=0.0, end=2.0, text="Hi there")]
        vtt = format_vtt(segs)
        assert "00:00:00.000 --> 00:00:02.000" in vtt
        assert "Hi there" in vtt
