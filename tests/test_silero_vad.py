"""Tests for silero_vad.py."""

from __future__ import annotations

from patterns.silero_vad import (
    SpeechSegment,
    VadConfig,
    VadEngine,
    VadResult,
    merge_speech_segments,
)


class TestVadConfig:
    def test_defaults(self):
        cfg = VadConfig()
        assert cfg.threshold == 0.5
        assert cfg.sample_rate == 16000

    def test_validate_valid(self):
        cfg = VadConfig()
        assert cfg.validate() == []

    def test_validate_bad_threshold(self):
        cfg = VadConfig(threshold=0.0)
        warnings = cfg.validate()
        assert any("threshold" in w for w in warnings)

    def test_validate_bad_sample_rate(self):
        cfg = VadConfig(sample_rate=44100)
        warnings = cfg.validate()
        assert any("sample_rate" in w for w in warnings)

    def test_validate_negative_min_speech(self):
        cfg = VadConfig(min_speech_ms=-1)
        warnings = cfg.validate()
        assert any("min_speech_ms" in w for w in warnings)


class TestSpeechSegment:
    def test_duration(self):
        seg = SpeechSegment(start_s=1.0, end_s=3.5)
        assert abs(seg.duration_s - 2.5) < 1e-9

    def test_start_ms(self):
        seg = SpeechSegment(start_s=1.5, end_s=2.0)
        assert seg.start_ms == 1500

    def test_end_ms(self):
        seg = SpeechSegment(start_s=0.0, end_s=2.3)
        assert seg.end_ms == 2300

    def test_overlaps_true(self):
        a = SpeechSegment(start_s=0.0, end_s=2.0)
        b = SpeechSegment(start_s=1.0, end_s=3.0)
        assert a.overlaps(b) is True

    def test_overlaps_false(self):
        a = SpeechSegment(start_s=0.0, end_s=1.0)
        b = SpeechSegment(start_s=2.0, end_s=3.0)
        assert a.overlaps(b) is False


class TestVadResult:
    def test_speech_duration(self):
        r = VadResult(
            segments=[
                SpeechSegment(start_s=0.0, end_s=1.0),
                SpeechSegment(start_s=2.0, end_s=3.5),
            ]
        )
        assert abs(r.speech_duration_s - 2.5) < 1e-9

    def test_has_speech_true(self):
        r = VadResult(segments=[SpeechSegment(start_s=0, end_s=1)])
        assert r.has_speech() is True

    def test_has_speech_false(self):
        r = VadResult(segments=[])
        assert r.has_speech() is False

    def test_segment_count(self):
        r = VadResult(segments=[SpeechSegment(0, 1), SpeechSegment(2, 3)])
        assert r.segment_count == 2


class TestVadEngine:
    def test_not_loaded_initially(self):
        engine = VadEngine()
        assert engine.is_loaded is False

    def test_detect_empty_audio(self):
        engine = VadEngine()
        result = engine.detect([])
        assert result.segment_count == 0
        assert result.total_duration_s == 0.0

    def test_detect_returns_result(self):
        engine = VadEngine()
        samples = [0.1] * 16000  # 1 second at 16kHz
        result = engine.detect(samples, sample_rate=16000)
        assert isinstance(result, VadResult)
        assert result.total_duration_s == 1.0

    def test_detect_realtime_empty(self):
        engine = VadEngine()
        prob = engine.detect_realtime([])
        assert prob == 0.0

    def test_detect_realtime_returns_float(self):
        engine = VadEngine()
        chunk = [0.5] * 512
        prob = engine.detect_realtime(chunk)
        assert 0.0 <= prob <= 1.0

    def test_validate_config_valid(self):
        engine = VadEngine()
        assert engine.validate_config() == []

    def test_config_property(self):
        cfg = VadConfig(threshold=0.7)
        engine = VadEngine(cfg)
        assert engine.config.threshold == 0.7


class TestMergeSpeechSegments:
    def test_empty(self):
        assert merge_speech_segments([]) == []

    def test_single_segment_unchanged(self):
        seg = SpeechSegment(start_s=0.0, end_s=1.0)
        merged = merge_speech_segments([seg])
        assert len(merged) == 1
        assert merged[0].start_s == 0.0

    def test_merge_close_segments(self):
        segs = [
            SpeechSegment(start_s=0.0, end_s=1.0),
            SpeechSegment(start_s=1.2, end_s=2.0),
        ]
        merged = merge_speech_segments(segs, gap_s=0.5)
        assert len(merged) == 1
        assert merged[0].start_s == 0.0
        assert merged[0].end_s == 2.0

    def test_no_merge_wide_gap(self):
        segs = [
            SpeechSegment(start_s=0.0, end_s=1.0),
            SpeechSegment(start_s=2.0, end_s=3.0),
        ]
        merged = merge_speech_segments(segs, gap_s=0.5)
        assert len(merged) == 2

    def test_merged_confidence_is_average(self):
        segs = [
            SpeechSegment(start_s=0.0, end_s=1.0, confidence=0.8),
            SpeechSegment(start_s=1.2, end_s=2.2, confidence=0.6),
        ]
        merged = merge_speech_segments(segs, gap_s=0.5)
        assert len(merged) == 1
        assert 0.6 <= merged[0].confidence <= 0.8
