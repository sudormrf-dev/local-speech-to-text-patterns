"""Tests for kokoro_tts.py."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from patterns.kokoro_tts import (
    KokoroConfig,
    KokoroVoice,
    SpeechSynthesizer,
    TTSResult,
    estimate_audio_duration,
)


class TestKokoroVoice:
    def test_female_voice_gender(self):
        assert KokoroVoice.AF_HEART.gender == "female"

    def test_male_voice_gender(self):
        assert KokoroVoice.AM_MICHAEL.gender == "male"

    def test_british_voice_accent(self):
        assert KokoroVoice.BF_EMMA.accent == "british"

    def test_american_voice_accent(self):
        assert KokoroVoice.AF_HEART.accent == "american"

    def test_british_male_accent(self):
        assert KokoroVoice.BM_GEORGE.accent == "british"
        assert KokoroVoice.BM_GEORGE.gender == "male"


class TestKokoroConfig:
    def test_defaults(self):
        cfg = KokoroConfig()
        assert cfg.speed == 1.0
        assert cfg.voice == KokoroVoice.AF_HEART
        assert cfg.sample_rate == 24000

    def test_validate_valid(self):
        cfg = KokoroConfig()
        assert cfg.validate() == []

    def test_validate_bad_speed_low(self):
        cfg = KokoroConfig(speed=0.1)
        warnings = cfg.validate()
        assert any("speed" in w for w in warnings)

    def test_validate_bad_speed_high(self):
        cfg = KokoroConfig(speed=5.0)
        warnings = cfg.validate()
        assert any("speed" in w for w in warnings)

    def test_validate_unusual_sample_rate(self):
        cfg = KokoroConfig(sample_rate=11025)
        warnings = cfg.validate()
        assert any("sample_rate" in w for w in warnings)


class TestTTSResult:
    def test_is_empty_true(self):
        r = TTSResult()
        assert r.is_empty() is True

    def test_is_empty_false(self):
        r = TTSResult(samples=[0.0, 0.1, -0.1])
        assert r.is_empty() is False

    def test_sample_count(self):
        r = TTSResult(samples=[0.0] * 100)
        assert r.sample_count == 100

    def test_save_creates_wav(self, tmp_path: Path):
        r = TTSResult(
            samples=[0.0, 0.5, -0.5, 0.1],
            sample_rate=24000,
            text="test",
        )
        out = tmp_path / "out.wav"
        r.save(out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_save_clamps_samples(self, tmp_path: Path):
        r = TTSResult(
            samples=[2.0, -2.0],  # Out of range — should clamp
            sample_rate=24000,
        )
        out = tmp_path / "clamped.wav"
        r.save(out)  # Should not raise
        assert out.exists()


class TestSpeechSynthesizer:
    def test_not_loaded_initially(self):
        synth = SpeechSynthesizer()
        assert synth.is_loaded is False

    def test_synthesize_returns_result(self):
        synth = SpeechSynthesizer()
        result = synth.synthesize("Hello world")
        assert isinstance(result, TTSResult)
        assert result.text == "Hello world"

    def test_synthesize_estimates_duration(self):
        synth = SpeechSynthesizer()
        result = synth.synthesize("Hello, this is a test sentence with some words.")
        assert result.duration_s > 0.0

    def test_synthesize_empty_text(self):
        synth = SpeechSynthesizer()
        result = synth.synthesize("")
        assert result.duration_s == 0.0

    def test_synthesize_chunks_single_chunk(self):
        synth = SpeechSynthesizer()
        results = synth.synthesize_chunks("Short text.", chunk_size=200)
        assert len(results) >= 1

    def test_synthesize_chunks_multiple(self):
        synth = SpeechSynthesizer()
        long_text = "Hello world. " * 30  # ~390 chars, should split
        results = synth.synthesize_chunks(long_text, chunk_size=100)
        assert len(results) > 1

    def test_estimate_duration_longer_for_more_text(self):
        synth = SpeechSynthesizer()
        short = synth.estimate_duration_s("Hi.")
        long_est = synth.estimate_duration_s("This is a much longer sentence with many words.")
        assert long_est > short

    def test_validate_config_valid(self):
        synth = SpeechSynthesizer()
        assert synth.validate_config() == []

    def test_config_property(self):
        cfg = KokoroConfig(speed=1.5)
        synth = SpeechSynthesizer(cfg)
        assert synth.config.speed == 1.5


class TestEstimateAudioDuration:
    def test_empty_text(self):
        assert estimate_audio_duration("") == 0.0

    def test_whitespace_text(self):
        assert estimate_audio_duration("   ") == 0.0

    def test_faster_speed_shorter_duration(self):
        slow = estimate_audio_duration("Hello world", speed=0.5)
        fast = estimate_audio_duration("Hello world", speed=2.0)
        assert fast < slow

    def test_duration_positive_for_text(self):
        assert estimate_audio_duration("Hello") > 0.0
