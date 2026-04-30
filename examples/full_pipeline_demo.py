"""Full pipeline demo: VAD -> STT -> LLM -> TTS (100% local, simulated).

Demonstrates the complete flow of a local voice assistant:
  1. Silence detection (Silero VAD)
  2. Wake-word trigger
  3. Speech capture and VAD segmentation
  4. Transcription (Whisper CTranslate2)
  5. LLM response generation (simulated)
  6. Speech synthesis (Kokoro TTS)

All external I/O is simulated with synthetic byte buffers so the demo
runs with no GPU, no model downloads, and zero dependencies beyond the
stdlib and the patterns in this repo.

Usage::

    python examples/full_pipeline_demo.py
"""

from __future__ import annotations

import io
import random
import time
from dataclasses import dataclass, field

from patterns.kokoro_tts import KokoroConfig, KokoroVoice, SpeechSynthesizer, TTSResult
from patterns.silero_vad import SpeechSegment, VadConfig, VadEngine, VadResult
from patterns.whisper_ctranslate2 import (
    ComputeType,
    TranscribeConfig,
    TranscribeResult,
    WhisperModel,
    WhisperSegment,
)

# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

PIPELINE_SEED = 42


@dataclass
class StageTimer:
    """Records wall-clock elapsed time for a pipeline stage.

    Attributes:
        name: Human-readable stage label.
        elapsed_ms: Measured latency in milliseconds (-1 if not yet run).
    """

    name: str
    elapsed_ms: float = -1.0

    def __enter__(self) -> StageTimer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: object) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1_000

    def __str__(self) -> str:
        return f"{self.name:30s} {self.elapsed_ms:7.1f} ms"


@dataclass
class PipelineRun:
    """Aggregated timing and output data for a single pipeline execution.

    Attributes:
        stage_timers: Ordered list of stage timers.
        transcript: Final transcription text from STT.
        response: LLM-generated response text.
        tts_duration_s: Duration of synthesized TTS audio.
        quality_score: Simulated transcription quality [0.0-1.0].
    """

    stage_timers: list[StageTimer] = field(default_factory=list)
    transcript: str = ""
    response: str = ""
    tts_duration_s: float = 0.0
    quality_score: float = 0.0

    @property
    def total_ms(self) -> float:
        """Sum of all measured stage latencies in milliseconds."""
        return sum(t.elapsed_ms for t in self.stage_timers if t.elapsed_ms >= 0)

    @property
    def ttfb_ms(self) -> float:
        """Time-to-first-byte: latency through VAD + STT stages."""
        vad_and_stt = [
            t for t in self.stage_timers if t.name in ("VAD Detection", "STT Transcription")
        ]
        return sum(t.elapsed_ms for t in vad_and_stt if t.elapsed_ms >= 0)


# ---------------------------------------------------------------------------
# Simulated audio helpers
# ---------------------------------------------------------------------------


def _make_audio_bytes(duration_s: float, sample_rate: int = 16_000) -> bytes:
    """Generate synthetic PCM audio bytes (silence + noise burst).

    Args:
        duration_s: Audio duration in seconds.
        sample_rate: Samples per second.

    Returns:
        Raw 16-bit little-endian PCM bytes.
    """
    rng = random.Random(PIPELINE_SEED)
    n_samples = int(duration_s * sample_rate)
    buf = io.BytesIO()
    for _ in range(n_samples):
        # Simulate silence (small amplitude) with occasional noise bursts
        amplitude = rng.gauss(0, 500)
        sample = max(-32768, min(32767, int(amplitude)))
        buf.write(sample.to_bytes(2, "little", signed=True))
    return buf.getvalue()


def _pcm_to_float_samples(pcm: bytes) -> list[float]:
    """Convert 16-bit PCM bytes to float samples in [-1.0, 1.0].

    Args:
        pcm: Raw 16-bit little-endian PCM bytes.

    Returns:
        List of float samples.
    """
    samples: list[float] = []
    for i in range(0, len(pcm) - 1, 2):
        raw = int.from_bytes(pcm[i : i + 2], "little", signed=True)
        samples.append(raw / 32768.0)
    return samples


# ---------------------------------------------------------------------------
# Simulated LLM inference
# ---------------------------------------------------------------------------

_LLM_RESPONSES: dict[str, str] = {
    "default": "I'm sorry, I didn't quite catch that. Could you please repeat?",
    "weather": "Right now it's 18 degrees Celsius and partly cloudy.",
    "time": "The current time is three twenty-seven in the afternoon.",
    "hello": "Hello! How can I help you today?",
    "help": "I can answer questions, set timers, or read you the news. What would you like?",
}


def _simulate_llm_inference(transcript: str, latency_ms: float = 450.0) -> str:
    """Simulate LLM response generation with a configurable delay.

    Args:
        transcript: User utterance from STT.
        latency_ms: Simulated inference latency in milliseconds.

    Returns:
        Generated response string.
    """
    time.sleep(latency_ms / 1_000)
    lower = transcript.lower()
    for keyword, response in _LLM_RESPONSES.items():
        if keyword in lower:
            return response
    return _LLM_RESPONSES["default"]


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------


def stage_silence_detection(audio_bytes: bytes, vad_engine: VadEngine) -> tuple[bool, float]:
    """Stage 0: Detect whether the input buffer contains any speech.

    Args:
        audio_bytes: Raw PCM audio bytes.
        vad_engine: Configured VAD engine.

    Returns:
        Tuple of (has_speech, speech_ratio).
    """
    samples = _pcm_to_float_samples(audio_bytes)
    result: VadResult = vad_engine.detect(samples)
    return result.has_speech(), result.speech_ratio


def stage_vad_segmentation(audio_bytes: bytes, vad_engine: VadEngine) -> list[SpeechSegment]:
    """Stage 1: Segment audio into speech regions.

    Args:
        audio_bytes: Raw PCM audio bytes.
        vad_engine: Configured VAD engine.

    Returns:
        List of detected speech segments.
    """
    samples = _pcm_to_float_samples(audio_bytes)
    result: VadResult = vad_engine.detect(samples)
    return result.segments


def stage_stt_transcription(
    audio_bytes: bytes,
    whisper: WhisperModel,
    config: TranscribeConfig,
    simulated_text: str,
) -> TranscribeResult:
    """Stage 2: Transcribe speech audio to text.

    Because we operate in stub mode (no real model), inject a simulated
    transcription result that matches the demo scenario.

    Args:
        audio_bytes: Raw PCM audio bytes (used for duration estimation).
        whisper: WhisperModel instance.
        config: Transcription configuration.
        simulated_text: Text to inject as the transcription result.

    Returns:
        :class:`TranscribeResult` with the simulated transcript.
    """
    # Simulate STT processing latency (proportional to audio length)
    n_samples = len(audio_bytes) // 2  # 16-bit = 2 bytes per sample
    duration_s = n_samples / 16_000
    processing_ratio = 0.15 if whisper.compute_type == ComputeType.INT8 else 0.30
    time.sleep(duration_s * processing_ratio)

    segment = WhisperSegment(
        start=0.0,
        end=duration_s,
        text=simulated_text,
        avg_logprob=-0.25,
        no_speech_prob=0.05,
    )
    return TranscribeResult(
        text=simulated_text,
        segments=[segment],
        language="en",
        language_probability=0.99,
        duration=duration_s,
    )


def stage_tts_synthesis(response_text: str, synth: SpeechSynthesizer) -> TTSResult:
    """Stage 3: Synthesize the LLM response to speech.

    Args:
        response_text: Text to synthesize.
        synth: SpeechSynthesizer instance.

    Returns:
        :class:`TTSResult` with duration metadata.
    """
    # Simulate TTS latency: ~150 ms base + text-proportional overhead
    char_latency_s = len(response_text) * 0.003
    time.sleep(0.15 + char_latency_s)
    return synth.synthesize(response_text)


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

_DEMO_SCENARIOS: list[dict[str, str]] = [
    {
        "label": "Wake word + weather query",
        "utterance": "Hey assistant, what's the weather like today?",
    },
    {"label": "Simple greeting", "utterance": "Hello there!"},
    {"label": "Help request", "utterance": "Can you help me with something?"},
    {"label": "Time query", "utterance": "What time is it right now?"},
]


def run_pipeline(
    scenario: dict[str, str],
    vad_engine: VadEngine,
    whisper: WhisperModel,
    stt_config: TranscribeConfig,
    synth: SpeechSynthesizer,
) -> PipelineRun:
    """Execute the full VAD -> STT -> LLM -> TTS pipeline for one scenario.

    Args:
        scenario: Dict with 'label' and 'utterance' keys.
        vad_engine: Initialised VAD engine.
        whisper: Initialised Whisper model.
        stt_config: STT transcription settings.
        synth: Initialised speech synthesiser.

    Returns:
        :class:`PipelineRun` with timing and output data.
    """
    run = PipelineRun()
    audio_bytes = _make_audio_bytes(duration_s=3.5)

    # Stage 0: Silence / activity detection
    t0 = StageTimer("Silence Detection")
    with t0:
        has_speech, _speech_ratio = stage_silence_detection(audio_bytes, vad_engine)
    run.stage_timers.append(t0)

    if not has_speech:
        print("  [SKIP] No speech detected — pipeline halted.")
        return run

    # Stage 1: VAD segmentation
    t1 = StageTimer("VAD Detection")
    with t1:
        stage_vad_segmentation(audio_bytes, vad_engine)
    run.stage_timers.append(t1)

    # Stage 2: STT transcription
    t2 = StageTimer("STT Transcription")
    with t2:
        stt_result = stage_stt_transcription(
            audio_bytes, whisper, stt_config, scenario["utterance"]
        )
    run.stage_timers.append(t2)

    run.transcript = stt_result.text
    # Quality score: combination of logprob and no_speech_prob
    if stt_result.segments:
        seg = stt_result.segments[0]
        run.quality_score = max(0.0, min(1.0, (seg.avg_logprob + 1.0) * (1 - seg.no_speech_prob)))

    # Stage 3: LLM inference
    t3 = StageTimer("LLM Inference")
    with t3:
        run.response = _simulate_llm_inference(run.transcript)
    run.stage_timers.append(t3)

    # Stage 4: TTS synthesis
    t4 = StageTimer("TTS Synthesis")
    with t4:
        tts_result = stage_tts_synthesis(run.response, synth)
    run.stage_timers.append(t4)
    run.tts_duration_s = tts_result.duration_s

    return run


def print_run_summary(scenario: dict[str, str], run: PipelineRun) -> None:
    """Print a formatted summary for one pipeline run.

    Args:
        scenario: Scenario metadata dict.
        run: Completed pipeline run.
    """
    print(f"\n{'=' * 62}")
    print(f"  Scenario : {scenario['label']}")
    print(f'  Utterance: "{scenario["utterance"]}"')
    print(f"{'=' * 62}")
    print(f"  {'Stage':<30} {'Latency':>10}")
    print(f"  {'-' * 44}")
    for timer in run.stage_timers:
        print(f"  {timer}")
    print(f"  {'-' * 44}")
    print(f"  {'TOTAL':30s} {run.total_ms:7.1f} ms")
    print(f"  {'TTFB (VAD+STT)':30s} {run.ttfb_ms:7.1f} ms")
    print()
    print(f'  Transcript : "{run.transcript}"')
    print(f'  Response   : "{run.response}"')
    print(f"  TTS audio  : {run.tts_duration_s:.2f}s estimated")
    print(f"  Quality    : {run.quality_score:.2%}")


def main() -> None:
    """Run the full pipeline demo across all scenarios."""
    print("\n" + "=" * 62)
    print("  Local STT Pipeline Demo — VAD -> STT -> LLM -> TTS")
    print("=" * 62)

    # Initialise components
    vad_engine = VadEngine(VadConfig(threshold=0.5, min_speech_ms=250, sample_rate=16_000))
    whisper = WhisperModel(model_size="base", device="cpu", compute_type=ComputeType.INT8)
    stt_config = TranscribeConfig(language="en", beam_size=5, vad_filter=True)
    synth = SpeechSynthesizer(KokoroConfig(voice=KokoroVoice.AF_HEART, speed=1.0))

    print(f"\n  STT model     : Whisper-{whisper.model_size} [{whisper.compute_type.value}]")
    print(f"  STT VRAM est. : {whisper.estimated_vram_mb()} MB")
    print(f"  TTS voice     : {synth.config.voice.value} ({synth.config.voice.gender})")
    print(f"  VAD threshold : {vad_engine.config.threshold}")

    all_runs: list[PipelineRun] = []

    for scenario in _DEMO_SCENARIOS:
        run = run_pipeline(scenario, vad_engine, whisper, stt_config, synth)
        print_run_summary(scenario, run)
        all_runs.append(run)

    # Aggregate summary
    finished = [r for r in all_runs if r.total_ms > 0]
    if finished:
        avg_total = sum(r.total_ms for r in finished) / len(finished)
        avg_ttfb = sum(r.ttfb_ms for r in finished) / len(finished)
        avg_quality = sum(r.quality_score for r in finished) / len(finished)

        print(f"\n{'=' * 62}")
        print("  AGGREGATE SUMMARY")
        print(f"{'=' * 62}")
        print(f"  Scenarios run     : {len(finished)}")
        print(f"  Avg total latency : {avg_total:.1f} ms")
        print(f"  Avg TTFB          : {avg_ttfb:.1f} ms")
        print(f"  Avg quality score : {avg_quality:.2%}")
        print(f"{'=' * 62}\n")


if __name__ == "__main__":
    main()
