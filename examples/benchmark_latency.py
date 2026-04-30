"""Latency benchmark: TTFB simulation for each pipeline component.

Measures simulated Time-To-First-Byte / First-Token latency across:
  - VAD detection
  - STT transcription (Whisper base, INT8 vs FP32)
  - LLM inference (fixed budget)
  - TTS synthesis (Kokoro)

Quantization impact (FP32 -> INT8) and device speedup (CPU vs RTX 5080)
are modelled via simple multiplier tables derived from published benchmarks.

Usage::

    python examples/benchmark_latency.py
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

from patterns.kokoro_tts import KokoroConfig, KokoroVoice, SpeechSynthesizer
from patterns.silero_vad import VadConfig, VadEngine
from patterns.whisper_ctranslate2 import ComputeType, TranscribeConfig, WhisperModel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SEED = 7
_WARMUP_ITERS = 2
_BENCH_ITERS = 5
_AUDIO_DURATION_S = 5.0  # simulated utterance length
_SAMPLE_RATE = 16_000

# Base latencies on a mid-range CPU (milliseconds) at FP32 reference
_BASE_LATENCY_MS: dict[str, float] = {
    "vad": 52.0,
    "stt_base": 210.0,
    "llm": 480.0,
    "tts": 145.0,
}

# Speedup factors relative to FP32 CPU baseline
_QUANTIZATION_SPEEDUP: dict[str, float] = {
    ComputeType.FLOAT32.value: 1.00,
    ComputeType.INT8.value: 3.80,
    ComputeType.FLOAT16.value: 2.20,
    ComputeType.INT8_FLOAT16.value: 4.10,
}

# Device speedup factors (RTX 5080 vs CPU reference)
_DEVICE_SPEEDUP: dict[str, float] = {
    "cpu": 1.0,
    "cuda_rtx5080": 18.5,  # RTX 5080 with INT8 vs CPU FP32
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ComponentResult:
    """Benchmark result for a single component configuration.

    Attributes:
        component: Component name (e.g., "STT", "TTS").
        config_label: Human-readable config description.
        latencies_ms: Per-iteration measured latencies.
        simulated: True when using the stub (no real model) implementation.
    """

    component: str
    config_label: str
    latencies_ms: list[float] = field(default_factory=list)
    simulated: bool = True

    @property
    def mean_ms(self) -> float:
        """Mean latency across iterations."""
        return sum(self.latencies_ms) / len(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def min_ms(self) -> float:
        """Minimum observed latency."""
        return min(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def max_ms(self) -> float:
        """Maximum observed latency."""
        return max(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def p95_ms(self) -> float:
        """95th-percentile latency."""
        if not self.latencies_ms:
            return 0.0
        sorted_lat = sorted(self.latencies_ms)
        idx = int(len(sorted_lat) * 0.95)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]


@dataclass
class BenchmarkSuite:
    """Full benchmark suite results.

    Attributes:
        results: All component benchmark results.
        device: Device label used for the run.
    """

    results: list[ComponentResult] = field(default_factory=list)
    device: str = "cpu"

    def ttfb_ms(self) -> float:
        """Estimated TTFB: VAD mean + STT mean (first config of each)."""
        vad = next((r for r in self.results if r.component == "VAD"), None)
        stt = next((r for r in self.results if r.component == "STT"), None)
        vad_lat = vad.mean_ms if vad else 0.0
        stt_lat = stt.mean_ms if stt else 0.0
        return vad_lat + stt_lat


# ---------------------------------------------------------------------------
# Simulated timing helpers
# ---------------------------------------------------------------------------


def _jitter(base_ms: float, rng: random.Random, factor: float = 0.08) -> float:
    """Apply Gaussian jitter to a base latency.

    Args:
        base_ms: Base latency in milliseconds.
        rng: Random number generator for reproducibility.
        factor: Standard deviation as a fraction of base_ms.

    Returns:
        Jittered latency in milliseconds (always > 0).
    """
    noise = rng.gauss(0, base_ms * factor)
    return max(1.0, base_ms + noise)


def _simulate_latency(target_ms: float, rng: random.Random) -> float:
    """Sleep for approximately target_ms and return actual elapsed time.

    Args:
        target_ms: Target sleep duration in milliseconds.
        rng: RNG for jitter.

    Returns:
        Actual elapsed time in milliseconds.
    """
    jittered = _jitter(target_ms, rng)
    t0 = time.perf_counter()
    time.sleep(jittered / 1_000)
    return (time.perf_counter() - t0) * 1_000


# ---------------------------------------------------------------------------
# Component benchmarks
# ---------------------------------------------------------------------------


def benchmark_vad(iterations: int = _BENCH_ITERS, device: str = "cpu") -> ComponentResult:
    """Benchmark Silero VAD detection latency.

    Args:
        iterations: Number of measurement iterations.
        device: Device label ('cpu' or 'cuda_rtx5080').

    Returns:
        :class:`ComponentResult` with per-iteration timings.
    """
    rng = random.Random(_SEED)
    result = ComponentResult(component="VAD", config_label=f"Silero v4 [{device}]")
    engine = VadEngine(VadConfig(threshold=0.5, sample_rate=_SAMPLE_RATE))
    n_samples = int(_AUDIO_DURATION_S * _SAMPLE_RATE)
    dummy_samples = [rng.gauss(0, 0.1) for _ in range(n_samples)]
    speedup = _DEVICE_SPEEDUP.get(device, 1.0)

    for _ in range(_WARMUP_ITERS):
        engine.detect(dummy_samples)

    for _ in range(iterations):
        base = _BASE_LATENCY_MS["vad"] / speedup
        elapsed = _simulate_latency(base, rng)
        result.latencies_ms.append(elapsed)

    return result


def benchmark_stt(
    compute_type: ComputeType,
    device: str = "cpu",
    iterations: int = _BENCH_ITERS,
) -> ComponentResult:
    """Benchmark Whisper STT transcription latency.

    Args:
        compute_type: CTranslate2 quantization type.
        device: Device label ('cpu' or 'cuda_rtx5080').
        iterations: Number of measurement iterations.

    Returns:
        :class:`ComponentResult` with per-iteration timings.
    """
    rng = random.Random(_SEED + 1)
    label = f"Whisper-base [{compute_type.value}] on {device}"
    result = ComponentResult(component="STT", config_label=label)
    _model = WhisperModel(model_size="base", device="cpu", compute_type=compute_type)
    _config = TranscribeConfig(language="en", beam_size=5)
    _ = (_model, _config)  # instantiate for realism

    q_speedup = _QUANTIZATION_SPEEDUP.get(compute_type.value, 1.0)
    d_speedup = _DEVICE_SPEEDUP.get(device, 1.0)
    total_speedup = q_speedup * d_speedup

    for _ in range(_WARMUP_ITERS):
        time.sleep(0.001)

    for _ in range(iterations):
        base = _BASE_LATENCY_MS["stt_base"] / total_speedup
        elapsed = _simulate_latency(base, rng)
        result.latencies_ms.append(elapsed)

    return result


def benchmark_llm(device: str = "cpu", iterations: int = _BENCH_ITERS) -> ComponentResult:
    """Benchmark LLM inference latency (fixed-budget simulation).

    Args:
        device: Device label ('cpu' or 'cuda_rtx5080').
        iterations: Number of measurement iterations.

    Returns:
        :class:`ComponentResult` with per-iteration timings.
    """
    rng = random.Random(_SEED + 2)
    result = ComponentResult(component="LLM", config_label=f"LLM inference [{device}]")
    speedup = _DEVICE_SPEEDUP.get(device, 1.0)

    for _ in range(iterations):
        base = _BASE_LATENCY_MS["llm"] / speedup
        elapsed = _simulate_latency(base, rng)
        result.latencies_ms.append(elapsed)

    return result


def benchmark_tts(device: str = "cpu", iterations: int = _BENCH_ITERS) -> ComponentResult:
    """Benchmark Kokoro TTS synthesis latency.

    Args:
        device: Device label ('cpu' or 'cuda_rtx5080').
        iterations: Number of measurement iterations.

    Returns:
        :class:`ComponentResult` with per-iteration timings.
    """
    rng = random.Random(_SEED + 3)
    result = ComponentResult(component="TTS", config_label=f"Kokoro-82M [{device}]")
    synth = SpeechSynthesizer(KokoroConfig(voice=KokoroVoice.AF_HEART, speed=1.0))
    sample_text = "The weather today is partly cloudy with a high of eighteen degrees."
    speedup = _DEVICE_SPEEDUP.get(device, 1.0)

    for _ in range(_WARMUP_ITERS):
        synth.synthesize(sample_text)

    for _ in range(iterations):
        base = _BASE_LATENCY_MS["tts"] / speedup
        elapsed = _simulate_latency(base, rng)
        result.latencies_ms.append(elapsed)

    return result


# ---------------------------------------------------------------------------
# Quantization comparison
# ---------------------------------------------------------------------------


def compare_quantization(iterations: int = _BENCH_ITERS) -> list[ComponentResult]:
    """Compare STT latency across FP32 vs INT8 quantization.

    Args:
        iterations: Measurement iterations per configuration.

    Returns:
        List of :class:`ComponentResult` objects, one per config.
    """
    configs = [
        ComputeType.FLOAT32,
        ComputeType.INT8,
        ComputeType.FLOAT16,
        ComputeType.INT8_FLOAT16,
    ]
    return [benchmark_stt(ct, device="cpu", iterations=iterations) for ct in configs]


def compare_devices(iterations: int = _BENCH_ITERS) -> dict[str, BenchmarkSuite]:
    """Compare full pipeline latency on CPU vs RTX 5080.

    Args:
        iterations: Measurement iterations per component.

    Returns:
        Dict mapping device label to :class:`BenchmarkSuite`.
    """
    suites: dict[str, BenchmarkSuite] = {}
    for device in ("cpu", "cuda_rtx5080"):
        suite = BenchmarkSuite(device=device)
        suite.results.append(benchmark_vad(iterations=iterations, device=device))
        suite.results.append(benchmark_stt(ComputeType.INT8, device=device, iterations=iterations))
        suite.results.append(benchmark_llm(device=device, iterations=iterations))
        suite.results.append(benchmark_tts(device=device, iterations=iterations))
        suites[device] = suite
    return suites


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

_COL = 42


def _print_table_header(title: str) -> None:
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")
    print(f"  {'Configuration':<{_COL}} {'mean':>8} {'min':>8} {'p95':>8}  (ms)")
    print(f"  {'-' * 76}")


def _print_result_row(r: ComponentResult) -> None:
    print(f"  {r.config_label:<{_COL}} {r.mean_ms:8.1f} {r.min_ms:8.1f} {r.p95_ms:8.1f}")


def print_quantization_report(results: list[ComponentResult]) -> None:
    """Print a markdown-style table for quantization comparison results.

    Args:
        results: List of STT ComponentResult objects by quantization type.
    """
    _print_table_header("STT Quantization Impact (Whisper-base, CPU)")
    for r in results:
        _print_result_row(r)

    baseline = next((r for r in results if "float32" in r.config_label), None)
    if baseline and baseline.mean_ms > 0:
        print(f"\n  Speedup vs FP32 baseline ({baseline.mean_ms:.1f} ms):")
        for r in results:
            speedup = baseline.mean_ms / r.mean_ms if r.mean_ms > 0 else 0.0
            print(f"    {r.config_label:<{_COL}} {speedup:5.2f}x")


def print_device_report(suites: dict[str, BenchmarkSuite]) -> None:
    """Print per-device TTFB and component breakdown.

    Args:
        suites: Device -> BenchmarkSuite mapping.
    """
    _print_table_header("Device Comparison: CPU vs RTX 5080 (INT8, 5 iterations)")
    for device, suite in suites.items():
        for r in suite.results:
            print(f"  {r.config_label:<{_COL}} {r.mean_ms:8.1f} {r.min_ms:8.1f} {r.p95_ms:8.1f}")
        total = sum(r.mean_ms for r in suite.results)
        print(f"  {'  TOTAL ' + device:<{_COL}} {total:8.1f}")
        print(f"  {'  TTFB ' + device:<{_COL}} {suite.ttfb_ms():8.1f}")
        print(f"  {'-' * 76}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run all latency benchmarks and print formatted reports."""
    print("\n" + "=" * 80)
    print("  Latency Benchmark — Local STT/TTS Pipeline")
    print(
        f"  Audio duration: {_AUDIO_DURATION_S}s | Iterations: {_BENCH_ITERS} (+ {_WARMUP_ITERS} warmup)"
    )
    print("=" * 80)

    print("\n[1/2] Running quantization comparison ...")
    quant_results = compare_quantization()
    print_quantization_report(quant_results)

    print("\n[2/2] Running device comparison ...")
    device_suites = compare_devices()
    print_device_report(device_suites)

    cpu_suite = device_suites.get("cpu")
    gpu_suite = device_suites.get("cuda_rtx5080")
    if cpu_suite and gpu_suite:
        cpu_total = sum(r.mean_ms for r in cpu_suite.results)
        gpu_total = sum(r.mean_ms for r in gpu_suite.results)
        overall_speedup = cpu_total / gpu_total if gpu_total > 0 else 0.0
        print(f"\n  Overall speedup (CPU -> RTX 5080, INT8): {overall_speedup:.1f}x")
        print(f"  CPU total  : {cpu_total:.1f} ms")
        print(f"  GPU total  : {gpu_total:.1f} ms")
    print()


if __name__ == "__main__":
    main()
