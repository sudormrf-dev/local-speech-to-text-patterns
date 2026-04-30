"""Model comparison benchmark: STT, TTS, and VAD model families.

Compares simulated performance metrics for the main model families:
  - STT : Whisper tiny / base / small / medium (WER, RTF, VRAM)
  - TTS : Kokoro-82M vs Piper ONNX (RTF, quality score)
  - VAD : Silero v4 vs WebRTC (latency, precision)

All numbers are simulated/referenced from published benchmarks.
No real models or audio files are required.

Usage::

    python benchmarks/model_comparison.py
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from patterns.kokoro_tts import KokoroConfig, KokoroVoice, SpeechSynthesizer
from patterns.silero_vad import VadConfig, VadEngine
from patterns.whisper_ctranslate2 import ComputeType, WhisperModel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SEED = 99

# ---------------------------------------------------------------------------
# STT comparison
# ---------------------------------------------------------------------------


@dataclass
class STTModelEntry:
    """Performance entry for a single STT model configuration.

    Attributes:
        model_size: Whisper model size label.
        compute_type: CTranslate2 quantization type.
        wer_percent: Word Error Rate on LibriSpeech test-clean (%).
        rtf: Real-Time Factor (< 1.0 = faster than real time).
        vram_mb: Peak GPU/RAM footprint in megabytes.
        params_m: Model parameter count in millions.
    """

    model_size: str
    compute_type: ComputeType
    wer_percent: float
    rtf: float
    vram_mb: int
    params_m: int

    @property
    def label(self) -> str:
        """Short display label for this entry."""
        return f"whisper-{self.model_size} [{self.compute_type.value}]"

    @property
    def realtime_capable(self) -> bool:
        """True if RTF < 1.0 (processes audio faster than it arrives)."""
        return self.rtf < 1.0


# Reference data derived from faster-whisper published benchmarks
_STT_ENTRIES: list[STTModelEntry] = [
    STTModelEntry("tiny",   ComputeType.INT8,    wer_percent=5.6,  rtf=0.06, vram_mb=75,   params_m=39),
    STTModelEntry("base",   ComputeType.INT8,    wer_percent=4.2,  rtf=0.12, vram_mb=145,  params_m=74),
    STTModelEntry("small",  ComputeType.INT8,    wer_percent=3.1,  rtf=0.30, vram_mb=466,  params_m=244),
    STTModelEntry("medium", ComputeType.INT8,    wer_percent=2.7,  rtf=0.72, vram_mb=1500, params_m=769),
    STTModelEntry("tiny",   ComputeType.FLOAT32, wer_percent=5.6,  rtf=0.23, vram_mb=155,  params_m=39),
    STTModelEntry("base",   ComputeType.FLOAT32, wer_percent=4.2,  rtf=0.46, vram_mb=290,  params_m=74),
    STTModelEntry("small",  ComputeType.FLOAT32, wer_percent=3.1,  rtf=1.14, vram_mb=932,  params_m=244),
    STTModelEntry("medium", ComputeType.FLOAT32, wer_percent=2.7,  rtf=2.88, vram_mb=3000, params_m=769),
]


def run_stt_comparison() -> list[STTModelEntry]:
    """Return STT model entries, optionally enriched by WhisperModel.estimated_vram_mb().

    Returns:
        List of :class:`STTModelEntry` objects.
    """
    enriched: list[STTModelEntry] = []
    for entry in _STT_ENTRIES:
        model = WhisperModel(model_size=entry.model_size, device="cpu", compute_type=entry.compute_type)
        # Cross-check our reference VRAM vs the pattern's own estimate
        pattern_vram = model.estimated_vram_mb()
        # Use max of reference and pattern estimate for a conservative figure
        final_vram = max(entry.vram_mb, pattern_vram)
        enriched.append(
            STTModelEntry(
                model_size=entry.model_size,
                compute_type=entry.compute_type,
                wer_percent=entry.wer_percent,
                rtf=entry.rtf,
                vram_mb=final_vram,
                params_m=entry.params_m,
            )
        )
    return enriched


# ---------------------------------------------------------------------------
# TTS comparison
# ---------------------------------------------------------------------------


@dataclass
class TTSModelEntry:
    """Performance entry for a single TTS model configuration.

    Attributes:
        model_name: Model identifier (e.g., 'kokoro-82m').
        voice: Voice label used for synthesis.
        rtf: Real-Time Factor (< 1.0 = faster than real time).
        quality_mos: Simulated Mean Opinion Score (1.0-5.0).
        vram_mb: Peak RAM/VRAM footprint in megabytes.
        latency_first_chunk_ms: Latency to first audio chunk (streaming).
        sample_rate: Output audio sample rate in Hz.
    """

    model_name: str
    voice: str
    rtf: float
    quality_mos: float
    vram_mb: int
    latency_first_chunk_ms: float
    sample_rate: int

    @property
    def label(self) -> str:
        """Short display label."""
        return f"{self.model_name} [{self.voice}]"

    @property
    def streaming_capable(self) -> bool:
        """True if first-chunk latency is below 200 ms."""
        return self.latency_first_chunk_ms < 200.0


def run_tts_comparison() -> list[TTSModelEntry]:
    """Run simulated TTS benchmarks for Kokoro and Piper.

    Uses :class:`SpeechSynthesizer` to confirm the synthesizer initialises
    correctly for each voice, then returns reference metric entries.

    Returns:
        List of :class:`TTSModelEntry` objects.
    """
    rng = random.Random(_SEED)
    entries: list[TTSModelEntry] = []

    kokoro_voices = [
        (KokoroVoice.AF_HEART, 0.92, 4.3),
        (KokoroVoice.BF_EMMA,  0.89, 4.2),
        (KokoroVoice.AM_MICHAEL, 0.87, 4.1),
    ]
    for voice, rtf, mos in kokoro_voices:
        synth = SpeechSynthesizer(KokoroConfig(voice=voice, speed=1.0))
        _ = synth.synthesize("benchmark text sample")
        jitter = rng.uniform(-0.01, 0.01)
        entries.append(
            TTSModelEntry(
                model_name="kokoro-82m",
                voice=voice.value,
                rtf=rtf + jitter,
                quality_mos=mos + rng.uniform(-0.05, 0.05),
                vram_mb=330,
                latency_first_chunk_ms=85.0 + rng.gauss(0, 5),
                sample_rate=24_000,
            )
        )

    # Piper ONNX reference entries (no Python class — simulated directly)
    piper_voices = [
        ("en_US-lessac-medium",  1.15, 3.8, 130),
        ("en_GB-alba-medium",    1.08, 3.7, 110),
    ]
    for voice_id, rtf, mos, first_chunk in piper_voices:
        entries.append(
            TTSModelEntry(
                model_name="piper-onnx",
                voice=voice_id,
                rtf=rtf + rng.uniform(-0.02, 0.02),
                quality_mos=mos + rng.uniform(-0.05, 0.05),
                vram_mb=60,
                latency_first_chunk_ms=float(first_chunk) + rng.gauss(0, 8),
                sample_rate=22_050,
            )
        )

    return entries


# ---------------------------------------------------------------------------
# VAD comparison
# ---------------------------------------------------------------------------


@dataclass
class VADModelEntry:
    """Performance entry for a single VAD model configuration.

    Attributes:
        model_name: Model identifier (e.g., 'silero-v4').
        latency_ms: Processing latency per 100 ms audio chunk.
        precision: Simulated precision (true positives / (TP + FP)).
        recall: Simulated recall (true positives / (TP + FN)).
        model_size_kb: On-disk model size in kilobytes.
        requires_gpu: True if GPU is required for real-time performance.
    """

    model_name: str
    latency_ms: float
    precision: float
    recall: float
    model_size_kb: int
    requires_gpu: bool = False

    @property
    def f1(self) -> float:
        """Harmonic mean of precision and recall."""
        denom = self.precision + self.recall
        return 2 * self.precision * self.recall / denom if denom > 0 else 0.0

    @property
    def label(self) -> str:
        """Short display label."""
        gpu_tag = " [GPU]" if self.requires_gpu else " [CPU]"
        return f"{self.model_name}{gpu_tag}"


def run_vad_comparison() -> list[VADModelEntry]:
    """Return VAD model comparison entries.

    Instantiates :class:`VadEngine` for Silero to verify the pattern
    loads correctly, then returns reference metric entries for both models.

    Returns:
        List of :class:`VADModelEntry` objects.
    """
    rng = random.Random(_SEED + 10)

    # Silero v4 via VadEngine
    _engine = VadEngine(VadConfig(threshold=0.5, sample_rate=16_000))
    silero_entry = VADModelEntry(
        model_name="silero-v4",
        latency_ms=9.5 + rng.gauss(0, 0.5),
        precision=0.967 + rng.uniform(-0.005, 0.005),
        recall=0.941 + rng.uniform(-0.005, 0.005),
        model_size_kb=900,
        requires_gpu=False,
    )

    # WebRTC VAD (reference only — no Python class in this repo)
    webrtc_entry = VADModelEntry(
        model_name="webrtc-vad",
        latency_ms=1.2 + rng.gauss(0, 0.2),
        precision=0.891 + rng.uniform(-0.01, 0.01),
        recall=0.875 + rng.uniform(-0.01, 0.01),
        model_size_kb=12,
        requires_gpu=False,
    )

    return [silero_entry, webrtc_entry]


# ---------------------------------------------------------------------------
# Markdown table printer
# ---------------------------------------------------------------------------

_MD_SEP = "|"


def _row(*cells: str) -> str:
    return _MD_SEP + _MD_SEP.join(f" {c} " for c in cells) + _MD_SEP


def _divider(widths: list[int]) -> str:
    return _MD_SEP + _MD_SEP.join("-" * (w + 2) for w in widths) + _MD_SEP


def print_stt_table(entries: list[STTModelEntry]) -> None:
    """Print a markdown table for STT model comparison.

    Args:
        entries: List of STT benchmark entries.
    """
    headers = ["Model", "Quant", "WER %", "RTF", "VRAM MB", "Params M", "RT?"]
    widths = [max(len(h), max(len(str(e.model_size)) + 10 for e in entries)) for h in headers]
    widths = [24, 14, 7, 6, 8, 8, 5]

    print("\n### STT Model Comparison (Whisper + CTranslate2)\n")
    print(_row(*[h.ljust(w) for h, w in zip(headers, widths)]))
    print(_divider(widths))
    for e in entries:
        rt = "yes" if e.realtime_capable else "no"
        row_cells = [
            f"whisper-{e.model_size}".ljust(widths[0]),
            e.compute_type.value.ljust(widths[1]),
            f"{e.wer_percent:.1f}".ljust(widths[2]),
            f"{e.rtf:.2f}".ljust(widths[3]),
            str(e.vram_mb).ljust(widths[4]),
            str(e.params_m).ljust(widths[5]),
            rt.ljust(widths[6]),
        ]
        print(_row(*row_cells))


def print_tts_table(entries: list[TTSModelEntry]) -> None:
    """Print a markdown table for TTS model comparison.

    Args:
        entries: List of TTS benchmark entries.
    """
    headers = ["Model", "Voice", "RTF", "MOS", "VRAM MB", "1st chunk ms", "Stream?"]
    widths = [14, 26, 6, 5, 8, 13, 8]

    print("\n### TTS Model Comparison (Kokoro-82M vs Piper ONNX)\n")
    print(_row(*[h.ljust(w) for h, w in zip(headers, widths)]))
    print(_divider(widths))
    for e in entries:
        stream = "yes" if e.streaming_capable else "no"
        row_cells = [
            e.model_name.ljust(widths[0]),
            e.voice.ljust(widths[1]),
            f"{e.rtf:.2f}".ljust(widths[2]),
            f"{e.quality_mos:.2f}".ljust(widths[3]),
            str(e.vram_mb).ljust(widths[4]),
            f"{e.latency_first_chunk_ms:.1f}".ljust(widths[5]),
            stream.ljust(widths[6]),
        ]
        print(_row(*row_cells))


def print_vad_table(entries: list[VADModelEntry]) -> None:
    """Print a markdown table for VAD model comparison.

    Args:
        entries: List of VAD benchmark entries.
    """
    headers = ["Model", "Latency ms", "Precision", "Recall", "F1", "Size KB", "GPU?"]
    widths = [16, 10, 10, 8, 6, 8, 5]

    print("\n### VAD Model Comparison (Silero v4 vs WebRTC)\n")
    print(_row(*[h.ljust(w) for h, w in zip(headers, widths)]))
    print(_divider(widths))
    for e in entries:
        gpu = "yes" if e.requires_gpu else "no"
        row_cells = [
            e.model_name.ljust(widths[0]),
            f"{e.latency_ms:.1f}".ljust(widths[1]),
            f"{e.precision:.3f}".ljust(widths[2]),
            f"{e.recall:.3f}".ljust(widths[3]),
            f"{e.f1:.3f}".ljust(widths[4]),
            str(e.model_size_kb).ljust(widths[5]),
            gpu.ljust(widths[6]),
        ]
        print(_row(*row_cells))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run all model comparisons and print markdown tables."""
    print("\n" + "=" * 80)
    print("  Model Comparison Benchmark — Local STT / TTS / VAD")
    print("=" * 80)

    stt_entries = run_stt_comparison()
    print_stt_table(stt_entries)

    tts_entries = run_tts_comparison()
    print_tts_table(tts_entries)

    vad_entries = run_vad_comparison()
    print_vad_table(vad_entries)

    # Summary recommendations
    best_stt = min(stt_entries, key=lambda e: e.wer_percent + e.rtf * 2)
    best_tts = max(tts_entries, key=lambda e: e.quality_mos)
    best_vad = max(vad_entries, key=lambda e: e.f1)

    print("\n### Recommendations\n")
    print(f"  STT: {best_stt.label}  — WER {best_stt.wer_percent:.1f}%, RTF {best_stt.rtf:.2f}")
    print(f"  TTS: {best_tts.label}  — MOS {best_tts.quality_mos:.2f}, 1st chunk {best_tts.latency_first_chunk_ms:.0f} ms")
    print(f"  VAD: {best_vad.label}  — F1 {best_vad.f1:.3f}, latency {best_vad.latency_ms:.1f} ms")
    print()


if __name__ == "__main__":
    main()
