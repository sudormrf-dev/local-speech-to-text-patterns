# local-speech-to-text-patterns

Production patterns for local speech processing: Whisper CTranslate2 transcription, Silero VAD, and Kokoro TTS — all running on CPU without cloud APIs.

## Patterns

### Whisper CTranslate2 (`patterns/whisper_ctranslate2.py`)
- `WhisperModel` — faster-whisper wrapper with compute-type auto-selection and VRAM estimation
- `ComputeType` — INT8 (CPU), FLOAT16 (GPU), BFLOAT16 (Ampere+)
- `TranscribeConfig` — beam size, VAD filter, language detection, word timestamps
- `TranscribeResult` — full text + segments with `filter_reliable()` to drop low-confidence segments
- `format_srt()` / `format_vtt()` — generate SubRip and WebVTT subtitle files

### Silero VAD (`patterns/silero_vad.py`)
- `VadEngine` — speech segment detection from raw PCM samples
- `VadConfig` — threshold, silence/speech duration filters, sample rate
- `SpeechSegment` — start/end times with `overlaps()` helper
- `VadResult` — `speech_ratio`, `speech_duration_s`, `has_speech()`
- `merge_speech_segments()` — bridge short silences between segments

### Kokoro TTS (`patterns/kokoro_tts.py`)
- `SpeechSynthesizer` — text-to-speech via Kokoro ONNX model
- `KokoroVoice` — 9 voices: American/British, male/female, multiple personas
- `KokoroConfig` — speed control, sample rate, GPU toggle
- `TTSResult` — PCM samples with `save(path)` → WAV output
- `estimate_audio_duration()` — predict audio length from text
- `synthesize_chunks()` — sentence-level streaming for low latency

## Quick Start

```python
from pathlib import Path
from patterns import WhisperModel, ComputeType, TranscribeConfig, format_srt

model = WhisperModel(model_size="base", compute_type=ComputeType.INT8)
config = TranscribeConfig(language="en", beam_size=5, vad_filter=True)
result = model.transcribe(Path("audio.wav"), config)
print(result.text)
print(format_srt(result.segments))
```

```python
from patterns import SpeechSynthesizer, KokoroConfig, KokoroVoice

synth = SpeechSynthesizer(KokoroConfig(voice=KokoroVoice.BF_EMMA, speed=1.0))
result = synth.synthesize("Hello from Kokoro TTS!")
result.save(Path("output.wav"))
```

## Installation

```bash
pip install -e ".[dev]"
# For actual inference:
pip install faster-whisper kokoro-onnx silero-vad
pytest -q
```

## Requirements

- Python 3.12+
- No runtime dependencies (stdlib only for patterns)
- faster-whisper, kokoro-onnx, silero-vad for real inference
