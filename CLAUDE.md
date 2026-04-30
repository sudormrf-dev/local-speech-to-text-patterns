# local-speech-to-text-patterns

## Purpose
Production patterns for local speech processing: Whisper CTranslate2 transcription, Silero VAD, and 

## Architecture
- `patterns/`: Core library code — dataclasses, protocols, type-annotated helpers
- `examples/`: End-to-end runnable demos
- `benchmarks/`: Comparative performance measurements
- `tests/`: pytest unit tests (CPU-only, no external deps required)

## Conventions
- Python 3.11+ with full type hints
- ruff lint (line-length = 120)
- mypy --strict
- PEP 257 docstrings
- pytest for all examples

## Key Patterns
See `patterns/` for the documented patterns.
Run `pytest tests/ -q` to verify all patterns are correct.
