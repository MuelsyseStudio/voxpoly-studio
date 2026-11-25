# Contributing to VoxPoly Studio

Thank you for your interest in contributing to VoxPoly Studio!

## Development Setup

1. Fork and clone the repository.
2. Run `install_windows.bat` or `./install_linux_mac.sh` to create `.venv`.
3. Run the portable unit test suite before submitting a PR:

```bash
pytest tests/test_basic.py -v
```

## Adding a New TTS Engine

VoxPoly Studio isolates third-party engines via `BaseVoiceCloneEngine` and `_SubprocessWorkerEngine` in `backend/tts_engine.py`:
- Keep heavy or version-conflicting dependencies inside a dedicated worker script (`backend/<engine>_worker.py`) communicating over JSON-lines stdio.
- Register supported languages in `backend/config.py` (`LANGUAGES`).
- Ensure `unload()` cleanly terminates the worker or frees GPU memory so switching engines never causes OOM on 8–16 GB GPUs.
