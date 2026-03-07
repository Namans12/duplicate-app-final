# Duplicate File Finder

Desktop duplicate finder built with Python and PyQt6. The maintained app entrypoint is `main.py`, with an isolated environment bootstrap in `.venv-app`.

## Current App

- Single maintained GUI entrypoint: `main.py`
- Exact duplicate detection: `MD5`, `SHA256`, `XXH3_128`
- Similarity detection: image perceptual hashing, audio fingerprinting, video keyframes, text/document comparison, archive comparison
- Optional acceleration: CUDA via CuPy, semantic text via `sentence-transformers`
- Supporting systems: undo, session import/export, plugin loading, database cache, memory management

## Recommended Setup

Create the dedicated app environment:

```powershell
powershell -ExecutionPolicy Bypass -File setup_app_env.ps1
```

Launch the app with:

```powershell
.\launch_app_env.bat
```

Direct launch options:

```powershell
& .\.venv-app\Scripts\python.exe launcher.py
& .\.venv-app\Scripts\python.exe bootstrap_launch.py main.py
```

## Key Files

- `main.py`: maintained GUI app
- `launcher.py`: single-entry launcher
- `bootstrap_launch.py`: dependency/bootstrap wrapper for consistent DLL loading
- `media_profiles.py`: per-media matching defaults and helper comparisons
- `dependencies.py`: optional dependency detection and shared imports
- `requirements-app-env.txt`: pinned package set for `.venv-app`
- `verify_app_env.py`: environment verification

## Notes

- Native Tesseract is still optional. Install `tesseract.exe` separately and add it to `PATH` if you want OCR fallback for scanned documents.
- The legacy `archive/` folder is not required for the maintained app.
