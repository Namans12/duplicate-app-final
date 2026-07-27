# Advanced Duplicate File Finder

Desktop app for finding duplicate files by **content, not filename** — perceptual image hashing, audio fingerprinting, video scene matching, and semantic text similarity, with a full undo system.

> **Canonical repository.** [`duplicate-app-final`](https://github.com/Namans12/duplicate-app-final) is a copy from 2026-03-07 with an isolated-environment installer, per-media tuning profiles, and substantial dedup-engine work. See [Related repositories](#related-repositories).

## Overview

Exact-hash deduplication only catches byte-identical files. It misses the same photo re-saved at different quality, the same song at a different bitrate, the same video re-encoded, and the same document lightly edited.

This tool runs six detection algorithms so near-duplicates are caught too, then gives you a review interface before anything is deleted — every operation is backed up and reversible.

## Features

### Detection

- **Perceptual image hashing** — visually identical images regardless of format or compression
- **Audio fingerprinting** — chromaprint + librosa, matches the same track across bitrates
- **Video scene detection** — multi-keyframe analysis for re-encoded video
- **Semantic text similarity** — transformer embeddings, threshold adjustable 0.50–0.95
- **MD5 / SHA-256** — exact matches
- **xxHash** — fast pre-filter pass

### Management

- **Complete undo** — every move and delete is backed up and restorable
- **Batch operations** with smart grouping and multi-select
- **Filtering** by file type, size range, modification date, and custom patterns
- **Live preview** for images, text, and file properties
- **Search within results** to filter duplicate groups instantly
- **GPU acceleration** where available
- **Plugin system** for custom hash algorithms

### Interface

PyQt6, tabbed layout, dark/light themes with persistent settings, keyboard shortcuts (`Ctrl+P`, `Ctrl+A`, `Del`, …), progress with ETA and current-file display.

## Tech Stack

Python · PyQt6 · SQLAlchemy + SQLite · Pillow · `imagehash` · OpenCV · `xxhash` · Ray (parallelism)

**Audio** — `pyacoustid`, `librosa`, `soundfile`, `pydub`, `mutagen`, `ffmpeg-python`
**ML** — `torch`, `transformers`, `sentence-transformers`

## Prerequisites

- Python 3.9+
- FFmpeg on `PATH` for audio and video analysis
- ~2 GB disk for the transformer models on first run
- Optional: CUDA-capable GPU

## Installation

```bash
git clone https://github.com/Namans12/duplicate-app.git
cd duplicate-app
pip install -r requirements.txt
```

## Usage

```bash
python launcher.py
```

The launcher lets you pick which build to run. On Windows, `launch.bat` does the same thing.

To skip the launcher and open the main PyQt6 window directly:

```bash
python main.py
```

> **Note on older instructions:** earlier versions of this README said `python main_enhanced.py` with `requirements_enhanced.txt`. That entry point now lives in [`archive/`](archive/) and is no longer the supported path — use `launcher.py` or `main.py`.

### Keyboard shortcuts

| Keys | Action |
|---|---|
| `Ctrl+P` | Preview selected file |
| `Ctrl+A` | Select all in group |
| `Del` | Delete selected |

## Project Structure

```
main.py                    PyQt6 application — the actual UI (~5900 lines)
launcher.py                version picker / entry point
launch.bat                 Windows launcher

duplicate_merger.py        merge and dedup logic
database_manager.py        SQLAlchemy persistence (duplicates.db)
audio_quality_analyzer.py  audio fingerprinting and quality ranking
gpu_accelerator.py         CUDA paths
memory_optimizer.py        large-library memory management
plugin_system.py           custom algorithm plugins
dependencies.py            runtime dependency checks

plugins/
  examples/                example hash and JSON plugins
  plugin_config.json
archive/                   superseded main.py variants, kept for reference
```

## Requirements files

| File | Use |
|---|---|
| `requirements.txt` | Standard install |
| `requirements_enhanced.txt` | Legacy — for the archived `main_enhanced.py` |
| `requirements_enterprise.txt` | Legacy — for the archived `main_enterprise.py` |

Use `requirements.txt` unless you are deliberately running something from `archive/`.

## Documentation

- [`SPEC.md`](SPEC.md) — specification
- [`DEVELOPMENT-LOG.md`](DEVELOPMENT-LOG.md) — build history
- [`AUDIO_ERROR_SUPPRESSION.md`](AUDIO_ERROR_SUPPRESSION.md) — silencing noisy audio-library warnings

## Branches

| Branch | Date | Notes |
|---|---|---|
| `new` *(default)* | 2026-03-06 | Current |
| `new2` | 2026-03-06 | +1 commit — GUI fix, 64 lines in one file. Never merged |

## Related Repositories

| Repo | Relationship |
|---|---|
| [`duplicate-app-final`](https://github.com/Namans12/duplicate-app-final) | Copy from 2026-03-07. Adds an isolated-venv installer, `media_profiles.py` for per-media-type tuning, real-folder dedup verification tooling, and a `final` branch with seven further commits of engine work. **Drops** this repo's `SPEC.md`, `DEVELOPMENT-LOG.md`, and `AUDIO_ERROR_SUPPRESSION.md` |
| [`PicSorter`](https://github.com/Namans12/PicSorter) | Focused image-only deduplicator — separate codebase, same problem space |

## Caution

This tool deletes files. The undo system is backed by `duplicates.db` and the backup directory — keep both until you've confirmed a run. Always dry-run against a copy before pointing it at an irreplaceable library.
