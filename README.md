# Advanced Duplicate File Finder — Final (Dupe of [duplicate-app](https://github.com/Namans12/duplicate-app))

Working copy of **[`duplicate-app`](https://github.com/Namans12/duplicate-app)**, taken 2026-03-07. Adds an isolated-environment installer, per-media-type tuning profiles, and — on the `final` branch — seven commits of dedup-engine work validated against real folders.

> ### ⚠️ Three branches, and they disagree
>
> `master` (2026-03-07) is close to the original. Three branches sit on top and **do not form a chain**:
>
> | Branch | Ahead | Date | Size |
> |---|---|---|---|
> | `final` | +7 | 2026-05-13 | 52 files, +45452/−10390 |
> | `final2` | +1 | 2026-05-13 | 14 files, +819/−363 |
> | `updated` | +1 | 2026-03-07 | 16 files, +428/−9020 |
>
> `final` is by far the most developed. `final2` is a separate single-commit pass from the same day. Diff them before merging either.

## What this repo is

Same product as the original: a PyQt6 desktop app that finds duplicate files by content — perceptual image hashing, audio fingerprinting, video scene matching, semantic text similarity — with full undo. For features, algorithms, and shortcuts, see the **[canonical README in `duplicate-app`](https://github.com/Namans12/duplicate-app#readme)**.

This README covers what changed.

## How it differs from `duplicate-app`

### Added

| File | Purpose |
|---|---|
| `setup_app_env.ps1` | Builds an isolated `.venv-app` so the app's heavy ML dependencies don't pollute your global Python |
| `launch_app_env.bat` | Launches from that isolated env; tells you to run the setup script if it's missing |
| `requirements-app-env.txt` | Pinned dependency set for the isolated env |
| `verify_app_env.py` | Verifies the environment is correctly built before launch |
| `bootstrap_launch.py` | Preloads shared dependency bootstrap, then runs any entry point via `runpy` |
| `media_profiles.py` | Per-media-type detection defaults — different thresholds and algorithms for images vs audio vs video vs documents |

### Removed

`SPEC.md`, `DEVELOPMENT-LOG.md`, `AUDIO_ERROR_SUPPRESSION.md`, and `.github/` instructions. **Those survive only in [`duplicate-app`](https://github.com/Namans12/duplicate-app)** — the audio-error-suppression notes in particular are still relevant here.

### Modified

| File | original | here | Δ |
|---|---|---|---|
| `main.py` | 5939 | 6065 | +126 |
| `dependencies.py` | 274 | 301 | +27 |
| `launcher.py` | 181 | 85 | **−96** |
| `gpu_accelerator.py` | 2448 | 2455 | +7 |

`launcher.py` shrank because version-picking moved out to the environment scripts.

## Branches

### `final` — 2026-05-13, +7 commits

The real continuation. Six commits of engine work, then a large validation pass:

| Date | Commit | What it did |
|---|---|---|
| 2026-03-08 | Best yet | Broad pass across 11 core modules |
| 2026-03-10 | Fixed Speed | `gpu_accelerator.py` and `main.py` performance |
| 2026-03-10 | Fixed more | GPU, main, media profiles |
| 2026-03-10 | Fixed Bugs | Bootstrap, database, merger, launcher, main, profiles |
| 2026-03-10 | 100% hash move updated | Hash-based move correctness in `main.py` |
| 2026-05-13 | Picsort issues | Real-folder validation — see below |

The final commit adds a verification harness rather than just more features:

- `run_real_folder_dedup.py` — runs deduplication against a real library
- `restore_dedup_moves.py` — reverses a run
- `audio_e2e_benchmark.py` — end-to-end audio dedup benchmark
- `tests/test_app_smoke_scan.py`, `test_core_integrity_hardening.py`, `test_image_selection_and_moves.py`

It also commits a series of dry-run reports (`real_folder_dedup_report*.{csv,json}`, `real_folder_dedup_report_summary.md`) recording what the engine *would* have done across successive tuning passes — strict-verified, same-track, and v2/v3 variants. Useful as a record of how the thresholds were arrived at; they are output artefacts rather than source.

### `final2` — 2026-05-13, +1 commit

A single "Claude" commit touching 14 core modules — `audio_quality_analyzer`, `bootstrap_launch`, `database_manager`, `dependencies`, `duplicate_merger`, `gpu_accelerator`, `launcher`, `main`, `media_profiles`, `memory_optimizer`, `plugin_system`, the example plugin, and `verify_app_env`. Also commits a stray `.DS_Store`.

Branches from `master`, **not** from `final` — so it does not include any of the March engine work above.

### `updated` — 2026-03-07, +1 commit

Housekeeping: deletes the eight `archive/main_*.py` variants and the `README_enterprise.md` / `README_new.md` duplicates, adds `SETTINGS_CHEATSHEET.md`. The −9020 lines are almost entirely archive removal.

## Installation

### Isolated environment (recommended — this repo's main addition)

```powershell
git clone https://github.com/Namans12/duplicate-app-final.git
cd duplicate-app-final
git checkout final
powershell -ExecutionPolicy Bypass -File setup_app_env.ps1
```

Then launch:

```
launch_app_env.bat
```

Verify the environment at any point:

```bash
python verify_app_env.py
```

### Standard install

```bash
pip install -r requirements.txt
python launcher.py
```

## Which repo should you use?

| You want | Use |
|---|---|
| Documentation — spec, dev log, audio-error notes | [`duplicate-app`](https://github.com/Namans12/duplicate-app) |
| The most developed engine, with real-folder validation | **this repo**, on `final` |
| Clean tree without the `archive/` cruft | this repo, on `updated` |
| Image-only deduplication | [`PicSorter`](https://github.com/Namans12/PicSorter) |

## Caution

This tool moves and deletes files. `final` ships `restore_dedup_moves.py` specifically to reverse a bad run — keep it and the generated reports until you have verified the results. Dry-run against a copy first.
