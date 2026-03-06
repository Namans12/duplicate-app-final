# Project Specification: Duplicate File Manager - Bulk Operations

This file defines the project requirements for the Ralph loop to implement.

## Project Overview

Enhance the existing Advanced Duplicate File Finder application with intelligent bulk move operations. The application currently identifies duplicates using multiple algorithms (hash-based, perceptual, audio fingerprinting, etc.) but requires manual selection for moving files. This project adds automated bulk operations with smart audio quality detection.

**Tech Stack:** Python 3.10+, PyQt6, Pillow, pydub, librosa/chromaprint for audio analysis

**Current State:** The application has:
- Duplicate detection with 6 algorithms (MD5, SHA256, Perceptual Image, Video Scene, Audio Fingerprint, Text Similarity)
- Manual file selection and move operations (`move_selected_files()`)
- Bulk delete preview functionality
- Audio fingerprinting capabilities via `gpu_accelerator.py`
- File metadata extraction and comparison

**Goal:** Add two new features to streamline duplicate management workflows.

---

## Features to Implement

List features in order of priority. The Ralph loop will pick one feature per iteration.

### Feature 1: Bulk Move Duplicates Button
**Priority:** High  
**Status:** Complete

**Description:**
Add a "Bulk Move Duplicates" button that automatically moves all duplicate files to a designated destination folder, intelligently keeping one original file in its current location for each duplicate group.

**Acceptance Criteria:**
- [ ] New button labeled "🚀 Bulk Move Duplicates" added to the results tab UI
- [ ] Button is disabled when no duplicates are found or no destination is set
- [ ] Clicking button shows a detailed preview dialog listing:
  - Total duplicate groups to process
  - For each group: which file will be kept (with reason: first found, largest, etc.)
  - Which files will be moved (with paths)
  - Total disk space that will be freed in kept location
  - Total number of files to be moved
- [ ] Preview dialog has "Proceed" and "Cancel" buttons
- [ ] On "Proceed", files are moved with error handling for:
  - Permission errors
  - Filename conflicts (auto-rename with counter suffix)
  - Files that no longer exist
  - Disk space issues
- [ ] Progress bar shows move operation status
- [ ] Success message shows: moved count, failed count, space freed
- [ ] Failed moves are logged and shown in a scrollable error dialog
- [ ] File tree view refreshes after operation completes
- [ ] Operation can be undone using existing undo system (if implemented)

**Technical Notes:**
- Leverage existing `move_selected_files()` function as reference in [main.py](main.py#L3721-L3795)
- Use existing `show_bulk_delete_preview()` as template for preview dialog ([main.py](main.py#L3493-L3607))
- Keep first file in each duplicate group (earliest in scan order)
- Handle filename conflicts same as current move: append `_1`, `_2`, etc.
- Use `os.makedirs(exist_ok=True)` for destination folder creation
- Maintain file permissions and timestamps where possible

**UI Location:**
- Add button next to existing "Move Selected" button in Duplicates Results tab
- Button should be visually distinct (different color or icon)

---

### Feature 2: Audio Smart Move - Keep Best Quality
**Priority:** High  
**Status:** Complete

**Description:**
Add an "Audio Smart Move" button specifically for audio files that analyzes audio quality across duplicate groups and automatically keeps the highest quality version while moving lower quality duplicates.

**Acceptance Criteria:**
- [ ] New button labeled "🎵 Audio Smart Move" added to the results tab UI
- [ ] Button is only enabled when:
  - Duplicates are found
  - At least one duplicate group contains audio files
  - A move destination is set
- [ ] Quality scoring algorithm implemented that evaluates:
  - Audio codec/format (priority: FLAC > WAV > ALAC > AAC > MP3 > OGG > WMA)
  - Bitrate (kbps) - higher is better
  - Sample rate (Hz) - higher is better (up to 48kHz, diminishing returns after)
  - File size as tiebreaker (larger = potentially better quality)
- [ ] Clicking button shows detailed preview dialog with:
  - List of audio duplicate groups (filter out non-audio groups)
  - For each group: quality scores for all files
  - Highlighted "best" file that will be kept (color-coded, bold text)
  - Audio properties displayed: format, bitrate, sample rate, file size
  - Visual indicator (stars, percentage) showing quality score
  - Total files to be moved
- [ ] Preview dialog has "Proceed" and "Cancel" buttons
- [ ] On "Proceed", executes bulk move keeping only highest quality files
- [ ] Handles edge cases:
  - Groups where all audio files have identical quality (keep first)
  - Mixed groups with audio and non-audio files (only process audio)
  - Corrupted audio files (skip and log warning)
  - Files that can't be analyzed (fall back to file size)
- [ ] Success message shows:
  - Audio files analyzed count
  - Files moved count
  - Files kept count
  - Average quality improvement (estimated)
- [ ] Detailed log shows which files were kept and why

**Acceptance Criteria (continued):**
- [ ] Audio quality analysis is performant (< 100ms per file for metadata extraction)
- [ ] Works with formats: MP3, FLAC, WAV, AAC, OGG, WMA, M4A, OPUS, AIFF, ALAC
- [ ] Does not re-encode or modify audio files, only moves them
- [ ] Integration with existing duplicate detection algorithms

**Technical Notes:**
- Use `pydub.AudioSegment` or `mutagen` library for metadata extraction
- Leverage existing audio fingerprinting code from `gpu_accelerator.py` `AudioFingerprinter` class
- Quality scoring formula (weighted):
  ```
  quality_score = (format_weight * 40) + (bitrate_kbps * 0.15) + (sample_rate_hz * 0.0005) + (file_size_mb * 0.5)
  
  Format weights:
  FLAC: 10, WAV: 9, ALAC: 9, AAC: 6, MP3: 5, OGG: 4, WMA: 3, Other: 1
  ```
- Cache audio metadata during duplicate scan to avoid re-analysis
- Add helper methods:
  - `analyze_audio_quality(file_path) -> Dict[str, Any]`
  - `compare_audio_files(files: List[str]) -> str` (returns path of best file)
  - `get_audio_metadata(file_path) -> Dict[str, Any]`
- Handle audio analysis errors gracefully (log and skip file)
- Display bitrate in kbps, sample rate in kHz for user-friendly output

**UI Location:**
- Add button next to "Bulk Move Duplicates" button
- Button should have music note icon (🎵) for easy identification
- Consider grouping related buttons together in a "Bulk Operations" section

---

## Technical Requirements

### Core Dependencies
- **Python:** 3.10+
- **GUI Framework:** PyQt6 >= 6.4.0
- **Image Processing:** Pillow (PIL), imagehash
- **Audio Analysis:** 
  - pydub (for format conversion and metadata)
  - mutagen (optional, for metadata parsing)
  - librosa or chromaprint (already used for fingerprinting)
  - numpy (for audio processing)

### Audio Library Installation
```bash
pip install pydub mutagen
# For pydub to work with all formats:
# - Windows: Install ffmpeg (choco install ffmpeg)
# - Linux: sudo apt install ffmpeg
# - macOS: brew install ffmpeg
```

### Code Architecture
- **Main Application:** `main.py` - `EnhancedDuplicateFinder` class
- **Audio Utilities:** Create new `audio_quality_analyzer.py` module
- **UI Integration:** Add buttons to existing results tab
- **Error Handling:** Consistent with existing error handling patterns

### Testing Strategy
- Manual testing with various audio file types
- Test edge cases: empty groups, single files, permission errors
- Test with large file sets (1000+ files) for performance
- Verify undo functionality works correctly
- Test on Windows, macOS, Linux (if multi-platform)

### Performance Goals
- Audio metadata extraction: < 100ms per file
- Bulk move operation: Handle 1000+ files without UI freeze
- Preview dialog: Display in < 500ms for typical scan results

---

## Non-Functional Requirements

### Usability
- Clear visual feedback during operations
- Informative error messages with actionable guidance
- Preview dialogs are easy to understand
- Operations can be cancelled mid-execution

### Reliability
- No data loss - files moved, not deleted
- Atomic operations where possible
- Clear audit trail of what was moved
- Undo capability maintained

### Performance
- UI remains responsive during operations (use threading)
- Memory efficient for large file sets
- Audio analysis is fast (metadata only, no full file read)

---

## Future Enhancements (Out of Scope)

These features are not part of the current ralph loop but could be added later:

- **Video Quality Analysis:** Similar to audio, keep best quality video
- **Custom Quality Rules:** Let users define their own quality priorities
- **Dry Run Mode:** Simulate bulk operations without moving files
- **Move to Multiple Destinations:** Organize by file type or quality tier
- **Cloud Storage Integration:** Move duplicates to cloud backup
- **Scheduled Scans:** Automatic periodic duplicate detection
- **Machine Learning Quality Detection:** Train model to detect quality from audio characteristics

---

## Completion Marker

When all features are implemented and tested, add "PROJECT COMPLETE" to DEVELOPMENT-LOG.md.

---

## Notes & Assumptions

1. **Existing Features Used:**
   - Move destination is set via existing "Select Move Destination" button
   - Duplicate detection is already complete before bulk operations
   - File tree views are already populated with duplicate groups
   
2. **Audio File Detection:**
   - Files are considered audio if extension matches: `.mp3, .wav, .flac, .aac, .ogg, .wma, .m4a, .opus, .aiff, .ape, .alac, .wv`
   - This list exists in `main.py` line 994 (`audio_extensions` set)

3. **Error Handling Philosophy:**
   - Fail gracefully - one file error shouldn't stop entire operation
   - Log all errors for debugging
   - Show summary of failures to user
   - Allow retry of failed operations

4. **Platform Compatibility:**
   - Primary target: Windows (based on user environment)
   - Path handling should use `os.path` or `pathlib` for cross-platform compatibility
   - Audio libraries (ffmpeg) must be installed separately by user

5. **Performance Considerations:**
   - Use threading for file operations (existing pattern in codebase)
   - Audio metadata extraction is I/O bound - consider thread pool
   - Cache audio metadata if possible to avoid redundant analysis
   - Show progress updates every 10 files or 100ms, whichever is longer

---

## Questions for Clarification (Resolved)

✅ **Q: For bulk move, should it keep one original or move everything?**  
A: Keep one original in place, move all other duplicates

✅ **Q: What defines "best" audio quality?**  
A: Combination of bitrate, sample rate, format priority, and file size

✅ **Q: Should there be a confirmation dialog?**  
A: Yes, detailed preview showing what will be kept vs moved

✅ **Q: Should audio quality detection work for all duplicates or specific scans?**  
A: Separate "Audio Smart Move" button for targeted audio processing

---

**Last Updated:** 2026-02-19  
**Version:** 1.0
