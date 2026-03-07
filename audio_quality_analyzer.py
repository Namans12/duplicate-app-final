import os
import logging
from collections import OrderedDict
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

try:
    from mutagen import File as MutagenFile
    HAS_MUTAGEN = True
except ImportError:  # AQ-1: was bare `except Exception`, now precise
    HAS_MUTAGEN = False
    MutagenFile = None


class AudioQualityAnalyzer:
    """Analyze audio file metadata and compute a quality score."""

    FORMAT_WEIGHTS = {
        'flac': 10,
        'wav': 9,
        'alac': 9,
        'aiff': 9,
        'ape': 8,
        'wv': 8,
        'aac': 6,
        'm4a': 6,
        'mp3': 5,
        'ogg': 4,
        'opus': 4,
        'wma': 3,
        'other': 1
    }

    # AQ-2: Bounded LRU cache to prevent unbounded memory growth on large libraries
    _CACHE_MAX_SIZE = 2048

    def __init__(self):
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()

    def analyze_audio_quality(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Return metadata + quality score for an audio file."""
        if not file_path:
            return None
        if file_path in self._cache:
            # Move to end to mark as recently used
            self._cache.move_to_end(file_path)
            return self._cache[file_path]

        metadata = self.get_audio_metadata(file_path)
        if metadata is None:
            return None

        metadata['quality_score'] = self.calculate_quality_score(metadata)

        # AQ-2: Evict oldest entry when cache is full (LRU eviction)
        if len(self._cache) >= self._CACHE_MAX_SIZE:
            self._cache.popitem(last=False)
        self._cache[file_path] = metadata
        return metadata

    def get_audio_metadata(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Extract audio metadata with mutagen when available."""
        try:
            if not os.path.exists(file_path):
                return None

            ext = os.path.splitext(file_path)[1].lower().lstrip('.')
            file_size_bytes = os.path.getsize(file_path)
            file_size_mb = file_size_bytes / (1024 * 1024) if file_size_bytes else 0.0

            bitrate_kbps = 0.0
            sample_rate_hz = 0.0
            duration_s = 0.0
            analysis_error = False

            if HAS_MUTAGEN and MutagenFile is not None:
                try:
                    audio = MutagenFile(file_path, easy=False)
                    if audio and hasattr(audio, "info") and audio.info:
                        info = audio.info
                        bitrate = getattr(info, "bitrate", None)
                        sample_rate = getattr(info, "sample_rate", None)
                        duration = getattr(info, "length", None)

                        if bitrate:
                            bitrate_kbps = bitrate / 1000.0
                        if sample_rate:
                            sample_rate_hz = float(sample_rate)
                        if duration:
                            duration_s = float(duration)
                except Exception as e:
                    analysis_error = True
                    logger.warning(f"Audio metadata read failed for {file_path}: {e}")

            # Estimate bitrate if missing and duration available
            if bitrate_kbps <= 0 and duration_s > 0 and file_size_bytes > 0:
                bitrate_kbps = (file_size_bytes * 8.0) / (duration_s * 1000.0)

            return {
                'format': ext or 'other',
                'bitrate_kbps': float(bitrate_kbps) if bitrate_kbps else 0.0,
                'sample_rate_hz': float(sample_rate_hz) if sample_rate_hz else 0.0,
                'duration_s': float(duration_s) if duration_s else 0.0,
                'file_size_mb': float(file_size_mb),
                'file_size_bytes': int(file_size_bytes),
                'analysis_error': analysis_error
            }
        except Exception as e:
            logger.warning(f"Audio metadata error for {file_path}: {e}")
            return None

    def calculate_quality_score(self, metadata: Dict[str, Any]) -> float:
        """Calculate weighted quality score from metadata."""
        format_key = (metadata.get('format') or 'other').lower()
        format_weight = self.FORMAT_WEIGHTS.get(format_key, self.FORMAT_WEIGHTS['other'])

        format_score = format_weight * 40.0
        bitrate_score = float(metadata.get('bitrate_kbps') or 0.0) * 0.15
        sample_rate_score = float(metadata.get('sample_rate_hz') or 0.0) * 0.0005
        size_score = float(metadata.get('file_size_mb') or 0.0) * 0.5

        score = format_score + bitrate_score + sample_rate_score + size_score

        # AQ-3: Penalise score when metadata could not be fully read, so files with
        # unreliable quality estimates rank lower than files with confirmed metadata.
        if metadata.get('analysis_error'):
            score *= 0.8

        return score

    def compare_audio_files(self, files):
        """Return the best audio file path based on quality score."""
        best_path = None
        best_score = -1.0

        for path in files:
            meta = self.analyze_audio_quality(path)
            if not meta:
                continue
            score = meta.get('quality_score', 0.0)
            if score > best_score:
                best_score = score
                best_path = path

        return best_path
