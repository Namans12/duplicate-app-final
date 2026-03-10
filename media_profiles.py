#!/usr/bin/env python3
"""Per-media profile defaults and helper utilities for duplicate detection."""
from __future__ import annotations

import hashlib
import os
import functools
import threading as _threading
import zipfile
import concurrent.futures as _cf
from copy import deepcopy
from typing import Any, Dict, Iterable, Optional, Tuple

try:
    import xxhash
    HAS_XXHASH = True
except ImportError:
    xxhash = None
    HAS_XXHASH = False

try:
    from PIL import Image
    import imagehash
    HAS_IMAGE = True
except ImportError:
    HAS_IMAGE = False
    Image = None
    imagehash = None

try:
    from skimage.metrics import structural_similarity
    HAS_SKIMAGE = True
except ImportError:
    structural_similarity = None
    HAS_SKIMAGE = False

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
    # Suppress MuPDF's internal error/warning printer — corrupt PDFs print
    # "MuPDF error: ..." to stderr even when the exception is caught.
    fitz.TOOLS.mupdf_display_errors(False)
except ImportError:
    fitz = None
    HAS_PYMUPDF = False

try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except ImportError:
    PdfReader = None
    HAS_PYPDF = False

MEDIA_ORDER = ["images", "videos", "audio", "documents", "code", "archives", "general"]

MEDIA_EXTENSIONS = {
    "images": {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"},
    "videos": {".mp4", ".mkv", ".mov", ".avi", ".wmv", ".webm", ".m4v"},
    "audio": {".mp3", ".flac", ".wav", ".aac", ".ogg", ".m4a", ".opus", ".wma"},
    "documents": {".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".txt", ".rtf", ".md"},
    "code": {".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".c", ".cpp", ".h", ".hpp", ".cs", ".html", ".css", ".json", ".xml", ".yaml", ".yml"},
    "archives": {".zip", ".rar", ".7z", ".tar", ".gz", ".bz2"},
}


def get_default_media_profiles() -> Dict[str, Dict[str, Any]]:
    return {
        "images": {"mode": "strict", "exact_hash": "XXH3_128", "enable_fuzzy": True, "ssim_threshold": 0.96},
        "videos": {"mode": "strict", "exact_hash": "XXH3_128", "enable_fuzzy": True, "keyframes": 12},
        "audio": {"mode": "strict", "exact_hash": "XXH3_128", "enable_fuzzy": True, "audio_method": "hybrid"},
        "documents": {"mode": "strict", "exact_hash": "SHA256", "enable_fuzzy": True, "semantic_threshold": 0.90},
        "code": {"mode": "strict", "exact_hash": "SHA256", "enable_fuzzy": True, "semantic_threshold": 0.92},
        "archives": {"mode": "strict", "exact_hash": "SHA256", "enable_fuzzy": True},
        "general": {"mode": "strict", "exact_hash": "XXH3_128", "enable_fuzzy": False},
    }


def merge_media_profiles(overrides: Optional[Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    profiles = get_default_media_profiles()
    if not overrides:
        return profiles
    for media_type, values in overrides.items():
        profiles.setdefault(media_type, {})
        profiles[media_type].update(values or {})
    return profiles


def summarize_media_profile(profile: Dict[str, Any]) -> str:
    mode = profile.get("mode", "strict").replace("-", " ").title()
    exact_hash = profile.get("exact_hash", "XXH3_128")
    fuzzy = "on" if profile.get("enable_fuzzy", False) else "off"
    return f"{mode} | exact: {exact_hash} | fuzzy: {fuzzy}"


def get_media_type(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    for media_type, extensions in MEDIA_EXTENSIONS.items():
        if ext in extensions:
            return media_type
    return "general"


def get_profile_for_path(file_path: str, profiles: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    profiles = merge_media_profiles(profiles)
    return profiles.get(get_media_type(file_path), profiles["general"])


def compute_exact_hash(file_path: str, hash_name: str = "XXH3_128") -> Optional[str]:
    hash_name = (hash_name or "XXH3_128").upper()
    try:
        if hash_name == "MD5":
            h = hashlib.md5()
        elif hash_name == "SHA256":
            h = hashlib.sha256()
        else:
            if HAS_XXHASH and xxhash is not None:
                h = xxhash.xxh3_128()
            else:
                # Fallback to SHA256 when xxhash is not installed
                h = hashlib.sha256()
        with open(file_path, "rb") as handle:
            for chunk in iter(lambda: handle.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def _read_text(file_path: str) -> str:
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as handle:
            return handle.read()
    except Exception:
        return ""


_PDF_EXTRACT_TIMEOUT = 12   # seconds per PDF file before giving up
_PDF_MAX_PAGES       = 300  # ignore absurdly large / corrupt documents
_TEXT_READ_MAX_BYTES = 10 * 1024 * 1024  # 10 MB cap for raw-text read


# ROOT-CAUSE fix: cache per-file text extraction so that each PDF (or other
# document) is extracted at most once during a similarity scan even when the
# same file appears in many pairs.  Without this, a scan with N document files
# would decode the same PDF N times, each potentially blocking for up to the
# daemon-thread timeout value.
@functools.lru_cache(maxsize=1024)
def extract_document_text(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()

    # Fast path: plain text / code / markup — read directly (with size cap)
    if ext in {".txt", ".md", ".py", ".json", ".yaml", ".yml", ".xml", ".html", ".css", ".js"}:
        return _read_text(file_path)

    if ext == ".pdf":
        # Skip huge files — text extraction would be too slow to be useful
        try:
            if os.path.getsize(file_path) > 100 * 1024 * 1024:   # 100 MB
                return ""
        except OSError:
            return ""

        # PyMuPDF — run in a daemon thread with a hard timeout so a corrupt
        # object stream can't hang the scan indefinitely (FIX-A).
        # threading.Thread.join(timeout) returns after the deadline regardless,
        # unlike ThreadPoolExecutor.__exit__ which calls shutdown(wait=True).
        if HAS_PYMUPDF:
            _fitz_result: list = [None]

            def _fitz_worker() -> None:
                try:
                    with fitz.open(file_path) as doc:
                        if len(doc) > _PDF_MAX_PAGES:
                            return
                        _fitz_result[0] = "\n".join(
                            page.get_text("text") for page in doc
                        )
                except Exception:
                    pass

            _t = _threading.Thread(target=_fitz_worker, daemon=True)
            _t.start()
            _t.join(timeout=_PDF_EXTRACT_TIMEOUT)
            if _fitz_result[0]:
                return _fitz_result[0]

        # pypdf fallback — same daemon-thread pattern
        if HAS_PYPDF:
            _pypdf_result: list = [None]

            def _pypdf_worker() -> None:
                try:
                    reader = PdfReader(file_path)
                    if len(reader.pages) > _PDF_MAX_PAGES:
                        return
                    _pypdf_result[0] = "\n".join(
                        page.extract_text() or "" for page in reader.pages
                    )
                except Exception:
                    pass

            _t = _threading.Thread(target=_pypdf_worker, daemon=True)
            _t.start()
            _t.join(timeout=_PDF_EXTRACT_TIMEOUT)
            if _pypdf_result[0]:
                return _pypdf_result[0]

        return ""   # could not extract text from this PDF

    return _read_text(file_path)


def compare_document_files(file1: str, file2: str, profile: Dict[str, Any], semantic_engine=None) -> Tuple[bool, float, list[str]]:
    text1 = extract_document_text(file1).strip()
    text2 = extract_document_text(file2).strip()
    if not text1 or not text2:
        return False, 0.0, []

    set1 = set(text1.split())
    set2 = set(text2.split())
    if not set1 or not set2:
        return False, 0.0, []

    lexical = len(set1 & set2) / max(1, len(set1 | set2))
    threshold = profile.get("semantic_threshold", 0.90)
    reasons = [f"lexical {lexical:.2f}"] if lexical >= threshold else []
    return lexical >= threshold, lexical, reasons


def compare_archive_files(file1: str, file2: str, profile: Dict[str, Any]) -> Tuple[bool, float, list[str]]:
    try:
        with zipfile.ZipFile(file1) as zip1, zipfile.ZipFile(file2) as zip2:
            set1 = {(item.filename, item.file_size) for item in zip1.infolist()}
            set2 = {(item.filename, item.file_size) for item in zip2.infolist()}
    except Exception:
        return False, 0.0, []

    if not set1 or not set2:
        return False, 0.0, []
    score = len(set1 & set2) / max(1, len(set1 | set2))
    return score >= 0.90, score, [f"archive {score:.2f}"] if score >= 0.90 else []


def compare_image_files(file1: str, file2: str, profile: Dict[str, Any]) -> Tuple[bool, float, list[str]]:
    if not HAS_IMAGE:
        return False, 0.0, []
    try:
        with Image.open(file1) as img1, Image.open(file2) as img2:
            hash1 = imagehash.phash(img1)
            hash2 = imagehash.phash(img2)
            distance = hash1 - hash2
            score = 1.0 - min(distance, 64) / 64.0
            reasons = [f"phash {distance}"]
            return score >= 0.85, score, reasons
    except Exception:
        return False, 0.0, []


def estimate_scene_based_keyframes(file_path: str, default_keyframes: int = 8) -> int:
    try:
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
    except Exception:
        return default_keyframes
    if size_mb < 100:
        return max(default_keyframes, 8)
    if size_mb < 1000:
        return max(default_keyframes, 12)
    return max(default_keyframes, 16)


def get_optional_engine_status() -> Dict[str, Dict[str, Any]]:
    return {
        "pymupdf": {"available": HAS_PYMUPDF, "status": "available" if HAS_PYMUPDF else "not installed"},
        "pypdf": {"available": HAS_PYPDF, "status": "available" if HAS_PYPDF else "not installed"},
        "skimage": {"available": HAS_SKIMAGE, "status": "available" if HAS_SKIMAGE else "not installed"},
    }
