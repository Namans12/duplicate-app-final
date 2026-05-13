#!/usr/bin/env python3
"""
Shared Dependencies Module
Centralizes optional imports and availability flags for the application
"""
import os
import sys
import logging
import multiprocessing
from typing import Optional, Any

# DEP-1: Do NOT call logging.basicConfig() here. Log configuration belongs in the
# application entry point (main.py / launcher.py). Calling it in a non-entry-point
# module takes precedence over and silently overrides the host app's log setup.
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

class AppConfig:
    """Centralized application configuration"""
    
    # File size thresholds (bytes)
    SMALL_FILE_THRESHOLD = 10 * 1024 * 1024      # 10 MB
    MEDIUM_FILE_THRESHOLD = 100 * 1024 * 1024    # 100 MB
    LARGE_FILE_THRESHOLD = 500 * 1024 * 1024     # 500 MB
    
    # Processing limits
    MAX_SEMANTIC_CHARS = 50000
    MAX_TEXT_CONTENT_CHARS = 10000
    
    # Memory settings
    DEFAULT_MAX_MEMORY_MB = 500
    DEFAULT_CHUNK_SIZE = 8192
    DEFAULT_HASH_CHUNK_SIZE = 65536
    
    # Database settings
    DEFAULT_DB_PATH = "scan_cache.db"
    CACHE_CLEANUP_DAYS = 30
    
    # GPU settings
    DEFAULT_GPU_BATCH_SIZE = 32
    MAX_GPU_BATCH_SIZE = 512
    MIN_GPU_BATCH_SIZE = 32
    
    # Undo settings
    MAX_UNDO_OPERATIONS = 100
    
    # Thread settings
    @staticmethod
    def get_optimal_workers(gpu_enabled: bool = False) -> int:
        """
        Get optimal worker thread count based on GPU mode.
        
        GPU-bound workloads need fewer CPU threads feeding the GPU to avoid
        oversubscription and context switching overhead.
        
        Args:
            gpu_enabled: Whether GPU is being used.
            
        Returns:
            Optimal number of worker threads.
        """
        cores = multiprocessing.cpu_count() or 4
        
        if gpu_enabled:
            # GPU mode: fewer threads to avoid oversubscription
            return max(2, min(8, cores - 1))
        else:
            # CPU-only mode: use all available cores minus one
            return max(1, cores - 1)


# =============================================================================
# OPTIONAL DEPENDENCY FLAGS
# =============================================================================

# PIL / Pillow for image processing
HAS_PILLOW = False
Image = None
try:
    from PIL import Image as _Image
    Image = _Image
    HAS_PILLOW = True
    logger.debug("PIL/Pillow available")
except ImportError:
    logger.info("PIL/Pillow not available - image processing disabled")

# imagehash for perceptual hashing
HAS_IMAGEHASH = False
imagehash = None
try:
    import imagehash as _imagehash
    imagehash = _imagehash
    HAS_IMAGEHASH = True
    logger.debug("imagehash available")
except ImportError:
    logger.info("imagehash not available - perceptual hashing will use fallback")

# OpenCV for video processing
HAS_CV2 = False
cv2 = None
try:
    import cv2 as _cv2
    cv2 = _cv2
    HAS_CV2 = True
    logger.debug("OpenCV available")
except ImportError:
    logger.info("OpenCV not available - video processing disabled")

# NumPy for numerical operations
HAS_NUMPY = False
np = None
try:
    import numpy as _np
    np = _np
    HAS_NUMPY = True
    logger.debug("NumPy available")
except ImportError:
    logger.warning("NumPy not available - some features will be limited")

# send2trash for recycle bin support
HAS_SEND2TRASH = False
send2trash_func = None
try:
    from send2trash import send2trash as _send2trash
    send2trash_func = _send2trash
    HAS_SEND2TRASH = True
    logger.debug("send2trash available")
except ImportError:
    logger.info("send2trash not available - files will be permanently deleted")

# Sentence transformers for semantic similarity
HAS_TRANSFORMERS = False
SentenceTransformer = None


def _prepare_windows_torch_dlls():
    """Preload Torch DLL directories on Windows before importing torch-backed libs."""
    if os.name != "nt":
        return

    candidates = [
        os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib"),
        os.path.join(sys.prefix, "lib", "site-packages", "torch", "lib"),
    ]
    seen = set()
    for candidate in candidates:
        if not os.path.isdir(candidate):
            continue
        norm = os.path.normcase(os.path.abspath(candidate))
        if norm in seen:
            continue
        seen.add(norm)
        try:
            os.add_dll_directory(candidate)
        except (AttributeError, OSError):
            pass


try:
    _prepare_windows_torch_dlls()
    import torch as _torch  # noqa: F401
    from sentence_transformers import SentenceTransformer as _ST
    SentenceTransformer = _ST
    HAS_TRANSFORMERS = True
    logger.debug("SentenceTransformers available")
except Exception as e:
    logger.info(f"SentenceTransformers unavailable - semantic similarity will use fallback if available: {e}")

# Normalize the availability flag to the imported class reference. In some Windows
# import sequences the class object is available even when the optimistic flag path
# above did not complete cleanly.
HAS_TRANSFORMERS = SentenceTransformer is not None

# Audio processing
HAS_AUDIO = False
AudioSegment = None
try:
    from pydub import AudioSegment as _AS
    AudioSegment = _AS
    HAS_AUDIO = True
    logger.debug("pydub available")
except Exception as e:
    logger.info(f"pydub unavailable - audio processing disabled: {e}")

# Chromaprint for audio fingerprinting
HAS_CHROMAPRINT = False
acoustid = None
try:
    import acoustid as _acoustid
    acoustid = _acoustid
    HAS_CHROMAPRINT = True
    logger.debug("acoustid available")
except Exception as e:
    logger.info(f"acoustid unavailable - audio fingerprinting disabled: {e}")

# librosa for audio analysis
HAS_LIBROSA = False
librosa = None
try:
    import librosa as _librosa
    librosa = _librosa
    HAS_LIBROSA = True
    logger.debug("librosa available")
except Exception as e:
    logger.info(f"librosa unavailable - MFCC analysis disabled: {e}")

# soundfile for audio I/O
HAS_SOUNDFILE = False
soundfile = None
try:
    import soundfile as _sf
    soundfile = _sf
    HAS_SOUNDFILE = True
    logger.debug("soundfile available")
except Exception as e:
    logger.info(f"soundfile unavailable: {e}")


# =============================================================================
# GPU / CUDA DETECTION
# =============================================================================

HAS_CUDA = False
CUDA_BACKEND = 'cpu'
cp = None  # CuPy module reference
torch = None  # PyTorch module reference


def detect_gpu_backend() -> tuple:
    """
    Detect available GPU acceleration backend with proper error handling.
    
    Returns:
        Tuple of (backend_name, is_cuda_available)
    """
    global cp, torch, HAS_CUDA, CUDA_BACKEND
    
    # Try CuPy first (preferred for numerical operations)
    try:
        import cupy as _cp
        n = _cp.cuda.runtime.getDeviceCount()
        if n > 0:
            try:
                device_props = _cp.cuda.runtime.getDeviceProperties(0)
                device_name = device_props.get('name', b'Unknown')
                if isinstance(device_name, bytes):
                    device_name = device_name.decode('utf-8', errors='ignore')
                logger.info(f"GPU: CuPy backend available: {n} device(s), {device_name}")
            except Exception:
                logger.info(f"GPU: CuPy backend available: {n} device(s)")
            cp = _cp
            HAS_CUDA = True
            CUDA_BACKEND = 'cupy'
            return "cupy", True
    except ImportError:
        logger.debug("CuPy not installed")
    except Exception as e:
        logger.warning(f"CuPy backend not available: {e!r}")

    # Try PyTorch CUDA as fallback
    try:
        import torch as _torch
        if _torch.cuda.is_available():
            device_count = _torch.cuda.device_count()
            device_name = _torch.cuda.get_device_name(0)
            logger.info(f"GPU: PyTorch CUDA backend available: {device_count} device(s), {device_name}")
            torch = _torch
            HAS_CUDA = True
            CUDA_BACKEND = 'torch'
            return "torch", True
        else:
            logger.debug("PyTorch installed but CUDA not available")
    except ImportError:
        logger.debug("PyTorch not installed")
    except Exception as e:
        logger.warning(f"PyTorch backend not available: {e!r}")

    logger.info("GPU: No GPU acceleration available, using CPU")
    return "cpu", False


# DEP-2: GPU backend detection runs once at import time. This is intentional so that
# all modules that `import dependencies` share the same detected backend without
# re-probing. Set the environment variable SKIP_GPU_DETECT=1 to skip detection
# (useful for test environments or machines without GPU drivers installed) — in that
# case CUDA_BACKEND stays 'cpu' and HAS_CUDA stays False.
# DEP-3: CUDA_BACKEND is named for historical reasons; its value can be 'cpu', 'cupy',
# or 'torch'. Think of it as the compute backend name, not strictly a CUDA flag.
if os.environ.get("SKIP_GPU_DETECT", "").strip() not in ("1", "true", "yes"):
    CUDA_BACKEND, HAS_CUDA = detect_gpu_backend()
else:
    logger.info("GPU detection skipped (SKIP_GPU_DETECT is set)")


def get_dependency_summary() -> dict:
    """Get a summary of all dependency availability."""
    return {
        "pillow": HAS_PILLOW,
        "imagehash": HAS_IMAGEHASH,
        "opencv": HAS_CV2,
        "numpy": HAS_NUMPY,
        "send2trash": HAS_SEND2TRASH,
        "transformers": HAS_TRANSFORMERS,
        "audio": HAS_AUDIO,
        "chromaprint": HAS_CHROMAPRINT,
        "librosa": HAS_LIBROSA,
        "soundfile": HAS_SOUNDFILE,
        "cuda": HAS_CUDA,
        "cuda_backend": CUDA_BACKEND,
    }
