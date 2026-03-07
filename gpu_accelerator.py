#!/usr/bin/env python3
"""
GPU Acceleration Module for Performance Enhancement
CUDA-based acceleration for perceptual hashing and similarity calculations
"""
import os
import sys
import multiprocessing
import warnings
import builtins
import numpy as np
from typing import List, Optional, Tuple, Any, Dict
import time


def safe_print(*args, **kwargs):
    """Print without crashing on non-UTF8 Windows consoles."""
    try:
        builtins.print(*args, **kwargs)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "utf-8"
        sep = kwargs.get("sep", " ")
        end = kwargs.get("end", "\n")
        text = sep.join(str(arg) for arg in args)
        safe_text = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
        builtins.print(safe_text, end=end)


print = safe_print

# Suppress common audio processing warnings
warnings.filterwarnings('ignore', category=UserWarning, module='librosa')
warnings.filterwarnings('ignore', category=UserWarning, module='audioread')
warnings.filterwarnings('ignore', message='.*Xing.*')
warnings.filterwarnings('ignore', message='.*PySoundFile failed.*')

# Try to import CUDA libraries (prefer CuPy, fall back to PyTorch)
HAS_CUDA = False
BACKEND = 'cpu'  # 'cupy' | 'torch' | 'cpu'
cp = None
torch = None


# Import shared configuration and utilities
from dependencies import (
    AppConfig,
    CUDA_BACKEND as DEP_CUDA_BACKEND,
    HAS_CUDA as DEP_HAS_CUDA,
    HAS_TRANSFORMERS as DEP_HAS_TRANSFORMERS,
    SentenceTransformer as DepSentenceTransformer,
)

# Use centralized get_optimal_workers
get_optimal_workers = AppConfig.get_optimal_workers


# =============================================================================
# PERCEPTUAL HASH THRESHOLDS - Centralized & Tunable
# =============================================================================

class PHashThresholds:
    """
    Centralized perceptual hash similarity thresholds.
    
    Hamming distance interpretation for 64-bit hashes:
    - 0: Identical images
    - 1-5: Very similar (minor edits, compression artifacts)
    - 6-10: Similar (resized, cropped, color adjustments)
    - 11-15: Somewhat similar (significant edits)
    - 16+: Different images
    
    Thresholds are expressed as maximum Hamming distance for "similar" classification.
    """
    
    # File type specific thresholds (Hamming distance)
    THRESHOLDS_BY_TYPE = {
        "images": 10,           # Standard image comparison
        "video_keyframe": 12,   # Video keyframes have more variance
        "thumbnails": 8,        # Thumbnails should match closely
        "icons": 6,             # Icons should be near-identical
        "screenshots": 10,      # Screenshots may have minor UI differences
    }
    
    # Preset levels for UI exposure
    PRESETS = {
        "strict": {
            "images": 5,
            "video_keyframe": 7,
            "thumbnails": 4,
            "icons": 3,
            "screenshots": 5,
            "description": "Only very similar files (minimal differences)",
        },
        "balanced": {
            "images": 10,
            "video_keyframe": 12,
            "thumbnails": 8,
            "icons": 6,
            "screenshots": 10,
            "description": "Good balance between precision and recall",
        },
        "aggressive": {
            "images": 16,
            "video_keyframe": 18,
            "thumbnails": 12,
            "icons": 10,
            "screenshots": 16,
            "description": "Catch more potential duplicates (may include false positives)",
        },
    }
    
    # Current active preset
    _current_preset = "balanced"
    _custom_thresholds: Dict[str, int] = {}
    
    @classmethod
    def get_threshold(cls, file_type: str = "images") -> int:
        """
        Get the Hamming distance threshold for a file type.
        
        Args:
            file_type: Type of file ("images", "video_keyframe", etc.)
            
        Returns:
            Maximum Hamming distance to consider files as similar.
        """
        # Check custom overrides first
        if file_type in cls._custom_thresholds:
            return cls._custom_thresholds[file_type]
        
        # Get from current preset
        preset = cls.PRESETS.get(cls._current_preset, cls.PRESETS["balanced"])
        return preset.get(file_type, preset.get("images", 10))
    
    @classmethod
    def set_preset(cls, preset_name: str) -> bool:
        """
        Set the active threshold preset.
        
        Args:
            preset_name: One of "strict", "balanced", "aggressive"
            
        Returns:
            True if preset was set successfully.
        """
        if preset_name in cls.PRESETS:
            cls._current_preset = preset_name
            cls._custom_thresholds.clear()  # Clear custom overrides
            print(f"[PHash] Threshold preset set to '{preset_name}': {cls.PRESETS[preset_name]['description']}")
            return True
        return False
    
    @classmethod
    def get_current_preset(cls) -> str:
        """Get the name of the current preset."""
        return cls._current_preset
    
    @classmethod
    def set_custom_threshold(cls, file_type: str, threshold: int) -> None:
        """
        Set a custom threshold for a specific file type.
        
        Args:
            file_type: Type of file
            threshold: Maximum Hamming distance (0-64)
        """
        cls._custom_thresholds[file_type] = max(0, min(64, threshold))
    
    @classmethod
    def get_all_thresholds(cls) -> Dict[str, int]:
        """Get all current thresholds (preset + custom overrides)."""
        preset = cls.PRESETS.get(cls._current_preset, cls.PRESETS["balanced"])
        result = {k: v for k, v in preset.items() if k != "description"}
        result.update(cls._custom_thresholds)
        return result
    
    @classmethod
    def get_preset_names(cls) -> List[str]:
        """Get list of available preset names."""
        return list(cls.PRESETS.keys())
    
    @classmethod
    def get_preset_description(cls, preset_name: str) -> str:
        """Get description of a preset."""
        return cls.PRESETS.get(preset_name, {}).get("description", "")


def hamming_distance(hash1: str, hash2: str) -> int:
    """
    Compute Hamming distance between two hash strings.
    
    Works with both binary strings ("0101...") and hex strings ("a3f2...").
    
    Args:
        hash1: First hash string
        hash2: Second hash string
        
    Returns:
        Number of differing bits.
    """
    if not hash1 or not hash2:
        return 64  # Maximum distance if either hash is empty
    
    # Detect if hex (16 chars) or binary (64 chars)
    if len(hash1) == 16 and all(c in '0123456789abcdefABCDEF' for c in hash1):
        # Hex format - convert to int
        try:
            int1 = int(hash1, 16)
            int2 = int(hash2, 16)
            xor = int1 ^ int2
            return bin(xor).count('1')
        except ValueError:
            return 64
    else:
        # Binary string format
        distance = 0
        for c1, c2 in zip(hash1, hash2):
            if c1 != c2:
                distance += 1
        return distance


def are_similar_phash(hash1: str, hash2: str, file_type: str = "images") -> bool:
    """
    Check if two perceptual hashes are similar based on file type threshold.
    
    Args:
        hash1: First perceptual hash
        hash2: Second perceptual hash
        file_type: Type of file for threshold lookup
        
    Returns:
        True if hashes are within the similarity threshold.
    """
    threshold = PHashThresholds.get_threshold(file_type)
    return hamming_distance(hash1, hash2) <= threshold


def get_similarity_level(hash1: str, hash2: str) -> Tuple[str, int]:
    """
    Get human-readable similarity level between two hashes.
    
    Args:
        hash1: First perceptual hash
        hash2: Second perceptual hash
        
    Returns:
        Tuple of (level_name, hamming_distance)
    """
    dist = hamming_distance(hash1, hash2)
    
    if dist == 0:
        return ("identical", dist)
    elif dist <= 5:
        return ("very_similar", dist)
    elif dist <= 10:
        return ("similar", dist)
    elif dist <= 15:
        return ("somewhat_similar", dist)
    else:
        return ("different", dist)


def detect_gpu_backend() -> Tuple[str, bool]:
    """Detect available GPU acceleration backend with proper error handling."""
    global cp, torch, HAS_CUDA, BACKEND
    
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
                print(f"[GPU] ✅ CuPy backend available: {n} device(s), {device_name}")
            except Exception:
                print(f"[GPU] ✅ CuPy backend available: {n} device(s)")
            cp = _cp
            return "cupy", True
    except ImportError:
        print("[GPU] ℹ️ CuPy not installed")
    except Exception as e:
        print(f"[GPU] ⚠️ CuPy backend not available: {e!r}")

    # Try PyTorch CUDA as fallback
    try:
        import torch as _torch
        if _torch.cuda.is_available():
            device_count = _torch.cuda.device_count()
            device_name = _torch.cuda.get_device_name(0)
            print(f"[GPU] ✅ PyTorch CUDA backend available: {device_count} device(s), {device_name}")
            torch = _torch
            return "pytorch", True
        else:
            print("[GPU] ⚠️ PyTorch installed but CUDA not available")
    except ImportError:
        print("[GPU] ℹ️ PyTorch not installed")
    except Exception as e:
        print(f"[GPU] ⚠️ PyTorch backend not available: {e!r}")

    print("[GPU] ℹ️ No GPU acceleration available, using CPU")
    return "cpu", False


def diagnose_gpu() -> Dict[str, Any]:
    """Run comprehensive GPU diagnostics for troubleshooting."""
    diagnostics = {
        "cuda_available": False,
        "backend": "cpu",
        "devices": [],
        "cupy": {"installed": False, "version": None, "error": None},
        "pytorch": {"installed": False, "version": None, "cuda_version": None, "error": None},
        "recommendations": []
    }
    
    # Check CuPy
    try:
        import cupy as _cp
        diagnostics["cupy"]["installed"] = True
        diagnostics["cupy"]["version"] = _cp.__version__
        try:
            n = _cp.cuda.runtime.getDeviceCount()
            if n > 0:
                diagnostics["cuda_available"] = True
                for i in range(n):
                    props = _cp.cuda.runtime.getDeviceProperties(i)
                    name = props.get('name', b'Unknown')
                    if isinstance(name, bytes):
                        name = name.decode('utf-8', errors='ignore')
                    mem = props.get('totalGlobalMem', 0) // (1024**3)
                    diagnostics["devices"].append({
                        "id": i, "name": name, "memory_gb": mem, "backend": "cupy"
                    })
        except Exception as e:
            diagnostics["cupy"]["error"] = str(e)
    except ImportError:
        diagnostics["recommendations"].append("Install CuPy: pip install cupy-cuda12x")
    except Exception as e:
        diagnostics["cupy"]["error"] = str(e)
    
    # Check PyTorch
    try:
        import torch as _torch
        diagnostics["pytorch"]["installed"] = True
        diagnostics["pytorch"]["version"] = _torch.__version__
        if _torch.cuda.is_available():
            diagnostics["cuda_available"] = True
            diagnostics["pytorch"]["cuda_version"] = _torch.version.cuda
            for i in range(_torch.cuda.device_count()):
                name = _torch.cuda.get_device_name(i)
                mem = _torch.cuda.get_device_properties(i).total_memory // (1024**3)
                # Avoid duplicates
                if not any(d["name"] == name for d in diagnostics["devices"]):
                    diagnostics["devices"].append({
                        "id": i, "name": name, "memory_gb": mem, "backend": "pytorch"
                    })
    except ImportError:
        diagnostics["recommendations"].append("Install PyTorch: pip install torch")
    except Exception as e:
        diagnostics["pytorch"]["error"] = str(e)
    
    # Set recommended backend
    if diagnostics["cupy"]["installed"] and not diagnostics["cupy"]["error"]:
        diagnostics["backend"] = "cupy"
    elif diagnostics["pytorch"]["installed"] and diagnostics["pytorch"]["cuda_version"]:
        diagnostics["backend"] = "pytorch"
    
    return diagnostics


# Initialize GPU backend on module load
BACKEND, HAS_CUDA = detect_gpu_backend()

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from PIL import Image
    HAS_PILLOW = True
except ImportError:
    HAS_PILLOW = False
    Image = None

# imagehash is optional - only needed for CPU fallback
try:
    import imagehash
    HAS_IMAGEHASH = True
except ImportError:
    HAS_IMAGEHASH = False
    imagehash = None

class GPUAccelerator:
    """GPU acceleration for image processing and similarity calculations"""
    
    # Image size for perceptual hashing (32x32 is standard for pHash)
    PHASH_SIZE = 32
    # Low-frequency DCT block size (8x8 = 64 bits for hash)
    DCT_LOW_FREQ = 8
    
    def __init__(self):
        self.cuda_available = HAS_CUDA
        self.device_info = None
        self.memory_pool = None
        self.gpu_memory_gb = 0
        self.optimal_batch_size = 64  # Default, will be computed dynamically

        if self.cuda_available:
            try:
                if BACKEND == 'cupy':
                    self.device_info = cp.cuda.runtime.getDeviceProperties(0)
                    # memory pool exists in CuPy
                    try:
                        self.memory_pool = cp.get_default_memory_pool()
                    except Exception:
                        self.memory_pool = None
                    
                    device_name = self.device_info.get('name', b'Unknown')
                    if isinstance(device_name, bytes):
                        device_name = device_name.decode('utf-8', errors='ignore')
                    print(f"🚀 GPU (CuPy): {device_name} initialized")
                    
                    if 'totalGlobalMem' in self.device_info:
                        self.gpu_memory_gb = self.device_info['totalGlobalMem'] / (1024**3)
                        print(f"💾 GPU Memory: {self.gpu_memory_gb:.1f} GB")
                        # Compute optimal batch size based on VRAM
                        # Each 32x32 float32 image = 4KB, plus DCT overhead ~3x
                        # Safe to use ~50% of VRAM for batch processing
                        safe_memory_mb = (self.gpu_memory_gb * 1024 * 0.5)
                        per_image_mb = (self.PHASH_SIZE * self.PHASH_SIZE * 4 * 3) / (1024 * 1024)
                        self.optimal_batch_size = max(32, min(512, int(safe_memory_mb / per_image_mb)))
                        print(f"📊 Optimal batch size: {self.optimal_batch_size} images")
                        
                elif BACKEND == 'torch':
                    import torch as _torch
                    device_idx = _torch.cuda.current_device()
                    self.device_info = {
                        'name': _torch.cuda.get_device_name(device_idx)
                    }
                    self.memory_pool = None
                    self.gpu_memory_gb = _torch.cuda.get_device_properties(device_idx).total_memory / (1024**3)
                    # Compute optimal batch size for PyTorch
                    safe_memory_mb = (self.gpu_memory_gb * 1024 * 0.5)
                    per_image_mb = (self.PHASH_SIZE * self.PHASH_SIZE * 4 * 3) / (1024 * 1024)
                    self.optimal_batch_size = max(32, min(512, int(safe_memory_mb / per_image_mb)))
                    print(f"🚀 GPU (PyTorch): {self.device_info['name']} initialized")
                    print(f"💾 GPU Memory: {self.gpu_memory_gb:.1f} GB")
                    print(f"📊 Optimal batch size: {self.optimal_batch_size} images")
            except Exception as e:
                print(f"⚠️ GPU initialization failed: {e}")
                self.cuda_available = False
                
    def is_cuda_available(self) -> bool:
        """Check if CUDA is available and working"""
        return self.cuda_available
        
    def get_gpu_memory_info(self) -> dict:
        """Get GPU memory usage information"""
        if not self.cuda_available:
            return {"status": "CUDA not available"}
        try:
            if BACKEND == 'cupy':
                meminfo = cp.cuda.runtime.memGetInfo()
                free_memory = meminfo[0]
                total_memory = meminfo[1]
                used_memory = total_memory - free_memory
                return {
                    "total_gb": total_memory / (1024**3),
                    "used_gb": used_memory / (1024**3),
                    "free_gb": free_memory / (1024**3),
                    "usage_percent": (used_memory / total_memory) * 100
                }
            elif BACKEND == 'torch':
                import torch as _torch
                total = _torch.cuda.get_device_properties(_torch.cuda.current_device()).total_memory
                allocated = _torch.cuda.memory_allocated()
                reserved = _torch.cuda.memory_reserved()
                free = total - reserved
                used = allocated
                return {
                    "total_gb": total / (1024**3),
                    "used_gb": used / (1024**3),
                    "free_gb": free / (1024**3),
                    "usage_percent": (used / total) * 100 if total > 0 else 0
                }
        except Exception as e:
            return {"error": str(e)}
            
    def batch_perceptual_hash_gpu(self, image_paths: List[str], batch_size: Optional[int] = None) -> List[str]:
        """
        Compute perceptual hashes for multiple images using TRUE GPU batch processing.
        
        This implementation:
        1. Decodes all images on CPU first
        2. Stacks them into a single (N, H, W) tensor
        3. Transfers once to GPU
        4. Runs batched DCT using cupyx.scipy.fft.dctn
        5. Vectorized threshold and hash generation
        
        Args:
            image_paths: List of image file paths
            batch_size: Images per GPU batch (default: auto-computed from VRAM)
            
        Returns:
            List of 16-character hex hash strings (64-bit pHash)
        """
        if not self.cuda_available or not HAS_PILLOW:
            return self._batch_perceptual_hash_cpu(image_paths)
        
        # Use optimal batch size if not specified
        if batch_size is None:
            batch_size = self.optimal_batch_size
            
        all_hashes: List[str] = []
        total_images = len(image_paths)
        
        try:
            for batch_start in range(0, total_images, batch_size):
                batch_end = min(batch_start + batch_size, total_images)
                batch_paths = image_paths[batch_start:batch_end]
                
                # Process this batch
                batch_hashes = self._process_true_batch_gpu(batch_paths)
                all_hashes.extend(batch_hashes)
                
                # Clear GPU memory between batches for large datasets
                if batch_start > 0 and batch_start % (batch_size * 4) == 0:
                    self._clear_gpu_memory()
                    
        except Exception as e:
            print(f"[GPU] ⚠️ Batch processing failed: {e!r}, falling back to CPU")
            return self._batch_perceptual_hash_cpu(image_paths)
            
        return all_hashes
    
    def _process_true_batch_gpu(self, image_paths: List[str]) -> List[str]:
        """
        Process a batch with TRUE GPU parallelism - single transfer, batched DCT.
        
        This is the high-performance path that achieves 10-50x speedup on large batches.
        """
        n = len(image_paths)
        
        # Phase 1: Decode all images on CPU and collect into numpy array
        cpu_images = []
        valid_indices = []  # Track which indices had valid images
        
        for idx, path in enumerate(image_paths):
            try:
                if not os.path.exists(path):
                    continue
                with Image.open(path) as img:
                    # Convert to grayscale and resize to standard pHash size
                    gray = img.convert("L").resize((self.PHASH_SIZE, self.PHASH_SIZE), Image.Resampling.LANCZOS)
                    cpu_images.append(np.array(gray, dtype=np.float32))
                    valid_indices.append(idx)
            except Exception as e:
                # Skip corrupted/unreadable images silently
                pass
        
        # If no valid images, return empty hashes
        if not cpu_images:
            return [""] * n
        
        # Phase 2: Stack into (N_valid, 32, 32) array - still on CPU
        batch_cpu = np.stack(cpu_images, axis=0)
        
        # Phase 3: Single host→device transfer
        if BACKEND == 'cupy':
            hashes = self._compute_phash_batch_cupy(batch_cpu)
        elif BACKEND == 'torch':
            hashes = self._compute_phash_batch_torch(batch_cpu)
        else:
            # Fallback - shouldn't happen if cuda_available is True
            return self._batch_perceptual_hash_cpu(image_paths)
        
        # Phase 4: Map hashes back to original indices
        result = [""] * n
        for i, orig_idx in enumerate(valid_indices):
            if i < len(hashes):
                result[orig_idx] = hashes[i]
                
        return result
    
    def _compute_phash_batch_cupy(self, batch_cpu: np.ndarray) -> List[str]:
        """
        Compute pHash for batch using CuPy with true GPU parallelism.
        
        Uses cupyx.scipy.fft.dctn for proper Type-II DCT (not FFT approximation).
        """
        try:
            from cupyx.scipy.fft import dctn
        except ImportError:
            # Fallback to FFT-based approximation if dctn not available
            return self._compute_phash_batch_cupy_fft(batch_cpu)
        
        # Single host→device transfer: (N, 32, 32)
        batch_gpu = cp.asarray(batch_cpu)
        
        # Batched Type-II DCT over last 2 dimensions (the image H, W)
        # This runs the entire batch in parallel on GPU
        dct_batch = dctn(batch_gpu, type=2, norm="ortho", axes=(-2, -1))
        
        # Extract low-frequency 8x8 block: (N, 8, 8)
        dct_low = dct_batch[:, :self.DCT_LOW_FREQ, :self.DCT_LOW_FREQ]
        
        # Compute per-image median for thresholding (vectorized)
        # Reshape to (N, 64) for median computation
        dct_flat = dct_low.reshape(dct_low.shape[0], -1)
        medians = cp.median(dct_flat, axis=1)  # (N,)
        
        # Vectorized binary threshold: (N, 64) bool
        binary = (dct_flat > medians[:, None])
        
        # Transfer only final bits to CPU (minimal data movement)
        binary_cpu = binary.get()  # (N, 64) bool array on CPU
        
        # Convert to hex hash strings
        hashes = []
        for bits in binary_cpu:
            # Convert 64 bits to 16-char hex string
            bit_str = "".join("1" if b else "0" for b in bits)
            hash_int = int(bit_str, 2)
            hashes.append(format(hash_int, "016x"))
            
        return hashes
    
    def _compute_phash_batch_cupy_fft(self, batch_cpu: np.ndarray) -> List[str]:
        """
        Fallback: Compute pHash using FFT approximation if dctn unavailable.
        """
        batch_gpu = cp.asarray(batch_cpu)
        
        # Use 2D FFT as DCT approximation
        fft_result = cp.fft.fft2(batch_gpu)
        dct_approx = cp.real(fft_result)[:, :self.DCT_LOW_FREQ, :self.DCT_LOW_FREQ]
        
        # Vectorized thresholding
        dct_flat = dct_approx.reshape(dct_approx.shape[0], -1)
        medians = cp.median(dct_flat, axis=1)
        binary = (dct_flat > medians[:, None])
        binary_cpu = binary.get()
        
        hashes = []
        for bits in binary_cpu:
            bit_str = "".join("1" if b else "0" for b in bits)
            hash_int = int(bit_str, 2)
            hashes.append(format(hash_int, "016x"))
            
        return hashes
    
    def _compute_phash_batch_torch(self, batch_cpu: np.ndarray) -> List[str]:
        """
        Compute pHash for batch using PyTorch CUDA with true GPU parallelism.
        
        Uses torch.fft for DCT approximation since torch doesn't have native DCT.
        """
        import torch as _torch
        
        # Single host→device transfer
        batch_gpu = _torch.from_numpy(batch_cpu).cuda()
        
        # 2D FFT as DCT approximation
        fft_result = _torch.fft.fft2(batch_gpu)
        dct_approx = _torch.real(fft_result)[:, :self.DCT_LOW_FREQ, :self.DCT_LOW_FREQ]
        
        # Vectorized thresholding
        dct_flat = dct_approx.reshape(dct_approx.shape[0], -1)
        medians = _torch.median(dct_flat, dim=1).values
        binary = (dct_flat > medians[:, None])
        
        # Transfer to CPU
        binary_cpu = binary.cpu().numpy()
        
        hashes = []
        for bits in binary_cpu:
            bit_str = "".join("1" if b else "0" for b in bits)
            hash_int = int(bit_str, 2)
            hashes.append(format(hash_int, "016x"))
            
        return hashes
            
    def _batch_perceptual_hash_cpu(self, image_paths: List[str]) -> List[str]:
        """Fallback CPU implementation for perceptual hashing.
        
        Uses imagehash library if available, otherwise manual DCT-based pHash.
        """
        if not HAS_PILLOW:
            return [""] * len(image_paths)
            
        results = []
        for path in image_paths:
            try:
                if not os.path.exists(path):
                    results.append("")
                    continue
                    
                with Image.open(path) as img:
                    # Convert to grayscale if needed
                    if img.mode != 'L':
                        img = img.convert('L')
                    
                    if HAS_IMAGEHASH:
                        # Use imagehash library if available
                        phash = imagehash.phash(img)
                        results.append(str(phash))
                    else:
                        # Manual CPU pHash implementation
                        img_resized = img.resize((self.PHASH_SIZE, self.PHASH_SIZE), Image.Resampling.LANCZOS)
                        pixels = np.array(img_resized, dtype=np.float32)
                        
                        # Simple DCT approximation using FFT
                        from scipy import fft
                        dct = fft.dctn(pixels, type=2, norm='ortho')
                        dct_low = dct[:self.DCT_LOW_FREQ, :self.DCT_LOW_FREQ].flatten()
                        
                        # Binary hash based on median
                        median = np.median(dct_low)
                        binary = dct_low > median
                        bit_str = "".join("1" if b else "0" for b in binary)
                        hash_int = int(bit_str, 2)
                        results.append(format(hash_int, "016x"))
                    
            except Exception as e:
                results.append("")
                
        return results
        
    def similarity_matrix_gpu(
        self, 
        hashes: List[str], 
        threshold: float = None,
        file_type: str = "images",
        use_hamming_threshold: bool = True
    ) -> List[Tuple[int, int, float, int]]:
        """
        Compute similarity matrix for hashes using GPU acceleration.
        
        Args:
            hashes: List of perceptual hash strings
            threshold: Similarity threshold (0.0-1.0) or None to use PHashThresholds
            file_type: File type for threshold lookup (when threshold is None)
            use_hamming_threshold: If True, use Hamming distance from PHashThresholds
            
        Returns:
            List of (index1, index2, similarity_score, hamming_distance) tuples
        """
        if not self.cuda_available or len(hashes) < 2:
            return self._similarity_matrix_cpu(hashes, threshold, file_type, use_hamming_threshold)
        
        # Get threshold
        if use_hamming_threshold and threshold is None:
            max_hamming = PHashThresholds.get_threshold(file_type)
            # Convert to normalized threshold (64-bit hash)
            threshold = 1.0 - (max_hamming / 64.0)
        elif threshold is None:
            threshold = 0.8  # Default fallback
            
        try:
            # Convert hashes to binary matrices
            binary_hashes = []
            for hash_str in hashes:
                if hash_str:
                    # Handle both hex (16 char) and binary (64 char) formats
                    if len(hash_str) == 16:
                        # Hex format - convert to binary
                        try:
                            int_val = int(hash_str, 16)
                            binary = [(int_val >> i) & 1 for i in range(63, -1, -1)]
                        except ValueError:
                            binary = [0] * 64
                    else:
                        binary = [1 if c == '1' else 0 for c in hash_str[:64]]
                        while len(binary) < 64:
                            binary.append(0)
                    binary_hashes.append(binary[:64])
                else:
                    binary_hashes.append([0] * 64)

            if not binary_hashes:
                return []

            if BACKEND == 'cupy':
                gpu_hashes = cp.array(binary_hashes, dtype=cp.float32)
                similarities, hamming_dists = self._compute_pairwise_similarities_gpu(gpu_hashes)
                cpu_similarities = cp.asnumpy(similarities)
                cpu_hamming = cp.asnumpy(hamming_dists)
            elif BACKEND == 'torch':
                import torch as _torch
                gpu_hashes = _torch.tensor(binary_hashes, dtype=_torch.float32).cuda()
                n = gpu_hashes.shape[0]
                h1 = gpu_hashes[:, None, :]
                h2 = gpu_hashes[None, :, :]
                hamming = _torch.sum(_torch.abs(h1 - h2), dim=2)
                max_distance = gpu_hashes.shape[1]
                similarities = 1.0 - (hamming / float(max_distance))
                cpu_similarities = similarities.cpu().numpy()
                cpu_hamming = hamming.cpu().numpy().astype(int)
            else:
                return self._similarity_matrix_cpu(hashes, threshold, file_type, use_hamming_threshold)

            similar_pairs = []
            for i in range(len(hashes)):
                for j in range(i + 1, len(hashes)):
                    similarity = float(cpu_similarities[i, j])
                    hamming_dist = int(cpu_hamming[i, j])
                    if similarity >= threshold:
                        similar_pairs.append((i, j, similarity, hamming_dist))

            return similar_pairs
            
        except Exception as e:
            print(f"GPU similarity computation failed: {e}")
            return self._similarity_matrix_cpu(hashes, threshold, file_type, use_hamming_threshold)
            
    def _compute_pairwise_similarities_gpu(self, binary_hashes: 'cp.ndarray') -> Tuple['cp.ndarray', 'cp.ndarray']:
        """Compute pairwise Hamming similarities on GPU"""
        n = binary_hashes.shape[0]

        # Expand dimensions for broadcasting
        hash1 = binary_hashes[:, None, :]  # Shape: (n, 1, 64)
        hash2 = binary_hashes[None, :, :]  # Shape: (1, n, 64)

        # Compute Hamming distance
        hamming_distances = cp.sum(cp.abs(hash1 - hash2), axis=2)

        # Convert to similarity (1 - normalized_distance)
        max_distance = binary_hashes.shape[1]  # 64 bits
        similarities = 1.0 - (hamming_distances / max_distance)

        return similarities, hamming_distances.astype(cp.int32)
        
    def _similarity_matrix_cpu(
        self, 
        hashes: List[str], 
        threshold: float,
        file_type: str = "images",
        use_hamming_threshold: bool = True
    ) -> List[Tuple[int, int, float, int]]:
        """CPU fallback for similarity computation"""
        # Get threshold
        if use_hamming_threshold and threshold is None:
            max_hamming = PHashThresholds.get_threshold(file_type)
            threshold = 1.0 - (max_hamming / 64.0)
        elif threshold is None:
            threshold = 0.8
            
        similar_pairs = []
        
        for i in range(len(hashes)):
            for j in range(i + 1, len(hashes)):
                if not hashes[i] or not hashes[j]:
                    continue
                
                # Compute Hamming distance using utility function
                dist = hamming_distance(hashes[i], hashes[j])
                similarity = 1.0 - (dist / 64.0)
                
                if similarity >= threshold:
                    similar_pairs.append((i, j, similarity, dist))
                    
        return similar_pairs
        
    def _clear_gpu_memory(self):
        """Clear GPU memory cache"""
        if not self.cuda_available:
            return

        try:
            if BACKEND == 'cupy' and self.memory_pool:
                self.memory_pool.free_all_blocks()
                cp.cuda.Stream.null.synchronize()
            elif BACKEND == 'torch':
                import torch as _torch
                _torch.cuda.empty_cache()
        except Exception as e:
            print(f"GPU memory clear warning: {e}")
                
    def benchmark_performance(self, test_image_paths: List[str] = None, num_test_images: int = 100) -> dict:
        """Benchmark GPU vs CPU performance"""
        if not test_image_paths:
            # Create dummy test images if none provided
            test_image_paths = self._create_test_images(num_test_images)
            
        results = {
            "cuda_available": self.cuda_available,
            "test_images": len(test_image_paths),
            "gpu_memory_info": self.get_gpu_memory_info() if self.cuda_available else None
        }
        
        # CPU benchmark
        start_time = time.time()
        cpu_hashes = self._batch_perceptual_hash_cpu(test_image_paths[:50])  # Limit for speed
        cpu_time = time.time() - start_time
        
        results["cpu_time"] = cpu_time
        results["cpu_images_per_second"] = len(test_image_paths[:50]) / cpu_time if cpu_time > 0 else 0
        
        # GPU benchmark if available
        if self.cuda_available:
            start_time = time.time()
            gpu_hashes = self.batch_perceptual_hash_gpu(test_image_paths[:50])
            gpu_time = time.time() - start_time
            
            results["gpu_time"] = gpu_time
            results["gpu_images_per_second"] = len(test_image_paths[:50]) / gpu_time if gpu_time > 0 else 0
            results["speedup_factor"] = cpu_time / gpu_time if gpu_time > 0 else 0
            
            # Memory usage after processing
            results["gpu_memory_after"] = self.get_gpu_memory_info()
            
        return results
        
    def _create_test_images(self, count: int) -> List[str]:
        """Create dummy test images for benchmarking"""
        import tempfile
        test_paths = []
        
        if not HAS_CV2:
            return test_paths
            
        temp_dir = tempfile.mkdtemp()
        
        for i in range(min(count, 10)):  # Limit actual file creation
            try:
                # Create random test image
                test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                test_path = os.path.join(temp_dir, f"test_image_{i}.jpg")
                cv2.imwrite(test_path, test_img)
                test_paths.append(test_path)
            except Exception as e:
                print(f"Test image creation error: {e}")
                
        return test_paths
        
    def cleanup(self):
        """Cleanup GPU resources"""
        if self.cuda_available:
            self._clear_gpu_memory()
            print("🧹 GPU resources cleaned up")


# =============================================================================
# VIDEO FINGERPRINTING - GPU-Accelerated
# =============================================================================

class VideoFingerprinter:
    """
    GPU-accelerated video fingerprinting using perceptual hashing.
    
    Pipeline:
    1. Extract keyframes on CPU (OpenCV) - fast enough, I/O bound
    2. Batch-send keyframes to GPU for pHash computation
    3. Combine keyframe hashes into a single video signature
    
    This provides consistency between image and video fingerprinting paths.
    """
    
    # Default number of keyframes to extract
    DEFAULT_KEYFRAMES = 5
    
    # Video extensions
    VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v', '.mpeg', '.mpg'}
    
    def __init__(self, gpu_accelerator_instance: GPUAccelerator = None):
        """
        Initialize video fingerprinter.
        
        Args:
            gpu_accelerator_instance: Existing GPUAccelerator to reuse, or None to create new
        """
        self.gpu_accel = gpu_accelerator_instance
        self.has_cv2 = HAS_CV2
        
    def is_video_file(self, file_path: str) -> bool:
        """Check if file is a supported video format."""
        ext = os.path.splitext(file_path)[1].lower()
        return ext in self.VIDEO_EXTENSIONS
    
    def extract_keyframes(
        self, 
        video_path: str, 
        num_frames: int = None,
        as_pil: bool = True
    ) -> List[Any]:
        """
        Extract keyframes from video at evenly-spaced intervals.
        
        Args:
            video_path: Path to video file
            num_frames: Number of keyframes to extract (default: DEFAULT_KEYFRAMES)
            as_pil: If True, return PIL Images; if False, return numpy arrays
            
        Returns:
            List of keyframe images (PIL or numpy)
        """
        if not self.has_cv2:
            print("[Video] ⚠️ OpenCV not available for keyframe extraction")
            return []
        
        if num_frames is None:
            num_frames = self.DEFAULT_KEYFRAMES
            
        frames = []
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"[Video] ⚠️ Cannot open video: {video_path}")
                return []
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            if total_frames < num_frames:
                num_frames = max(1, total_frames)
            
            # Calculate frame positions (evenly distributed)
            if num_frames == 1:
                positions = [total_frames // 2]  # Middle frame
            else:
                # Avoid first and last 5% to skip intros/credits
                start_frame = int(total_frames * 0.05)
                end_frame = int(total_frames * 0.95)
                positions = [
                    start_frame + int((end_frame - start_frame) * i / (num_frames - 1))
                    for i in range(num_frames)
                ]
            
            for pos in positions:
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                ret, frame = cap.read()
                
                if ret and frame is not None:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    if as_pil and HAS_PILLOW:
                        frames.append(Image.fromarray(frame_rgb))
                    else:
                        frames.append(frame_rgb)
            
            cap.release()
            
        except Exception as e:
            print(f"[Video] ⚠️ Keyframe extraction failed for {video_path}: {e!r}")
            
        return frames
    
    def compute_video_fingerprint(
        self, 
        video_path: str, 
        num_keyframes: int = None,
        combine_method: str = "concat"
    ) -> Optional[str]:
        """
        Compute a perceptual fingerprint for a video using GPU-accelerated pHash.
        
        Args:
            video_path: Path to video file
            num_keyframes: Number of keyframes to use
            combine_method: How to combine keyframe hashes:
                - "concat": Concatenate all hashes (default, most discriminative)
                - "xor": XOR all hashes together (single hash output)
                - "average": Average the hash bits (robust to minor changes)
            
        Returns:
            Video fingerprint string, or None if failed
        """
        if num_keyframes is None:
            num_keyframes = self.DEFAULT_KEYFRAMES
        
        # Step 1: Extract keyframes (CPU, I/O bound)
        keyframes = self.extract_keyframes(video_path, num_keyframes, as_pil=True)
        
        if not keyframes:
            return None
        
        # Step 2: Compute pHash for all keyframes using GPU
        keyframe_hashes = self._compute_keyframe_hashes_gpu(keyframes)
        
        if not keyframe_hashes or all(h == "" for h in keyframe_hashes):
            return None
        
        # Filter out empty hashes
        valid_hashes = [h for h in keyframe_hashes if h]
        
        if not valid_hashes:
            return None
        
        # Step 3: Combine hashes into single signature
        if combine_method == "concat":
            # Concatenate all hashes - most discriminative
            combined = "|".join(valid_hashes)
            return f"video:{combined}"
            
        elif combine_method == "xor":
            # XOR all hashes together - single 64-bit hash
            result = 0
            for h in valid_hashes:
                try:
                    result ^= int(h, 16)
                except ValueError:
                    continue
            return f"video_xor:{format(result, '016x')}"
            
        elif combine_method == "average":
            # Average hash bits - robust to minor changes
            return self._average_hashes(valid_hashes)
            
        else:
            # Default to concat
            separator = "|"
            return f"video:{separator.join(valid_hashes)}"
    
    def _compute_keyframe_hashes_gpu(self, keyframes: List[Any]) -> List[str]:
        """
        Compute pHash for keyframes using GPU batch processing.
        
        This reuses the same GPU pipeline as image pHash for consistency.
        """
        if not keyframes:
            return []
        
        # Convert PIL images to numpy arrays for GPU processing
        import tempfile
        temp_paths = []
        
        try:
            # Save keyframes as temp files (required by current GPU pipeline)
            # TODO: Future optimization - add direct numpy array support to GPU pipeline
            temp_dir = tempfile.mkdtemp(prefix="video_keyframes_")
            
            for i, frame in enumerate(keyframes):
                temp_path = os.path.join(temp_dir, f"keyframe_{i}.png")
                if isinstance(frame, Image.Image):
                    frame.save(temp_path)
                elif HAS_PILLOW:
                    Image.fromarray(frame).save(temp_path)
                else:
                    # Fallback to OpenCV
                    if self.has_cv2:
                        cv2.imwrite(temp_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                temp_paths.append(temp_path)
            
            # Use GPU accelerator for batch pHash
            if self.gpu_accel and self.gpu_accel.cuda_available:
                hashes = self.gpu_accel.batch_perceptual_hash_gpu(temp_paths)
            else:
                # Fallback to CPU
                hashes = self._compute_keyframe_hashes_cpu(keyframes)
                
        except Exception as e:
            print(f"[Video] ⚠️ GPU keyframe hashing failed: {e!r}")
            hashes = self._compute_keyframe_hashes_cpu(keyframes)
            
        finally:
            # Cleanup temp files
            for path in temp_paths:
                try:
                    os.remove(path)
                except Exception:
                    pass
            try:
                os.rmdir(temp_dir)
            except Exception:
                pass
        
        return hashes
    
    def _compute_keyframe_hashes_cpu(self, keyframes: List[Any]) -> List[str]:
        """CPU fallback for keyframe hashing."""
        hashes = []
        
        for frame in keyframes:
            try:
                if HAS_IMAGEHASH:
                    if isinstance(frame, Image.Image):
                        img = frame
                    else:
                        img = Image.fromarray(frame)
                    
                    if img.mode != 'L':
                        img = img.convert('L')
                    
                    phash = imagehash.phash(img)
                    hashes.append(str(phash))
                else:
                    # Very basic fallback - average hash
                    if isinstance(frame, Image.Image):
                        img = frame.convert('L').resize((8, 8))
                    else:
                        img = Image.fromarray(frame).convert('L').resize((8, 8))
                    
                    pixels = np.array(img)
                    avg = pixels.mean()
                    bits = (pixels > avg).flatten()
                    hash_val = sum(1 << i for i, b in enumerate(bits) if b)
                    hashes.append(format(hash_val, '016x'))
                    
            except Exception as e:
                hashes.append("")
                
        return hashes
    
    def _average_hashes(self, hashes: List[str]) -> str:
        """
        Compute average hash from multiple hashes.
        
        Each bit position is set if >50% of input hashes have that bit set.
        """
        if not hashes:
            return ""
        
        # Convert to integers
        int_hashes = []
        for h in hashes:
            try:
                int_hashes.append(int(h, 16))
            except ValueError:
                continue
        
        if not int_hashes:
            return ""
        
        # Count bits at each position
        n = len(int_hashes)
        result = 0
        
        for bit in range(64):
            count = sum(1 for h in int_hashes if (h >> bit) & 1)
            if count > n // 2:
                result |= (1 << bit)
        
        return f"video_avg:{format(result, '016x')}"
    
    def are_videos_similar(
        self, 
        fingerprint1: str, 
        fingerprint2: str,
        threshold: int = None
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Compare two video fingerprints for similarity.
        
        Args:
            fingerprint1: First video fingerprint
            fingerprint2: Second video fingerprint
            threshold: Max Hamming distance (default: from PHashThresholds)
            
        Returns:
            Tuple of (is_similar, similarity_score, details)
        """
        if threshold is None:
            threshold = PHashThresholds.get_threshold("video_keyframe")
        
        # Parse fingerprints
        hashes1 = self._parse_fingerprint(fingerprint1)
        hashes2 = self._parse_fingerprint(fingerprint2)
        
        if not hashes1 or not hashes2:
            return False, 0.0, {"error": "Invalid fingerprint"}
        
        # For concatenated fingerprints, compare corresponding keyframes
        if len(hashes1) > 1 and len(hashes2) > 1:
            return self._compare_multi_hash(hashes1, hashes2, threshold)
        
        # For single hashes (xor/average), simple comparison
        dist = hamming_distance(hashes1[0], hashes2[0])
        similarity = 1.0 - (dist / 64.0)
        is_similar = dist <= threshold
        
        return is_similar, similarity, {
            "hamming_distance": dist,
            "threshold": threshold,
            "method": "single_hash"
        }
    
    def _parse_fingerprint(self, fingerprint: str) -> List[str]:
        """Parse video fingerprint into list of hashes."""
        if not fingerprint:
            return []
        
        # Remove prefix
        if fingerprint.startswith("video:"):
            content = fingerprint[6:]
            return content.split("|")
        elif fingerprint.startswith("video_xor:"):
            return [fingerprint[10:]]
        elif fingerprint.startswith("video_avg:"):
            return [fingerprint[10:]]
        elif fingerprint.startswith("scenes:"):
            # Compatibility with old format
            content = fingerprint[7:]
            return content.split("|")
        else:
            # Assume single hash
            return [fingerprint]
    
    def _compare_multi_hash(
        self, 
        hashes1: List[str], 
        hashes2: List[str],
        threshold: int
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Compare multi-hash fingerprints (keyframe sequences).
        
        Uses a sliding window approach to handle slight timing differences.
        """
        n1, n2 = len(hashes1), len(hashes2)
        
        if n1 != n2:
            # Different number of keyframes - use best match approach
            pass
        
        # Compute pairwise distances
        distances = []
        for i, h1 in enumerate(hashes1):
            best_dist = 64
            for j, h2 in enumerate(hashes2):
                dist = hamming_distance(h1, h2)
                if dist < best_dist:
                    best_dist = dist
            distances.append(best_dist)
        
        # Average distance across keyframes
        avg_distance = sum(distances) / len(distances) if distances else 64
        similarity = 1.0 - (avg_distance / 64.0)
        
        # Count how many keyframes are within threshold
        matching_frames = sum(1 for d in distances if d <= threshold)
        match_ratio = matching_frames / len(distances) if distances else 0
        
        # Videos are similar if majority of keyframes match
        is_similar = match_ratio >= 0.6  # 60% of keyframes must match
        
        return is_similar, similarity, {
            "avg_hamming_distance": avg_distance,
            "matching_frames": matching_frames,
            "total_frames": len(distances),
            "match_ratio": match_ratio,
            "frame_distances": distances,
            "threshold": threshold,
            "method": "multi_hash"
        }
    
    def batch_fingerprint_videos(
        self, 
        video_paths: List[str],
        num_keyframes: int = None,
        combine_method: str = "concat"
    ) -> Dict[str, Optional[str]]:
        """
        Compute fingerprints for multiple videos efficiently.
        
        Args:
            video_paths: List of video file paths
            num_keyframes: Keyframes per video
            combine_method: Hash combination method
            
        Returns:
            Dict mapping video path to fingerprint (or None if failed)
        """
        results = {}
        
        # Extract all keyframes first (batch I/O)
        all_keyframes = {}
        for path in video_paths:
            keyframes = self.extract_keyframes(path, num_keyframes)
            all_keyframes[path] = keyframes
        
        # Flatten all keyframes for GPU batch processing
        flat_keyframes = []
        keyframe_map = []  # Track which video each keyframe belongs to
        
        for path, keyframes in all_keyframes.items():
            for kf in keyframes:
                flat_keyframes.append(kf)
                keyframe_map.append(path)
        
        # Batch compute all hashes on GPU
        if flat_keyframes:
            all_hashes = self._compute_keyframe_hashes_gpu(flat_keyframes)
        else:
            all_hashes = []
        
        # Reconstruct per-video hashes
        video_hashes = {path: [] for path in video_paths}
        for i, hash_val in enumerate(all_hashes):
            video_hashes[keyframe_map[i]].append(hash_val)
        
        # Combine hashes for each video
        for path in video_paths:
            hashes = video_hashes.get(path, [])
            valid_hashes = [h for h in hashes if h]
            
            if not valid_hashes:
                results[path] = None
            elif combine_method == "concat":
                separator = "|"
                results[path] = f"video:{separator.join(valid_hashes)}"
            elif combine_method == "xor":
                result = 0
                for h in valid_hashes:
                    try:
                        result ^= int(h, 16)
                    except ValueError:
                        continue
                results[path] = f"video_xor:{format(result, '016x')}"
            elif combine_method == "average":
                results[path] = self._average_hashes(valid_hashes)
            else:
                separator = "|"
                results[path] = f"video:{separator.join(valid_hashes)}"
        
        return results


# =============================================================================
# AUDIO FINGERPRINTING - Hybrid Chromaprint + GPU-accelerated MFCC
# =============================================================================

# Audio processing dependencies
HAS_CHROMAPRINT = False
HAS_LIBROSA = False
HAS_SOUNDFILE = False

# Try to find fpcalc in project directory before importing acoustid
_project_dir = os.path.dirname(os.path.abspath(__file__))
_fpcalc_dirs = [
    os.path.join(_project_dir, 'chromaprint-fpcalc-1.5.1-windows-x86_64'),
    _project_dir,
]
for _dir in _fpcalc_dirs:
    _fpcalc = os.path.join(_dir, 'fpcalc.exe')
    if os.path.exists(_fpcalc):
        os.environ['PATH'] = _dir + os.pathsep + os.environ.get('PATH', '')
        break

try:
    import acoustid
    import subprocess
    import sys
    import os as _os_module
    HAS_CHROMAPRINT = True
    
    # Monkey-patch acoustid to suppress fpcalc stderr output
    # This prevents mpg123 decoder errors from flooding the console
    _original_fingerprint_file = acoustid.fingerprint_file
    
    def _silent_fingerprint_file(path, maxlength=120):
        """Wrapper to suppress fpcalc stderr (mpg123 errors)"""
        try:
            # Save original stderr
            old_stderr = sys.stderr
            
            try:
                # Redirect stderr to null during fpcalc execution
                sys.stderr = open(_os_module.devnull, 'w')
                result = _original_fingerprint_file(path, maxlength)
                return result
            finally:
                # Restore stderr
                if sys.stderr != old_stderr:
                    sys.stderr.close()
                sys.stderr = old_stderr
                
        except Exception:
            # Silently ignore fpcalc failures (corrupted audio)
            return None, None
    
    # Apply the patch
    acoustid.fingerprint_file = _silent_fingerprint_file
    
except (ImportError, AttributeError) as e:
    acoustid = None
    HAS_CHROMAPRINT = False

try:
    import librosa
    import warnings
    
    # Suppress librosa deprecation warnings
    warnings.filterwarnings('ignore', category=FutureWarning, module='librosa')
    warnings.filterwarnings('ignore', category=FutureWarning, message='.*audioread.*')
    warnings.filterwarnings('ignore', category=UserWarning, module='librosa')
    
    HAS_LIBROSA = True
except ImportError:
    librosa = None
    HAS_LIBROSA = False

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    sf = None


class AudioFingerprinter:
    """
    Hybrid audio fingerprinting with Chromaprint + GPU-accelerated MFCC.
    
    Pipeline:
    1. **Quick filter**: File size similarity check (±10%)
    2. **Chromaprint**: Fast perceptual fingerprinting (catches 95% of duplicates)
       - Format-agnostic: handles MP3→FLAC, bitrate changes, etc.
    3. **MFCC fallback**: GPU-accelerated spectral features for edge cases
       - More robust: catches different masterings, slight tempo changes
    
    Chromaprint uses the AcoustID algorithm - same as MusicBrainz.
    MFCC (Mel-frequency cepstral coefficients) captures timbral characteristics.
    """
    
    # Supported audio extensions
    AUDIO_EXTENSIONS = {
        '.mp3', '.flac', '.wav', '.aac', '.ogg', '.wma', '.m4a', 
        '.opus', '.aiff', '.ape', '.alac', '.wv'
    }
    
    # MFCC parameters
    MFCC_N_COEFFS = 13       # Number of MFCC coefficients (standard)
    MFCC_SAMPLE_RATE = 22050  # Sample rate for analysis
    MFCC_DURATION = 30        # Seconds to analyze (first 30s usually enough)
    MFCC_HOP_LENGTH = 512     # Hop length for STFT
    MFCC_N_FFT = 2048         # FFT window size
    
    # Similarity thresholds
    CHROMAPRINT_THRESHOLD = 0.85   # Chromaprint similarity (0-1)
    MFCC_THRESHOLD = 0.80          # MFCC cosine similarity (0-1)
    
    def __init__(self):
        self.has_chromaprint = HAS_CHROMAPRINT
        self.has_librosa = HAS_LIBROSA
        self.has_soundfile = HAS_SOUNDFILE
        self.cuda_available = HAS_CUDA
        
        # Try to find fpcalc in the project directory
        self._setup_fpcalc()
        
    def _setup_fpcalc(self):
        """Setup fpcalc path for Chromaprint fingerprinting."""
        if self.has_chromaprint:
            # Check if fpcalc is in PATH first
            try:
                import subprocess
                result = subprocess.run(['fpcalc', '--version'], 
                                       capture_output=True, timeout=5)
                if result.returncode == 0:
                    return  # fpcalc already in PATH
            except Exception:
                pass
            
            # Try to find fpcalc in project directory
            project_dir = os.path.dirname(os.path.abspath(__file__))
            fpcalc_dirs = [
                os.path.join(project_dir, 'chromaprint-fpcalc-1.5.1-windows-x86_64'),
                project_dir,
            ]
            
            for dir_path in fpcalc_dirs:
                fpcalc_path = os.path.join(dir_path, 'fpcalc.exe')
                if os.path.exists(fpcalc_path):
                    # Add to PATH for acoustid
                    os.environ['PATH'] = dir_path + os.pathsep + os.environ.get('PATH', '')
                    print(f"[Audio] ✅ Found fpcalc at: {fpcalc_path}")
                    return
            
            # Check if fpcalc.exe exists without .exe extension lookup
            for dir_path in fpcalc_dirs:
                fpcalc_path = os.path.join(dir_path, 'fpcalc')
                if os.path.exists(fpcalc_path):
                    os.environ['PATH'] = dir_path + os.pathsep + os.environ.get('PATH', '')
                    print(f"[Audio] ✅ Found fpcalc at: {fpcalc_path}")
                    return
        
    def is_audio_file(self, file_path: str) -> bool:
        """Check if file is a supported audio format."""
        ext = os.path.splitext(file_path)[1].lower()
        return ext in self.AUDIO_EXTENSIONS
    
    def compute_fingerprint(
        self, 
        audio_path: str,
        method: str = "hybrid"
    ) -> Optional[Dict[str, Any]]:
        """
        Compute audio fingerprint using specified method.
        
        Args:
            audio_path: Path to audio file
            method: "chromaprint", "mfcc", or "hybrid" (both)
            
        Returns:
            Dict with fingerprint data, or None if failed
        """
        if not os.path.exists(audio_path):
            return None
            
        result = {
            "path": audio_path,
            "chromaprint": None,
            "mfcc": None,
            "duration": None
        }
        
        # Chromaprint fingerprinting
        if method in ("chromaprint", "hybrid") and self.has_chromaprint:
            cp_result = self._compute_chromaprint(audio_path)
            if cp_result:
                result["chromaprint"] = cp_result["fingerprint"]
                result["duration"] = cp_result["duration"]
        
        # MFCC fingerprinting
        if method in ("mfcc", "hybrid") and self.has_librosa:
            mfcc_result = self._compute_mfcc_fingerprint(audio_path)
            if mfcc_result is not None:
                # Convert to list for JSON serialization
                result["mfcc"] = mfcc_result.tolist() if isinstance(mfcc_result, np.ndarray) else mfcc_result
        
        # Return None if both methods failed
        if result["chromaprint"] is None and result["mfcc"] is None:
            return None
            
        return result
    
    def _compute_chromaprint(self, audio_path: str) -> Optional[Dict[str, Any]]:
        """
        Compute Chromaprint (AcoustID) fingerprint.
        
        This is the primary fingerprinting method - fast and robust.
        Handles format conversions, bitrate changes, volume normalization.
        """
        if not self.has_chromaprint:
            return None
            
        try:
            # acoustid.fingerprint_file returns (duration, fingerprint)
            duration, fingerprint = acoustid.fingerprint_file(audio_path)
            
            # Silently ignore if fingerprint_file returns None (corrupted audio)
            if duration is None or fingerprint is None:
                return None
            
            return {
                "fingerprint": fingerprint,
                "duration": duration
            }
        except Exception:
            # Silently ignore failures - usually corrupted/unsupported audio
            return None
    
    def _compute_mfcc_fingerprint(self, audio_path: str) -> Optional[np.ndarray]:
        """
        Compute MFCC-based fingerprint with GPU acceleration.
        
        MFCCs capture the timbral/spectral characteristics of audio.
        We compute mean + std of MFCCs across time for a fixed-size fingerprint.
        
        GPU acceleration is applied to:
        1. FFT computation (via CuPy if available)
        2. Matrix operations for normalization
        """
        if not self.has_librosa:
            return None
            
        try:
            # Suppress librosa/audioread warnings during load
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning)
                warnings.filterwarnings('ignore', message='.*Xing.*')
                warnings.filterwarnings('ignore', message='.*PySoundFile.*')
                
                # Load audio (first N seconds, mono, resampled)
                y, sr = librosa.load(
                    audio_path, 
                    sr=self.MFCC_SAMPLE_RATE, 
                    mono=True, 
                    duration=self.MFCC_DURATION
                )
            
            if len(y) == 0:
                return None
            
            # Compute MFCCs
            if self.cuda_available and BACKEND == 'cupy':
                mfccs = self._compute_mfcc_gpu(y, sr)
            else:
                mfccs = self._compute_mfcc_cpu(y, sr)
            
            if mfccs is None or mfccs.size == 0:
                return None
            
            # Create fixed-size fingerprint: mean + std of each coefficient
            # Shape: (13,) + (13,) = (26,) dimensional fingerprint
            mfcc_mean = np.mean(mfccs, axis=1)
            mfcc_std = np.std(mfccs, axis=1)
            fingerprint = np.concatenate([mfcc_mean, mfcc_std])
            
            # Normalize to unit vector for cosine similarity
            norm = np.linalg.norm(fingerprint)
            if norm > 0:
                fingerprint = fingerprint / norm
                
            return fingerprint
            
        except Exception:
            # Silently ignore failures - usually corrupted/unsupported audio
            return None
    
    def _compute_mfcc_gpu(self, y: np.ndarray, sr: int) -> Optional[np.ndarray]:
        """
        GPU-accelerated MFCC computation using CuPy.
        
        Accelerates:
        - STFT (Short-Time Fourier Transform)
        - Mel filterbank application
        - DCT for cepstral coefficients
        """
        try:
            # Transfer audio to GPU
            y_gpu = cp.asarray(y)
            
            # Compute STFT on GPU
            # Using librosa's STFT but with GPU arrays where possible
            # Note: librosa doesn't directly support CuPy, so we do hybrid
            
            # For now, use librosa on CPU for STFT (complex operation)
            # and GPU for mel filterbank + DCT
            stft = librosa.stft(
                y, 
                n_fft=self.MFCC_N_FFT, 
                hop_length=self.MFCC_HOP_LENGTH
            )
            
            # Magnitude spectrum
            S = np.abs(stft) ** 2
            
            # Transfer to GPU for mel filterbank
            S_gpu = cp.asarray(S)
            
            # Create mel filterbank on GPU
            n_mels = 128
            mel_basis = librosa.filters.mel(
                sr=sr, 
                n_fft=self.MFCC_N_FFT, 
                n_mels=n_mels
            )
            mel_basis_gpu = cp.asarray(mel_basis)
            
            # Apply mel filterbank (GPU matrix multiply)
            mel_spec_gpu = cp.dot(mel_basis_gpu, S_gpu)
            
            # Log mel spectrogram
            mel_spec_gpu = cp.log(mel_spec_gpu + 1e-10)
            
            # DCT on GPU to get MFCCs
            try:
                from cupyx.scipy.fft import dct
                mfccs_gpu = dct(mel_spec_gpu, type=2, axis=0, norm='ortho')[:self.MFCC_N_COEFFS]
            except ImportError:
                # Fallback: transfer to CPU for DCT
                mel_spec_cpu = cp.asnumpy(mel_spec_gpu)
                from scipy.fft import dct
                mfccs_cpu = dct(mel_spec_cpu, type=2, axis=0, norm='ortho')[:self.MFCC_N_COEFFS]
                return mfccs_cpu
            
            # Transfer result back to CPU
            mfccs = cp.asnumpy(mfccs_gpu)
            return mfccs
            
        except Exception:
            # Silently fall back to CPU
            return self._compute_mfcc_cpu(y, sr)
    
    def _compute_mfcc_cpu(self, y: np.ndarray, sr: int) -> Optional[np.ndarray]:
        """CPU fallback for MFCC computation using librosa."""
        try:
            mfccs = librosa.feature.mfcc(
                y=y, 
                sr=sr, 
                n_mfcc=self.MFCC_N_COEFFS,
                n_fft=self.MFCC_N_FFT,
                hop_length=self.MFCC_HOP_LENGTH
            )
            return mfccs
        except Exception:
            # Silently ignore failures
            return None
    
    def compare_fingerprints(
        self, 
        fp1: Dict[str, Any], 
        fp2: Dict[str, Any],
        method: str = "hybrid"
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Compare two audio fingerprints.
        
        Args:
            fp1, fp2: Fingerprint dicts from compute_fingerprint()
            method: Comparison method ("chromaprint", "mfcc", "hybrid")
            
        Returns:
            Tuple of (is_similar, similarity_score, details)
        """
        details = {
            "chromaprint_score": None,
            "mfcc_score": None,
            "method_used": None,
            "duration_match": None
        }
        
        # Check duration similarity first (quick filter)
        if fp1.get("duration") and fp2.get("duration"):
            d1, d2 = fp1["duration"], fp2["duration"]
            if d1 > 0 and d2 > 0:
                duration_ratio = min(d1, d2) / max(d1, d2)
                details["duration_match"] = duration_ratio
                
                # Skip if durations are too different (>20% difference)
                if duration_ratio < 0.8:
                    return False, 0.0, details
        
        # Chromaprint comparison
        cp_score = None
        if method in ("chromaprint", "hybrid"):
            cp1, cp2 = fp1.get("chromaprint"), fp2.get("chromaprint")
            if cp1 and cp2:
                cp_score = self._compare_chromaprint(cp1, cp2)
                details["chromaprint_score"] = cp_score
                details["method_used"] = "chromaprint"
                
                # If chromaprint matches well, we're done
                if cp_score >= self.CHROMAPRINT_THRESHOLD:
                    return True, cp_score, details
        
        # MFCC comparison (fallback or primary)
        mfcc_score = None
        if method in ("mfcc", "hybrid"):
            mfcc1, mfcc2 = fp1.get("mfcc"), fp2.get("mfcc")
            if mfcc1 is not None and mfcc2 is not None:
                mfcc_score = self._compare_mfcc(mfcc1, mfcc2)
                details["mfcc_score"] = mfcc_score
                
                if mfcc_score >= self.MFCC_THRESHOLD:
                    details["method_used"] = "mfcc" if cp_score is None else "hybrid"
                    return True, mfcc_score, details
        
        # Combine scores if both available
        if cp_score is not None and mfcc_score is not None:
            # Weighted average: chromaprint is more reliable
            combined = 0.7 * cp_score + 0.3 * mfcc_score
            details["method_used"] = "hybrid_combined"
            is_similar = combined >= 0.8
            return is_similar, combined, details
        
        # Single method result
        final_score = cp_score if cp_score is not None else (mfcc_score or 0.0)
        return False, final_score, details
    
    def _compare_chromaprint(self, fp1: str, fp2: str) -> float:
        """
        Compare two Chromaprint fingerprints.
        
        Returns similarity score 0.0-1.0.
        
        Uses base64 decoding to compare raw fingerprint data.
        Chromaprint fingerprints are base64-encoded arrays of 32-bit integers.
        """
        if not self.has_chromaprint:
            return 0.0
            
        try:
            import base64
            import struct
            
            # Decode base64 fingerprints to raw bytes
            # Chromaprint uses a custom encoding, try to decode
            def decode_fp(fp_str):
                """Decode chromaprint fingerprint string to int array."""
                try:
                    # Standard base64
                    raw = base64.b64decode(fp_str + '==')  # Add padding if needed
                    # Unpack as 32-bit unsigned integers
                    count = len(raw) // 4
                    return list(struct.unpack(f'<{count}I', raw[:count*4]))
                except Exception:
                    try:
                        raw = base64.b64decode(fp_str)
                        count = len(raw) // 4
                        return list(struct.unpack(f'<{count}I', raw[:count*4]))
                    except Exception:
                        return None
            
            raw1 = decode_fp(fp1)
            raw2 = decode_fp(fp2)
            
            if not raw1 or not raw2:
                # Fallback to string comparison
                return self._compare_fingerprint_strings(fp1, fp2)
            
            # Compare using bit population count (Hamming distance)
            min_len = min(len(raw1), len(raw2))
            if min_len == 0:
                return 0.0
            
            # Count matching bits
            total_bits = 0
            matching_bits = 0
            
            for i in range(min_len):
                xor = raw1[i] ^ raw2[i]
                diff_bits = bin(xor).count('1')
                matching_bits += 32 - diff_bits
                total_bits += 32
            
            similarity = matching_bits / total_bits if total_bits > 0 else 0.0
            return similarity
            
        except Exception as e:
            # Fallback to string comparison
            return self._compare_fingerprint_strings(fp1, fp2)
    
    def _compare_fingerprint_strings(self, fp1: str, fp2: str) -> float:
        """Simple string-based fingerprint comparison fallback."""
        if not fp1 or not fp2:
            return 0.0
        
        min_len = min(len(fp1), len(fp2))
        max_len = max(len(fp1), len(fp2))
        
        if min_len == 0:
            return 0.0
        
        matches = sum(c1 == c2 for c1, c2 in zip(fp1[:min_len], fp2[:min_len]))
        
        length_penalty = min_len / max_len
        char_similarity = matches / min_len
        
        return char_similarity * 0.7 + length_penalty * 0.3
    
    def _compare_mfcc(
        self, 
        mfcc1: Any, 
        mfcc2: Any
    ) -> float:
        """
        Compare two MFCC fingerprints using cosine similarity.
        
        GPU-accelerated for batch comparisons.
        """
        try:
            # Convert to numpy arrays if needed
            if isinstance(mfcc1, list):
                mfcc1 = np.array(mfcc1)
            if isinstance(mfcc2, list):
                mfcc2 = np.array(mfcc2)
            
            # Ensure same shape
            if mfcc1.shape != mfcc2.shape:
                return 0.0
            
            # Cosine similarity (fingerprints should already be normalized)
            if self.cuda_available and BACKEND == 'cupy' and len(mfcc1) > 100:
                # GPU for large batches
                m1_gpu = cp.asarray(mfcc1)
                m2_gpu = cp.asarray(mfcc2)
                similarity = float(cp.dot(m1_gpu, m2_gpu))
            else:
                # CPU
                similarity = float(np.dot(mfcc1, mfcc2))
            
            # Clamp to [0, 1]
            return max(0.0, min(1.0, similarity))
            
        except Exception:
            # Silently return 0 similarity on comparison failure
            return 0.0
    
    def batch_fingerprint_audio(
        self, 
        audio_paths: List[str],
        method: str = "hybrid"
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        """
        Compute fingerprints for multiple audio files.
        
        Args:
            audio_paths: List of audio file paths
            method: Fingerprinting method
            
        Returns:
            Dict mapping path to fingerprint (or None if failed)
        """
        results = {}
        
        for path in audio_paths:
            results[path] = self.compute_fingerprint(path, method)
            
        return results
    
    def find_similar_audio(
        self, 
        audio_paths: List[str],
        method: str = "hybrid",
        threshold: float = None
    ) -> List[Tuple[str, str, float, Dict[str, Any]]]:
        """
        Find all pairs of similar audio files.
        
        Args:
            audio_paths: List of audio file paths
            method: Comparison method
            threshold: Similarity threshold (default: based on method)
            
        Returns:
            List of (path1, path2, similarity, details) tuples
        """
        if threshold is None:
            threshold = self.CHROMAPRINT_THRESHOLD if method == "chromaprint" else self.MFCC_THRESHOLD
        
        # Compute all fingerprints
        fingerprints = self.batch_fingerprint_audio(audio_paths, method)
        
        # Filter out failed fingerprints
        valid_fps = {k: v for k, v in fingerprints.items() if v is not None}
        paths = list(valid_fps.keys())
        
        similar_pairs = []
        
        # Pairwise comparison
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                p1, p2 = paths[i], paths[j]
                is_similar, score, details = self.compare_fingerprints(
                    valid_fps[p1], valid_fps[p2], method
                )
                
                if score >= threshold:
                    similar_pairs.append((p1, p2, score, details))
        
        return similar_pairs
    
    def get_status(self) -> Dict[str, Any]:
        """Get audio fingerprinting status."""
        return {
            "available": self.has_chromaprint or self.has_librosa,
            "chromaprint_available": self.has_chromaprint,
            "librosa_available": self.has_librosa,
            "gpu_accelerated": self.cuda_available and self.has_librosa,
            "backend": BACKEND if self.cuda_available else "cpu",
            "supported_formats": list(self.AUDIO_EXTENSIONS)
        }


# Global audio fingerprinter instance
audio_fingerprinter = AudioFingerprinter()


# =============================================================================
# SEMANTIC SIMILARITY - GPU-accelerated text embeddings
# =============================================================================

# Lazy-loaded singleton for sentence transformer model
_SEMANTIC_MODEL = None
_SEMANTIC_MODEL_DEVICE = None
HAS_SENTENCE_TRANSFORMERS = False

try:
    if DEP_HAS_TRANSFORMERS and DepSentenceTransformer is not None:
        SentenceTransformer = DepSentenceTransformer
        HAS_SENTENCE_TRANSFORMERS = True
except Exception:
    pass


def get_semantic_model():
    """
    Get the singleton semantic similarity model.
    
    Lazy-loads the model on first call to avoid startup overhead.
    Uses CUDA if available for GPU-accelerated embeddings.
    
    Returns:
        SentenceTransformer model or None if not available.
    """
    global _SEMANTIC_MODEL, _SEMANTIC_MODEL_DEVICE
    
    if not HAS_SENTENCE_TRANSFORMERS:
        return None
    
    if _SEMANTIC_MODEL is None:
        # Determine device
        if HAS_CUDA and BACKEND == 'torch' and torch is not None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif HAS_CUDA and BACKEND == 'cupy':
            # CuPy available means CUDA is available, use it
            device = "cuda"
        else:
            device = "cpu"
        
        try:
            print(f"[Semantic] Loading SentenceTransformer model on {device}...")
            _SEMANTIC_MODEL = SentenceTransformer('all-MiniLM-L6-v2', device=device)
            _SEMANTIC_MODEL_DEVICE = device
            print(f"[Semantic] ✅ Model loaded successfully on {device}")
        except Exception as e:
            print(f"[Semantic] ⚠️ Failed to load model: {e}")
            # Try CPU fallback
            try:
                print("[Semantic] Attempting CPU fallback...")
                _SEMANTIC_MODEL = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")
                _SEMANTIC_MODEL_DEVICE = "cpu"
                print("[Semantic] ✅ Model loaded on CPU")
            except Exception as e2:
                print(f"[Semantic] ❌ CPU fallback failed: {e2}")
                return None
    
    return _SEMANTIC_MODEL


def compute_semantic_similarity(text1: str, text2: str) -> float:
    """
    Compute semantic similarity between two texts using sentence embeddings.
    
    Uses GPU acceleration if available for fast embedding computation.
    Returns cosine similarity (0.0 to 1.0).
    
    Args:
        text1: First text to compare.
        text2: Second text to compare.
        
    Returns:
        Cosine similarity score (0.0 = completely different, 1.0 = identical meaning).
    """
    model = get_semantic_model()
    if model is None:
        return 0.0
    
    try:
        # Encode both texts in a single batch for efficiency
        embeddings = model.encode(
            [text1, text2], 
            convert_to_numpy=True, 
            normalize_embeddings=True  # Pre-normalize for fast cosine
        )
        e1, e2 = embeddings
        # Cosine similarity (since embeddings are normalized, dot product = cosine)
        return float((e1 * e2).sum())
    except Exception as e:
        print(f"[Semantic] Error computing similarity: {e}")
        return 0.0


def batch_compute_embeddings(texts: List[str], batch_size: int = 32) -> np.ndarray:
    """
    Compute embeddings for a batch of texts.
    
    Efficient for comparing many texts - compute embeddings once,
    then use cosine similarity matrix.
    
    Args:
        texts: List of texts to embed.
        batch_size: Batch size for encoding (larger = faster but more VRAM).
        
    Returns:
        Numpy array of shape (len(texts), embedding_dim) with normalized embeddings.
    """
    model = get_semantic_model()
    if model is None:
        return np.array([])
    
    try:
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100
        )
        return embeddings
    except Exception as e:
        print(f"[Semantic] Error computing batch embeddings: {e}")
        return np.array([])


def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarity matrix from embeddings.
    
    Args:
        embeddings: Normalized embeddings array of shape (n, dim).
        
    Returns:
        Similarity matrix of shape (n, n) where [i,j] is similarity between i and j.
    """
    if len(embeddings) == 0:
        return np.array([])
    
    # For normalized embeddings, cosine similarity = dot product
    # Use GPU if available for large matrices
    if HAS_CUDA and BACKEND == 'cupy' and len(embeddings) > 100:
        try:
            embeddings_gpu = cp.asarray(embeddings)
            similarity_matrix = cp.dot(embeddings_gpu, embeddings_gpu.T)
            return cp.asnumpy(similarity_matrix)
        except Exception:
            pass
    
    # CPU fallback
    return np.dot(embeddings, embeddings.T)


def find_similar_texts(
    texts: List[str], 
    threshold: float = 0.8,
    batch_size: int = 32
) -> List[Tuple[int, int, float]]:
    """
    Find all pairs of semantically similar texts above a threshold.
    
    Args:
        texts: List of texts to compare.
        threshold: Minimum similarity score (0.0 to 1.0).
        batch_size: Batch size for embedding computation.
        
    Returns:
        List of tuples (idx1, idx2, similarity) for all similar pairs.
    """
    if len(texts) < 2:
        return []
    
    embeddings = batch_compute_embeddings(texts, batch_size)
    if len(embeddings) == 0:
        return []
    
    similarity_matrix = compute_similarity_matrix(embeddings)
    
    # Find pairs above threshold
    similar_pairs = []
    n = len(texts)
    for i in range(n):
        for j in range(i + 1, n):
            sim = similarity_matrix[i, j]
            if sim >= threshold:
                similar_pairs.append((i, j, float(sim)))
    
    return similar_pairs


class SemanticSimilarityEngine:
    """
    GPU-accelerated semantic similarity engine for text comparison.
    
    Uses sentence-transformers with CUDA acceleration when available.
    Optimized for batch processing of many texts.
    """
    
    def __init__(self):
        self.model = None
        self.device = None
        self._initialized = False
    
    def initialize(self) -> bool:
        """
        Initialize the semantic model.
        
        Returns:
            True if initialization successful.
        """
        self.model = get_semantic_model()
        if self.model is not None:
            self.device = _SEMANTIC_MODEL_DEVICE
            self._initialized = True
            return True
        return False
    
    @property
    def is_available(self) -> bool:
        """Check if semantic similarity is available."""
        return HAS_SENTENCE_TRANSFORMERS
    
    @property
    def is_initialized(self) -> bool:
        """Check if model is loaded."""
        return self._initialized and self.model is not None
    
    @property
    def is_gpu_accelerated(self) -> bool:
        """Check if using GPU acceleration."""
        return self.device == "cuda"
    
    def compare(self, text1: str, text2: str) -> float:
        """Compare two texts and return similarity score."""
        if not self._initialized:
            self.initialize()
        return compute_semantic_similarity(text1, text2)
    
    def batch_embed(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Compute embeddings for multiple texts."""
        if not self._initialized:
            self.initialize()
        return batch_compute_embeddings(texts, batch_size)
    
    def find_similar(
        self, 
        texts: List[str], 
        threshold: float = 0.8,
        batch_size: int = 32
    ) -> List[Tuple[int, int, float]]:
        """Find all pairs of similar texts."""
        if not self._initialized:
            self.initialize()
        return find_similar_texts(texts, threshold, batch_size)
    
    def get_status(self) -> Dict[str, Any]:
        """Get engine status information."""
        return {
            "available": self.is_available,
            "initialized": self.is_initialized,
            "device": self.device,
            "gpu_accelerated": self.is_gpu_accelerated,
            "model_name": "all-MiniLM-L6-v2"
        }


# =============================================================================
# PIPELINE STATUS - Comprehensive GPU acceleration logging
# =============================================================================

def get_pipeline_status() -> Dict[str, Dict[str, Any]]:
    """
    Get detailed status of all processing pipelines.
    
    Returns:
        Dictionary with status for each pipeline (gpu_enabled, backend, notes).
    """
    pipelines = {}
    
    # Image pHash pipeline
    pipelines["image_phash"] = {
        "name": "Image pHash",
        "gpu_enabled": HAS_CUDA and BACKEND in ('cupy', 'torch'),
        "backend": BACKEND if HAS_CUDA else "cpu",
        "available": HAS_PILLOW,
        "notes": "Batched DCT-based perceptual hashing"
    }
    
    # Video fingerprinting pipeline
    pipelines["video_fingerprint"] = {
        "name": "Video Keyframes",
        "gpu_enabled": HAS_CUDA and BACKEND in ('cupy', 'torch'),
        "backend": BACKEND if HAS_CUDA else "cpu",
        "available": HAS_PILLOW,  # Uses OpenCV for extraction, GPU for pHash
        "notes": "GPU pHash on extracted keyframes"
    }
    
    # Semantic similarity pipeline
    semantic_gpu = False
    semantic_backend = "cpu"
    if HAS_SENTENCE_TRANSFORMERS:
        # Check if CUDA is available for sentence-transformers
        try:
            import torch as _t
            if _t.cuda.is_available():
                semantic_gpu = True
                semantic_backend = "pytorch-cuda"
        except Exception:
            pass
    
    pipelines["semantic_similarity"] = {
        "name": "Text Similarity",
        "gpu_enabled": semantic_gpu,
        "backend": semantic_backend,
        "available": HAS_SENTENCE_TRANSFORMERS,
        "notes": "SentenceTransformer embeddings"
    }
    
    # Audio fingerprinting (Chromaprint + GPU MFCC)
    audio_gpu = HAS_CUDA and BACKEND in ('cupy', 'torch')  # MFCC uses GPU FFT
    pipelines["audio_fingerprint"] = {
        "name": "Audio Fingerprint",
        "gpu_enabled": audio_gpu and HAS_LIBROSA,
        "backend": BACKEND if (audio_gpu and HAS_LIBROSA) else "cpu",
        "available": HAS_CHROMAPRINT or HAS_LIBROSA,
        "notes": f"{'Chromaprint + ' if HAS_CHROMAPRINT else ''}{'GPU MFCC' if audio_gpu and HAS_LIBROSA else 'CPU MFCC' if HAS_LIBROSA else 'N/A'}"
    }
    
    # Document hashing (always CPU)
    pipelines["document_hash"] = {
        "name": "Documents",
        "gpu_enabled": False,
        "backend": "cpu",
        "available": True,
        "notes": "MD5/SHA256 hashing"
    }
    
    return pipelines


def print_pipeline_status():
    """Print a formatted summary of GPU acceleration status for all pipelines."""
    pipelines = get_pipeline_status()
    
    gpu_pipelines = []
    cpu_pipelines = []
    
    for key, info in pipelines.items():
        if not info["available"]:
            continue
        if info["gpu_enabled"]:
            gpu_pipelines.append(info["name"])
        else:
            cpu_pipelines.append(info["name"])
    
    # Print GPU-accelerated pipelines
    if gpu_pipelines:
        print(f"🚀 GPU accelerated: {', '.join(gpu_pipelines)}")
    
    # Print CPU-only pipelines
    if cpu_pipelines:
        print(f"💻 CPU only: {', '.join(cpu_pipelines)}")


def get_acceleration_summary() -> str:
    """Get a one-line summary of GPU acceleration status."""
    pipelines = get_pipeline_status()
    
    gpu_count = sum(1 for p in pipelines.values() if p["available"] and p["gpu_enabled"])
    cpu_count = sum(1 for p in pipelines.values() if p["available"] and not p["gpu_enabled"])
    
    if gpu_count == 0:
        return "⚠️ No GPU acceleration available"
    elif cpu_count == 0:
        return f"✅ Full GPU acceleration ({gpu_count} pipelines)"
    else:
        return f"✅ Partial GPU acceleration ({gpu_count} GPU, {cpu_count} CPU)"


# Global instances
gpu_accelerator = GPUAccelerator()
video_fingerprinter = VideoFingerprinter(gpu_accelerator)
semantic_engine = SemanticSimilarityEngine()

# Print pipeline status on module load
print_pipeline_status()
