#!/usr/bin/env python3
"""Verify optional app environment dependencies."""
import importlib
import sys


def try_import(name):
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "n/a")
        print(f"{name}: OK ({version})")
        return module, True
    except Exception as exc:
        # VE-2: Show exception type so ImportError vs SyntaxError are distinguishable
        print(f"{name}: FAIL [{type(exc).__name__}] ({exc})")
        return None, False


def main():
    """Run dependency verification. Returns True if all core deps are OK."""
    print("Python:", sys.executable)
    print("Version:", sys.version.replace("\n", " "))
    print()

    all_ok = True
    
    # VE-1: Do NOT import 'main' here — doing so executes all module-level code in
    # main.py including Qt widget creation, GPU init, and database connections, which
    # is wrong for a dependency-verification script.
    deps, _ = try_import("dependencies")
    gpu, _ = try_import("gpu_accelerator")
    try_import("torch")
    try_import("torchvision")
    try_import("sentence_transformers")
    try_import("pydub")
    try_import("librosa")
    try_import("cupy")
    
    # Core required packages - these affect the exit code
    _, qt_ok = try_import("PyQt6")
    _, xxhash_ok = try_import("xxhash")
    _, mutagen_ok = try_import("mutagen")
    _, send2trash_ok = try_import("send2trash")
    
    # VE-FIX: Track core dependency failures
    core_ok = qt_ok and xxhash_ok and mutagen_ok and send2trash_ok
    
    print()

    if deps is not None:
        print("dependencies.HAS_TRANSFORMERS =", getattr(deps, "HAS_TRANSFORMERS", None))
        print("dependencies.HAS_AUDIO =", getattr(deps, "HAS_AUDIO", None))
        print("dependencies.HAS_CUDA =", getattr(deps, "HAS_CUDA", None))
        print("dependencies.CUDA_BACKEND =", getattr(deps, "CUDA_BACKEND", None))

    return core_ok


if __name__ == "__main__":
    # VE-FIX: Return proper exit code
    success = main()
    sys.exit(0 if success else 1)
