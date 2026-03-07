#!/usr/bin/env python3
"""Verify optional app environment dependencies."""
import importlib
import sys


def try_import(name):
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "n/a")
        print(f"{name}: OK ({version})")
        return module
    except Exception as exc:
        # VE-2: Show exception type so ImportError vs SyntaxError are distinguishable
        print(f"{name}: FAIL [{type(exc).__name__}] ({exc})")
        return None


def main():
    print("Python:", sys.executable)
    print("Version:", sys.version.replace("\n", " "))
    print()

    # VE-1: Do NOT import 'main' here — doing so executes all module-level code in
    # main.py including Qt widget creation, GPU init, and database connections, which
    # is wrong for a dependency-verification script.
    deps = try_import("dependencies")
    gpu = try_import("gpu_accelerator")
    try_import("torch")
    try_import("torchvision")
    try_import("sentence_transformers")
    try_import("pydub")
    try_import("librosa")
    try_import("cupy")
    # Core required packages
    try_import("PyQt6")
    try_import("xxhash")
    try_import("mutagen")
    try_import("send2trash")
    print()

    if deps is not None:
        print("dependencies.HAS_TRANSFORMERS =", getattr(deps, "HAS_TRANSFORMERS", None))
        print("dependencies.HAS_AUDIO =", getattr(deps, "HAS_AUDIO", None))
        print("dependencies.HAS_CUDA =", getattr(deps, "HAS_CUDA", None))
        print("dependencies.CUDA_BACKEND =", getattr(deps, "CUDA_BACKEND", None))


if __name__ == "__main__":
    main()
