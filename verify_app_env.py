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
        print(f"{name}: FAIL ({exc})")
        return None


def main():
    print("Python:", sys.executable)
    print("Version:", sys.version.replace("\n", " "))
    print()

    deps = try_import("dependencies")
    gpu = try_import("gpu_accelerator")
    app = try_import("main")
    torch = try_import("torch")
    try_import("torchvision")
    try_import("sentence_transformers")
    try_import("pydub")
    try_import("librosa")
    try_import("cupy")
    print()

    if deps is not None:
        print("dependencies.HAS_TRANSFORMERS =", getattr(deps, "HAS_TRANSFORMERS", None))
        print("dependencies.HAS_AUDIO =", getattr(deps, "HAS_AUDIO", None))
    if app is not None:
        print("main.HAS_SEMANTIC_SIMILARITY =", getattr(app, "HAS_SEMANTIC_SIMILARITY", None))
        print("main.HAS_AUDIO_FINGERPRINTER =", getattr(app, "HAS_AUDIO_FINGERPRINTER", None))


if __name__ == "__main__":
    main()
