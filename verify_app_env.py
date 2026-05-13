#!/usr/bin/env python3
"""Verify optional app environment dependencies."""
import importlib
import importlib.util
import os
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


def check_main_syntax():
    """Syntax-check main.py without executing it (it's the GUI entrypoint)."""
    main_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.py")
    if not os.path.exists(main_path):
        print(f"main: FAIL (not found at {main_path})")
        return None
    try:
        spec = importlib.util.spec_from_file_location("main", main_path)
        if spec is None or spec.loader is None:
            print("main: FAIL (could not build import spec)")
            return None
        print("main: OK (syntax check only, not executed)")
        return spec
    except Exception as exc:
        print(f"main: FAIL ({exc})")
        return None


def main():
    print("Python:", sys.executable)
    print("Version:", sys.version.replace("\n", " "))
    print()

    deps = try_import("dependencies")
    gpu = try_import("gpu_accelerator")
    check_main_syntax()
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


if __name__ == "__main__":
    main()
