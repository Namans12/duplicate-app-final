#!/usr/bin/env python3
"""Launch an app entrypoint after preloading shared dependency bootstrap."""
import os
import runpy
import sys


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: bootstrap_launch.py <script> [args...]")
        return 1

    script = sys.argv[1]
    if not os.path.exists(script):
        print(f"Script not found: {script}")
        return 1

    # Import the shared dependency bootstrap first so optional DLL-backed
    # packages resolve consistently before the GUI entrypoint runs.
    import dependencies  # noqa: F401

    sys.argv = [script, *sys.argv[2:]]
    runpy.run_path(script, run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
