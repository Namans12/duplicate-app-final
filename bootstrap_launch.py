#!/usr/bin/env python3
"""Launch an app entrypoint after preloading shared dependency bootstrap."""
import os
import runpy
import sys


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: bootstrap_launch.py <script> [args...]")
        return 1

    base_dir = os.path.dirname(os.path.abspath(__file__))
    script = sys.argv[1]
    if not os.path.isabs(script):
        script = os.path.join(base_dir, script)
    script = os.path.abspath(script)
    if not os.path.exists(script):
        print(f"Script not found: {script}")
        return 1

    # Import the shared dependency bootstrap first so optional DLL-backed
    # packages resolve consistently before the GUI entrypoint runs.
    try:
        import dependencies  # noqa: F401
    except ImportError as e:
        print(f"Warning: Could not load dependencies module: {e}", file=sys.stderr)

    # BL-FIX: Add proper exception handling for runpy.run_path()
    sys.argv = [script, *sys.argv[2:]]
    try:
        runpy.run_path(script, run_name="__main__")
        return 0
    except FileNotFoundError as e:
        print(f"Error: Script not found: {e}", file=sys.stderr)
        return 1
    except SyntaxError as e:
        print(f"Error: Syntax error in script: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        return 130
    except SystemExit as e:
        # Let SystemExit propagate with its exit code
        return e.code if isinstance(e.code, int) else (1 if e.code else 0)
    except Exception as e:
        print(f"Error running script: {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
