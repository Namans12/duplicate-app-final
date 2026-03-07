#!/usr/bin/env python3
"""Single-entry launcher for the maintained Duplicate File Finder app."""
import os
import subprocess
import sys


def write_line(text=""):
    """Print text without crashing on non-UTF8 Windows consoles."""
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "utf-8"
        safe_text = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
        print(safe_text)


def check_dependencies():
    """Check if required dependencies are available."""
    missing = []
    try:
        import PyQt6  # noqa: F401
    except ImportError:
        missing.append("PyQt6 (install: pip install PyQt6)")
    try:
        import xxhash  # noqa: F401 — L-2: xxhash is a required, not optional dep
    except ImportError:
        missing.append("xxhash (install: pip install xxhash)")
    if missing:
        return False, "Missing required packages: " + ", ".join(missing)
    return True, "PyQt6 and xxhash available"


def launch_application(script_name):
    """Launch the maintained application."""
    try:
        write_line(f"Launching {script_name}...")
        bootstrap = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bootstrap_launch.py")
        target = bootstrap if os.path.exists(bootstrap) else script_name
        command = [sys.executable, target]
        if target == bootstrap:
            command.append(script_name)
        result = subprocess.run(command)  # L-1: removed check=True — non-zero exit on close is not an error
        if result.returncode != 0:
            write_line(f"Note: {script_name} exited with code {result.returncode}")
    except KeyboardInterrupt:
        write_line(f"\n{script_name} interrupted by user")
        return False
    return True


def main():
    """Launch the single maintained app entrypoint."""
    write_line("=" * 60)
    write_line("DUPLICATE FILE FINDER")
    write_line("=" * 60)
    write_line(f"Python: {sys.executable}")

    env_python = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv-app", "Scripts", "python.exe")
    if os.path.exists(env_python) and os.path.normcase(sys.executable) != os.path.normcase(env_python):
        write_line("Note: .venv-app exists but is not the active interpreter.")
        write_line("Launch with .\\launch_app_env.bat to use the isolated app environment.\n")

    deps_ok, deps_msg = check_dependencies()
    if not deps_ok:
        write_line(f"Error: {deps_msg}")
        write_line("Please install dependencies first:")
        write_line("pip install -r requirements.txt")
        return

    write_line(f"OK: {deps_msg}")

    script = "main.py"
    if not os.path.exists(script):
        write_line("Error: main.py not found.")
        return

    write_line("\nMaintained App:")
    write_line("-" * 30)
    write_line(f"Script: {script}")
    write_line("Includes scanner, results, settings, search, import/export, undo, GPU, plugins, and merge analysis.")
    write_line()

    success = launch_application(script)
    if success:
        write_line("Duplicate File Finder closed successfully")


if __name__ == "__main__":
    main()
