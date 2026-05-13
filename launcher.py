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
    try:
        import PyQt6  # noqa: F401

        return True, "PyQt6 available"
    except ImportError:
        return False, "PyQt6 not found. Install with: pip install PyQt6"


def launch_application(script_name):
    """Launch the maintained application."""
    here = os.path.dirname(os.path.abspath(__file__))
    # Resolve both bootstrap and the target script relative to this launcher
    # so the launcher works regardless of the caller's working directory.
    bootstrap = os.path.join(here, "bootstrap_launch.py")
    script_abs = script_name if os.path.isabs(script_name) else os.path.join(here, script_name)
    target = bootstrap if os.path.exists(bootstrap) else script_abs
    try:
        write_line(f"Launching {script_name}...")
        command = [sys.executable, target]
        if target == bootstrap:
            command.append(script_abs)
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        write_line(f"Error launching {script_name}: {e}")
        return False
    except KeyboardInterrupt:
        write_line(f"\n{script_name} interrupted by user")
        return False
    except OSError as e:
        write_line(f"Error launching {script_name}: {e}")
        return False
    return True


def main():
    """Launch the single maintained app entrypoint."""
    write_line("=" * 60)
    write_line("DUPLICATE FILE FINDER")
    write_line("=" * 60)
    write_line(f"Python: {sys.executable}")

    here = os.path.dirname(os.path.abspath(__file__))
    if os.name == "nt":
        env_python = os.path.join(here, ".venv-app", "Scripts", "python.exe")
    else:
        env_python = os.path.join(here, ".venv-app", "bin", "python")
    if os.path.exists(env_python) and os.path.normcase(sys.executable) != os.path.normcase(env_python):
        write_line("Note: .venv-app exists but is not the active interpreter.")
        if os.name == "nt":
            write_line("Launch with .\\launch_app_env.bat to use the isolated app environment.\n")
        else:
            write_line("Activate it with: source .venv-app/bin/activate\n")

    deps_ok, deps_msg = check_dependencies()
    if not deps_ok:
        write_line(f"Error: {deps_msg}")
        write_line("Please install dependencies first:")
        write_line("pip install -r requirements.txt")
        return

    write_line(f"OK: {deps_msg}")

    script = os.path.join(here, "main.py")
    if not os.path.exists(script):
        write_line(f"Error: main.py not found at {script}.")
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
