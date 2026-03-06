#!/usr/bin/env python3
"""
Duplicate File Finder Launcher
Easy launcher for choosing between different versions
"""
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


def find_script(*candidates):
    """Return the first existing candidate path."""
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def check_dependencies():
    """Check if required dependencies are available."""
    try:
        import PyQt6  # noqa: F401

        return True, "PyQt6 available"
    except ImportError:
        return False, "PyQt6 not found. Install with: pip install PyQt6"


def check_enterprise_modules():
    """Check if enterprise modules are available."""
    missing = []

    modules = [
        "database_manager.py",
        "gpu_accelerator.py",
        "plugin_system.py",
        "memory_optimizer.py",
        "duplicate_merger.py",
    ]

    for module in modules:
        if not os.path.exists(module):
            missing.append(module)

    return len(missing) == 0, missing


def launch_application(script_name):
    """Launch the specified application."""
    try:
        write_line(f"Launching {script_name}...")
        subprocess.run([sys.executable, script_name], check=True)
    except subprocess.CalledProcessError as e:
        write_line(f"Error launching {script_name}: {e}")
        return False
    except KeyboardInterrupt:
        write_line(f"\n{script_name} interrupted by user")
        return False
    return True


def main():
    """Main launcher interface."""
    write_line("=" * 60)
    write_line("DUPLICATE FILE FINDER - VERSION LAUNCHER")
    write_line("=" * 60)

    deps_ok, deps_msg = check_dependencies()
    if not deps_ok:
        write_line(f"Error: {deps_msg}")
        write_line("Please install dependencies first:")
        write_line("pip install -r requirements.txt")
        return

    write_line(f"OK: {deps_msg}")

    enterprise_ok, missing_modules = check_enterprise_modules()

    write_line("\nAVAILABLE VERSIONS:")
    write_line("-" * 30)

    options = []

    basic_script = find_script("main.py")
    if basic_script:
        options.append(("1", "Basic Duplicate Finder", basic_script, "Simple duplicate detection"))
        write_line("1. Basic Duplicate Finder")
        write_line(f"   Script: {basic_script}")
        write_line("   - Simple duplicate detection")
        write_line("   - Basic PyQt6 interface")
        write_line()

    enhanced_script = find_script("main_enhanced.py", os.path.join("archive", "main_enhanced.py"))
    if enhanced_script:
        options.append(("2", "Enhanced UX Version", enhanced_script, "All UX improvements"))
        write_line("2. Enhanced UX Version")
        write_line(f"   Script: {enhanced_script}")
        write_line("   - Search within duplicates")
        write_line("   - Progress details with ETA")
        write_line("   - Dark/Light theme toggle")
        write_line("   - Keyboard shortcuts")
        write_line("   - Professional interface")
        write_line()

    enterprise_script = find_script("main_enterprise.py", os.path.join("archive", "main_enterprise.py"))
    if enterprise_script:
        if enterprise_ok:
            options.append(("3", "Enterprise Edition", enterprise_script, "Full enterprise features"))
            write_line("3. Enterprise Edition")
            write_line(f"   Script: {enterprise_script}")
            write_line("   - All UX enhancements")
            write_line("   - Database persistence")
            write_line("   - GPU acceleration")
            write_line("   - Plugin system")
            write_line("   - Memory optimization")
            write_line("   - File merging")
            write_line()
        else:
            write_line("3. Enterprise Edition - UNAVAILABLE")
            write_line("   Missing enterprise modules:")
            for module in missing_modules:
                write_line(f"   - {module}")
            write_line()

    if not options:
        write_line("Error: No application versions found!")
        return

    write_line("SELECT VERSION TO LAUNCH:")
    write_line("-" * 30)

    while True:
        try:
            choice = input(f"Enter choice (1-{len(options)}) or 'q' to quit: ").strip().lower()

            if choice == "q":
                write_line("Goodbye!")
                return

            selected_option = None
            for opt in options:
                if opt[0] == choice:
                    selected_option = opt
                    break

            if selected_option:
                _, description, script, details = selected_option
                write_line(f"\nSelected: {description}")
                write_line(f"Features: {details}")

                confirm = input("Launch this version? (y/n): ").strip().lower()
                if confirm in ["y", "yes"]:
                    success = launch_application(script)
                    if success:
                        write_line(f"{description} closed successfully")
                    break

                write_line("Launch cancelled\n")
                continue

            write_line(f"Invalid choice '{choice}'. Please try again.\n")

        except KeyboardInterrupt:
            write_line("\nGoodbye!")
            break
        except Exception as e:
            write_line(f"Error: {e}")


if __name__ == "__main__":
    main()
