#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import sys


def unique_path(path: str) -> str:
    if not os.path.exists(path):
        return path
    stem, ext = os.path.splitext(path)
    i = 1
    while True:
        candidate = f"{stem}__restored{i}{ext}"
        if not os.path.exists(candidate):
            return candidate
        i += 1


def main():
    parser = argparse.ArgumentParser(description="Restore files moved by run_real_folder_dedup.py")
    parser.add_argument("--report", required=True, help="Path to real_folder_dedup_report.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # RDM-FIX: Add proper error handling for JSON loading
    try:
        with open(args.report, "r", encoding="utf-8") as f:
            report = json.load(f)
    except FileNotFoundError:
        print(f"Error: Report file not found: {args.report}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in report file: {e}", file=sys.stderr)
        sys.exit(1)
    except PermissionError:
        print(f"Error: Permission denied reading report file: {args.report}", file=sys.stderr)
        sys.exit(1)

    moved = report.get("moved_files", [])
    restored = 0
    missing = 0
    errors = []

    for item in moved:
        src = item.get("destination")
        dst = item.get("source")
        if not src or not dst:
            continue
        if not os.path.exists(src):
            missing += 1
            continue

        dst = unique_path(dst)
        if args.dry_run:
            restored += 1
            continue

        # RDM-FIX: Add error handling for file operations
        try:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)
            restored += 1
        except (OSError, shutil.Error) as e:
            errors.append({"source": src, "destination": dst, "error": str(e)})

    # RDM-FIX: Output proper JSON instead of Python dict repr
    result = {
        "report": os.path.abspath(args.report),
        "dry_run": args.dry_run,
        "entries": len(moved),
        "restored_or_planned": restored,
        "missing_sources": missing,
    }
    if errors:
        result["errors"] = errors
    print(json.dumps(result, indent=2))

    # RDM-FIX: Return proper exit code
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
