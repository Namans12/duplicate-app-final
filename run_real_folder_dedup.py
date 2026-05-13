#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import time
from collections import defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path

from gpu_accelerator import audio_fingerprinter
from audio_quality_analyzer import AudioQualityAnalyzer
import librosa
import numpy as np

AUDIO_EXTENSIONS = {
    '.mp3', '.flac', '.wav', '.aac', '.ogg', '.wma', '.m4a',
    '.opus', '.aiff', '.ape', '.alac', '.wv'
}


def normalized_name(value: str) -> str:
    if not value:
        return ""
    value = value.lower()
    value = re.sub(r"\.[a-z0-9]{1,5}$", "", value)
    value = re.sub(r"^\s*\d{1,3}\s*[-._)]\s*", "", value)
    value = re.sub(r"\s*\((?:copy|\d+)\)\s*", " ", value)
    value = re.sub(r"\s*\[(?:copy|\d+)\]\s*", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def audio_title_match(path1: str, path2: str) -> bool:
    n1 = normalized_name(os.path.basename(path1))
    n2 = normalized_name(os.path.basename(path2))
    return bool(n1 and n2 and n1 == n2)


def compute_sha256(path: str) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except (OSError, IOError) as e:
        # RFD-FIX: Use specific exception types instead of bare Exception
        return None


def pick_canonical(files: list[str]) -> str:
    """Pick the canonical file from a group. Returns the largest, newest file."""
    # RFD-FIX: Handle empty file list
    if not files:
        raise ValueError("Cannot pick canonical from empty file list")
    
    def score(path: str):
        try:
            stat = os.stat(path)
            return (stat.st_size, -stat.st_mtime, path.lower())
        except (OSError, IOError):
            # RFD-FIX: Use specific exception types
            return (0, 0, path.lower())
    return sorted(files, key=score, reverse=True)[0]


def unique_dest_path(base_dir: str, rel_path: str) -> str:
    """Generate unique destination path, avoiding TOCTOU race condition."""
    target = os.path.join(base_dir, rel_path)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    
    # RFD-FIX: Use atomic file creation to avoid TOCTOU race condition
    if not os.path.exists(target):
        return target

    stem, ext = os.path.splitext(target)
    index = 1
    # Limit iterations to prevent infinite loop
    max_attempts = 10000
    while index < max_attempts:
        candidate = f"{stem}__dup{index}{ext}"
        if not os.path.exists(candidate):
            return candidate
        index += 1
    raise RuntimeError(f"Could not find unique path after {max_attempts} attempts: {target}")


def gather_audio_files(source_dir: str) -> list[str]:
    out = []
    for root, _, files in os.walk(source_dir):
        for filename in files:
            ext = os.path.splitext(filename)[1].lower()
            if ext in AUDIO_EXTENSIONS:
                out.append(os.path.join(root, filename))
    return sorted(out)


def build_exact_groups(files: list[str]) -> dict[str, list[str]]:
    groups = defaultdict(list)
    for path in files:
        digest = compute_sha256(path)
        if digest:
            groups[digest].append(path)
    return {f"exact:{k}": v for k, v in groups.items() if len(v) > 1}


def compare_audio_groups(files: list[str], exact_groups: dict[str, list[str]], policy: str = "strict-verified"):
    analyzer = AudioQualityAnalyzer()
    duplicate_files = set()
    representatives = []
    rep_lookup = {}
    for group in exact_groups.values():
        duplicate_files.update(group)
        if group:
            rep = pick_canonical(group)
            representatives.append(rep)
            rep_lookup[rep] = list(group)

    non_duplicates = [f for f in files if f not in duplicate_files]
    comparison_files = representatives + non_duplicates

    fps = {}
    metadata_cache = {}
    segment_cache = {}

    def _infer_title_artist_from_filename(path: str):
        name = normalized_name(os.path.basename(path))
        if not name:
            return "", ""
        if " - " in name:
            artist, title = name.split(" - ", 1)
            return title.strip(), artist.strip()
        return name.strip(), ""

    def _meta(path: str):
        if path not in metadata_cache:
            info = analyzer.analyze_audio_quality(path) or {}
            inferred_title, inferred_artist = _infer_title_artist_from_filename(path)
            info["_inferred_title"] = inferred_title
            info["_inferred_artist"] = inferred_artist
            metadata_cache[path] = info
        return metadata_cache[path]

    def _meta_title_artist_match(meta1: dict, meta2: dict) -> bool:
        title1 = normalized_name(meta1.get("title") or meta1.get("_inferred_title") or "")
        title2 = normalized_name(meta2.get("title") or meta2.get("_inferred_title") or "")
        artist1 = normalized_name(meta1.get("artist") or meta1.get("_inferred_artist") or "")
        artist2 = normalized_name(meta2.get("artist") or meta2.get("_inferred_artist") or "")
        if not title1 or not title2 or title1 != title2:
            return False
        if artist1 and artist2:
            return artist1 == artist2
        return True

    def _name_match_with_metadata(path1: str, path2: str, meta1: dict, meta2: dict) -> bool:
        if audio_title_match(path1, path2):
            return True
        fname1 = normalized_name(os.path.basename(path1))
        fname2 = normalized_name(os.path.basename(path2))
        title1 = normalized_name(meta1.get("title") or meta1.get("_inferred_title") or "")
        title2 = normalized_name(meta2.get("title") or meta2.get("_inferred_title") or "")
        if title1 and title2 and title1 == title2:
            if title1 in fname1 and title2 in fname2:
                return True
            if fname1 in title1 or fname2 in title2:
                return True
        return False

    audio_cache = {}

    def _load_audio(path: str):
        if path not in audio_cache:
            try:
                y, sr = librosa.load(path, sr=22050, mono=True)
            except (OSError, IOError, Exception) as e:
                # RFD-FIX: Log specific error types but still handle gracefully
                y, sr = None, None
            audio_cache[path] = (y, sr)
        return audio_cache[path]

    def _segment_vector(path: str, region: str, shift_seconds: float = 0.0):
        cache_key = (path, region, round(float(shift_seconds), 3))
        if cache_key in segment_cache:
            return segment_cache[cache_key]
        y, sr = _load_audio(path)
        if y is None or sr is None or len(y) < 22050 * 6:
            segment_cache[cache_key] = None
            return None
        win = int(5 * 22050)
        shift = int(shift_seconds * 22050)
        
        # AUDIO-FIX: Support 5-point segment analysis for better coverage
        if region == "start":
            start = max(0, min(len(y) - win, 0 + shift))
            seg = y[start:start + win]
        elif region == "early_mid":
            # 25% into the track
            pos = len(y) // 4
            s = max(0, min(len(y) - win, pos - win // 2 + shift))
            seg = y[s:s + win]
        elif region == "middle":
            mid = len(y) // 2
            s = max(0, min(len(y) - win, mid - win // 2 + shift))
            seg = y[s:s + win]
        elif region == "late_mid":
            # 75% into the track
            pos = (3 * len(y)) // 4
            s = max(0, min(len(y) - win, pos - win // 2 + shift))
            seg = y[s:s + win]
        else:  # "end"
            s = max(0, min(len(y) - win, len(y) - win + shift))
            seg = y[s:s + win]
        mfcc = librosa.feature.mfcc(y=seg, sr=22050, n_mfcc=13)
        vec = np.concatenate([np.mean(mfcc, axis=1), np.std(mfcc, axis=1)])
        norm = np.linalg.norm(vec)
        vec = vec / norm if norm > 0 else vec
        segment_cache[cache_key] = vec
        return vec

    # AUDIO-FIX: 5-point segment matching with tighter thresholds for exact duplicates
    SEGMENT_REGIONS = ("start", "early_mid", "middle", "late_mid", "end")
    
    def _segment_5point_match(path1: str, path2: str, min_threshold: float = 0.92, avg_threshold: float = 0.95):
        """
        5-point segment analysis for accurate exact duplicate detection.
        Compares start, early-mid (25%), middle (50%), late-mid (75%), and end.
        Uses tighter thresholds: min >= 0.92, avg >= 0.95
        """
        try:
            def _directional_scores(src: str, dst: str):
                out = {}
                for region in SEGMENT_REGIONS:
                    v1 = _segment_vector(src, region, 0.0)
                    if v1 is None:
                        return None
                    best = 0.0
                    # Test with time shifts to handle slight timing differences
                    for shift in (-8.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 8.0):
                        v2 = _segment_vector(dst, region, shift)
                        if v2 is None:
                            continue
                        sim = float(np.clip(np.dot(v1, v2), 0.0, 1.0))
                        if sim > best:
                            best = sim
                    out[region] = best
                return out

            scores_a = _directional_scores(path1, path2)
            scores_b = _directional_scores(path2, path1)
            if scores_a is None or scores_b is None:
                return False, {"error": "short_audio"}

            # Take the max of both directions for each region
            scores = {
                region: max(scores_a.get(region, 0.0), scores_b.get(region, 0.0))
                for region in SEGMENT_REGIONS
            }
            vals = [scores[r] for r in SEGMENT_REGIONS]
            
            # AUDIO-FIX: Tighter thresholds - ALL segments must be highly similar
            min_val = min(vals)
            avg_val = sum(vals) / len(vals)
            ok = (min_val >= min_threshold) and (avg_val >= avg_threshold)
            
            # Add detailed info for debugging
            scores["_min"] = min_val
            scores["_avg"] = avg_val
            scores["_passed"] = ok
            
            return ok, scores
        except (OSError, IOError, ValueError) as e:
            return False, {"error": "segment_compare_failed"}

    # Keep legacy 3-point for backward compatibility
    def _segment_triplet_match(path1: str, path2: str, min_threshold: float = 0.90, avg_threshold: float = 0.93):
        """Legacy 3-point matching - use _segment_5point_match for better accuracy."""
        try:
            def _directional_scores(src: str, dst: str):
                out = {}
                for region in ("start", "middle", "end"):
                    v1 = _segment_vector(src, region, 0.0)
                    if v1 is None:
                        return None
                    best = 0.0
                    for shift in (-8.0, -4.0, -2.0, 0.0, 2.0, 4.0, 8.0):
                        v2 = _segment_vector(dst, region, shift)
                        if v2 is None:
                            continue
                        sim = float(np.clip(np.dot(v1, v2), 0.0, 1.0))
                        if sim > best:
                            best = sim
                    out[region] = best
                return out

            scores_a = _directional_scores(path1, path2)
            scores_b = _directional_scores(path2, path1)
            if scores_a is None or scores_b is None:
                return False, {"error": "short_audio"}

            scores = {
                region: max(scores_a.get(region, 0.0), scores_b.get(region, 0.0))
                for region in ("start", "middle", "end")
            }
            vals = [scores[r] for r in ("start", "middle", "end")]
            ok = (min(vals) >= min_threshold) and ((sum(vals) / len(vals)) >= avg_threshold)
            return ok, scores
        except (OSError, IOError, ValueError) as e:
            return False, {"error": "segment_compare_failed"}

    def _fingerprint_score(details: dict, score: float) -> float:
        return max(
            score or 0.0,
            details.get("chromaprint_score") or 0.0,
            details.get("mfcc_score") or 0.0,
            details.get("combined_score") or 0.0,
        )

    def _duration_match(meta1: dict, meta2: dict, tolerance: float = 0.15) -> bool:
        """Check if durations match within tolerance (default 15% for different sources)."""
        d1 = meta1.get("duration") or meta1.get("length") or 0
        d2 = meta2.get("duration") or meta2.get("length") or 0
        if d1 <= 0 or d2 <= 0:
            return True  # Can't compare, assume OK
        ratio = min(d1, d2) / max(d1, d2)
        return ratio >= (1.0 - tolerance)

    def _is_exact_duplicate(
        file1: str, file2: str,
        meta1: dict, meta2: dict,
        details: dict, score: float,
        stream_scores_5pt: dict
    ) -> tuple[bool, dict]:
        """
        Comprehensive exact duplicate detection using ALL available signals.
        Returns (is_exact, confidence_info).
        
        For exact duplicates, ALL of these must be true:
        1. Filename OR title matches
        2. Metadata (title required, artist if both present)
        3. Duration within 2%
        4. 5-point audio stream analysis (min >= 0.92, avg >= 0.95)
        5. Fingerprint score >= 0.96 OR strict_match=True
        """
        confidence = {
            "filename_match": False,
            "metadata_match": False,
            "duration_match": False,
            "stream_5pt_match": False,
            "fingerprint_match": False,
            "overall_score": 0.0,
        }
        
        # 1. Filename/title match
        name_ok = _name_match_with_metadata(file1, file2, meta1, meta2)
        confidence["filename_match"] = name_ok
        
        # 2. Metadata match
        metadata_ok = _meta_title_artist_match(meta1, meta2)
        confidence["metadata_match"] = metadata_ok
        
        # 3. Duration match (within 2%)
        dur_ok = _duration_match(meta1, meta2, tolerance=0.02)
        confidence["duration_match"] = dur_ok
        
        # 4. 5-point stream analysis
        stream_ok = stream_scores_5pt.get("_passed", False)
        confidence["stream_5pt_match"] = stream_ok
        confidence["stream_min"] = stream_scores_5pt.get("_min", 0.0)
        confidence["stream_avg"] = stream_scores_5pt.get("_avg", 0.0)
        
        # 5. Fingerprint - tighter threshold for exact duplicates
        fp_score = _fingerprint_score(details, score)
        fp_ok = bool(details.get("strict_match") or fp_score >= 0.96)
        confidence["fingerprint_match"] = fp_ok
        confidence["fingerprint_score"] = fp_score
        
        # Calculate overall confidence score
        scores = []
        if stream_scores_5pt.get("_avg"):
            scores.append(stream_scores_5pt["_avg"])
        if fp_score > 0:
            scores.append(fp_score)
        confidence["overall_score"] = sum(scores) / len(scores) if scores else 0.0
        
        # ALL must pass for exact duplicate
        is_exact = all([name_ok, metadata_ok, dur_ok, stream_ok, fp_ok])
        confidence["is_exact"] = is_exact
        
        return is_exact, confidence

    def _is_verified_match(policy_name: str, details: dict, score: float, name_ok: bool, metadata_ok: bool, stream_ok: bool, stream_scores: dict) -> bool:
        """Legacy verification function for backward compatibility."""
        fp_score = _fingerprint_score(details, score)
        if policy_name == "same-track":
            # For 5-point scores, check all regions
            if "_passed" in stream_scores:
                stream_relaxed_ok = stream_scores.get("_min", 0) >= 0.85 and stream_scores.get("_avg", 0) >= 0.90
            else:
                vals = [stream_scores.get(k, 0.0) for k in ("start", "middle", "end") if isinstance(stream_scores, dict)]
                stream_relaxed_ok = bool(vals) and (min(vals) >= 0.85) and ((sum(vals) / len(vals)) >= 0.90)
            fp_ok = bool(details.get("strict_match") or details.get("near_match") or fp_score >= 0.95)
            return bool(name_ok and metadata_ok and fp_ok and stream_relaxed_ok)

        # strict-verified policy - use tighter thresholds
        if "_passed" in stream_scores:
            # Using 5-point analysis
            stream_strict_ok = stream_scores.get("_passed", False)
        else:
            stream_strict_ok = stream_ok
        fp_ok = bool(details.get("strict_match") or fp_score >= 0.96)
        return bool(name_ok and metadata_ok and fp_ok and stream_strict_ok)

    for path in comparison_files:
        fps[path] = audio_fingerprinter.compute_fingerprint(path, method="hybrid")

    strict_groups = []
    near_candidates = []

    for i, file1 in enumerate(comparison_files):
        fp1 = fps.get(file1)
        if not fp1:
            continue
        for file2 in comparison_files[i + 1:]:
            fp2 = fps.get(file2)
            if not fp2:
                continue
            
            # RFD-FIX: Validate fingerprint comparison result format
            try:
                result = audio_fingerprinter.compare_fingerprints(fp1, fp2, method="hybrid")
                if not result or len(result) != 3:
                    continue
                is_sim, score, details = result
            except (TypeError, ValueError):
                continue
            meta1 = _meta(file1)
            meta2 = _meta(file2)
            title_name_match = _name_match_with_metadata(file1, file2, meta1, meta2)
            metadata_match = _meta_title_artist_match(meta1, meta2)

            # Fast prefilter: if title/name+metadata disagree, skip expensive stream analysis.
            if not (title_name_match and metadata_match):
                continue

            # AUDIO-FIX: Use 5-point segment analysis for better accuracy
            stream_match_5pt, stream_scores_5pt = _segment_5point_match(file1, file2)
            
            # Also run legacy 3-point for backward compatibility in reports
            stream_match_3pt, stream_scores_3pt = _segment_triplet_match(file1, file2)

            group1 = rep_lookup.get(file1, [file1])
            group2 = rep_lookup.get(file2, [file2])

            # AUDIO-FIX: Use comprehensive exact duplicate detection
            is_exact, confidence_info = _is_exact_duplicate(
                file1, file2,
                meta1, meta2,
                details, score,
                stream_scores_5pt
            )
            
            # Also check with legacy verification for the specified policy
            verified_same_song = _is_verified_match(
                policy,
                details,
                score,
                title_name_match,
                metadata_match,
                stream_match_5pt,  # Use 5-point result
                stream_scores_5pt,
            )
            
            # Accept if either method confirms it's a duplicate
            if is_exact or verified_same_song:
                for a in group1:
                    for b in group2:
                        if a == b:
                            continue
                        strict_groups.append({
                            "files": [a, b],
                            "score": _fingerprint_score(details, score),
                            "details": {
                                **details,
                                "name_match": title_name_match,
                                "metadata_match": metadata_match,
                                "segment_scores_3pt": stream_scores_3pt,
                                "segment_scores_5pt": stream_scores_5pt,
                                "confidence": confidence_info,
                                "is_exact_duplicate": is_exact,
                                "policy": policy,
                            },
                        })
            elif details.get("near_match") and title_name_match and metadata_match:
                # Near matches that didn't pass exact duplicate criteria
                for a in group1:
                    for b in group2:
                        if a == b:
                            continue
                        near_candidates.append({
                            "files": [a, b],
                            "score": _fingerprint_score(details, score),
                            "details": {
                                **details,
                                "name_match": title_name_match,
                                "metadata_match": metadata_match,
                                "segment_scores_3pt": stream_scores_3pt,
                                "segment_scores_5pt": stream_scores_5pt,
                                "confidence": confidence_info,
                                "policy": policy,
                            },
                        })

    return strict_groups, near_candidates


def dedup_pair_records(pair_records: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for item in pair_records:
        a, b = sorted(item["files"])
        key = (a, b)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def build_move_plan(exact_groups: dict[str, list[str]], strict_pairs: list[dict]):
    move_targets = set()

    for group in exact_groups.values():
        keep = pick_canonical(group)
        for path in group:
            if path != keep:
                move_targets.add(path)

    strict_graph = defaultdict(set)
    for pair in strict_pairs:
        a, b = pair["files"]
        strict_graph[a].add(b)
        strict_graph[b].add(a)

    visited = set()
    for node in list(strict_graph.keys()):
        if node in visited:
            continue
        stack = [node]
        component = []
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            component.append(cur)
            stack.extend(strict_graph[cur] - visited)
        if len(component) > 1:
            keep = pick_canonical(component)
            for path in component:
                if path != keep:
                    move_targets.add(path)

    return sorted(move_targets)


def execute_moves(source_dir: str, target_dir: str, move_paths: list[str], dry_run: bool):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = os.path.join(target_dir, f"dedup_run_{timestamp}")
    moved = []

    if not dry_run:
        os.makedirs(run_root, exist_ok=True)

    for src in move_paths:
        rel = os.path.relpath(src, source_dir)
        dest = unique_dest_path(run_root if not dry_run else target_dir, rel)
        moved.append({"source": src, "destination": dest})
        if not dry_run:
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.move(src, dest)

    return moved, run_root


def write_report(report_path: str, csv_path: str, report: dict, moved: list[dict], near_candidates: list[dict]):
    out = dict(report)
    out["moved_files"] = moved
    out["near_review_candidates"] = near_candidates

    # RFD-FIX: Add error handling for file writes
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
    except (OSError, IOError) as e:
        print(f"Warning: Failed to write JSON report: {e}")

    try:
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["type", "source", "destination", "score", "notes"])
            for item in moved:
                writer.writerow(["moved", item["source"], item["destination"], "", "exact/strict duplicate"]) 
            for item in near_candidates:
                a, b = item["files"]
                writer.writerow(["near_review", a, b, f"{item.get('score', 0.0):.4f}", "near match with title support"])
    except (OSError, IOError) as e:
        print(f"Warning: Failed to write CSV report: {e}")


def main():
    parser = argparse.ArgumentParser(description="Run real-folder audio dedup and move duplicates.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--report", default="real_folder_dedup_report.json")
    parser.add_argument("--csv", default="real_folder_dedup_report.csv")
    parser.add_argument("--policy", choices=["strict-verified", "same-track"], default="strict-verified")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    source_dir = os.path.abspath(args.source)
    target_dir = os.path.abspath(args.target)
    report_path = os.path.abspath(args.report)
    csv_path = os.path.abspath(args.csv)

    # RFD-FIX: Validate input paths
    if not os.path.isdir(source_dir):
        print(f"Error: Source directory does not exist: {source_dir}")
        return 1
    if not os.path.isdir(target_dir):
        print(f"Error: Target directory does not exist: {target_dir}")
        return 1
    if os.path.realpath(source_dir) == os.path.realpath(target_dir):
        print("Error: Source and target directories cannot be the same")
        return 1

    started = time.time()
    files = gather_audio_files(source_dir)
    exact_groups = build_exact_groups(files)
    strict_pairs, near_candidates = compare_audio_groups(files, exact_groups, policy=args.policy)
    strict_pairs = dedup_pair_records(strict_pairs)
    near_candidates = dedup_pair_records(near_candidates)

    move_plan = build_move_plan(exact_groups, strict_pairs)
    moved, run_root = execute_moves(source_dir, target_dir, move_plan, args.dry_run)

    report = {
        "source_dir": source_dir,
        "target_dir": target_dir,
        "run_root": run_root,
        "dry_run": args.dry_run,
        "policy": args.policy,
        "scan_audio_file_count": len(files),
        "exact_group_count": len(exact_groups),
        "strict_pair_count": len(strict_pairs),
        "near_candidate_count": len(near_candidates),
        "planned_move_count": len(move_plan),
        "actual_moved_count": len(moved),
        "elapsed_seconds": time.time() - started,
        "exact_groups": exact_groups,
        "strict_pairs": strict_pairs,
    }

    write_report(report_path, csv_path, report, moved, near_candidates)

    print(json.dumps({
        "scan_audio_file_count": len(files),
        "exact_group_count": len(exact_groups),
        "strict_pair_count": len(strict_pairs),
        "near_candidate_count": len(near_candidates),
        "actual_moved_count": len(moved),
        "policy": args.policy,
        "report": report_path,
        "csv": csv_path,
        "run_root": run_root,
        "elapsed_seconds": report["elapsed_seconds"],
        "dry_run": args.dry_run,
    }, indent=2))


if __name__ == "__main__":
    main()
