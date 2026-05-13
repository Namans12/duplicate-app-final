#!/usr/bin/env python3
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from itertools import combinations
from pathlib import Path

import numpy as np

# AB-FIX: Handle soundfile import with proper error handling
try:
    import soundfile as sf
except ImportError:
    print("Error: soundfile library required for audio_e2e_benchmark")
    import sys
    sys.exit(1)

from duplicate_merger import FileMerger
from gpu_accelerator import audio_fingerprinter

try:
    import librosa
except ImportError:
    librosa = None


def _sine(freq, seconds, sr=22050, amp=0.25):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _mix(*waves):
    arr = np.sum(np.stack(waves, axis=0), axis=0)
    peak = np.max(np.abs(arr))
    if peak > 0:
        arr = arr / max(peak, 1.0) * 0.8
    return arr.astype(np.float32)


def _build_dataset(root: Path):
    sr = 22050
    intro = _mix(_sine(220, 5, sr), _sine(440, 5, sr, 0.15))
    middle_a = _mix(_sine(330, 20, sr), _sine(550, 20, sr, 0.2))
    outro = _mix(_sine(260, 5, sr), _sine(520, 5, sr, 0.15))

    song_a = np.concatenate([intro, middle_a, outro])
    song_a_alt_container = song_a.copy()

    middle_b = _mix(_sine(880, 20, sr), _sine(1760, 20, sr, 0.2))
    song_b_hard = np.concatenate([intro, middle_b, outro])

    files = {
        "song_a.wav": song_a,
        "song_a_copy.wav": song_a.copy(),
        "song_a.flac": song_a_alt_container,
        "song_a_pcm24.wav": song_a_alt_container,
        "song_b_hard.wav": song_b_hard,
        "totally_different.wav": _mix(_sine(1230, 30, sr), _sine(90, 30, sr, 0.08)),
    }

    # AB-FIX: Add error handling for soundfile writes
    for name, audio in files.items():
        out = root / name
        try:
            if name == "song_a_pcm24.wav":
                sf.write(str(out), audio, sr, subtype="PCM_24")
            else:
                sf.write(str(out), audio, sr)
        except Exception as e:
            raise RuntimeError(f"Failed to write {out}: {e}")

    gt = {
        "exact_pairs": {tuple(sorted(("song_a.wav", "song_a_copy.wav")))},
        "near_pairs": {
            tuple(sorted(("song_a.wav", "song_a.flac"))),
            tuple(sorted(("song_a.wav", "song_a_pcm24.wav"))),
            tuple(sorted(("song_a_copy.wav", "song_a.flac"))),
            tuple(sorted(("song_a_copy.wav", "song_a_pcm24.wav"))),
            tuple(sorted(("song_a.flac", "song_a_pcm24.wav"))),
        },
        "hard_negative_pairs": {
            tuple(sorted(("song_a.wav", "song_b_hard.wav"))),
            tuple(sorted(("song_a_copy.wav", "song_b_hard.wav"))),
            tuple(sorted(("song_a.flac", "song_b_hard.wav"))),
            tuple(sorted(("song_a_pcm24.wav", "song_b_hard.wav"))),
            tuple(sorted(("song_a.wav", "totally_different.wav"))),
            tuple(sorted(("song_a_copy.wav", "totally_different.wav"))),
            tuple(sorted(("song_a.flac", "totally_different.wav"))),
            tuple(sorted(("song_a_pcm24.wav", "totally_different.wav"))),
        },
    }
    return gt


def _pairs_from_groups(groups):
    pairs = set()
    for group in groups.values():
        for a, b in combinations(group, 2):
            pairs.add(tuple(sorted((os.path.basename(a), os.path.basename(b)))))
    return pairs


def _segment_similarity(path1, path2):
    if librosa is None:
        return {"error": "librosa unavailable"}

    y1, sr1 = librosa.load(path1, sr=22050, mono=True)
    y2, sr2 = librosa.load(path2, sr=22050, mono=True)
    if sr1 != sr2 or len(y1) < 22050 * 6 or len(y2) < 22050 * 6:
        return {"error": "too_short_or_rate_mismatch"}

    def seg(y, which):
        dur = len(y) / 22050
        win = int(5 * 22050)
        if which == "start":
            return y[:win]
        if which == "middle":
            mid = int((dur / 2) * 22050)
            return y[max(0, mid - win // 2):max(0, mid - win // 2) + win]
        return y[-win:]

    def feat(v):
        m = librosa.feature.mfcc(y=v, sr=22050, n_mfcc=13)
        f = np.concatenate([np.mean(m, axis=1), np.std(m, axis=1)])
        n = np.linalg.norm(f)
        return f / n if n > 0 else f

    out = {}
    for key in ("start", "middle", "end"):
        f1 = feat(seg(y1, key))
        f2 = feat(seg(y2, key))
        out[key] = float(np.clip(np.dot(f1, f2), 0.0, 1.0))
    return out


def _evaluate(dataset_dir: Path, mode: str = "current"):
    files = sorted(str(p) for p in dataset_dir.glob("*") if p.is_file())

    t0 = time.perf_counter()
    hashes = {}
    fingerprints = {}

    def normalized_name(value: str) -> str:
        if not value:
            return ""
        value = value.lower()
        value = re.sub(r"\.[a-z0-9]{1,5}$", "", value)
        value = re.sub(r"\s*\((?:copy|\d+)\)\s*", " ", value)
        value = re.sub(r"\s*\[(?:copy|\d+)\]\s*", " ", value)
        value = re.sub(r"\s+", " ", value)
        return value.strip()

    def audio_title_match(path1: str, path2: str) -> bool:
        n1 = normalized_name(os.path.basename(path1))
        n2 = normalized_name(os.path.basename(path2))
        return bool(n1 and n2 and n1 == n2)

    for path in files:
        fp = audio_fingerprinter.compute_fingerprint(path, method="hybrid")
        fingerprints[path] = fp
        if not fp:
            continue
        if mode == "legacy":
            parts = []
            if fp.get("chromaprint"):
                parts.append(f"cp:{fp['chromaprint'][:32]}")
            mfcc = fp.get("mfcc")
            if isinstance(mfcc, list) and len(mfcc) >= 4:
                parts.append("mfcc:" + "_".join(f"{v:.2f}" for v in mfcc[:4]))
            if parts:
                hashes.setdefault("audio:" + "|".join(parts), []).append(path)
        else:
            # AB-FIX: Move hashlib import to top of file (already done)
            with open(path, "rb") as handle:
                h = hashlib.sha256()
                for chunk in iter(lambda: handle.read(65536), b""):
                    h.update(chunk)
            hashes.setdefault("audio_exact:" + h.hexdigest(), []).append(path)

    duplicates = {k: v for k, v in hashes.items() if len(v) > 1}
    duplicate_files = {p for group in duplicates.values() for p in group}
    if mode == "legacy":
        rep_lookup = {}
        comparison_files = [p for p in files if p not in duplicate_files]
    else:
        rep_lookup = {group[0]: list(group) for group in duplicates.values() if group}
        representatives = list(rep_lookup.keys())
        non_duplicates = [p for p in files if p not in duplicate_files]
        comparison_files = representatives + non_duplicates

    similar = {}
    sid = 0
    for i, file1 in enumerate(comparison_files):
        fp1 = fingerprints.get(file1)
        if not fp1:
            continue
        for file2 in comparison_files[i + 1:]:
            fp2 = fingerprints.get(file2)
            if not fp2:
                continue
            is_sim, sim, details = audio_fingerprinter.compare_fingerprints(fp1, fp2, method="hybrid")
            if mode == "legacy":
                is_near_same_song = bool(details.get("near_match"))
            else:
                is_near_same_song = bool(details.get("near_match") and audio_title_match(file1, file2))
            if not (is_sim or is_near_same_song):
                continue

            reason = "100% audio match" if details.get("strict_match") else "same song near match"
            group1 = rep_lookup.get(file1, [file1])
            group2 = rep_lookup.get(file2, [file2])
            for member1 in group1:
                for member2 in group2:
                    if member1 == member2:
                        continue
                    similar[f"similar_{sid}"] = {
                        "files": [member1, member2],
                        "reasons": [reason],
                        "match_type": "content",
                        "audio_match": {
                            "tier": "strict_recording_match" if details.get("strict_match") else "same_song_match",
                            "score": max(
                                sim or 0.0,
                                details.get("chromaprint_score") or 0.0,
                                details.get("mfcc_score") or 0.0,
                                details.get("combined_score") or 0.0,
                            ),
                            "chromaprint_score": details.get("chromaprint_score"),
                            "mfcc_score": details.get("mfcc_score"),
                            "combined_score": details.get("combined_score"),
                            "duration_ratio": details.get("duration_match"),
                            "label": "100% Audio Match" if details.get("strict_match") else "Same Song Match",
                            "reason": reason,
                        },
                    }
                    sid += 1

    runtime = time.perf_counter() - t0

    dup_pairs = _pairs_from_groups(duplicates)
    sim_pairs = set()
    for info in similar.values():
        f1, f2 = info["files"]
        sim_pairs.add(tuple(sorted((os.path.basename(f1), os.path.basename(f2)))))

    groups = list(duplicates.values()) + [v["files"] for v in similar.values()]
    merger = FileMerger()
    mergeable = 0
    for group in groups:
        result = merger.analyze_mergeable_files(group)
        if result.get("mergeable"):
            mergeable += 1

    return {
        "runtime_seconds": runtime,
        "duplicates": duplicates,
        "similar": similar,
        "duplicate_pairs": sorted(dup_pairs),
        "similar_pairs": sorted(sim_pairs),
        "mergeable_group_count": mergeable,
    }


def _metrics(gt, result, dataset_dir: Path):
    dup_pairs = set(tuple(x) for x in result["duplicate_pairs"])
    sim_pairs = set(tuple(x) for x in result["similar_pairs"])

    exact_hits = gt["exact_pairs"] & dup_pairs
    near_hits = gt["near_pairs"] & (dup_pairs | sim_pairs)

    predicted_audio_matches = dup_pairs | sim_pairs
    true_positive_space = gt["exact_pairs"] | gt["near_pairs"]

    false_positive_pairs = sorted(predicted_audio_matches - true_positive_space)
    hard_negative_hits = sorted(sim_pairs & gt["hard_negative_pairs"])

    missed_exact = sorted(gt["exact_pairs"] - dup_pairs)
    missed_near = sorted(gt["near_pairs"] - (dup_pairs | sim_pairs))

    missed_segments = {}
    for a, b in missed_exact + missed_near:
        missed_segments[f"{a} :: {b}"] = _segment_similarity(str(dataset_dir / a), str(dataset_dir / b))

    fp_segments = {}
    for a, b in hard_negative_hits:
        fp_segments[f"{a} :: {b}"] = _segment_similarity(str(dataset_dir / a), str(dataset_dir / b))

    return {
        "exact_duplicate_recall": len(exact_hits) / max(1, len(gt["exact_pairs"])),
        "near_duplicate_recall": len(near_hits) / max(1, len(gt["near_pairs"])),
        "false_positives": len(false_positive_pairs),
        "runtime_seconds": result["runtime_seconds"],
        "details": {
            "missed_exact_pairs": missed_exact,
            "missed_near_pairs": missed_near,
            "hard_negative_false_positives": hard_negative_hits,
            "false_positive_pairs_all": false_positive_pairs,
            "missed_segment_similarity": missed_segments,
            "hard_negative_segment_similarity": fp_segments,
            "mergeable_group_count": result["mergeable_group_count"],
        },
    }


def _compile_report(gt, baseline_result, current_result, dataset_dir: Path):
    baseline = _metrics(gt, baseline_result, dataset_dir)
    current = _metrics(gt, current_result, dataset_dir)
    return {
        "dataset_counts": {
            "exact_duplicates": len(gt["exact_pairs"]),
            "encoding_variants": len(gt["near_pairs"]),
            "hard_negatives": len(gt["hard_negative_pairs"]),
        },
        "baseline_metrics": {
            "exact_duplicate_recall": baseline["exact_duplicate_recall"],
            "near_duplicate_recall": baseline["near_duplicate_recall"],
            "false_positives": baseline["false_positives"],
            "runtime_seconds": baseline["runtime_seconds"],
        },
        "post_change_metrics": {
            "exact_duplicate_recall": current["exact_duplicate_recall"],
            "near_duplicate_recall": current["near_duplicate_recall"],
            "false_positives": current["false_positives"],
            "runtime_seconds": current["runtime_seconds"],
        },
        "findings": {
            "baseline": baseline["details"],
            "post_change": current["details"],
        },
        "net_improvement": {
            "exact_duplicate_recall_delta": current["exact_duplicate_recall"] - baseline["exact_duplicate_recall"],
            "near_duplicate_recall_delta": current["near_duplicate_recall"] - baseline["near_duplicate_recall"],
            "false_positives_delta": current["false_positives"] - baseline["false_positives"],
            "runtime_seconds_delta": current["runtime_seconds"] - baseline["runtime_seconds"],
        },
    }


def run_once(output_json: str | None = None):
    # AB-FIX: Add proper error handling for temp directory
    try:
        tmp = Path(tempfile.mkdtemp(prefix="audio_e2e_"))
    except OSError as e:
        print(f"Error: Failed to create temp directory: {e}")
        return None
    
    try:
        gt = _build_dataset(tmp)
        baseline_result = _evaluate(tmp, mode="legacy")
        current_result = _evaluate(tmp, mode="current")
        report = _compile_report(gt, baseline_result, current_result, tmp)
        report["dataset_dir"] = str(tmp)

        if output_json:
            try:
                with open(output_json, "w", encoding="utf-8") as f:
                    json.dump(report, f, indent=2)
            except (OSError, IOError) as e:
                print(f"Warning: Failed to write output file: {e}")
        print(json.dumps(report, indent=2))
        return report
    finally:
        try:
            shutil.rmtree(tmp)
        except (OSError, IOError) as e:
            print(f"Warning: Failed to clean temp dir {tmp}: {e}")


if __name__ == "__main__":
    run_once(output_json="audio_e2e_report.json")
