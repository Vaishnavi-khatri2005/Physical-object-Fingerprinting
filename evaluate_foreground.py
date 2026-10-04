#!/usr/bin/env python3
"""
Shoeprint: Foreground-Isolated Evaluation Pipeline.

Executes the identical evaluation protocol as evaluate.py (same split, same seed=42,
same SIFT/RANSAC parameters), but extracts SIFT keypoints exclusively within the
segmented shoe foreground mask to eliminate background floor/table noise.

Compares results query-by-query against baseline outputs/evaluation_results.csv.

Usage:
    python evaluate_foreground.py
"""

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate import discover_shoe_images, plot_and_save_confusion_matrix, split_dataset
from prototype_foreground import create_foreground_mask_opencv, load_image_rgb
from src.config import DEFAULT_OUTPUTS_DIR
from src.features import SIFTFeatureExtractor
from src.matcher import SIFTRansacMatcher, aggregate_shoe_scores


class MaskedSIFTFeatureExtractor:
    """
    Extracts SIFT features exclusively within the OpenCV-segmented foreground mask.
    Supports caching to avoid redundant image segmentation on subsequent runs.
    """

    def __init__(self, max_features: int = 2500, cache_dir: Optional[Path] = None):
        self.max_features = max_features
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.sift = cv2.SIFT_create(nfeatures=self.max_features)

    def extract(self, image_path: Path, use_cache: bool = True) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray], str, float]:
        image_path = Path(image_path)

        cache_file = None
        if self.cache_dir and use_cache:
            stat = image_path.stat()
            cache_stem = f"fg_{image_path.stem}_{stat.st_size}_{stat.st_mtime_ns}"
            cache_file = self.cache_dir / f"{cache_stem}.npz"
            if cache_file.exists():
                kps, desc = self._load_from_cache(cache_file)
                return kps, desc, "cached", 0.0

        # Load RGB image and grayscale
        img_rgb = load_image_rgb(image_path)
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        # Generate foreground mask
        mask, status, coverage = create_foreground_mask_opencv(img_rgb)

        # Detect SIFT keypoints with mask
        kps, desc = self.sift.detectAndCompute(gray, mask=mask)

        if kps is None:
            kps = []

        if cache_file is not None and desc is not None:
            self._save_to_cache(cache_file, kps, desc)

        return kps, desc, status, coverage

    def _save_to_cache(self, cache_file: Path, keypoints: List[cv2.KeyPoint], descriptors: np.ndarray) -> None:
        try:
            kp_data = np.array([
                (kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave, kp.class_id)
                for kp in keypoints
            ], dtype=np.float32)
            np.savez_compressed(cache_file, kp=kp_data, desc=descriptors)
        except Exception:
            pass

    def _load_from_cache(self, cache_file: Path) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        try:
            data = np.load(cache_file)
            kp_data = data["kp"]
            descriptors = data["desc"]
            keypoints = [
                cv2.KeyPoint(
                    x=float(row[0]),
                    y=float(row[1]),
                    size=float(row[2]),
                    angle=float(row[3]),
                    response=float(row[4]),
                    octave=int(row[5]),
                    class_id=int(row[6])
                )
                for row in kp_data
            ]
            return keypoints, descriptors
        except Exception:
            return [], None


def main():
    dataset_dir = Path("photos").resolve()
    baseline_csv_path = DEFAULT_OUTPUTS_DIR / "evaluation_results.csv"
    eval_dir = DEFAULT_OUTPUTS_DIR / "evaluation"
    fg_csv_path = eval_dir / "foreground_results.csv"
    cm_path = eval_dir / "confusion_matrix_foreground.png"

    eval_dir.mkdir(parents=True, exist_ok=True)

    if not baseline_csv_path.exists():
        print(f"[X] Error: Baseline evaluation CSV not found: {baseline_csv_path}")
        return

    # 1. Load baseline results
    baseline_records: Dict[str, dict] = {}
    with open(baseline_csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            baseline_records[r["query_image"]] = r

    # 2. Re-establish identical split (seed=42, test_ratio=0.30)
    shoe_dict = discover_shoe_images(dataset_dir)
    shoe_names = sorted(shoe_dict.keys())
    shoe_to_idx = {name: idx for idx, name in enumerate(shoe_names)}

    reference_dict, test_dict = split_dataset(shoe_dict, test_ratio=0.30, seed=42)
    total_ref = sum(len(imgs) for imgs in reference_dict.values())
    total_test = sum(len(imgs) for imgs in test_dict.values())

    print("=" * 65)
    print("  Shoeprint: Foreground-Isolated Full Evaluation")
    print("=" * 65)
    print(f"[*] Dataset:         {dataset_dir}")
    print(f"[*] Evaluation Split: Same 108 reference / 46 test (Seed 42)")
    print(f"[*] Matching Params: Same SIFT (2500), Lowe (0.75), RANSAC (5.0)")
    print(f"[*] Baseline file:   {baseline_csv_path.name}")
    print("=" * 65)

    # Setup masked extractor and matcher
    cache_dir = DEFAULT_OUTPUTS_DIR / "cache" / "sift_foreground"
    extractor = MaskedSIFTFeatureExtractor(max_features=2500, cache_dir=cache_dir)
    matcher = SIFTRansacMatcher(ratio_thresh=0.75, ransac_reproj_thresh=5.0)

    # 3. Pre-extract masked reference features
    print(f"\n[*] Pre-segmenting & extracting masked SIFT for {total_ref} reference images...")
    t_ref_start = time.time()
    ref_features: Dict[Path, Tuple[List, Optional[np.ndarray], str]] = {}
    for shoe_id, ref_imgs in reference_dict.items():
        for r_path in ref_imgs:
            kp, desc, _, _ = extractor.extract(r_path, use_cache=True)
            ref_features[r_path] = (kp, desc, shoe_id)
    print(f"    Completed reference extraction in {time.time() - t_ref_start:.2f}s.")

    # 4. Run evaluation on identical 46 query test images
    all_queries = []
    for shoe_id, test_imgs in test_dict.items():
        for t_path in test_imgs:
            all_queries.append((t_path, shoe_id))

    print(f"\n[*] Running foreground-isolated matching on {len(all_queries)} queries...")
    correct_top1 = 0
    correct_top3 = 0
    incorrect_top1 = 0

    fg_results = []
    confusion_matrix = np.zeros((len(shoe_names), len(shoe_names)), dtype=int)

    query_durations = []

    for query_path, true_shoe in tqdm(all_queries, desc="Evaluating Foreground Queries", unit="query"):
        t0 = time.time()

        # Extract masked query features
        q_kp, q_desc, _, _ = extractor.extract(query_path, use_cache=True)

        if q_desc is None or len(q_desc) < 4:
            predicted_shoe = "none"
            top1_score = 0
            is_correct = False
            top3_hit = False
        else:
            match_results = []
            for r_path, (r_kp, r_desc, ref_shoe_id) in ref_features.items():
                res = matcher.match_pair(
                    query_kp=q_kp,
                    query_desc=q_desc,
                    ref_kp=r_kp,
                    ref_desc=r_desc,
                    shoe_id=ref_shoe_id,
                    ref_path=r_path,
                )
                match_results.append(res)

            ranked = aggregate_shoe_scores(match_results)

            if ranked and ranked[0].score > 0:
                best_shoe = ranked[0]
                predicted_shoe = best_shoe.shoe_id
                top1_score = best_shoe.score
                top3_candidates = [s.shoe_id for s in ranked[:3]]
            else:
                predicted_shoe = "none"
                top1_score = 0
                top3_candidates = []

            is_correct = (predicted_shoe == true_shoe)
            top3_hit = (true_shoe in top3_candidates)

        elapsed = time.time() - t0
        query_durations.append(elapsed)

        if is_correct:
            correct_top1 += 1
        else:
            incorrect_top1 += 1

        if top3_hit:
            correct_top3 += 1

        # Update confusion matrix
        true_idx = shoe_to_idx[true_shoe]
        if predicted_shoe in shoe_to_idx:
            pred_idx = shoe_to_idx[predicted_shoe]
            confusion_matrix[true_idx, pred_idx] += 1

        fg_results.append({
            "query_image": query_path.name,
            "true_shoe": true_shoe,
            "predicted_shoe": predicted_shoe,
            "correct": is_correct,
            "top1_score": top1_score,
            "elapsed_s": elapsed,
        })

    # 5. Save outputs/evaluation/foreground_results.csv
    with open(fg_csv_path, mode="w", newline="", encoding="utf-8") as f:
        fieldnames = ["query_image", "true_shoe", "predicted_shoe", "correct", "top1_score"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in fg_results:
            writer.writerow({
                "query_image": r["query_image"],
                "true_shoe": r["true_shoe"],
                "predicted_shoe": r["predicted_shoe"],
                "correct": r["correct"],
                "top1_score": r["top1_score"],
            })

    # 6. Save confusion matrix
    plot_and_save_confusion_matrix(shoe_names, confusion_matrix, cm_path)

    # 7. Compare Baseline vs Foreground
    total_q = len(all_queries)
    base_correct_top1 = sum(1 for r in baseline_records.values() if r["correct"] == "True")
    base_top1_acc = (base_correct_top1 / total_q) * 100
    fg_top1_acc = (correct_top1 / total_q) * 100
    fg_top3_acc = (correct_top3 / total_q) * 100

    # Query-by-query comparison: promoted vs demoted
    promoted_queries = []
    demoted_queries = []
    both_correct = []
    both_failed = []

    for r in fg_results:
        q_name = r["query_image"]
        base_r = baseline_records[q_name]
        b_corr = (base_r["correct"] == "True")
        f_corr = r["correct"]

        if not b_corr and f_corr:
            promoted_queries.append({
                "query": q_name,
                "true_shoe": r["true_shoe"],
                "base_pred": base_r["predicted_shoe"],
                "base_score": base_r["top1_score"],
                "fg_pred": r["predicted_shoe"],
                "fg_score": r["top1_score"],
            })
        elif b_corr and not f_corr:
            demoted_queries.append({
                "query": q_name,
                "true_shoe": r["true_shoe"],
                "base_pred": base_r["predicted_shoe"],
                "base_score": base_r["top1_score"],
                "fg_pred": r["predicted_shoe"],
                "fg_score": r["top1_score"],
            })
        elif b_corr and f_corr:
            both_correct.append(q_name)
        else:
            both_failed.append(q_name)

    # Per-shoe breakdown
    per_shoe_base = {}
    per_shoe_fg = {}
    per_shoe_total = {}

    for s in shoe_names:
        per_shoe_total[s] = 0
        per_shoe_base[s] = 0
        per_shoe_fg[s] = 0

    for r in fg_results:
        s = r["true_shoe"]
        per_shoe_total[s] += 1
        if r["correct"]:
            per_shoe_fg[s] += 1
        if baseline_records[r["query_image"]]["correct"] == "True":
            per_shoe_base[s] += 1

    avg_fg_time = sum(query_durations) / len(query_durations)

    # 8. Print comprehensive comparison
    print("\n" + "=" * 65)
    print("EVALUATION COMPARISON: BASELINE vs FOREGROUND-ISOLATED")
    print("=" * 65)
    print(f"{'Metric':<28} | {'Baseline':<14} | {'Foreground-Isolated':<18}")
    print("-" * 65)
    print(f"{'Correct Predictions':<28} | {base_correct_top1}/{total_q:<11} | {correct_top1}/{total_q:<15}")
    print(f"{'Top-1 Accuracy':<28} | {base_top1_acc:.1f}%{'':<9} | {fg_top1_acc:.1f}%{'':<13}")
    print(f"{'Top-3 Accuracy':<28} | {'78.3%':<14} | {fg_top3_acc:.1f}%{'':<13}")
    print(f"{'Avg Time / Query':<28} | {'~2.23s':<14} | {avg_fg_time:.2f}s{'':<14}")
    print("-" * 65)

    print("\nPer-Shoe Top-1 Accuracy Breakdown:")
    print("-" * 65)
    print(f"{'Shoe ID':<10} | {'Baseline':<16} | {'Foreground':<16} | {'Delta':<10}")
    print("-" * 65)
    for s in shoe_names:
        tot = per_shoe_total[s]
        b_acc = (per_shoe_base[s] / tot) * 100
        f_acc = (per_shoe_fg[s] / tot) * 100
        delta = f_acc - b_acc
        delta_str = f"+{delta:.1f}%" if delta > 0 else f"{delta:.1f}%"
        print(f"{s:<10} | {per_shoe_base[s]}/{tot} ({b_acc:.1f}%){'':<4} | {per_shoe_fg[s]}/{tot} ({f_acc:.1f}%){'':<4} | {delta_str:<10}")
    print("-" * 65)

    print(f"\n[+] Promoted Queries (Failed in Baseline -> CORRECT in Foreground): {len(promoted_queries)}")
    for p in promoted_queries:
        print(f"    * {p['query'][:30]}... ({p['true_shoe']}): Baseline predicted {p['base_pred']} (score {p['base_score']}) -> Foreground predicted {p['fg_pred']} (score {p['fg_score']})")

    print(f"\n[-] Demoted Queries (Correct in Baseline -> FAILED in Foreground): {len(demoted_queries)}")
    for d in demoted_queries:
        print(f"    * {d['query'][:30]}... ({d['true_shoe']}): Baseline predicted {d['base_pred']} (score {d['base_score']}) -> Foreground predicted {d['fg_pred']} (score {d['fg_score']})")

    print("\n[*] Artifacts Saved:")
    print(f"    - Results CSV:     {fg_csv_path}")
    print(f"    - Confusion Matrix:{cm_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
