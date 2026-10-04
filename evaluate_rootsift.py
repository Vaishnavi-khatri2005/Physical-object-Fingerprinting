#!/usr/bin/env python3
"""
Shoeprint: RootSIFT Evaluation Pipeline.

Implements the RootSIFT descriptor transformation (Arandjelovic & Zisserman, CVPR 2012):
  1. Extract normal SIFT descriptors.
  2. L1-normalize each descriptor.
  3. Apply element-wise square root.
  4. Perform matching using the exact same BFMatcher + Lowe ratio (0.75) + RANSAC (5.0).

Evaluates on the exact same 108 reference / 46 test partition (seed=42),
compares query-by-query against the unmasked baseline, and saves results.

Usage:
    python evaluate_rootsift.py
"""

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
from src.config import DEFAULT_OUTPUTS_DIR
from src.features import SIFTFeatureExtractor
from src.matcher import SIFTRansacMatcher, aggregate_shoe_scores


def transform_to_rootsift(descriptors: Optional[np.ndarray], eps: float = 1e-7) -> Optional[np.ndarray]:
    """
    Transforms standard SIFT descriptors into RootSIFT:
      1. L1 normalization: desc / (||desc||_1 + eps)
      2. Element-wise square root: sqrt(desc)
    
    Using Euclidean distance on RootSIFT vectors is mathematically equivalent
    to using the Hellinger distance / Bhattacharyya kernel on probability distributions.
    """
    if descriptors is None or len(descriptors) == 0:
        return descriptors

    # L1 normalization along feature dimension (axis 1)
    l1_norm = np.linalg.norm(descriptors, ord=1, axis=1, keepdims=True) + eps
    desc_l1 = descriptors / l1_norm

    # Element-wise square root (clipping at 0 to avoid numerical negatives)
    desc_rootsift = np.sqrt(np.maximum(desc_l1, 0.0))

    return desc_rootsift.astype(np.float32)


def main():
    dataset_dir = Path("photos").resolve()
    baseline_csv_path = DEFAULT_OUTPUTS_DIR / "evaluation_results.csv"
    eval_dir = DEFAULT_OUTPUTS_DIR / "evaluation"
    rootsift_csv_path = eval_dir / "rootsift_results.csv"
    cm_path = eval_dir / "confusion_matrix_rootsift.png"

    eval_dir.mkdir(parents=True, exist_ok=True)

    if not baseline_csv_path.exists():
        print(f"[X] Error: Baseline evaluation results not found: {baseline_csv_path}")
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
    print("  Shoeprint: RootSIFT Evaluation Pipeline")
    print("=" * 65)
    print(f"[*] Dataset:         {dataset_dir}")
    print(f"[*] Evaluation Split: Same 108 reference / 46 test (Seed 42)")
    print(f"[*] Matching Params: Same SIFT (2500), Lowe (0.75), RANSAC (5.0)")
    print(f"[*] Feature Space:   RootSIFT (L1-norm + sqrt / Hellinger kernel)")
    print(f"[*] Baseline file:   {baseline_csv_path.name}")
    print("=" * 65)

    # 3. Setup feature extractor and matcher
    cache_dir = DEFAULT_OUTPUTS_DIR / "cache" / "sift"
    extractor = SIFTFeatureExtractor(max_features=2500, cache_dir=cache_dir)
    matcher = SIFTRansacMatcher(ratio_thresh=0.75, ransac_reproj_thresh=5.0)

    # 4. Pre-load reference features and convert to RootSIFT
    print(f"\n[*] Pre-loading & transforming {total_ref} reference descriptors to RootSIFT...")
    t_ref_start = time.time()
    ref_features: Dict[Path, Tuple[List, Optional[np.ndarray], str]] = {}
    for shoe_id, ref_imgs in reference_dict.items():
        for r_path in ref_imgs:
            kp, desc = extractor.extract(r_path, use_cache=True)
            desc_root = transform_to_rootsift(desc)
            ref_features[r_path] = (kp, desc_root, shoe_id)
    print(f"    Completed in {time.time() - t_ref_start:.2f}s.")

    # 5. Run evaluation on identical 46 query test images
    all_queries = []
    for shoe_id, test_imgs in test_dict.items():
        for t_path in test_imgs:
            all_queries.append((t_path, shoe_id))

    print(f"\n[*] Running RootSIFT matching on {len(all_queries)} queries...")
    correct_top1 = 0
    correct_top3 = 0
    incorrect_top1 = 0

    rootsift_results = []
    confusion_matrix = np.zeros((len(shoe_names), len(shoe_names)), dtype=int)
    query_durations = []

    for query_path, true_shoe in tqdm(all_queries, desc="Evaluating RootSIFT", unit="query"):
        t0 = time.time()

        # Extract raw SIFT, then convert to RootSIFT
        q_kp, q_desc = extractor.extract(query_path, use_cache=True)
        q_desc_root = transform_to_rootsift(q_desc)

        if q_desc_root is None or len(q_desc_root) < 4:
            predicted_shoe = "none"
            top1_score = 0
            is_correct = False
            top3_hit = False
        else:
            match_results = []
            for r_path, (r_kp, r_desc_root, ref_shoe_id) in ref_features.items():
                res = matcher.match_pair(
                    query_kp=q_kp,
                    query_desc=q_desc_root,
                    ref_kp=r_kp,
                    ref_desc=r_desc_root,
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

        rootsift_results.append({
            "query_image": query_path.name,
            "true_shoe": true_shoe,
            "predicted_shoe": predicted_shoe,
            "correct": is_correct,
            "top1_score": top1_score,
            "elapsed_s": elapsed,
        })

    # 6. Save outputs/evaluation/rootsift_results.csv
    with open(rootsift_csv_path, mode="w", newline="", encoding="utf-8") as f:
        fieldnames = ["query_image", "true_shoe", "predicted_shoe", "correct", "top1_score"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rootsift_results:
            writer.writerow({
                "query_image": r["query_image"],
                "true_shoe": r["true_shoe"],
                "predicted_shoe": r["predicted_shoe"],
                "correct": r["correct"],
                "top1_score": r["top1_score"],
            })

    # 7. Save confusion matrix
    plot_and_save_confusion_matrix(shoe_names, confusion_matrix, cm_path)

    # 8. Compare Baseline vs RootSIFT
    total_q = len(all_queries)
    base_correct_top1 = sum(1 for r in baseline_records.values() if r["correct"] == "True")
    base_top1_acc = (base_correct_top1 / total_q) * 100
    root_top1_acc = (correct_top1 / total_q) * 100
    root_top3_acc = (correct_top3 / total_q) * 100

    promoted_queries = []
    demoted_queries = []

    for r in rootsift_results:
        q_name = r["query_image"]
        base_r = baseline_records[q_name]
        b_corr = (base_r["correct"] == "True")
        r_corr = r["correct"]

        if not b_corr and r_corr:
            promoted_queries.append({
                "query": q_name,
                "true_shoe": r["true_shoe"],
                "base_pred": base_r["predicted_shoe"],
                "base_score": base_r["top1_score"],
                "root_pred": r["predicted_shoe"],
                "root_score": r["top1_score"],
            })
        elif b_corr and not r_corr:
            demoted_queries.append({
                "query": q_name,
                "true_shoe": r["true_shoe"],
                "base_pred": base_r["predicted_shoe"],
                "base_score": base_r["top1_score"],
                "root_pred": r["predicted_shoe"],
                "root_score": r["top1_score"],
            })

    # Per-shoe breakdown
    per_shoe_base = {}
    per_shoe_root = {}
    per_shoe_total = {}

    for s in shoe_names:
        per_shoe_total[s] = 0
        per_shoe_base[s] = 0
        per_shoe_root[s] = 0

    for r in rootsift_results:
        s = r["true_shoe"]
        per_shoe_total[s] += 1
        if r["correct"]:
            per_shoe_root[s] += 1
        if baseline_records[r["query_image"]]["correct"] == "True":
            per_shoe_base[s] += 1

    avg_root_time = sum(query_durations) / len(query_durations)

    # 9. Print comprehensive comparison
    print("\n" + "=" * 65)
    print("EVALUATION COMPARISON: SIFT BASELINE vs ROOTSIFT")
    print("=" * 65)
    print(f"{'Metric':<28} | {'Baseline SIFT':<14} | {'RootSIFT':<18}")
    print("-" * 65)
    print(f"{'Correct Predictions':<28} | {base_correct_top1}/{total_q:<11} | {correct_top1}/{total_q:<15}")
    print(f"{'Top-1 Accuracy':<28} | {base_top1_acc:.1f}%{'':<9} | {root_top1_acc:.1f}%{'':<13}")
    print(f"{'Top-3 Accuracy':<28} | {'78.3%':<14} | {root_top3_acc:.1f}%{'':<13}")
    print(f"{'Avg Latency / Query':<28} | {'~2.23s':<14} | {avg_root_time:.2f}s{'':<14}")
    print("-" * 65)

    print("\nPer-Shoe Top-1 Accuracy Breakdown:")
    print("-" * 65)
    print(f"{'Shoe ID':<10} | {'Baseline SIFT':<16} | {'RootSIFT':<16} | {'Delta':<10}")
    print("-" * 65)
    for s in shoe_names:
        tot = per_shoe_total[s]
        b_acc = (per_shoe_base[s] / tot) * 100
        r_acc = (per_shoe_root[s] / tot) * 100
        delta = r_acc - b_acc
        delta_str = f"+{delta:.1f}%" if delta > 0 else f"{delta:.1f}%"
        print(f"{s:<10} | {per_shoe_base[s]}/{tot} ({b_acc:.1f}%){'':<4} | {per_shoe_root[s]}/{tot} ({r_acc:.1f}%){'':<4} | {delta_str:<10}")
    print("-" * 65)

    print(f"\n[+] Number of Queries Improved (Baseline FAILED -> RootSIFT CORRECT): {len(promoted_queries)}")
    for p in promoted_queries:
        print(f"    * {p['query'][:30]}... ({p['true_shoe']}): Baseline predicted {p['base_pred']} (score {p['base_score']}) -> RootSIFT predicted {p['root_pred']} (score {p['root_score']})")

    print(f"\n[-] Number of Queries Degraded (Baseline CORRECT -> RootSIFT FAILED): {len(demoted_queries)}")
    for d in demoted_queries:
        print(f"    * {d['query'][:30]}... ({d['true_shoe']}): Baseline predicted {d['base_pred']} (score {d['base_score']}) -> RootSIFT predicted {d['root_pred']} (score {d['root_score']})")

    print("\n[*] Artifacts Saved:")
    print(f"    - Results CSV:     {rootsift_csv_path}")
    print(f"    - Confusion Matrix:{cm_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
