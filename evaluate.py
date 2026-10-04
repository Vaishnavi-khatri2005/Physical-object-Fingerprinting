#!/usr/bin/env python3
"""
Shoeprint: Automated Evaluation Pipeline for SIFT + RANSAC Baseline.

Splits each physical shoe's photos into disjoint Reference (Gallery) and Test (Query) sets
using a fixed random seed, performs exhaustive gallery matching, computes Top-1 and Top-3
accuracies, generates a confusion matrix, and exports detailed evaluation results.

Usage:
    python evaluate.py --dataset photos
    python evaluate.py --dataset photos --test-ratio 0.3 --seed 42
"""

import argparse
import csv
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_OUTPUTS_DIR, SUPPORTED_IMAGE_EXTENSIONS
from src.features import SIFTFeatureExtractor
from src.matcher import SIFTRansacMatcher, aggregate_shoe_scores


def parse_args():
    parser = argparse.ArgumentParser(
        description="Automated Evaluation Pipeline for SIFT + RANSAC Shoe Matching.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="photos",
        help="Path to the dataset directory containing shoe_xx subfolders.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.30,
        help="Proportion of images per shoe to hold out for testing/querying.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Fixed random seed for reproducible train/test splitting.",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.75,
        help="Lowe's ratio test threshold.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=2500,
        help="Maximum SIFT keypoints per image.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUTS_DIR / "evaluation"),
        help="Directory to save evaluation artifacts (confusion matrix, plots).",
    )
    parser.add_argument(
        "--results-csv",
        type=str,
        default=str(DEFAULT_OUTPUTS_DIR / "evaluation_results.csv"),
        help="Path to save evaluation_results.csv.",
    )
    return parser.parse_args()


def discover_shoe_images(dataset_dir: Path) -> Dict[str, List[Path]]:
    """Discovers all valid image files grouped by shoe folder."""
    shoe_dict = {}
    subdirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])
    for subdir in subdirs:
        shoe_id = subdir.name
        img_files = sorted([
            f for f in subdir.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS and not f.name.startswith(".")
        ])
        if img_files:
            shoe_dict[shoe_id] = img_files
    return shoe_dict


def split_dataset(
    shoe_dict: Dict[str, List[Path]],
    test_ratio: float = 0.30,
    seed: int = 42,
) -> Tuple[Dict[str, List[Path]], Dict[str, List[Path]]]:
    """
    Splits images of EACH shoe into disjoint Reference (Gallery) and Test (Query) sets.
    Ensures reproducibility via a fixed random seed.
    Guarantees no test image appears in the reference set.
    """
    rng = random.Random(seed)
    reference_dict = {}
    test_dict = {}

    for shoe_id, images in sorted(shoe_dict.items()):
        shuffled = list(images)
        rng.shuffle(shuffled)

        n_total = len(shuffled)
        # Allocate at least 1 test image and at least 1 reference image
        n_test = max(1, int(round(n_total * test_ratio)))
        if n_test >= n_total:
            n_test = n_total - 1

        test_images = sorted(shuffled[:n_test])
        reference_images = sorted(shuffled[n_test:])

        test_dict[shoe_id] = test_images
        reference_dict[shoe_id] = reference_images

    return reference_dict, test_dict


def plot_and_save_confusion_matrix(
    shoe_names: List[str],
    matrix: np.ndarray,
    output_path: Path,
):
    """
    Renders and saves a clean confusion matrix heatmap.
    Rows: Actual Shoe ID, Columns: Predicted Shoe ID.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_classes = len(shoe_names)

    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    im = ax.imshow(matrix, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Set tick labels
    ax.set(
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=shoe_names,
        yticklabels=shoe_names,
        title="SIFT + RANSAC Baseline - Confusion Matrix",
        ylabel="True Physical Shoe ID",
        xlabel="Predicted Shoe ID",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate numbers inside each cell
    thresh = matrix.max() / 2.0 if matrix.max() > 0 else 1.0
    for i in range(n_classes):
        for j in range(n_classes):
            val = int(matrix[i, j])
            color = "white" if val > thresh else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=11, fontweight="bold")

    fig.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset).resolve()
    output_dir = Path(args.output_dir).resolve()
    results_csv_path = Path(args.results_csv).resolve()

    if not dataset_dir.exists():
        print(f"[X] Error: Dataset directory not found: {dataset_dir}")
        sys.exit(1)

    shoe_dict = discover_shoe_images(dataset_dir)
    if not shoe_dict:
        print(f"[X] Error: No shoe folders or images found in {dataset_dir}")
        sys.exit(1)

    shoe_names = sorted(shoe_dict.keys())
    shoe_to_idx = {name: idx for idx, name in enumerate(shoe_names)}

    print("=" * 65)
    print("  Shoeprint: Automated SIFT + RANSAC Evaluation Pipeline")
    print("=" * 65)
    print(f"[*] Dataset:       {dataset_dir}")
    print(f"[*] Random Seed:   {args.seed}")
    print(f"[*] Test Ratio:    {args.test_ratio:.2f}")
    print(f"[*] Lowe Ratio:    {args.ratio}")
    print(f"[*] Total Shoes:   {len(shoe_names)}")

    # 1. Perform disjoint split per shoe
    reference_dict, test_dict = split_dataset(shoe_dict, test_ratio=args.test_ratio, seed=args.seed)

    total_ref = sum(len(imgs) for imgs in reference_dict.values())
    total_test = sum(len(imgs) for imgs in test_dict.values())

    print("\n[*] Shoe Split Breakdown (Fair Distribution):")
    print("-" * 55)
    print(f"{'Shoe ID':<12} | {'Reference':<12} | {'Test':<10} | {'Total':<8}")
    print("-" * 55)
    for shoe_id in shoe_names:
        r_cnt = len(reference_dict[shoe_id])
        t_cnt = len(test_dict[shoe_id])
        tot = r_cnt + t_cnt
        print(f"{shoe_id:<12} | {r_cnt:<12} | {t_cnt:<10} | {tot:<8}")
    print("-" * 55)
    print(f"{'TOTAL':<12} | {total_ref:<12} | {total_test:<10} | {total_ref + total_test:<8}")
    print("-" * 55)

    # Setup feature extractor and matcher
    cache_dir = DEFAULT_OUTPUTS_DIR / "cache" / "sift"
    extractor = SIFTFeatureExtractor(max_features=args.max_features, cache_dir=cache_dir)
    matcher = SIFTRansacMatcher(ratio_thresh=args.ratio, ransac_reproj_thresh=5.0)

    # Pre-extract / cache reference features
    print(f"\n[*] Pre-loading / caching features for {total_ref} reference images...")
    ref_features: Dict[Path, Tuple[List, np.ndarray, str]] = {}
    for shoe_id, ref_imgs in reference_dict.items():
        for r_path in ref_imgs:
            kp, desc = extractor.extract(r_path, use_cache=True)
            ref_features[r_path] = (kp, desc, shoe_id)

    # Flatten test queries
    all_queries = []
    for shoe_id, test_imgs in test_dict.items():
        for t_path in test_imgs:
            all_queries.append((t_path, shoe_id))

    print(f"[*] Running evaluation on {len(all_queries)} query test images against {total_ref} references...")

    # Evaluation counters
    correct_top1 = 0
    correct_top3 = 0
    incorrect_top1 = 0

    evaluation_rows = []
    confusion_matrix = np.zeros((len(shoe_names), len(shoe_names)), dtype=int)

    t_eval_start = time.time()

    for query_path, true_shoe in tqdm(all_queries, desc="Evaluating Queries", unit="query"):
        # Extract query features
        q_kp, q_desc = extractor.extract(query_path, use_cache=True)

        if q_desc is None or len(q_desc) < 4:
            # Query has no keypoints
            predicted_shoe = "none"
            top1_score = 0
            is_correct = False
            top3_hit = False
        else:
            # Match against every reference image in gallery
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

            # Aggregate scores by shoe ID
            ranked_shoes = aggregate_shoe_scores(match_results)

            if ranked_shoes and ranked_shoes[0].score > 0:
                best_shoe = ranked_shoes[0]
                predicted_shoe = best_shoe.shoe_id
                top1_score = best_shoe.score
                top3_candidates = [s.shoe_id for s in ranked_shoes[:3]]
            else:
                predicted_shoe = "none"
                top1_score = 0
                top3_candidates = []

            is_correct = (predicted_shoe == true_shoe)
            top3_hit = (true_shoe in top3_candidates)

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

        # Record for CSV: query_image, true_shoe, predicted_shoe, correct, top1_score
        evaluation_rows.append({
            "query_image": query_path.name,
            "true_shoe": true_shoe,
            "predicted_shoe": predicted_shoe,
            "correct": is_correct,
            "top1_score": top1_score,
        })

    eval_duration = time.time() - t_eval_start
    total_queries = len(all_queries)
    top1_acc = (correct_top1 / total_queries * 100) if total_queries > 0 else 0.0
    top3_acc = (correct_top3 / total_queries * 100) if total_queries > 0 else 0.0

    # 2. Save CSV results
    results_csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["query_image", "true_shoe", "predicted_shoe", "correct", "top1_score"]
    with open(results_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(evaluation_rows)

    # 3. Save Confusion Matrix plot
    cm_path = output_dir / "confusion_matrix.png"
    plot_and_save_confusion_matrix(shoe_names, confusion_matrix, cm_path)

    # 4. Print requested clean summary
    print("\n" + "=" * 45)
    print("SIFT + RANSAC EVALUATION")
    print("=" * 45)
    print(f"Shoes: {len(shoe_names)}")
    print(f"Reference images: {total_ref}")
    print(f"Test images: {total_test}\n")
    print(f"Correct predictions: {correct_top1}")
    print(f"Incorrect predictions: {incorrect_top1}\n")
    print(f"Top-1 accuracy: {top1_acc:.1f}%")
    print(f"Top-3 accuracy: {top3_acc:.1f}%")
    print("=" * 45)

    print(f"\n[*] Detailed results saved to:")
    print(f"    - CSV: {results_csv_path}")
    print(f"    - Confusion Matrix: {cm_path}")
    print(f"[*] Total evaluation time: {eval_duration:.2f}s ({eval_duration/total_queries:.2f}s per query).")

    print("\nImportant Evaluation Context:")
    print("  * This is an instance-level closed-set baseline evaluation.")
    print("  * Biometric metrics (FAR, FRR, EER) are not calculated at this stage.")
    print("  * Results reflect classical local feature matching and do not claim production readiness.")
    print("=" * 65)


if __name__ == "__main__":
    main()
