#!/usr/bin/env python3
"""
Shoeprint: Biometric & Instance Verification Evaluation (FAR, FRR, EER).

Uses the RootSIFT pipeline to compute genuine and impostor matching score distributions,
sweeps decision thresholds to compute False Acceptance Rate (FAR) and False Rejection Rate (FRR),
locates the Equal Error Rate (EER) threshold, generates diagnostic verification curves,
and supports open-set unknown-object rejection testing on photos_unknown/.

Usage:
    python evaluate_biometrics.py
"""

import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate import discover_shoe_images, split_dataset
from evaluate_rootsift import transform_to_rootsift
from src.config import DEFAULT_OUTPUTS_DIR, SUPPORTED_IMAGE_EXTENSIONS
from src.features import SIFTFeatureExtractor, load_grayscale_image
from src.matcher import SIFTRansacMatcher, aggregate_shoe_scores


def main():
    dataset_dir = Path("photos").resolve()
    unknown_dir = Path("photos_unknown").resolve()
    eval_dir = DEFAULT_OUTPUTS_DIR / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    far_frr_csv_path = eval_dir / "far_frr_eer.csv"
    threshold_results_csv_path = eval_dir / "threshold_results.csv"
    far_frr_plot_path = eval_dir / "far_frr_curve.png"
    score_dist_plot_path = eval_dir / "score_distribution.png"

    # 1. Re-establish deterministic split (seed=42, test_ratio=0.30)
    shoe_dict = discover_shoe_images(dataset_dir)
    shoe_names = sorted(shoe_dict.keys())

    reference_dict, test_dict = split_dataset(shoe_dict, test_ratio=0.30, seed=42)
    total_ref = sum(len(imgs) for imgs in reference_dict.values())
    total_test = sum(len(imgs) for imgs in test_dict.values())

    print("=" * 65)
    print("  Shoeprint: FAR, FRR & EER Verification Evaluation (RootSIFT)")
    print("=" * 65)
    print(f"[*] Dataset:         {dataset_dir}")
    print(f"[*] Known Shoes:     {len(shoe_names)} ({total_ref} ref, {total_test} test)")
    print(f"[*] Evaluation Split: Seed 42, 30% Test Ratio (Identical to previous tests)")
    print(f"[*] Matcher:         RootSIFT (L1 + sqrt) + BFMatcher + Lowe (0.75) + RANSAC (5.0)")
    print("=" * 65)

    # 2. Setup cached feature extractor and matcher
    cache_dir = DEFAULT_OUTPUTS_DIR / "cache" / "sift"
    extractor = SIFTFeatureExtractor(max_features=2500, cache_dir=cache_dir)
    matcher = SIFTRansacMatcher(ratio_thresh=0.75, ransac_reproj_thresh=5.0)

    # 3. Pre-load reference features and convert to RootSIFT
    print(f"\n[*] Pre-loading & transforming {total_ref} reference descriptors to RootSIFT...")
    ref_features: Dict[Path, Tuple[List, Optional[np.ndarray], str]] = {}
    for shoe_id, ref_imgs in reference_dict.items():
        for r_path in ref_imgs:
            kp, desc = extractor.extract(r_path, use_cache=True)
            desc_root = transform_to_rootsift(desc)
            ref_features[r_path] = (kp, desc_root, shoe_id)

    # 4. For each test query, compute genuine score and highest impostor score
    all_queries = []
    for shoe_id, test_imgs in test_dict.items():
        for t_path in test_imgs:
            all_queries.append((t_path, shoe_id))

    print(f"\n[*] Evaluating genuine and impostor scores for {len(all_queries)} queries...")
    query_scores = []
    correct_top1 = 0
    correct_top3 = 0

    for query_path, true_shoe in tqdm(all_queries, desc="Computing Verification Scores", unit="query"):
        q_kp, q_desc = extractor.extract(query_path, use_cache=True)
        q_desc_root = transform_to_rootsift(q_desc)

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

        # Genuine score: score for true_shoe
        genuine_score = 0
        for s in ranked:
            if s.shoe_id == true_shoe:
                genuine_score = s.score
                break

        # Highest impostor score: max score among all other shoes
        impostor_scores = [s.score for s in ranked if s.shoe_id != true_shoe]
        highest_impostor_score = max(impostor_scores) if impostor_scores else 0

        # Identification metrics
        predicted_shoe = ranked[0].shoe_id if ranked and ranked[0].score > 0 else "none"
        top1_score = ranked[0].score if ranked else 0
        top3_candidates = [s.shoe_id for s in ranked[:3]]

        if predicted_shoe == true_shoe:
            correct_top1 += 1
        if true_shoe in top3_candidates:
            correct_top3 += 1

        query_scores.append({
            "query_image": query_path.name,
            "true_shoe": true_shoe,
            "genuine_score": genuine_score,
            "highest_impostor_score": highest_impostor_score,
            "predicted_shoe": predicted_shoe,
            "score_of_predicted_shoe": top1_score,
        })

    total_queries = len(all_queries)
    top1_acc = (correct_top1 / total_queries) * 100
    top3_acc = (correct_top3 / total_queries) * 100

    genuine_list = np.array([q["genuine_score"] for q in query_scores])
    impostor_list = np.array([q["highest_impostor_score"] for q in query_scores])

    # 5. Threshold sweep to calculate FAR, FRR, and EER
    max_observed_score = int(max(np.max(genuine_list), np.max(impostor_list)))
    thresholds = np.arange(0, max_observed_score + 2)

    sweep_records = []
    best_diff = float("inf")
    best_thresh = 0
    best_eer = 0.0
    best_far = 0.0
    best_frr = 0.0

    far_values = []
    frr_values = []

    for theta in thresholds:
        # FRR: genuine comparisons rejected (< theta) / total genuine comparisons
        frr = float(np.mean(genuine_list < theta))
        # FAR: impostor comparisons accepted (>= theta) / total impostor comparisons
        far = float(np.mean(impostor_list >= theta))

        far_values.append(far * 100)
        frr_values.append(frr * 100)

        sweep_records.append({
            "threshold": int(theta),
            "FAR_percent": round(far * 100, 2),
            "FRR_percent": round(frr * 100, 2),
        })

        diff = abs(far - frr)
        if diff < best_diff:
            best_diff = diff
            best_thresh = int(theta)
            best_far = far * 100
            best_frr = frr * 100
            best_eer = ((far + frr) / 2.0) * 100

    # 6. Save outputs/evaluation/far_frr_eer.csv
    with open(far_frr_csv_path, mode="w", newline="", encoding="utf-8") as f:
        fieldnames = ["threshold", "FAR_percent", "FRR_percent"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sweep_records)

    # 7. Assign verification decision at EER threshold and save threshold_results.csv
    for q in query_scores:
        # A genuine query is accepted if its score meets or exceeds the verification threshold
        if q["genuine_score"] >= best_thresh:
            q["decision"] = "ACCEPTED"
        else:
            q["decision"] = "REJECTED"

    with open(threshold_results_csv_path, mode="w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "query_image",
            "true_shoe",
            "genuine_score",
            "highest_impostor_score",
            "predicted_shoe",
            "decision",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for q in query_scores:
            writer.writerow({
                "query_image": q["query_image"],
                "true_shoe": q["true_shoe"],
                "genuine_score": q["genuine_score"],
                "highest_impostor_score": q["highest_impostor_score"],
                "predicted_shoe": q["predicted_shoe"],
                "decision": q["decision"],
            })

    # 8. Generate FAR vs FRR Curve Plot
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    ax.plot(thresholds, far_values, color="#ef4444", linewidth=2.2, label="FAR (False Acceptance Rate)")
    ax.plot(thresholds, frr_values, color="#2563eb", linewidth=2.2, label="FRR (False Rejection Rate)")

    # Mark EER intersection
    ax.axvline(best_thresh, color="#475569", linestyle="--", linewidth=1.5, alpha=0.8, label=f"EER Threshold (theta={best_thresh})")
    ax.plot(best_thresh, best_eer, marker="o", markersize=8, color="#0f172a", zorder=5)
    ax.annotate(
        f"EER = {best_eer:.1f}%\nThreshold = {best_thresh}",
        xy=(best_thresh, best_eer),
        xytext=(best_thresh + max(2, max_observed_score * 0.04), best_eer + 8),
        arrowprops=dict(facecolor="#0f172a", shrink=0.08, width=1.5, headwidth=6),
        fontsize=10,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8fafc", edgecolor="#cbd5e1"),
    )

    ax.set_title("RootSIFT Verification: FAR vs FRR Trade-off Curve", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Decision Threshold (Minimum Inlier Score)", fontsize=11)
    ax.set_ylabel("Error Rate (%)", fontsize=11)
    ax.set_xlim(0, min(120, max_observed_score + 5))
    ax.set_ylim(-2, 102)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="center right", fontsize=10)

    fig.tight_layout()
    plt.savefig(far_frr_plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 9. Generate Score Distribution Plot
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
    bins = np.linspace(0, min(160, max_observed_score), 35)

    ax.hist(impostor_list, bins=bins, alpha=0.65, color="#f97316", edgecolor="#ea580c", label=f"Highest Impostor Scores (Mean: {np.mean(impostor_list):.1f})")
    ax.hist(genuine_list, bins=bins, alpha=0.65, color="#10b981", edgecolor="#059669", label=f"Genuine Scores (Mean: {np.mean(genuine_list):.1f})")

    ax.axvline(best_thresh, color="#dc2626", linestyle="--", linewidth=2.0, label=f"EER Cutoff Threshold ({best_thresh})")

    ax.set_title("RootSIFT Verification: Genuine vs Impostor Score Distribution", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("RANSAC Inlier Matching Score", fontsize=11)
    ax.set_ylabel("Number of Query Images", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=10)

    fig.tight_layout()
    plt.savefig(score_dist_plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 10. Unknown-Object Rejection Evaluation
    unknown_exists = unknown_dir.exists() and any(unknown_dir.iterdir())
    unknown_summary = {}

    if unknown_exists:
        unknown_files = sorted([
            f for f in unknown_dir.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ])
        print(f"\n[*] Evaluating open-set rejection on {len(unknown_files)} unknown images from photos_unknown/...")

        unknown_records = []
        n_correct_rejected = 0
        n_incorrect_accepted = 0

        for u_path in tqdm(unknown_files, desc="Matching Unknown Images", unit="img"):
            u_img = load_grayscale_image(u_path)
            u_kp, u_desc = extractor.sift.detectAndCompute(u_img, None)
            u_desc_root = transform_to_rootsift(u_desc)

            highest_known_score = 0
            if u_desc_root is not None and len(u_desc_root) >= 4:
                match_results = []
                for r_path, (r_kp, r_desc_root, ref_shoe_id) in ref_features.items():
                    res = matcher.match_pair(
                        query_kp=u_kp,
                        query_desc=u_desc_root,
                        ref_kp=r_kp,
                        ref_desc=r_desc_root,
                        shoe_id=ref_shoe_id,
                        ref_path=r_path,
                    )
                    match_results.append(res)
                ranked = aggregate_shoe_scores(match_results)
                highest_known_score = ranked[0].score if ranked else 0

            # Decision: reject if score < EER threshold
            if highest_known_score < best_thresh:
                decision = "REJECTED"
                n_correct_rejected += 1
            else:
                decision = "ACCEPTED"
                n_incorrect_accepted += 1

            unknown_records.append({
                "query_image": u_path.name,
                "highest_known_score": highest_known_score,
                "decision": decision,
            })

        rejection_rate = (n_correct_rejected / max(1, len(unknown_files))) * 100
        unknown_summary = {
            "total_unknown": len(unknown_files),
            "correctly_rejected": n_correct_rejected,
            "incorrectly_accepted": n_incorrect_accepted,
            "rejection_rate": rejection_rate,
        }

        # Save unknown results
        unknown_csv_path = eval_dir / "unknown_results.csv"
        with open(unknown_csv_path, mode="w", newline="", encoding="utf-8") as f:
            fieldnames = ["query_image", "highest_known_score", "decision"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(unknown_records)
    else:
        print("\n[*] 'photos_unknown/' directory does not exist. Open-set unknown rejection skipped (no images invented).")

    # 11. Final Report
    print("\n" + "=" * 65)
    print("BIOMETRIC & INSTANCE VERIFICATION REPORT (ROOTSIFT)")
    print("=" * 65)
    print(f"Top-1 accuracy:           {top1_acc:.1f}% ({correct_top1}/{total_queries})")
    print(f"Top-3 accuracy:           {top3_acc:.1f}% ({correct_top3}/{total_queries})")
    print(f"EER threshold:            {best_thresh}")
    print(f"EER (Equal Error Rate):   {best_eer:.1f}%")
    print(f"  * FAR at threshold:     {best_far:.1f}%")
    print(f"  * FRR at threshold:     {best_frr:.1f}%")

    if unknown_exists:
        print(f"Unknown rejection rate:   {unknown_summary['rejection_rate']:.1f}% ({unknown_summary['correctly_rejected']}/{unknown_summary['total_unknown']} correctly rejected)")
    else:
        print("Unknown rejection rate:   N/A ('photos_unknown/' not provided)")
    print("=" * 65)

    print("\nScore Distributions Summary:")
    print(f"  * Genuine Scores:   Mean = {np.mean(genuine_list):.1f} | Median = {np.median(genuine_list):.1f} | Min = {np.min(genuine_list)} | Max = {np.max(genuine_list)}")
    print(f"  * Impostor Scores:  Mean = {np.mean(impostor_list):.1f} | Median = {np.median(impostor_list):.1f} | Min = {np.min(impostor_list)} | Max = {np.max(impostor_list)}")

    print("\n[*] Generated Artifacts:")
    print(f"    - FAR/FRR/EER CSV:       {far_frr_csv_path}")
    print(f"    - Threshold Results CSV: {threshold_results_csv_path}")
    print(f"    - FAR vs FRR Curve:      {far_frr_plot_path}")
    print(f"    - Score Distribution:    {score_dist_plot_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
