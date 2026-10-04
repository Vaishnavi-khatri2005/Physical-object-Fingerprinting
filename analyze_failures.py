#!/usr/bin/env python3
"""
Shoeprint: Failure Analysis Module for SIFT + RANSAC Baseline.

Analyzes every incorrectly classified query from evaluation_results.csv,
computes score margins against the true physical shoe, generates comparative
visualizations (Query vs Predicted vs True reference), exports failures.csv,
and computes quantitative failure statistics.

Usage:
    python analyze_failures.py
"""

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
from evaluate import discover_shoe_images, split_dataset
from src.config import DEFAULT_OUTPUTS_DIR
from src.features import SIFTFeatureExtractor
from src.matcher import ImageMatchResult, SIFTRansacMatcher, aggregate_shoe_scores


def parse_args():
    parser = argparse.ArgumentParser(
        description="Failure Analysis for SIFT + RANSAC Shoe Matching.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="photos",
        help="Path to the dataset directory containing shoe_xx subfolders.",
    )
    parser.add_argument(
        "--eval-results",
        type=str,
        default=str(DEFAULT_OUTPUTS_DIR / "evaluation_results.csv"),
        help="Path to the evaluation_results.csv file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUTS_DIR / "failure_analysis"),
        help="Directory to save failure analysis CSV and visualizations.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.30,
        help="Test ratio used during evaluation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used during evaluation.",
    )
    return parser.parse_args()


def _load_display_image(path: Path) -> np.ndarray:
    """Loads image respecting EXIF orientation, converts to BGR for OpenCV."""
    with Image.open(path) as pil_img:
        pil_img = ImageOps.exif_transpose(pil_img)
        rgb_arr = np.array(pil_img.convert("RGB"), dtype=np.uint8)
    return cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)


def _get_font(size: int = 15) -> ImageFont.ImageFont:
    """Loads a TTF font or falls back gracefully to default PIL font."""
    font_names = ["arial.ttf", "segoeui.ttf", "DejaVuSans.ttf", "calibri.ttf"]
    for fn in font_names:
        try:
            return ImageFont.truetype(fn, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_match_panel(
    query_bgr: np.ndarray,
    ref_bgr: np.ndarray,
    match_result: ImageMatchResult,
    banner_title: str,
    banner_color: Tuple[int, int, int],
    line_color: Tuple[int, int, int] = (0, 230, 0),
) -> Image.Image:
    """Renders a single side-by-side match panel with inlier lines and a banner header."""
    matches_mask = None
    if match_result.inlier_mask is not None and match_result.inlier_count > 0:
        matches_mask = match_result.inlier_mask.ravel().tolist()

    draw_params = dict(
        matchColor=line_color,
        singlePointColor=None,
        matchesMask=matches_mask,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    canvas_bgr = cv2.drawMatches(
        query_bgr,
        match_result.query_keypoints,
        ref_bgr,
        match_result.reference_keypoints,
        match_result.good_matches,
        None,
        **draw_params,
    )

    canvas_rgb = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB)
    pil_canvas = Image.fromarray(canvas_rgb)

    banner_h = 45
    w = pil_canvas.width
    panel = Image.new("RGB", (w, pil_canvas.height + banner_h), banner_color)
    draw = ImageDraw.Draw(panel)
    font = _get_font(15)
    draw.text((15, 12), banner_title, fill=(255, 255, 255), font=font)
    panel.paste(pil_canvas, (0, banner_h))

    return panel


def create_comparative_failure_visualization(
    query_path: Path,
    true_shoe: str,
    predicted_shoe: str,
    pred_match_result: ImageMatchResult,
    true_match_result: Optional[ImageMatchResult],
    true_ref_fallback: Optional[Path],
    output_path: Path,
):
    """
    Renders a comparative 2-panel image:
      Top panel: Query vs Predicted Reference (False match)
      Bottom panel: Query vs Best Reference of TRUE shoe
    """
    query_bgr = _load_display_image(query_path)
    pred_ref_bgr = _load_display_image(pred_match_result.reference_path)

    # Top Panel: False match
    top_title = (
        f"[PREDICTED FALSE MATCH] Shoe: {predicted_shoe} ({pred_match_result.reference_path.name})  |  "
        f"Score: {pred_match_result.inlier_count} inliers"
    )
    top_panel = render_match_panel(
        query_bgr,
        pred_ref_bgr,
        pred_match_result,
        banner_title=top_title,
        banner_color=(185, 28, 28),  # Deep red banner
        line_color=(0, 220, 0),
    )

    # Bottom Panel: True shoe match
    if true_match_result is not None:
        true_ref_bgr = _load_display_image(true_match_result.reference_path)
        bot_title = (
            f"[TRUE SHOE REFERENCE] Shoe: {true_shoe} ({true_match_result.reference_path.name})  |  "
            f"Score: {true_match_result.inlier_count} inliers  |  "
            f"Margin Deficit: -{pred_match_result.inlier_count - true_match_result.inlier_count}"
        )
        bot_panel = render_match_panel(
            query_bgr,
            true_ref_bgr,
            true_match_result,
            banner_title=bot_title,
            banner_color=(30, 64, 175),  # Deep blue banner
            line_color=(56, 189, 248),   # Cyan/light-blue inliers
        )
    else:
        # Fallback if no true matches had keypoints
        ref_path = true_ref_fallback or query_path
        true_ref_bgr = _load_display_image(ref_path)
        dummy_res = ImageMatchResult(
            shoe_id=true_shoe,
            reference_path=ref_path,
            total_raw_matches=0,
            good_matches_count=0,
            inlier_count=0,
            good_matches=[],
            inlier_mask=None,
            homography=None,
            query_keypoints=pred_match_result.query_keypoints,
            reference_keypoints=[],
        )
        bot_title = f"[TRUE SHOE REFERENCE] Shoe: {true_shoe} ({ref_path.name})  |  Score: 0 inliers (No geometric match)"
        bot_panel = render_match_panel(
            query_bgr,
            true_ref_bgr,
            dummy_res,
            banner_title=bot_title,
            banner_color=(71, 85, 105),
            line_color=(128, 128, 128),
        )

    # Combine vertically
    max_w = max(top_panel.width, bot_panel.width)
    total_h = top_panel.height + bot_panel.height + 15

    combined = Image.new("RGB", (max_w, total_h), (15, 23, 42))
    combined.paste(top_panel, (0, 0))
    combined.paste(bot_panel, (0, top_panel.height + 15))

    combined.save(output_path, quality=90)


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset).resolve()
    eval_csv_path = Path(args.eval_results).resolve()
    output_dir = Path(args.output_dir).resolve()
    failures_csv_path = output_dir / "failures.csv"

    output_dir.mkdir(parents=True, exist_ok=True)

    if not eval_csv_path.exists():
        print(f"[X] Error: Evaluation results CSV not found: {eval_csv_path}")
        return

    # 1. Read evaluation_results.csv
    eval_rows = []
    with open(eval_csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            eval_rows.append(r)

    correct_rows = [r for r in eval_rows if r["correct"] == "True"]
    incorrect_rows = [r for r in eval_rows if r["correct"] == "False"]

    print("=" * 65)
    print("  Shoeprint: Failure Analysis Module (SIFT + RANSAC Baseline)")
    print("=" * 65)
    print(f"[*] Total evaluated queries: {len(eval_rows)}")
    print(f"[*] Correct predictions:    {len(correct_rows)}")
    print(f"[*] Incorrect predictions:  {len(incorrect_rows)}")
    print("=" * 65)

    # 2. Re-establish deterministic train/test split to retrieve reference images
    shoe_dict = discover_shoe_images(dataset_dir)
    reference_dict, test_dict = split_dataset(shoe_dict, test_ratio=args.test_ratio, seed=args.seed)

    # Find file path mapping for test images
    test_file_map = {}
    for shoe_id, imgs in test_dict.items():
        for p in imgs:
            test_file_map[p.name] = (p, shoe_id)

    # Setup feature extractor and matcher with cache
    cache_dir = DEFAULT_OUTPUTS_DIR / "cache" / "sift"
    extractor = SIFTFeatureExtractor(max_features=2500, cache_dir=cache_dir)
    matcher = SIFTRansacMatcher(ratio_thresh=0.75, ransac_reproj_thresh=5.0)

    # Pre-extract reference features
    ref_features: Dict[Path, Tuple[List, np.ndarray, str]] = {}
    for shoe_id, ref_imgs in reference_dict.items():
        for r_path in ref_imgs:
            kp, desc = extractor.extract(r_path, use_cache=True)
            ref_features[r_path] = (kp, desc, shoe_id)

    # 3. Analyze each failed query
    print("\n[*] Re-evaluating 16 failure cases against gallery to extract margins...")
    failures_data = []

    for idx, inc in enumerate(incorrect_rows, start=1):
        q_name = inc["query_image"]
        true_shoe = inc["true_shoe"]
        pred_shoe = inc["predicted_shoe"]
        pred_score = int(inc["top1_score"])

        q_path, _ = test_file_map[q_name]
        q_kp, q_desc = extractor.extract(q_path, use_cache=True)

        # Match against gallery
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

        # Find best result for predicted shoe
        pred_match_result = None
        for s in ranked:
            if s.shoe_id == pred_shoe:
                pred_match_result = s.best_match_result
                break

        # Find best result for true shoe
        true_score = 0
        true_match_result = None
        for s in ranked:
            if s.shoe_id == true_shoe:
                true_score = s.score
                true_match_result = s.best_match_result
                break

        margin = pred_score - true_score

        # Save comparative visualization
        viz_filename = f"failure_{idx:02d}_{true_shoe}_misclassified_as_{pred_shoe}.jpg"
        viz_path = output_dir / viz_filename
        true_fallback = reference_dict[true_shoe][0] if reference_dict[true_shoe] else None

        if pred_match_result is not None:
            create_comparative_failure_visualization(
                query_path=q_path,
                true_shoe=true_shoe,
                predicted_shoe=pred_shoe,
                pred_match_result=pred_match_result,
                true_match_result=true_match_result,
                true_ref_fallback=true_fallback,
                output_path=viz_path,
            )

        failures_data.append({
            "query_image": q_name,
            "true_shoe": true_shoe,
            "predicted_shoe": pred_shoe,
            "predicted_score": pred_score,
            "correct_shoe_score": true_score,
            "score_margin": margin,
            "viz_path": str(viz_path.resolve()),
        })

    # 4. Save failures.csv
    # Exact required columns: query_image, true_shoe, predicted_shoe, predicted_score, correct_shoe_score, score_margin
    with open(failures_csv_path, mode="w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "query_image",
            "true_shoe",
            "predicted_shoe",
            "predicted_score",
            "correct_shoe_score",
            "score_margin",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for fd in failures_data:
            writer.writerow({
                "query_image": fd["query_image"],
                "true_shoe": fd["true_shoe"],
                "predicted_shoe": fd["predicted_shoe"],
                "predicted_score": fd["predicted_score"],
                "correct_shoe_score": fd["correct_shoe_score"],
                "score_margin": fd["score_margin"],
            })

    # 5. Calculate statistics
    avg_correct_score = (
        sum(int(r["top1_score"]) for r in correct_rows) / len(correct_rows)
        if correct_rows else 0.0
    )
    avg_incorrect_score = (
        sum(fd["predicted_score"] for fd in failures_data) / len(failures_data)
        if failures_data else 0.0
    )
    avg_margin = (
        sum(fd["score_margin"] for fd in failures_data) / len(failures_data)
        if failures_data else 0.0
    )

    failures_per_true_shoe = Counter(fd["true_shoe"] for fd in failures_data)
    confusion_pairs = Counter(f"{fd['true_shoe']} -> {fd['predicted_shoe']}" for fd in failures_data)

    print("\n" + "=" * 55)
    print("FAILURE ANALYSIS STATISTICS")
    print("=" * 55)
    print(f"Average score for CORRECT matches:    {avg_correct_score:.1f} inliers")
    print(f"Average score for INCORRECT matches:  {avg_incorrect_score:.1f} inliers")
    print(f"Average score margin (pred - true):   {avg_margin:.1f} inliers")
    print("-" * 55)

    print("\nNumber of Failures for Each True Shoe:")
    for shoe_id, count in failures_per_true_shoe.most_common():
        total_test_for_shoe = sum(1 for r in eval_rows if r["true_shoe"] == shoe_id)
        pct = (count / total_test_for_shoe * 100) if total_test_for_shoe > 0 else 0
        print(f"  * {shoe_id}: {count} failures / {total_test_for_shoe} queries ({pct:.1f}% failure rate)")

    print("\nMost Common Confusion Pairs (True -> Predicted):")
    for pair, count in confusion_pairs.most_common():
        print(f"  * {pair}: {count} occurrence(s)")
    print("-" * 55)

    # 6. Print 10 most difficult queries (smallest difference between predicted and correct-shoe score)
    # Sort by score_margin ascending (borderline / closest decisions first)
    difficult_queries = sorted(failures_data, key=lambda x: (x["score_margin"], x["predicted_score"]))

    print("\nTop 10 Most Difficult Queries (Smallest Margin Between Predicted & Correct Shoe):")
    print("-" * 90)
    print(f"{'#':<3} | {'Query Image':<32} | {'True Shoe':<10} | {'Predicted':<10} | {'Pred Score':<10} | {'True Score':<10} | {'Margin':<8}")
    print("-" * 90)
    for i, q in enumerate(difficult_queries[:10], start=1):
        q_short = q["query_image"]
        if len(q_short) > 30:
            q_short = q_short[:14] + "..." + q_short[-13:]
        print(
            f"{i:<3} | {q_short:<32} | {q['true_shoe']:<10} | {q['predicted_shoe']:<10} | "
            f"{q['predicted_score']:<10} | {q['correct_shoe_score']:<10} | +{q['score_margin']:<7}"
        )
    print("-" * 90)

    print(f"\n[*] Artifacts successfully saved:")
    print(f"    - Failures CSV: {failures_csv_path}")
    print(f"    - Visualizations: {output_dir} ({len(failures_data)} comparative plots)")
    print("=" * 65)


if __name__ == "__main__":
    main()
