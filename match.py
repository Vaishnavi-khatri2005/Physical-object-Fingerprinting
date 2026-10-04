#!/usr/bin/env python3
"""
Shoeprint: Instance-Level Physical Object Fingerprinting
SIFT + RANSAC Feature Matching CLI Tool

Usage:
    python match.py --query "path/to/query.jpg" --dataset "photos"
    python match.py --query "path/to/query.jpg" --dataset "photos" --top-k 3
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_OUTPUTS_DIR, SUPPORTED_IMAGE_EXTENSIONS
from src.features import SIFTFeatureExtractor
from src.matcher import SIFTRansacMatcher, aggregate_shoe_scores
from src.visualizer import save_match_visualization


def parse_args():
    parser = argparse.ArgumentParser(
        description="Match a query shoe photograph against a dataset using SIFT + RANSAC.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Path to the query shoe photograph.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="photos",
        help="Path to the dataset directory containing shoe_xx subfolders.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top shoe candidates to return.",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.75,
        help="Lowe's ratio test threshold (closer to 0.7 is stricter).",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=2500,
        help="Maximum SIFT keypoints to extract per image.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Custom file path for the match visualization image.",
    )
    return parser.parse_args()


def discover_dataset_images(dataset_dir: Path):
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


def main():
    args = parse_args()
    query_path = Path(args.query).resolve()
    dataset_dir = Path(args.dataset).resolve()

    # Validate query image path
    if not query_path.exists():
        print(f"[X] Error: Query image does not exist: {query_path}")
        sys.exit(1)

    # Validate dataset path
    if not dataset_dir.exists():
        print(f"[X] Error: Dataset directory does not exist: {dataset_dir}")
        sys.exit(1)

    shoe_dict = discover_dataset_images(dataset_dir)
    if not shoe_dict:
        print(f"[X] Error: No shoe folders or images found in {dataset_dir}")
        sys.exit(1)

    total_ref_images = sum(len(imgs) for imgs in shoe_dict.values())
    print("=" * 65)
    print("  Shoeprint: SIFT + RANSAC Physical Object Fingerprinting")
    print("=" * 65)
    print(f"[*] Query image:       {query_path.name}")
    print(f"[*] Dataset directory: {dataset_dir}")
    print(f"[*] Total shoes:       {len(shoe_dict)} ({total_ref_images} reference photos)")
    print(f"[*] Lowe ratio test:   {args.ratio}")
    print(f"[*] Max SIFT features: {args.max_features}")
    print("=" * 65)

    # Setup feature extractor and matcher
    cache_dir = DEFAULT_OUTPUTS_DIR / "cache" / "sift"
    extractor = SIFTFeatureExtractor(max_features=args.max_features, cache_dir=cache_dir)
    matcher = SIFTRansacMatcher(ratio_thresh=args.ratio, ransac_reproj_thresh=5.0)

    # 1. Extract query features
    print("\n[1/3] Extracting SIFT keypoints and descriptors from query...")
    t0 = time.time()
    query_kp, query_desc = extractor.extract(query_path, use_cache=False)
    if query_desc is None or len(query_desc) == 0:
        print("[X] Error: No SIFT features could be detected in the query image.")
        sys.exit(1)
    print(f"      Detected {len(query_kp)} keypoints in query ({time.time() - t0:.2f}s).")

    # 2. Compare against every image in the dataset
    print(f"\n[2/3] Comparing query against {total_ref_images} reference images...")
    match_results = []
    skipped_self = False

    t_start = time.time()
    for shoe_id, img_paths in shoe_dict.items():
        for ref_path in img_paths:
            # If the query itself is from the dataset, skip matching the exact identical file
            if ref_path.resolve() == query_path.resolve():
                skipped_self = True
                continue

            ref_kp, ref_desc = extractor.extract(ref_path, use_cache=True)
            res = matcher.match_pair(
                query_kp,
                query_desc,
                ref_kp,
                ref_desc,
                shoe_id=shoe_id,
                ref_path=ref_path
            )
            match_results.append(res)

    duration = time.time() - t_start
    print(f"      Completed comparisons in {duration:.2f}s.")
    if skipped_self:
        print("      (Note: Excluded query image itself from reference search to prevent self-matching).")

    # 3. Aggregate shoe-level scores
    print("\n[3/3] Aggregating scores by physical shoe ID...")
    ranked_shoes = aggregate_shoe_scores(match_results)

    # 4. Display results
    print("\n" + "=" * 45)
    print(f"Query: {query_path.name}\n")
    print(f"Top matches:")
    for rank, shoe_score in enumerate(ranked_shoes[:args.top_k], start=1):
        print(f"{rank}. {shoe_score.shoe_id} - score: {shoe_score.score}")
    print("=" * 45)

    # Detailed explanation of the score
    print("\nExplanation of Score:")
    print("  * The score represents the NUMBER of geometrically consistent SIFT keypoint inliers")
    print("    verified by RANSAC homography between the query and reference image.")
    print("  * This is an absolute count of physical surface correspondences (creases, sole textures,")
    print("    wear abrasion, and stitch intersections) - NOT a percentage or statistical accuracy.")
    print("  * Higher scores indicate more shared geometric details on the physical shoe surface.")

    # 5. Save visualization of the best match
    if ranked_shoes and ranked_shoes[0].score > 0:
        best_shoe = ranked_shoes[0]
        best_result = best_shoe.best_match_result

        matches_dir = DEFAULT_OUTPUTS_DIR / "matches"
        if args.output:
            viz_out = Path(args.output).resolve()
        else:
            viz_out = matches_dir / f"best_match_{query_path.stem}.jpg"

        print(f"\n[*] Saving visual match correspondence to: {viz_out.name}...")
        save_match_visualization(query_path, best_result, viz_out)
        print(f"    - Saved: {viz_out}")
        print(f"    - Best matching photo: {best_shoe.shoe_id}/{best_shoe.best_reference_image.name}")
        print(f"    - Inliers drawn: {best_shoe.score} green connecting lines")
    else:
        print("\n[!] No significant geometric matches found across any shoe in the dataset.")

    print("\n[OK] Matching pipeline complete! Original dataset untouched.")
    print("=" * 65)


if __name__ == "__main__":
    main()
