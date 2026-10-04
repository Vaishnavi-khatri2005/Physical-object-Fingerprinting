"""
Feature Matching and RANSAC Geometric Verification Module.
Implements Brute-Force matching with Lowe's ratio test,
RANSAC homography estimation, and shoe-level score aggregation.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class ImageMatchResult:
    """Stores matching statistics between a query image and one reference dataset image."""
    shoe_id: str
    reference_path: Path
    total_raw_matches: int
    good_matches_count: int
    inlier_count: int
    good_matches: List[cv2.DMatch]
    inlier_mask: Optional[np.ndarray]
    homography: Optional[np.ndarray]
    query_keypoints: List[cv2.KeyPoint]
    reference_keypoints: List[cv2.KeyPoint]


@dataclass
class ShoeMatchScore:
    """Aggregated match result for one physical shoe candidate."""
    shoe_id: str
    score: int  # Top inlier count achieved by any reference image of this shoe
    best_reference_image: Path
    best_match_result: ImageMatchResult
    total_inliers_across_shoe: int
    num_images_compared: int


class SIFTRansacMatcher:
    """
    Performs pairwise descriptor matching using BFMatcher,
    applies Lowe's ratio test, and verifies spatial consistency via RANSAC homography.
    """

    def __init__(self, ratio_thresh: float = 0.75, ransac_reproj_thresh: float = 5.0):
        self.ratio_thresh = ratio_thresh
        self.ransac_reproj_thresh = ransac_reproj_thresh
        # SIFT descriptors are float32 vectors, so Euclidean distance (NORM_L2) is appropriate
        self.bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    def match_pair(
        self,
        query_kp: List[cv2.KeyPoint],
        query_desc: Optional[np.ndarray],
        ref_kp: List[cv2.KeyPoint],
        ref_desc: Optional[np.ndarray],
        shoe_id: str = "",
        ref_path: Optional[Path] = None,
    ) -> ImageMatchResult:
        """
        Matches keypoint descriptors between query and reference images,
        filters via Lowe's ratio test, and computes RANSAC inliers.
        """
        ref_path = ref_path or Path("reference")

        # Edge cases: missing descriptors or too few keypoints
        if query_desc is None or ref_desc is None or len(query_desc) < 4 or len(ref_desc) < 4:
            return ImageMatchResult(
                shoe_id=shoe_id,
                reference_path=ref_path,
                total_raw_matches=0,
                good_matches_count=0,
                inlier_count=0,
                good_matches=[],
                inlier_mask=None,
                homography=None,
                query_keypoints=query_kp,
                reference_keypoints=ref_kp,
            )

        # 1. K-Nearest Neighbor matching (k=2)
        try:
            raw_matches = self.bf.knnMatch(query_desc, ref_desc, k=2)
        except Exception:
            raw_matches = []

        # 2. Lowe's Ratio Test
        # Keeps match if nearest neighbor is significantly closer than second nearest neighbor
        good_matches = []
        for match_pair in raw_matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.ratio_thresh * n.distance:
                    good_matches.append(m)

        # 3. RANSAC Geometric Verification
        # At least 4 point correspondences are required to compute a 2D planar homography
        if len(good_matches) < 4:
            return ImageMatchResult(
                shoe_id=shoe_id,
                reference_path=ref_path,
                total_raw_matches=len(raw_matches),
                good_matches_count=len(good_matches),
                inlier_count=0,
                good_matches=good_matches,
                inlier_mask=None,
                homography=None,
                query_keypoints=query_kp,
                reference_keypoints=ref_kp,
            )

        src_pts = np.float32([query_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([ref_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        try:
            homography, mask = cv2.findHomography(
                src_pts,
                dst_pts,
                cv2.RANSAC,
                self.ransac_reproj_thresh
            )
        except Exception:
            homography, mask = None, None

        inliers = int(mask.sum()) if mask is not None else 0

        return ImageMatchResult(
            shoe_id=shoe_id,
            reference_path=ref_path,
            total_raw_matches=len(raw_matches),
            good_matches_count=len(good_matches),
            inlier_count=inliers,
            good_matches=good_matches,
            inlier_mask=mask,
            homography=homography,
            query_keypoints=query_kp,
            reference_keypoints=ref_kp,
        )


def aggregate_shoe_scores(match_results: List[ImageMatchResult]) -> List[ShoeMatchScore]:
    """
    Aggregates image-level match results by unique shoe ID.
    
    The shoe score represents the maximum number of geometrically verified
    inlier keypoints found against any reference photograph of that physical shoe.
    Shoes are ranked in descending order of their inlier score.
    """
    grouped: Dict[str, List[ImageMatchResult]] = {}
    for res in match_results:
        grouped.setdefault(res.shoe_id, []).append(res)

    shoe_scores: List[ShoeMatchScore] = []
    for shoe_id, results in grouped.items():
        # Find best reference image for this shoe based on inlier count
        best_res = max(results, key=lambda r: r.inlier_count)
        total_inliers = sum(r.inlier_count for r in results)

        shoe_scores.append(ShoeMatchScore(
            shoe_id=shoe_id,
            score=best_res.inlier_count,
            best_reference_image=best_res.reference_path,
            best_match_result=best_res,
            total_inliers_across_shoe=total_inliers,
            num_images_compared=len(results),
        ))

    # Rank descending by score (inlier count)
    shoe_scores.sort(key=lambda s: s.score, reverse=True)
    return shoe_scores
