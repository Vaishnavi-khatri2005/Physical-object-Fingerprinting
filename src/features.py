"""
SIFT Feature Extraction Module.
Detects local scale-invariant keypoints and computes 128-dimensional
gradient orientation descriptors from physical shoe images.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps


def load_grayscale_image(image_path: Path) -> np.ndarray:
    """
    Loads an image safely, respects EXIF orientation, and converts to grayscale.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Use Pillow to handle EXIF rotation tags common in mobile phone photos
    with Image.open(image_path) as pil_img:
        pil_img = ImageOps.exif_transpose(pil_img)
        # Convert to single-channel 8-bit grayscale
        gray_pil = pil_img.convert("L")
        gray_arr = np.array(gray_pil, dtype=np.uint8)

    return gray_arr


class SIFTFeatureExtractor:
    """
    Extracts SIFT (Scale-Invariant Feature Transform) keypoints and descriptors.
    Supports optional disk caching to speed up subsequent queries.
    """

    def __init__(self, max_features: int = 2000, cache_dir: Optional[Path] = None):
        self.max_features = max_features
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Initialize OpenCV SIFT detector
        self.sift = cv2.SIFT_create(nfeatures=self.max_features)

    def extract(self, image_path: Path, use_cache: bool = True) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        """
        Extracts SIFT keypoints and descriptors for an image file.
        Returns:
            keypoints: List of cv2.KeyPoint objects
            descriptors: (N, 128) float32 numpy array or None if no features found
        """
        image_path = Path(image_path)

        # Check cache if enabled
        cache_file = None
        if self.cache_dir and use_cache:
            # Hash filename and file size to generate cache key
            stat = image_path.stat()
            cache_stem = f"{image_path.stem}_{stat.st_size}_{stat.st_mtime_ns}"
            cache_file = self.cache_dir / f"{cache_stem}.npz"
            if cache_file.exists():
                return self._load_from_cache(cache_file)

        # Read image in grayscale
        gray = load_grayscale_image(image_path)

        # Detect keypoints and compute descriptors
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)

        # Save to cache if enabled
        if cache_file is not None and keypoints is not None and descriptors is not None:
            self._save_to_cache(cache_file, keypoints, descriptors)

        return keypoints, descriptors

    def _save_to_cache(self, cache_file: Path, keypoints: List[cv2.KeyPoint], descriptors: np.ndarray) -> None:
        """Serializes OpenCV KeyPoints and descriptor array to compressed .npz file."""
        try:
            kp_data = np.array([
                (kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave, kp.class_id)
                for kp in keypoints
            ], dtype=np.float32)
            np.savez_compressed(cache_file, kp=kp_data, desc=descriptors)
        except Exception:
            pass  # Fall back to un-cached if writing fails

    def _load_from_cache(self, cache_file: Path) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        """Deserializes KeyPoints and descriptors from compressed .npz file."""
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
            # If cache corrupted, return empty to trigger re-computation
            return [], None
