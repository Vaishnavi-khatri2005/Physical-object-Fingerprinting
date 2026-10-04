"""
Visualization Module for SIFT+RANSAC Feature Matches.
Draws matched keypoints and connecting lines between query and reference images.
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .matcher import ImageMatchResult


def _load_display_image(path: Path) -> np.ndarray:
    """Loads image respecting EXIF orientation, converts to BGR for OpenCV drawing."""
    with Image.open(path) as pil_img:
        pil_img = ImageOps.exif_transpose(pil_img)
        rgb_arr = np.array(pil_img.convert("RGB"), dtype=np.uint8)
    return cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)


def _get_font(size: int = 16) -> ImageFont.ImageFont:
    """Fallback font loader."""
    font_names = ["arial.ttf", "segoeui.ttf", "DejaVuSans.ttf", "calibri.ttf"]
    for fn in font_names:
        try:
            return ImageFont.truetype(fn, size)
        except OSError:
            continue
    return ImageFont.load_default()


def save_match_visualization(
    query_path: Path,
    match_result: ImageMatchResult,
    output_path: Path,
) -> Path:
    """
    Renders and saves a side-by-side visualization of the query and reference images,
    drawing connecting lines for all geometrically verified RANSAC inliers.
    Original images remain strictly untouched.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    query_img_bgr = _load_display_image(query_path)
    ref_img_bgr = _load_display_image(match_result.reference_path)

    # Inlier mask for cv2.drawMatches
    matches_mask = None
    if match_result.inlier_mask is not None:
        matches_mask = match_result.inlier_mask.ravel().tolist()

    # Draw matches: green lines for inliers
    draw_params = dict(
        matchColor=(0, 230, 0),       # Green lines for inliers
        singlePointColor=None,
        matchesMask=matches_mask,      # Only draw verified inliers
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    matched_canvas = cv2.drawMatches(
        query_img_bgr,
        match_result.query_keypoints,
        ref_img_bgr,
        match_result.reference_keypoints,
        match_result.good_matches,
        None,
        **draw_params
    )

    # Convert canvas to PIL to add clean top banner
    canvas_rgb = cv2.cvtColor(matched_canvas, cv2.COLOR_BGR2RGB)
    pil_canvas = Image.fromarray(canvas_rgb)

    banner_height = 65
    total_w = pil_canvas.width
    total_h = pil_canvas.height + banner_height

    final_img = Image.new("RGB", (total_w, total_h), (24, 32, 47))
    draw = ImageDraw.Draw(final_img)

    title_font = _get_font(18)
    sub_font = _get_font(13)

    # Header text
    draw.text((15, 10), "SIFT + RANSAC Feature Fingerprint Verification", fill=(255, 255, 255), font=title_font)
    info_line = (
        f"Query: {query_path.name}  |  "
        f"Matched: {match_result.shoe_id} ({match_result.reference_path.name})  |  "
        f"Geometrically Verified Inliers (Score): {match_result.inlier_count}"
    )
    draw.text((15, 36), info_line, fill=(147, 197, 253), font=sub_font)

    # Paste matched visualization
    final_img.paste(pil_canvas, (0, banner_height))

    # Save visualization
    final_img.save(output_path, quality=92)
    return output_path
