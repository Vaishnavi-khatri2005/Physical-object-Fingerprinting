#!/usr/bin/env python3
"""
Shoeprint: Lightweight Foreground Isolation Prototype.

Tests fast OpenCV-based shoe segmentation on exactly 3 sample images,
generates binary masks, applies them to SIFT keypoint extraction to
exclude background keypoints, and saves visual comparison composites.

Usage:
    python prototype_foreground.py
"""

import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


def _get_font(size: int = 14) -> ImageFont.ImageFont:
    """Loads a TTF font or falls back gracefully to PIL default font."""
    font_names = ["arial.ttf", "segoeui.ttf", "DejaVuSans.ttf", "calibri.ttf"]
    for fn in font_names:
        try:
            return ImageFont.truetype(fn, size)
        except OSError:
            continue
    return ImageFont.load_default()


def load_image_rgb(path: Path) -> np.ndarray:
    """Loads image respecting EXIF orientation as RGB numpy array."""
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        return np.array(img.convert("RGB"), dtype=np.uint8)


def create_foreground_mask_opencv(img_rgb: np.ndarray) -> Tuple[np.ndarray, str, float]:
    """
    Lightweight, fast OpenCV approach to isolate the shoe from floor/table backgrounds.
    
    Steps:
      1. Downscale to max 250px for sub-second processing.
      2. K-Means clustering (K=4) in CIELAB color space.
      3. Identify background clusters by sampling 4 image corners (where floor/table resides).
      4. Invert to retain foreground, morphological closing to fill interior shoe holes.
      5. Extract largest connected component (the physical shoe).
      6. Safe dilation (7px) to preserve outer sole tread and perimeter stitching.
      7. Fallback guard: if mask is degenerate (<10% or >95%), fallback safely to full image.
    """
    h, w = img_rgb.shape[:2]
    max_dim = 250
    scale = max_dim / max(h, w)
    sw, sh = int(w * scale), int(h * scale)
    small = cv2.resize(img_rgb, (sw, sh), interpolation=cv2.INTER_AREA)

    # 1. CIELAB color space
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).astype(np.float32)
    pixels = lab.reshape(-1, 3)

    # 2. K-means clustering (K=4)
    K = 4
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, K, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    label_img = labels.reshape(sh, sw)

    # 3. Sample 4 image corners (outer 8%) as background reference
    cw, ch = max(4, int(sw * 0.08)), max(4, int(sh * 0.08))
    corners = np.concatenate([
        label_img[:ch, :cw].ravel(),
        label_img[:ch, -cw:].ravel(),
        label_img[-ch:, :cw].ravel(),
        label_img[-ch:, -cw:].ravel()
    ])

    corner_counts = np.bincount(corners, minlength=K)
    # Clusters that make up >25% of corners are background
    bg_clusters = [c for c in range(K) if corner_counts[c] / len(corners) > 0.25]
    if not bg_clusters:
        bg_clusters = [int(np.argmax(corner_counts))]

    # Initial foreground
    fg = np.isin(label_img, bg_clusters, invert=True).astype(np.uint8) * 255

    # 4. Morphological hole filling & noise cleanup
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel_close)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel_open)

    # 5. Connected component filtering: keep largest component (the shoe)
    num_labels, comp_labels, stats, centroids = cv2.connectedComponentsWithStats(fg)
    clean_mask = np.zeros_like(fg)

    if num_labels > 1:
        largest_idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        clean_mask[comp_labels == largest_idx] = 255

    # Slight dilation so outer edge stitching & sole tread are not clipped
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    clean_mask = cv2.dilate(clean_mask, kernel_dilate)

    # 6. Fallback guard
    coverage = float(np.mean(clean_mask > 0))
    if coverage < 0.10 or coverage > 0.95:
        # Fallback to full image if segmentation failed/collapsed
        status = "fallback_full"
        full_mask = np.ones((h, w), dtype=np.uint8) * 255
    else:
        status = "segmented"
        full_mask = cv2.resize(clean_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    return full_mask, status, coverage


def render_prototype_composite(
    img_rgb: np.ndarray,
    mask: np.ndarray,
    sift_raw_kps: List[cv2.KeyPoint],
    sift_masked_kps: List[cv2.KeyPoint],
    shoe_id: str,
    filename: str,
    status: str,
    coverage: float,
    output_path: Path,
):
    """
    Renders a 4-panel visual comparison:
      [1] Original Image
      [2] Binary Mask (White = Shoe, Black = Background)
      [3] Masked Overlay (Background dimmed 70%, shoe outlined in green)
      [4] SIFT Keypoints (Green = Kept on shoe, Red = Eliminated background keypoints)
    """
    h, w = img_rgb.shape[:2]
    target_h = 360
    target_w = int(w * (target_h / h))

    # Panel 1: Original Image
    p1 = cv2.resize(img_rgb, (target_w, target_h))

    # Panel 2: Binary Mask
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    p2 = cv2.resize(mask_3ch, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    # Panel 3: Masked Overlay
    overlay = img_rgb.copy()
    overlay[mask == 0] = (overlay[mask == 0] * 0.25).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (34, 197, 94), 3)  # green border
    p3 = cv2.resize(overlay, (target_w, target_h))

    # Panel 4: Keypoints comparison
    kp_canvas = img_rgb.copy()
    # Draw eliminated keypoints in red
    masked_set = set((round(kp.pt[0], 1), round(kp.pt[1], 1)) for kp in sift_masked_kps)
    for kp in sift_raw_kps:
        pt_rounded = (round(kp.pt[0], 1), round(kp.pt[1], 1))
        x, y = int(kp.pt[0]), int(kp.pt[1])
        if pt_rounded not in masked_set:
            cv2.circle(kp_canvas, (x, y), 3, (239, 68, 68), -1)  # Red = eliminated background
        else:
            cv2.circle(kp_canvas, (x, y), 3, (34, 197, 94), -1)  # Green = kept shoe feature
    p4 = cv2.resize(kp_canvas, (target_w, target_h))

    # Combine the 4 panels horizontally
    composite = np.hstack([p1, p2, p3, p4])

    # Convert to PIL to add header banners
    banner_h = 60
    total_w = composite.shape[1]
    total_h = composite.shape[0] + banner_h

    canvas = Image.new("RGB", (total_w, total_h), (15, 23, 42))
    canvas.paste(Image.fromarray(composite), (0, banner_h))

    draw = ImageDraw.Draw(canvas)
    title_font = _get_font(16)
    sub_font = _get_font(12)

    # Header Title
    title = f"Foreground Isolation Prototype: {shoe_id} ({filename})"
    draw.text((15, 10), title, fill=(255, 255, 255), font=title_font)

    # Header Subtitle metrics
    eliminated = len(sift_raw_kps) - len(sift_masked_kps)
    pct_eliminated = (eliminated / max(1, len(sift_raw_kps))) * 100
    info_text = (
        f"Status: {status}  |  Shoe Area: {coverage*100:.1f}%  |  "
        f"Raw SIFT: {len(sift_raw_kps)} pts  |  "
        f"Masked SIFT: {len(sift_masked_kps)} pts  |  "
        f"Background Eliminated: {eliminated} pts ({pct_eliminated:.1f}%)"
    )
    draw.text((15, 34), info_text, fill=(147, 197, 253), font=sub_font)

    # Panel bottom label strip
    panel_labels = [
        "(1) Original Photograph",
        "(2) Binary Mask",
        "(3) Isolated Shoe Overlay",
        "(4) SIFT: Green=Shoe, Red=Filtered",
    ]
    for i, lbl in enumerate(panel_labels):
        lx = i * target_w + 10
        ly = banner_h + 10
        draw.rectangle([lx - 4, ly - 2, lx + len(lbl) * 7 + 4, ly + 18], fill=(0, 0, 0, 180))
        draw.text((lx, ly), lbl, fill=(255, 255, 255), font=sub_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)


def main():
    start_total = time.time()

    # Select exactly 3 sample images from 3 different shoe folders
    samples = [
        ("shoe2", Path("photos/shoe2/WhatsApp Image 2026-10-03 at 10.29.34 PM (2).jpeg")),
        ("shoe3", Path("photos/shoe3/WhatsApp Image 2026-10-03 at 10.32.15 PM.jpeg")),
        ("shoe4", Path("photos/shoe4/WhatsApp Image 2026-10-03 at 10.29.49 PM (1).jpeg")),
    ]

    out_dir = Path("outputs/foreground_masks")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  ShoePrint: Foreground Isolation Prototype (3 Samples)")
    print("=" * 65)
    print(f"[*] Processing 3 sample images...")
    print(f"[*] Output directory: {out_dir.resolve()}\n")

    sift = cv2.SIFT_create(nfeatures=2500)
    results = []

    for idx, (shoe_id, img_path) in enumerate(samples, start=1):
        t0 = time.time()
        img_rgb = load_image_rgb(img_path)
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        # 1. Generate foreground mask
        mask, status, coverage = create_foreground_mask_opencv(img_rgb)

        # 2. Extract SIFT WITHOUT mask
        kps_raw, _ = sift.detectAndCompute(gray, None)

        # 3. Extract SIFT WITH mask
        kps_masked, _ = sift.detectAndCompute(gray, mask=mask)

        elapsed = time.time() - t0

        # 4. Render and save composite visualization
        out_filename = f"prototype_{idx}_{shoe_id}_{img_path.stem}.jpg"
        out_path = out_dir / out_filename
        render_prototype_composite(
            img_rgb=img_rgb,
            mask=mask,
            sift_raw_kps=kps_raw,
            sift_masked_kps=kps_masked,
            shoe_id=shoe_id,
            filename=img_path.name,
            status=status,
            coverage=coverage,
            output_path=out_path,
        )

        eliminated = len(kps_raw) - len(kps_masked)
        pct_eliminated = (eliminated / max(1, len(kps_raw))) * 100

        results.append({
            "sample_num": idx,
            "shoe_id": shoe_id,
            "filename": img_path.name,
            "status": status,
            "coverage_pct": coverage * 100,
            "kps_raw": len(kps_raw),
            "kps_masked": len(kps_masked),
            "eliminated": eliminated,
            "pct_eliminated": pct_eliminated,
            "elapsed_s": elapsed,
            "saved_path": out_path,
        })

        print(f"[{idx}/3] {shoe_id} ({img_path.name[:25]}...):")
        print(f"      - Segmentation Status: {status} ({coverage * 100:.1f}% shoe area)")
        print(f"      - SIFT Raw: {len(kps_raw)} pts -> SIFT Masked: {len(kps_masked)} pts")
        print(f"      - Background keypoints eliminated: {eliminated} ({pct_eliminated:.1f}%)")
        print(f"      - Time: {elapsed:.3f}s")
        print(f"      - Saved: {out_path.name}\n")

    total_time = time.time() - start_total

    print("=" * 65)
    print("PROTOTYPE RESULTS SUMMARY")
    print("=" * 65)
    print(f"Total time elapsed: {total_time:.2f}s (Average: {total_time / 3:.2f}s per image)")
    print(f"Output folder:      {out_dir}")
    print("=" * 65)


if __name__ == "__main__":
    main()
