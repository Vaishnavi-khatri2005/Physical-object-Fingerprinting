"""
Reporting module for the Shoeprint dataset inspection.
Formats console summary statistics and exports structured CSV reports.
Uses standard library csv with optional pandas integration for maximum robustness.
"""

import csv
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from .config import DEFAULT_INVENTORY_CSV, DEFAULT_SUMMARY_CSV
from .dataset_inspector import ShoeRecord


def export_reports(
    shoes: Dict[str, ShoeRecord],
    summary_csv_path: Path = DEFAULT_SUMMARY_CSV,
    inventory_csv_path: Path = DEFAULT_INVENTORY_CSV,
) -> Tuple[Path, Path]:
    """
    Exports summary and detailed image inventory CSVs.
    Outputs:
      1. dataset_summary.csv with exact columns: shoe_id, image_count, valid_images, invalid_images
      2. image_inventory.csv with detailed image-level audit data.
    """
    summary_csv_path = Path(summary_csv_path)
    inventory_csv_path = Path(inventory_csv_path)

    summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
    inventory_csv_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Write dataset_summary.csv
    summary_headers = ["shoe_id", "image_count", "valid_images", "invalid_images"]
    sorted_shoe_ids = sorted(shoes.keys())

    with open(summary_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_headers)
        writer.writeheader()
        for shoe_id in sorted_shoe_ids:
            record = shoes[shoe_id]
            writer.writerow({
                "shoe_id": shoe_id,
                "image_count": record.total_images,
                "valid_images": record.valid_images,
                "invalid_images": record.invalid_images,
            })

    # 2. Write image_inventory.csv
    inventory_headers = [
        "shoe_id",
        "filename",
        "is_valid",
        "error_message",
        "width",
        "height",
        "channels",
        "format",
        "file_size_kb",
        "is_duplicate",
        "duplicate_of",
        "file_path",
    ]

    with open(inventory_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=inventory_headers)
        writer.writeheader()
        for shoe_id in sorted_shoe_ids:
            shoe_rec = shoes[shoe_id]
            for img_rec in shoe_rec.records:
                writer.writerow({
                    "shoe_id": shoe_id,
                    "filename": img_rec.filename,
                    "is_valid": img_rec.is_valid,
                    "error_message": img_rec.error_message or "",
                    "width": img_rec.width if img_rec.width is not None else "",
                    "height": img_rec.height if img_rec.height is not None else "",
                    "channels": img_rec.channels if img_rec.channels is not None else "",
                    "format": img_rec.format or "",
                    "file_size_kb": round(img_rec.file_size_bytes / 1024, 2),
                    "is_duplicate": img_rec.is_duplicate,
                    "duplicate_of": str(img_rec.duplicate_of.name) if img_rec.duplicate_of else "",
                    "file_path": str(img_rec.path.resolve()),
                })

    return summary_csv_path, inventory_csv_path


def print_dataset_summary(shoes: Dict[str, ShoeRecord]) -> None:
    """
    Prints the dataset summary to console in the clean, readable format requested:
    
    Total shoes: 10
    Total images: 127
    Average images/shoe: 12.7
    Minimum images/shoe: 10
    Maximum images/shoe: 15
    Invalid images: 0
    """
    total_shoes = len(shoes)

    if total_shoes == 0:
        print("\n" + "=" * 45)
        print("DATASET INSPECTION SUMMARY")
        print("=" * 45)
        print("Total shoes: 0")
        print("Total images: 0")
        print("Average images/shoe: 0.0")
        print("Minimum images/shoe: 0")
        print("Maximum images/shoe: 0")
        print("Invalid images: 0")
        print("=" * 45)
        return

    counts = [rec.total_images for rec in shoes.values()]
    total_images = sum(counts)
    avg_images = total_images / total_shoes if total_shoes > 0 else 0.0
    min_images = min(counts) if counts else 0
    max_images = max(counts) if counts else 0
    total_invalid = sum(rec.invalid_images for rec in shoes.values())
    total_duplicates = sum(rec.duplicate_images for rec in shoes.values())

    # Format breakdown and resolution stats
    formats = Counter()
    resolutions = []
    for shoe_rec in shoes.values():
        for img_rec in shoe_rec.records:
            if img_rec.is_valid and img_rec.format:
                formats[img_rec.format] += 1
            if img_rec.is_valid and img_rec.width and img_rec.height:
                resolutions.append((img_rec.width, img_rec.height))

    print("\n" + "=" * 45)
    print("DATASET INSPECTION SUMMARY")
    print("=" * 45)
    print(f"Total shoes: {total_shoes}")
    print(f"Total images: {total_images}")
    print(f"Average images/shoe: {avg_images:.1f}")
    print(f"Minimum images/shoe: {min_images}")
    print(f"Maximum images/shoe: {max_images}")
    print(f"Invalid images: {total_invalid}")
    print("=" * 45)

    if total_duplicates > 0:
        print(f"[!] Warning: Detected {total_duplicates} duplicate/near-duplicate image(s).")
        print("    Check outputs/image_inventory.csv for specific filenames.")

    if formats:
        format_strs = [f"{fmt}: {cnt}" for fmt, cnt in formats.most_common()]
        print(f"Detected formats: {', '.join(format_strs)}")

    if resolutions:
        widths, heights = zip(*resolutions)
        min_res = f"{min(widths)}x{min(heights)}"
        max_res = f"{max(widths)}x{max(heights)}"
        print(f"Resolution range: {min_res} to {max_res}")

    print("=" * 45 + "\n")
