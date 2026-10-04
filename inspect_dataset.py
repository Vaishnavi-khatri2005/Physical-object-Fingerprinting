#!/usr/bin/env python3
"""
Shoeprint: Instance-Level Physical Object Fingerprinting
Dataset Management & Image Inspection CLI Tool

Usage:
    python inspect_dataset.py
    python inspect_dataset.py --dataset path/to/dataset
    python inspect_dataset.py --dataset "C:/Users/.../dataset" --output-dir outputs
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path so modules can be imported directly
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_DATASET_DIR, DEFAULT_OUTPUTS_DIR
from src.contact_sheet import generate_all_contact_sheets
from src.dataset_inspector import DatasetInspector
from src.reporter import export_reports, print_dataset_summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect and validate dataset for physical shoe fingerprinting.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(DEFAULT_DATASET_DIR),
        help="Path to the dataset directory containing shoe_xx folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUTS_DIR),
        help="Directory where summary CSV and contact-sheet previews will be saved.",
    )
    parser.add_argument(
        "--no-previews",
        action="store_true",
        help="Skip generating image contact sheets (faster for large datasets).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_path = Path(args.dataset).resolve()
    output_dir = Path(args.output_dir).resolve()
    previews_dir = output_dir / "previews"
    summary_csv = output_dir / "dataset_summary.csv"
    inventory_csv = output_dir / "image_inventory.csv"

    print("=" * 60)
    print("  Shoeprint: Dataset Management & Image Inspection Module")
    print("=" * 60)
    print(f"[*] Dataset Directory: {dataset_path}")
    print(f"[*] Output Directory:  {output_dir}")

    # Check if dataset path exists
    if not dataset_path.exists():
        print(f"\n[X] Error: Dataset directory not found: {dataset_path}")
        print("    Please create the directory or specify it with --dataset <path>.")
        sys.exit(1)

    print("\n[*] Scanning dataset and verifying image integrity...")
    inspector = DatasetInspector(dataset_path)
    shoes = inspector.scan()

    if not shoes:
        print("[!] No shoe folders found to inspect. Check dataset directory structure.")
        print_dataset_summary({})
        return

    # Print requested console summary
    print_dataset_summary(shoes)

    # Export CSV reports
    print(f"[*] Exporting summary report to: {summary_csv.name}...")
    export_reports(shoes, summary_csv_path=summary_csv, inventory_csv_path=inventory_csv)
    print(f"    - Saved: {summary_csv}")
    print(f"    - Saved detailed inventory: {inventory_csv}")

    # Generate contact sheet previews
    if not args.no_previews:
        print(f"\n[*] Generating visual contact-sheet previews in: {previews_dir}...")
        preview_paths = generate_all_contact_sheets(shoes, output_dir=previews_dir)
        print(f"    - Generated {len(preview_paths)} contact sheet(s).")
        for p in preview_paths[:5]:  # print first few as examples
            print(f"      * {p.name}")
        if len(preview_paths) > 5:
            print(f"      * ... and {len(preview_paths) - 5} more.")
    else:
        print("\n[*] Previews skipped (--no-previews flag passed).")

    print("\n[OK] Inspection complete! Original images were NOT modified.")
    print("=" * 60)


if __name__ == "__main__":
    main()
