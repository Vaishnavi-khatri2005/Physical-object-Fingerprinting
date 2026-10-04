"""
Core inspection module for Shoeprint dataset management.
Responsible for folder discovery, image integrity verification,
metadata extraction, and duplicate detection.
"""

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .config import SUPPORTED_IMAGE_EXTENSIONS


@dataclass
class ImageRecord:
    """Stores metadata and validation status for a single image file."""
    path: Path
    shoe_id: str
    filename: str
    is_valid: bool
    error_message: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    channels: Optional[int] = None
    format: Optional[str] = None
    file_size_bytes: int = 0
    sha256_hash: Optional[str] = None
    dhash: Optional[str] = None
    is_duplicate: bool = False
    duplicate_of: Optional[Path] = None


@dataclass
class ShoeRecord:
    """Stores aggregated inspection metrics for one physical shoe folder."""
    shoe_id: str
    folder_path: Path
    total_images: int = 0
    valid_images: int = 0
    invalid_images: int = 0
    duplicate_images: int = 0
    records: List[ImageRecord] = field(default_factory=list)


def compute_sha256(file_path: Path, chunk_size: int = 65536) -> str:
    """
    Computes the SHA-256 cryptographic hash of a file to detect exact byte-level duplicates.
    Reads in chunks to handle large high-resolution images without excessive memory usage.
    """
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_dhash(image: Image.Image, hash_size: int = 8) -> str:
    """
    Computes a Difference Hash (dHash) for perceptual visual duplicate detection.
    
    How dHash works:
    1. Converts image to grayscale.
    2. Resizes to (hash_size + 1, hash_size) e.g., 9x8 = 72 pixels.
    3. Compares adjacent pixels horizontally: if left pixel > right pixel, bit is 1, else 0.
    4. Yields a 64-bit integer representation resistant to minor lighting/compression differences.
    """
    try:
        # Convert to grayscale and resize smoothly
        resized = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
        pixels = list(resized.getdata())

        # If image has almost zero variance (e.g. flat color or blank exposure), skip perceptual hashing
        if np.std(pixels) < 3.0:
            return ""
        
        # Calculate horizontal gradient differences
        difference = []
        width = hash_size + 1
        for row in range(hash_size):
            row_start = row * width
            for col in range(hash_size):
                pixel_left = pixels[row_start + col]
                pixel_right = pixels[row_start + col + 1]
                difference.append(pixel_left > pixel_right)
                
        # Convert boolean list to a hexadecimal string
        decimal_val = 0
        for index, value in enumerate(difference):
            if value:
                decimal_val |= 1 << index

        # If all bits are 0 or 1, there is zero texture/pattern
        if decimal_val in (0, (1 << (hash_size * hash_size)) - 1):
            return ""

        return f"{decimal_val:016x}"
    except Exception:
        return ""


def hamming_distance(hex_hash1: str, hex_hash2: str) -> int:
    """Calculates bit difference between two hex perceptual hashes."""
    if not hex_hash1 or not hex_hash2:
        return 999
    try:
        n1 = int(hex_hash1, 16)
        n2 = int(hex_hash2, 16)
        return bin(n1 ^ n2).count("1")
    except ValueError:
        return 999


def validate_image_file(file_path: Path, shoe_id: str) -> ImageRecord:
    """
    Safely inspects and verifies an image file without modifying it.
    
    Checks performed:
    1. File existence and non-zero size.
    2. Extension compatibility.
    3. PIL verification (checks file header and structure).
    4. Full buffer decode (catches truncated/partially corrupted files).
    5. Dimensions (width, height) and color mode extraction.
    6. SHA-256 exact hash and dHash calculation.
    """
    record = ImageRecord(
        path=file_path,
        shoe_id=shoe_id,
        filename=file_path.name,
        is_valid=False,
    )

    # Check 1: Check existence and non-zero size
    try:
        stat = file_path.stat()
        record.file_size_bytes = stat.st_size
        if record.file_size_bytes == 0:
            record.error_message = "File is empty (0 bytes)"
            return record
    except OSError as e:
        record.error_message = f"Cannot read file stats: {e}"
        return record

    # Check 2: Check extension
    ext = file_path.suffix.lower()
    if ext not in SUPPORTED_IMAGE_EXTENSIONS:
        record.error_message = f"Unsupported file extension '{ext}'"
        return record

    # Check 3 & 4: Deep validation with Pillow
    try:
        with Image.open(file_path) as img:
            # Check headers
            img.verify()

        # Re-open to actually decode image pixels and extract metadata
        # (verify() closes or invalidates the file pointer in Pillow)
        with Image.open(file_path) as img:
            # Respect EXIF orientation tag if present
            img = ImageOps.exif_transpose(img)
            record.width, record.height = img.size
            record.format = img.format or ext.lstrip(".").upper()
            record.channels = len(img.getbands())

            if record.width <= 0 or record.height <= 0:
                record.error_message = f"Invalid image dimensions: {record.width}x{record.height}"
                return record

            # Compute dHash on decoded image
            record.dhash = compute_dhash(img)

        # Check 5: Secondary check using OpenCV to ensure OpenCV can read it as well
        # Using cv2.imdecode ensures multi-engine compatibility
        img_np = cv2.imdecode(np.fromfile(str(file_path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if img_np is None:
            record.error_message = "OpenCV failed to decode image buffer (corrupted file)"
            return record

        # Compute SHA-256 for exact duplicates
        record.sha256_hash = compute_sha256(file_path)
        record.is_valid = True
        return record

    except UnidentifiedImageError:
        record.error_message = "Unidentified image format or corrupted header"
        return record
    except (OSError, SyntaxError) as e:
        record.error_message = f"Corrupted image data: {e}"
        return record
    except Exception as e:
        record.error_message = f"Unexpected error reading image: {e}"
        return record


class DatasetInspector:
    """
    High-level scanner and analyzer for the physical shoe dataset.
    """

    def __init__(self, dataset_dir: Path):
        self.dataset_dir = Path(dataset_dir)
        self.shoes: Dict[str, ShoeRecord] = {}
        self.seen_sha256: Dict[str, Path] = {}
        self.seen_dhashes: List[Tuple[str, Path]] = []

    def scan(self) -> Dict[str, ShoeRecord]:
        """
        Scans all subfolders in the dataset directory.
        Each subfolder name is treated as a unique physical shoe identifier.
        """
        self.shoes.clear()
        self.seen_sha256.clear()
        self.seen_dhashes.clear()

        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory does not exist: {self.dataset_dir.resolve()}")

        # Discover all subdirectories
        entries = sorted(os.listdir(self.dataset_dir))
        subdirs = [
            self.dataset_dir / entry
            for entry in entries
            if (self.dataset_dir / entry).is_dir()
        ]

        if not subdirs:
            print(f"[!] Warning: No shoe subdirectories found in: {self.dataset_dir.resolve()}")
            return {}

        for folder in subdirs:
            shoe_id = folder.name
            shoe_record = self._inspect_shoe_folder(folder, shoe_id)
            self.shoes[shoe_id] = shoe_record

        return self.shoes

    def _inspect_shoe_folder(self, folder_path: Path, shoe_id: str) -> ShoeRecord:
        """Inspects all files within a single shoe folder."""
        shoe_record = ShoeRecord(shoe_id=shoe_id, folder_path=folder_path)

        # Find all files in the folder (ignoring hidden files)
        all_files = sorted([
            f for f in folder_path.iterdir()
            if f.is_file() and not f.name.startswith(".")
        ])

        shoe_record.total_images = len(all_files)

        for file_path in all_files:
            record = validate_image_file(file_path, shoe_id)

            # Duplicate detection (only for valid images)
            if record.is_valid and record.sha256_hash:
                # 1. Exact SHA-256 match
                if record.sha256_hash in self.seen_sha256:
                    record.is_duplicate = True
                    record.duplicate_of = self.seen_sha256[record.sha256_hash]
                    shoe_record.duplicate_images += 1
                else:
                    self.seen_sha256[record.sha256_hash] = file_path

                    # 2. Visual perceptual duplicate check (exact perceptual match)
                    if record.dhash:
                        for existing_dhash, existing_path in self.seen_dhashes:
                            if hamming_distance(record.dhash, existing_dhash) == 0:
                                record.is_duplicate = True
                                record.duplicate_of = existing_path
                                shoe_record.duplicate_images += 1
                                break
                        if not record.is_duplicate:
                            self.seen_dhashes.append((record.dhash, file_path))

            if record.is_valid:
                shoe_record.valid_images += 1
            else:
                shoe_record.invalid_images += 1

            shoe_record.records.append(record)

        return shoe_record
