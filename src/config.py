"""
Configuration settings for the Shoeprint dataset inspection module.
"""

from pathlib import Path

# Supported image file extensions (all lowercase with leading dot)
SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tiff",
    ".tif",
}

# Default directory paths relative to the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_DIR = PROJECT_ROOT / "dataset"
DEFAULT_OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DEFAULT_PREVIEWS_DIR = DEFAULT_OUTPUTS_DIR / "previews"
DEFAULT_SUMMARY_CSV = DEFAULT_OUTPUTS_DIR / "dataset_summary.csv"
DEFAULT_INVENTORY_CSV = DEFAULT_OUTPUTS_DIR / "image_inventory.csv"

# Contact sheet rendering configuration
CONTACT_SHEET_CONFIG = {
    "thumb_size": (240, 240),      # (width, height) of each thumbnail
    "cols": 4,                     # Number of thumbnails per row
    "padding": 12,                 # Padding between thumbnails in pixels
    "bg_color": (245, 247, 250),   # Light neutral background
    "card_bg": (255, 255, 255),    # White card background behind each image
    "border_color": (220, 225, 230),
    "header_height": 70,           # Height of the title banner
    "footer_height": 28,           # Height of the thumbnail label banner
    "text_color": (30, 41, 59),    # Dark slate text color
    "accent_color": (37, 99, 235), # Blue accent
    "quality": 92                  # Output JPEG/PNG quality
}
