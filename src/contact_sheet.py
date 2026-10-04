"""
Contact sheet and preview montage generator.
Creates visual contact sheets for each physical shoe class,
allowing rapid visual verification without touching original images.
"""

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont, ImageOps

from .config import CONTACT_SHEET_CONFIG, DEFAULT_PREVIEWS_DIR
from .dataset_inspector import ImageRecord, ShoeRecord


def _get_font(size: int = 14) -> ImageFont.ImageFont:
    """Attempts to load a standard TrueType font, falling back gracefully to PIL default font."""
    font_names = [
        "arial.ttf",
        "segoeui.ttf",
        "DejaVuSans.ttf",
        "Helvetica.ttf",
        "calibri.ttf",
    ]
    for font_name in font_names:
        try:
            return ImageFont.truetype(font_name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _create_thumbnail_card(
    record: ImageRecord,
    card_width: int,
    card_height: int,
    footer_height: int,
    font: ImageFont.ImageFont,
) -> Image.Image:
    """
    Renders an individual image card containing the thumbnail and file info footer.
    """
    card = Image.new("RGB", (card_width, card_height), (255, 255, 255))
    draw = ImageDraw.Draw(card)

    thumb_area_height = card_height - footer_height

    if record.is_valid:
        try:
            with Image.open(record.path) as img:
                img = ImageOps.exif_transpose(img)
                # Keep aspect ratio while fitting into thumbnail area
                img.thumbnail((card_width - 8, thumb_area_height - 8), Image.Resampling.LANCZOS)
                
                # Center the thumbnail in the upper card area
                offset_x = (card_width - img.width) // 2
                offset_y = (thumb_area_height - img.height) // 2
                
                if img.mode != "RGB":
                    img = img.convert("RGB")
                card.paste(img, (offset_x, offset_y))
        except Exception:
            draw.rectangle([4, 4, card_width - 4, thumb_area_height - 4], fill=(254, 242, 242))
            draw.text((10, thumb_area_height // 2 - 10), "Read Error", fill=(220, 38, 38), font=font)
    else:
        # Invalid image placeholder card
        draw.rectangle([4, 4, card_width - 4, thumb_area_height - 4], fill=(254, 242, 242))
        err_msg = record.error_message or "Invalid File"
        if len(err_msg) > 26:
            err_msg = err_msg[:24] + "..."
        draw.text((10, thumb_area_height // 2 - 12), "CORRUPTED / INVALID", fill=(220, 38, 38), font=font)
        draw.text((10, thumb_area_height // 2 + 6), err_msg, fill=(153, 27, 27), font=font)

    # Footer banner with file info
    footer_bg = (241, 245, 249) if record.is_valid else (254, 226, 226)
    draw.rectangle([0, thumb_area_height, card_width, card_height], fill=footer_bg)

    # Filename
    name = record.filename
    if len(name) > 22:
        name = name[:10] + "..." + name[-9:]

    # Text details
    if record.is_valid:
        dim_str = f"{record.width}x{record.height}"
        info_str = f"{name} ({dim_str})"
        text_color = (30, 41, 59)
    else:
        info_str = f"{name} [FAILED]"
        text_color = (220, 38, 38)

    draw.text((6, thumb_area_height + 6), info_str, fill=text_color, font=font)

    # If duplicate, draw a warning tag in top-right corner
    if record.is_duplicate:
        tag_w, tag_h = 75, 20
        draw.rectangle([card_width - tag_w - 4, 4, card_width - 4, tag_h + 4], fill=(234, 88, 12))
        draw.text((card_width - tag_w + 2, 7), "DUPLICATE", fill=(255, 255, 255), font=font)

    # Border around card
    border_color = (203, 213, 225) if not record.is_duplicate else (249, 115, 22)
    draw.rectangle([0, 0, card_width - 1, card_height - 1], outline=border_color, width=1)

    return card


def generate_shoe_contact_sheet(
    shoe_record: ShoeRecord,
    output_dir: Path = DEFAULT_PREVIEWS_DIR,
    config: Optional[dict] = None,
) -> Optional[Path]:
    """
    Generates a high-quality visual contact sheet for a single shoe.
    Saves the output preview image to outputs/previews/<shoe_id>_preview.png.
    Original dataset images are NEVER modified.
    """
    cfg = config or CONTACT_SHEET_CONFIG
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = shoe_record.records
    total_items = len(records)

    # Fonts
    title_font = _get_font(20)
    meta_font = _get_font(13)
    card_font = _get_font(11)

    thumb_w, thumb_h = cfg["thumb_size"]
    footer_h = cfg["footer_height"]
    card_w = thumb_w
    card_h = thumb_h + footer_h
    padding = cfg["padding"]
    header_h = cfg["header_height"]

    # If no images found, create an informational empty card
    if total_items == 0:
        canvas_w = 600
        canvas_h = header_h + 120
        canvas = Image.new("RGB", (canvas_w, canvas_h), (248, 250, 252))
        draw = ImageDraw.Draw(canvas)
        draw.text((20, 20), f"Shoe ID: {shoe_record.shoe_id}", fill=(15, 23, 42), font=title_font)
        draw.text((20, 50), "Status: Empty folder (No images detected)", fill=(220, 38, 38), font=meta_font)
        out_path = output_dir / f"{shoe_record.shoe_id}_preview.png"
        canvas.save(out_path)
        return out_path

    cols = min(cfg["cols"], total_items)
    rows = math.ceil(total_items / cols)

    canvas_w = (cols * card_w) + ((cols + 1) * padding)
    canvas_h = header_h + (rows * card_h) + ((rows + 1) * padding)

    canvas = Image.new("RGB", (canvas_w, canvas_h), cfg["bg_color"])
    draw = ImageDraw.Draw(canvas)

    # Draw Header Banner
    draw.rectangle([0, 0, canvas_w, header_h], fill=(30, 41, 59))
    draw.text(
        (padding, 12),
        f"Shoe ID: {shoe_record.shoe_id}",
        fill=(255, 255, 255),
        font=title_font,
    )
    meta_text = (
        f"Total: {shoe_record.total_images} photos  |  "
        f"Valid: {shoe_record.valid_images}  |  "
        f"Invalid: {shoe_record.invalid_images}  |  "
        f"Duplicates: {shoe_record.duplicate_images}"
    )
    draw.text((padding, 42), meta_text, fill=(203, 213, 225), font=meta_font)

    # Draw each card in grid
    for idx, record in enumerate(records):
        col_idx = idx % cols
        row_idx = idx // cols

        x = padding + col_idx * (card_w + padding)
        y = header_h + padding + row_idx * (card_h + padding)

        card = _create_thumbnail_card(record, card_w, card_h, footer_h, card_font)
        canvas.paste(card, (x, y))

    out_path = output_dir / f"{shoe_record.shoe_id}_preview.png"
    canvas.save(out_path, quality=cfg.get("quality", 92))
    return out_path


def generate_all_contact_sheets(
    shoes: Dict[str, ShoeRecord],
    output_dir: Path = DEFAULT_PREVIEWS_DIR,
) -> List[Path]:
    """Generates contact sheet previews for all scanned shoes."""
    generated_paths = []
    for shoe_record in shoes.values():
        path = generate_shoe_contact_sheet(shoe_record, output_dir=output_dir)
        if path:
            generated_paths.append(path)
    return generated_paths
