"""Shared image labeling for judge evidence.

Burns a banner (e.g. 'FRAME 3 | STEP 6' or 'IMAGE 2 | code cell 14') onto the
top of an evidence PNG so the image is *self-identifying*: the judge reads the
id it cites from the pixels instead of inferring it from attachment order.
Falls back to the raw image on any error.
"""
from __future__ import annotations

from pathlib import Path


def label_image(src: str | Path, dst: str | Path, banner: str) -> str:
    """Return the path of a copy of `src` with `banner` burned on top."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        im = Image.open(src).convert("RGB")
        w, h = im.size
        band = 30
        canvas = Image.new("RGB", (w, h + band), "white")
        canvas.paste(im, (0, band))
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype(
                "/System/Library/Fonts/Supplemental/Arial.ttf", 20)
        except Exception:  # noqa: BLE001
            try:
                import matplotlib
                font = ImageFont.truetype(
                    str(Path(matplotlib.get_data_path(), "fonts", "ttf",
                             "DejaVuSans.ttf")), 20)
            except Exception:  # noqa: BLE001
                font = ImageFont.load_default()
        draw.rectangle([0, 0, w, band], fill="black")
        draw.text((6, 4), banner, fill="white", font=font)
        canvas.save(dst)
        return str(dst)
    except Exception:  # noqa: BLE001
        return str(src)
