"""Crop-around-mask and feathered paste-back.

Inpainting a crop at the model's native size, instead of the whole image, is the main
memory and speed lever on a 24 GB Mac. It also keeps pixels outside the crop untouched.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


def mask_bbox(mask: Image.Image) -> tuple[int, int, int, int]:
    a = np.asarray(mask) > 127
    if not a.any():
        raise ValueError("empty mask")
    ys, xs = np.where(a)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def context_box(mask: Image.Image, context_scale: float, min_side: int = 64) -> tuple[int, int, int, int]:
    """Square box centred on the mask, clipped to the image, as large as the image allows."""
    W, H = mask.size
    x0, y0, x1, y1 = mask_bbox(mask)
    side = int(max(x1 - x0, y1 - y0) * context_scale)
    side = max(min_side, min(side, W, H))
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    bx0 = min(max(cx - side // 2, 0), W - side)
    by0 = min(max(cy - side // 2, 0), H - side)
    return bx0, by0, bx0 + side, by0 + side


def crop_for_model(image: Image.Image, mask: Image.Image, box, res: int):
    img = image.crop(box).resize((res, res), Image.LANCZOS)
    m = mask.crop(box).resize((res, res), Image.NEAREST)
    return img, m


def paste_back(image: Image.Image, mask: Image.Image, generated: Image.Image, box, feather_px: int) -> Image.Image:
    """Resize the generated crop back and blend it in only where the (feathered) mask is set."""
    w, h = box[2] - box[0], box[3] - box[1]
    gen = generated.convert("RGB").resize((w, h), Image.LANCZOS)
    m = mask.crop(box)
    if feather_px > 0:
        m = m.filter(ImageFilter.MaxFilter(2 * (feather_px // 2) + 1)).filter(ImageFilter.GaussianBlur(feather_px / 2))
    out = image.convert("RGB").copy()
    region = Image.composite(gen, out.crop(box), m)
    out.paste(region, box[:2])
    return out
