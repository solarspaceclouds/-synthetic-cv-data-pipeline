"""Mask shapes for the generation region. The exact instance outline leaks the old object's
silhouette into the new one; dilated/bbox/convex-hull masks give the generator room."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def dilate(mask: Image.Image, px: int) -> Image.Image:
    return mask.filter(ImageFilter.MaxFilter(2 * px + 1)) if px > 0 else mask


def bbox_mask(mask: Image.Image) -> Image.Image:
    a = np.asarray(mask) > 127
    ys, xs = np.where(a)
    out = Image.new("L", mask.size, 0)
    ImageDraw.Draw(out).rectangle((xs.min(), ys.min(), xs.max(), ys.max()), fill=255)
    return out


def _hull(points: np.ndarray) -> list[tuple[int, int]]:
    pts = sorted(set(map(tuple, points.tolist())))
    if len(pts) < 3:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def convex_hull_mask(mask: Image.Image) -> Image.Image:
    a = np.asarray(mask) > 127
    edge = a & ~np.asarray(Image.fromarray(a.astype(np.uint8) * 255).filter(ImageFilter.MinFilter(3)), dtype=bool)
    ys, xs = np.where(edge if edge.any() else a)
    out = Image.new("L", mask.size, 0)
    ImageDraw.Draw(out).polygon(_hull(np.stack([xs, ys], 1)), fill=255)
    return out


def shape_mask(mask: Image.Image, shape: str, dilate_px: int) -> Image.Image:
    if shape == "exact":
        base = mask
    elif shape == "bbox":
        base = bbox_mask(mask)
    elif shape == "convex_hull":
        base = convex_hull_mask(mask)
    else:
        raise ValueError(f"unknown mask shape {shape!r}")
    return dilate(base, dilate_px)
