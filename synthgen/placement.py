"""Placement planner for `insert`: where a new object goes and how big it is.

Heuristic (v1), using the objects already in the scene as evidence of the ground plane:
  * support instances = annotated objects of the support classes (default: the inserted class)
  * the new object's bottom edge is drawn near a support's bottom edge (same "row" of the scene),
    so it stands where comparable objects already stand
  * height follows perspective: with >= 2 supports at different depths, a line height = a*y_bottom + b
    is fitted (lower in the frame = closer = bigger); otherwise the support's height is jittered
  * width from the class's median aspect ratio in the dataset
  * the box must be free space: inside the image, not touching any annotated object (incl. crowd)
  * optional ground check: the bottom strip of the box must lie mostly on a ground mask (e.g. "road",
    segmented by the open-vocabulary labeller), so a car in the right "row" can't land on a pier

Limits: free space only knows annotated objects (unannotated signs or poles can be covered), and
supports must be of a class with similar size to the inserted one.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from synthgen.data.coco import CocoSource


@dataclass
class Placement:
    box: list[float]           # x, y, w, h
    support_ann_ids: list[int]
    size_model: str            # "perspective_fit" | "support_jitter"


def class_aspect(source: CocoSource, cls: str) -> float:
    cid = source.cat_by_name[cls]
    ars = [a["bbox"][2] / a["bbox"][3] for anns in source.anns_by_image.values() for a in anns
           if a["category_id"] == cid and not a["iscrowd"] and a["bbox"][3] > 0 and a["area"] > 1000]
    return float(np.median(ars)) if ars else 1.0


def _expand(b, f):
    x, y, w, h = b
    return (x - f * w, y - f * h, x + w * (1 + f), y + h * (1 + f))


def _intersects(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def sharpness(gray: np.ndarray, box) -> float:
    """Mean absolute Laplacian of the 0-255 grey image around `box` (x, y, w, h), padded by half its size.
    COCO val2017 inserts: an out-of-focus alley 7.3, plain road surfaces 13-19, textured ground 25-70."""
    from scipy import ndimage
    x, y, w, h = box
    win = gray[int(max(0, y - 0.5 * h)):int(y + 1.5 * h), int(max(0, x - 0.5 * w)):int(x + 1.5 * w)]
    return float(np.abs(ndimage.laplace(win.astype(np.float32))).mean()) if win.size else 0.0


def place(source: CocoSource, image_id: int, cls: str, support_classes: list[str], aspect: float,
          rng: random.Random, y_jitter: float = 0.08, size_jitter: float = 0.15, gap: float = 0.05,
          min_height_px: float = 24, min_support_area: float = 1000, attempts: int = 60,
          ground: np.ndarray | None = None, ground_min_frac: float = 0.6) -> Placement | None:
    im = source.images[image_id]
    W, H = im["width"], im["height"]
    anns = source.anns_by_image[image_id]
    sup_ids = {source.cat_by_name[c] for c in support_classes}
    supports = [a for a in anns if a["category_id"] in sup_ids and not a["iscrowd"] and a["area"] >= min_support_area]
    if not supports:
        return None
    bottoms = np.array([a["bbox"][1] + a["bbox"][3] for a in supports])
    heights = np.array([a["bbox"][3] for a in supports])
    fit = None
    if len(supports) >= 2 and np.ptp(bottoms) > 0.05 * H:
        a_, b_ = np.polyfit(bottoms, heights, 1)
        if a_ > 0:              # bigger when lower in the frame, as perspective requires
            fit = (a_, b_)
    occupied = [_expand(a["bbox"], gap) for a in anns]
    edge = max(3, 0.01 * max(W, H))
    for _ in range(attempts):
        s = rng.choice(supports)
        yb = s["bbox"][1] + s["bbox"][3] + rng.uniform(-y_jitter, y_jitter) * H
        if fit:
            h = (fit[0] * yb + fit[1]) * (1 + rng.uniform(-size_jitter, size_jitter) / 2)
        else:
            h = s["bbox"][3] * (1 + rng.uniform(-size_jitter, size_jitter))
        w = h * aspect * (1 + rng.uniform(-size_jitter, size_jitter))
        if h < min_height_px or w > W - 2 * edge:
            continue
        x = rng.uniform(edge, W - edge - w)
        y = yb - h
        if y <= edge or yb >= H - edge:
            continue
        cand = (x, y, x + w, yb)
        if any(_intersects(cand, o) for o in occupied):
            continue
        if ground is not None:
            strip = ground[int(yb - 0.15 * h):int(yb) + 1, int(x):int(x + w) + 1]
            if strip.size == 0 or strip.mean() < ground_min_frac:
                continue
        return Placement([round(x, 1), round(y, 1), round(w, 1), round(h, 1)], [s["id"] for s in supports],
                         "perspective_fit" if fit else "support_jitter")
    return None
