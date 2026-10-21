"""Label recovery: the inpainting mask is not the label.

After generation, re-detect the intended class and keep the detection that best matches the
generated region. The tight box/mask from the detector becomes the label; the sample is
rejected if nothing of that class is found there.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from synthgen.adapters.base import LabelerAdapter
from synthgen.types import Detection


@dataclass
class Verdict:
    accepted: bool
    reason: str
    detection: Detection | None = None
    inside_frac: float = 0.0     # fraction of the detected mask that lies inside the generated region
    fill_frac: float = 0.0       # fraction of the generated region the detected object occupies
    score: float | None = None


def _box_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def detect_class(labeler: LabelerAdapter, image: Image.Image, cls: str, phrases: dict[str, list[str]] | None = None):
    """Detections for one class, querying each of its phrases separately and relabelling to `cls`.
    Detector vocabularies don't match dataset taxonomies: COCO calls a pickup a "truck", while an
    open-vocabulary detector scores it higher as "car" unless asked about "pickup truck"."""
    dets = []
    for p in (phrases or {}).get(cls, [cls]):
        for d in labeler.detect(image, [p]):
            d.label = cls
            dets.append(d)
    return dets


def recover_label(labeler: LabelerAdapter, image: Image.Image, gen_mask: Image.Image, cls: str,
                  contrast: tuple[str, ...] = (), min_score: float = 0.35, min_inside: float = 0.7,
                  min_fill: float = 0.15, same_object_iou: float = 0.5, margin: float = 0.1,
                  phrases: dict[str, list[str]] | None = None) -> Verdict:
    """`contrast`: confusable classes, typically the original class of a replaced object. Each class is
    queried on its own, because open-vocabulary detectors given several phrases assign one label per box
    and hide near-ties. If a contrast class scores within `margin` of the target on the same object,
    the sample is ambiguous and rejected."""
    region = np.asarray(gen_mask.convert("L")) > 127
    best, best_inside, best_fill = None, -1.0, 0.0
    for d in detect_class(labeler, image, cls, phrases):
        if d.score < min_score or d.mask is None:
            continue
        m = np.asarray(d.mask) > 127
        area = m.sum()
        if area == 0:
            continue
        inside = (m & region).sum() / area
        fill = (m & region).sum() / region.sum()
        if inside > best_inside or (inside == best_inside and fill > best_fill):
            best, best_inside, best_fill = d, inside, fill
    if best is None:
        return Verdict(False, f"no '{cls}' detected above {min_score}")
    if best_inside < min_inside:
        return Verdict(False, f"best '{cls}' lies mostly outside the generated region", best, best_inside, best_fill)
    if best_fill < min_fill:
        return Verdict(False, f"'{cls}' too small for the generated region", best, best_inside, best_fill)
    for c in contrast:
        for d in detect_class(labeler, image, c, phrases):
            if d.score > best.score - margin and _box_iou(d.box, best.box) >= same_object_iou:
                return Verdict(False, f"ambiguous: '{cls}' {best.score:.2f} vs '{c}' {d.score:.2f}",
                               best, best_inside, best_fill)
    return Verdict(True, "ok", best, best_inside, best_fill)


def verify_removal(labeler: LabelerAdapter, image: Image.Image, gen_mask: Image.Image, cls: str,
                   min_score: float = 0.35, max_inside: float = 0.5,
                   phrases: dict[str, list[str]] | None = None) -> Verdict:
    """Remove succeeds when no detection of `cls` lies mostly inside the generated region."""
    region = np.asarray(gen_mask.convert("L")) > 127
    for d in detect_class(labeler, image, cls, phrases):
        if d.score < min_score or d.mask is None:
            continue
        m = np.asarray(d.mask) > 127
        if m.sum() and (m & region).sum() / m.sum() >= max_inside:
            return Verdict(False, f"'{cls}' still present in region ({d.score:.2f})", d, float((m & region).sum() / m.sum()))
    return Verdict(True, "ok")


def unlabelled_instances(labeler: LabelerAdapter, image: Image.Image, cls: str, known_boxes: list[tuple],
                         min_score: float = 0.5, phrases: dict[str, list[str]] | None = None) -> list[Detection]:
    """Detections of `cls` anywhere in the image that match no known box (xyxy).

    Source datasets miss instances (COCO left a person behind horses unlabelled). In a hard-negative
    image that teaches the detector to ignore a real object, so such samples are flagged or rejected.
    A detection counts as known if it overlaps a known box (IoU >= 0.3) or lies mostly inside one."""
    out = []
    for d in detect_class(labeler, image, cls, phrases):
        if d.score < min_score:
            continue
        x0, y0, x1, y1 = d.box
        area = max((x1 - x0) * (y1 - y0), 1e-6)
        known = False
        for k in known_boxes:
            ix = max(0.0, min(x1, k[2]) - max(x0, k[0]))
            iy = max(0.0, min(y1, k[3]) - max(y0, k[1]))
            if _box_iou(d.box, k) >= 0.3 or ix * iy / area >= 0.6:
                known = True
                break
        if not known:
            out.append(d)
    return out


def _lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB (uint8, ..., 3) -> CIELAB (D65)."""
    c = rgb.astype(np.float64) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    xyz = c @ np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]).T
    xyz /= np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def color_change(original: Image.Image, result: Image.Image, mask: Image.Image) -> float:
    """75th percentile of per-pixel CIELAB delta E inside the edit mask: how much the edit changed the
    colour. A percentile, not the median, because windows, tyres and shadows keep their colour. On COCO
    cars: no visible change 3, faint 10, clear recolours 19-61."""
    m = np.asarray(mask.convert("L")) > 127
    if not m.any():
        return 0.0
    a = _lab(np.asarray(original.convert("RGB"))[m])
    b = _lab(np.asarray(result.convert("RGB"))[m])
    return float(np.percentile(np.linalg.norm(a - b, axis=1), 75))
