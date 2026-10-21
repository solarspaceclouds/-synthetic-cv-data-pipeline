"""Reference bank for reference-conditioned operations: crops of other instances of a class,
drawn only from the same source split so test-set objects never leak into training images."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
from PIL import Image

from synthgen.data.coco import CocoSource


@dataclass
class RefEntry:
    image_id: int
    ann_id: int
    bbox: list[float]


def _iou(a, b) -> float:
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


class ReferenceBank:
    """Only clean instances qualify as references: not cut off by the border and not overlapped by
    another annotated object. An occluded reference makes the generator copy the occluder too."""

    def __init__(self, source: CocoSource, classes: list[str], min_area: float, rng: random.Random,
                 max_overlap_iou: float = 0.05, max_uses: int = 2, embedder=None):
        self.source, self.rng, self.max_uses, self.embedder = source, rng, max_uses, embedder
        self.reject = None                  # optional (entry, cls) -> bool, e.g. seen through a fence or window
        self._rejected: dict[int, bool] = {}
        self.uses: dict[int, int] = {}
        self._lum: dict[tuple, float] = {}
        self.entries: dict[str, list[RefEntry]] = {c: [] for c in classes}
        for image_id, anns in source.anns_by_image.items():
            im = source.images[image_id]
            edge = max(3, 0.01 * max(im["width"], im["height"]))
            for a in anns:
                c = source.class_name(a["category_id"])
                if c not in self.entries or a["iscrowd"] or a["area"] < min_area or not a.get("segmentation"):
                    continue
                x, y, w, h = a["bbox"]
                if x <= edge or y <= edge or x + w >= im["width"] - edge or y + h >= im["height"] - edge:
                    continue
                if any(_iou(a["bbox"], o["bbox"]) > max_overlap_iou for o in anns if o["id"] != a["id"]):
                    continue
                self.entries[c].append(RefEntry(image_id, a["id"], a["bbox"]))

    def _is_greyscale(self, image_id: int) -> bool:
        """Black-and-white photos make the generator insert a grey object into a colour scene."""
        key = ("grey", image_id)
        if key not in self._lum:
            hsv = np.asarray(self.source.load_image(image_id).resize((64, 64)).convert("HSV"))
            self._lum[key] = float(hsv[..., 1].mean()) < 12
        return bool(self._lum[key])

    def _luminance(self, image_id: int, bbox: list[float]) -> float:
        """Mean luminance in [0, 1] of the area around an object (bbox grown 2x): a cheap proxy for the
        local lighting. Whole-image brightness misses e.g. a car in shadow on a sunny street."""
        key = (image_id, tuple(bbox))
        if key not in self._lum:
            im = self.source.load_image(image_id).convert("L")
            x, y, w, h = bbox
            box = (max(0, x - w / 2), max(0, y - h / 2), min(im.width, x + 1.5 * w), min(im.height, y + 1.5 * h))
            self._lum[key] = float(np.asarray(im.crop(tuple(map(int, box)))).mean() / 255)
        return self._lum[key]

    def _embedding(self, image_id: int, ann_id: int):
        crop, _ = reference_crop_and_mask(self.source, image_id, ann_id, pad=0.05, white_bg=True)
        return self.embedder.embed(crop, key=(image_id, ann_id))

    def pick(self, cls: str, exclude_image: int, target_bbox: list[float], match: str,
             target_ann_id: int | None = None) -> RefEntry | None:
        """`exclude_image` is the target image: never its own instances, and the lighting reference.
        `target_ann_id` (the object being replaced) enables the visual term of `match: visual`."""
        pool = [e for e in self.entries.get(cls, [])
                if e.image_id != exclude_image and self.uses.get(e.ann_id, 0) < self.max_uses]
        if not pool:
            return None
        e = self._pick(pool, exclude_image, target_bbox, match, target_ann_id, cls)
        if e is not None:
            self.uses[e.ann_id] = self.uses.get(e.ann_id, 0) + 1
        return e

    def _usable(self, e, cls) -> bool:
        if self._is_greyscale(e.image_id):
            return False
        if self.reject is not None:
            if e.ann_id not in self._rejected:
                self._rejected[e.ann_id] = bool(self.reject(e, cls))
            return not self._rejected[e.ann_id]
        return True

    def _pick(self, pool, exclude_image, target_bbox, match, target_ann_id=None, cls=None):
        if match == "random":
            for e in self.rng.sample(pool, len(pool)):
                if self._usable(e, cls):
                    return e
            return None
        # best match among a random shortlist, so picks vary across seeds
        ar = target_bbox[2] / max(target_bbox[3], 1)
        if match not in ("aspect", "aspect_light", "visual"):
            raise ValueError(f"unknown reference match {match!r} (random | aspect | aspect_light | visual)")
        short = self.rng.sample(pool, min({"aspect": 16, "aspect_light": 32, "visual": 48}[match], len(pool)))
        target_vec = None
        if match == "visual" and target_ann_id is not None and self.embedder is not None:
            target_vec = self._embedding(exclude_image, target_ann_id)

        def cost(e):
            c = abs(np.log((e.bbox[2] / max(e.bbox[3], 1)) / ar))
            if match in ("aspect_light", "visual"):
                # AnyDoor/MimicBrush copy the reference's lighting: a night reference gives a dark object in daylight
                c += 3.0 * abs(self._luminance(e.image_id, e.bbox) - self._luminance(exclude_image, target_bbox))
            if target_vec is not None:
                # viewpoint/shape: a car seen from above shouldn't replace one seen from street level
                c += 2.0 * (1.0 - float(self._embedding(e.image_id, e.ann_id) @ target_vec))
            return c

        short = [e for e in short if self._usable(e, cls)]
        return min(short, key=cost) if short else None

    def crop(self, entry: RefEntry, pad: float = 0.1, white_bg: bool = True) -> Image.Image:
        return reference_crop(self.source, entry.image_id, entry.ann_id, pad, white_bg)


def reference_crop(source: CocoSource, image_id: int, ann_id: int, pad: float = 0.1, white_bg: bool = True) -> Image.Image:
    """Padded crop of one instance, optionally cut out onto white (what AnyDoor-style models expect)."""
    return reference_crop_and_mask(source, image_id, ann_id, pad, white_bg)[0]


def reference_crop_and_mask(source: CocoSource, image_id: int, ann_id: int, pad: float = 0.1,
                            white_bg: bool = True) -> tuple[Image.Image, Image.Image]:
    image = source.load_image(image_id)
    ann = next(a for a in source.anns_by_image[image_id] if a["id"] == ann_id)
    x, y, w, h = ann["bbox"]
    box = (int(max(0, x - pad * w)), int(max(0, y - pad * h)),
           int(min(image.width, x + w * (1 + pad))), int(min(image.height, y + h * (1 + pad))))
    crop = image.crop(box)
    m = CocoSource.ann_mask(ann, image.size).crop(box)
    if white_bg:
        crop = Image.composite(crop, Image.new("RGB", crop.size, (255, 255, 255)), m)
    return crop, m
