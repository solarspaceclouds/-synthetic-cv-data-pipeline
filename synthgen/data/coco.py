"""COCO-format source and writer. No pycocotools dependency: polygons are rasterised with PIL,
RLE (compressed or uncompressed, as used for crowd regions) is decoded here, output masks are
written as uncompressed RLE."""

from __future__ import annotations

import json
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


class CocoSource:
    def __init__(self, annotations: str | Path, images_dir: str | Path, download_missing: bool = False):
        self.root = Path(images_dir)
        self.download_missing = download_missing
        d = json.loads(Path(annotations).read_text())
        self.images = {im["id"]: im for im in d["images"]}
        self.categories = {c["id"]: c for c in d["categories"]}
        self.cat_by_name = {c["name"]: c["id"] for c in d["categories"]}
        self.anns_by_image: dict[int, list[dict]] = defaultdict(list)
        for a in d["annotations"]:
            self.anns_by_image[a["image_id"]].append(a)

    def class_name(self, cat_id: int) -> str:
        return self.categories[cat_id]["name"]

    def image_path(self, image_id: int) -> Path:
        return self.root / self.images[image_id]["file_name"]

    def load_image(self, image_id: int) -> Image.Image:
        p = self.image_path(image_id)
        if not p.exists():
            url = self.images[image_id].get("coco_url")
            if not (self.download_missing and url):
                raise FileNotFoundError(p)
            p.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(url, p)
        return Image.open(p).convert("RGB")

    @staticmethod
    def ann_mask(ann: dict, size: tuple[int, int]) -> Image.Image | None:
        """Polygon or RLE annotation (crowd included) -> "L" mask. None if there is no segmentation."""
        seg = ann.get("segmentation")
        if isinstance(seg, dict):
            return rle_to_mask(seg)
        if not isinstance(seg, list) or not seg:
            return None
        m = Image.new("L", size, 0)
        draw = ImageDraw.Draw(m)
        for poly in seg:
            if len(poly) >= 6:
                draw.polygon(list(zip(poly[0::2], poly[1::2])), fill=255)
        return m


def _rle_counts(s: str | bytes) -> list[int]:
    """Decode the compressed COCO RLE string (pycocotools rleFrString)."""
    s = s.encode() if isinstance(s, str) else s
    counts, p = [], 0
    while p < len(s):
        x, k, more = 0, 0, True
        while more:
            c = s[p] - 48
            x |= (c & 0x1F) << (5 * k)
            more = bool(c & 0x20)
            p += 1
            k += 1
            if not more and c & 0x10:
                x |= -1 << (5 * k)
        if len(counts) > 2:
            x += counts[-2]
        counts.append(x)
    return counts


def rle_to_mask(rle: dict) -> Image.Image:
    """COCO RLE (column-major; counts as a list or a compressed string) -> "L" mask."""
    h, w = rle["size"]
    counts = rle["counts"] if isinstance(rle["counts"], list) else _rle_counts(rle["counts"])
    flat = np.zeros(h * w, np.uint8)
    pos = 0
    for i, n in enumerate(counts):
        if i % 2:
            flat[pos:pos + n] = 255
        pos += n
    return Image.fromarray(flat.reshape((h, w), order="F"))


def mask_to_rle(mask: Image.Image) -> dict:
    """Uncompressed COCO RLE (column-major), readable by pycocotools."""
    a = (np.asarray(mask) > 127).astype(np.uint8)
    h, w = a.shape
    flat = a.flatten(order="F")
    change = np.flatnonzero(np.diff(flat)) + 1
    bounds = np.concatenate([[0], change, [flat.size]])
    counts = np.diff(bounds).tolist()
    if flat[0] == 1:
        counts = [0] + counts
    return {"size": [h, w], "counts": counts}


def mask_area_bbox(mask: Image.Image) -> tuple[int, list[float]]:
    a = np.asarray(mask) > 127
    if not a.any():
        return 0, [0, 0, 0, 0]
    ys, xs = np.where(a)
    return int(a.sum()), [float(xs.min()), float(ys.min()), float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)]


class CocoWriter:
    """Accumulates synthetic images and annotations; every record carries provenance fields."""

    def __init__(self, categories: list[dict]):
        self.categories = categories
        self.images: list[dict] = []
        self.annotations: list[dict] = []

    def add_image(self, file_name: str, size: tuple[int, int], source_image_id: int, task_id: str) -> int:
        image_id = len(self.images) + 1
        self.images.append({"id": image_id, "file_name": file_name, "width": size[0], "height": size[1],
                            "synthetic": True, "source_image_id": source_image_id, "task_id": task_id})
        return image_id

    def add_ann(self, image_id: int, category_id: int, bbox: list[float], area: float, segmentation,
                iscrowd: int = 0, synthetic: bool = False, **extra) -> None:
        self.annotations.append({"id": len(self.annotations) + 1, "image_id": image_id, "category_id": category_id,
                                 "bbox": bbox, "area": area, "segmentation": segmentation, "iscrowd": iscrowd,
                                 "synthetic": synthetic, **extra})

    def save(self, path: Path, info: dict) -> None:
        path.write_text(json.dumps({"info": info, "images": self.images, "annotations": self.annotations,
                                    "categories": self.categories}))
