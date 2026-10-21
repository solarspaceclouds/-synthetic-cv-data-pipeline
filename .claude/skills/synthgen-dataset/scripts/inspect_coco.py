"""Summarise a COCO annotation file to choose target classes and thresholds for a synthgen job.

For each class: instance count, how many pass the default target filters (non-crowd polygon,
area >= min_area, not touching the border), and how many would qualify as clean references
(area >= ref_min_area, not truncated, no box overlap with other objects).

    .venv/bin/python .claude/skills/synthgen-dataset/scripts/inspect_coco.py data/coco/annotations/instances_val2017.json
    ... --classes car truck person --min-area 4000 --ref-min-area 16000
"""

import argparse
import json
from collections import defaultdict


def iou(a, b):
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("annotations")
    ap.add_argument("--classes", nargs="*", help="only these classes (default: all)")
    ap.add_argument("--min-area", type=float, default=4000)
    ap.add_argument("--ref-min-area", type=float, default=16000)
    ap.add_argument("--top", type=int, default=30, help="rows to print when --classes is not given")
    a = ap.parse_args()

    d = json.load(open(a.annotations))
    cats = {c["id"]: c["name"] for c in d["categories"]}
    images = {im["id"]: im for im in d["images"]}
    by_image = defaultdict(list)
    for ann in d["annotations"]:
        by_image[ann["image_id"]].append(ann)

    stats = defaultdict(lambda: {"instances": 0, "images": set(), "targets": 0, "refs": 0, "crowd_or_rle": 0})
    for image_id, anns in by_image.items():
        im = images[image_id]
        W, H = im["width"], im["height"]
        edge = max(3, 0.01 * max(W, H))
        for ann in anns:
            name = cats[ann["category_id"]]
            if a.classes and name not in a.classes:
                continue
            s = stats[name]
            s["instances"] += 1
            s["images"].add(image_id)
            if ann.get("iscrowd") or not isinstance(ann.get("segmentation"), list):
                s["crowd_or_rle"] += 1
                continue
            x, y, w, h = ann["bbox"]
            inside = x > edge and y > edge and x + w < W - edge and y + h < H - edge
            if ann["area"] >= a.min_area and inside:
                s["targets"] += 1
            if ann["area"] >= a.ref_min_area and inside and \
                    not any(iou(ann["bbox"], o["bbox"]) > 0.05 for o in anns if o["id"] != ann["id"]):
                s["refs"] += 1

    print(f"{a.annotations}: {len(images)} images, {len(d['annotations'])} annotations, {len(cats)} classes")
    print(f"targets = non-crowd polygon, area >= {a.min_area:g}, not touching the border "
          f"(before the overlap filter, which typically removes another 30-50% in street scenes)")
    print(f"refs    = also area >= {a.ref_min_area:g} and no box overlap with other objects\n")
    rows = sorted(stats.items(), key=lambda kv: -kv[1]["instances"])
    if not a.classes:
        rows = rows[: a.top]
    print(f"{'class':20s} {'instances':>9s} {'images':>7s} {'targets':>8s} {'refs':>6s} {'crowd/rle':>9s}")
    for name, s in rows:
        print(f"{name:20s} {s['instances']:9d} {len(s['images']):7d} {s['targets']:8d} {s['refs']:6d} {s['crowd_or_rle']:9d}")
    missing = [c for c in (a.classes or []) if c not in stats]
    if missing:
        print(f"\nnot found: {missing}; available: {sorted(set(cats.values()))}")


if __name__ == "__main__":
    main()
