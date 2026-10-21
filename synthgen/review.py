"""Contact sheet for a job: original | region | result (+ recovered box) [| reference], one row per task.

    python -m synthgen.review out/coco_car_regenerate [--max 8]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

from synthgen.config import _build, JobConfig
from synthgen.run import _source

TILE = 320


def _fit(im: Image.Image) -> Image.Image:
    im = im.copy()
    im.thumbnail((TILE, TILE))
    canvas = Image.new("RGB", (TILE, TILE), (24, 24, 24))
    canvas.paste(im, ((TILE - im.width) // 2, (TILE - im.height) // 2))
    return canvas


def sheet(out: Path, max_rows: int) -> Path:
    cfg = _build(JobConfig, json.loads((out / "job.json").read_text())["config"], "job.json")
    source = _source(cfg)
    verdicts = {json.loads(l)["task_id"]: json.loads(l) for l in (out / "verdicts.jsonl").read_text().splitlines()}
    tasks = [json.loads(l) for l in (out / "tasks.jsonl").read_text().splitlines()][:max_rows]
    cols = 4 if any(t.get("reference") for t in tasks) else 3
    rows = []
    for t in tasks:
        tid = t["task_id"]
        orig = source.load_image(t["image_id"])
        mask = Image.open(out / "masks" / f"{tid}.png")
        region = Image.composite(Image.new("RGB", orig.size, (255, 60, 60)), orig, mask.point(lambda v: 110 if v else 0))
        result = Image.open(out / "images" / f"{tid}.jpg").convert("RGB")
        v = verdicts.get(tid, {})
        d = ImageDraw.Draw(result)
        if v.get("box"):
            d.rectangle(v["box"], outline=(0, 255, 0) if v["accepted"] else (255, 0, 0), width=3)
        tiles = [orig, region, result]
        if cols == 4:
            ref = t.get("reference")
            tiles.append(source.load_image(ref["image_id"]) if ref else Image.new("RGB", (8, 8)))
        row = Image.new("RGB", (TILE * cols, TILE + 22), (0, 0, 0))
        for i, im in enumerate(tiles):
            row.paste(_fit(im), (i * TILE, 22))
        status = "ACCEPT" if v.get("accepted") else f"REJECT: {v.get('reason', 'not verified')}"
        extra = f" area x{v['area_ratio']}" if "area_ratio" in v else ""
        ImageDraw.Draw(row).text((4, 4), f"{tid}  '{t['prompt']}'  {status}  score={v.get('score')}{extra}", fill=(255, 255, 255))
        rows.append(row)
    grid = Image.new("RGB", (TILE * cols, len(rows) * (TILE + 22)))
    for i, r in enumerate(rows):
        grid.paste(r, (0, i * (TILE + 22)))
    path = out / "review.jpg"
    grid.save(path, quality=90)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="synthgen.review")
    ap.add_argument("out")
    ap.add_argument("--max", type=int, default=8)
    a = ap.parse_args()
    print(sheet(Path(a.out), a.max))
