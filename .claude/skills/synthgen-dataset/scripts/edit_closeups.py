"""Full-resolution before/after close-ups of every edit in a synthgen job.

The contact sheet (synthgen.review) shrinks whole images to thumbnails, which hides exactly the
failures verification can't catch: leftover shadows and reflections, fragments of a removed object,
smears, objects left floating. This writes one image per task: original crop | result crop, around
the edit region with context, at native resolution (upscaled if the edit is small).

    .venv/bin/python .claude/skills/synthgen-dataset/scripts/edit_closeups.py out/<job>
    ... --accepted-only --context 1.5 --min-side 384

Writes out/<job>/closeups/<task>.jpg and prints the paths; open them one by one.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--accepted-only", action="store_true")
    ap.add_argument("--context", type=float, default=1.0, help="margin around the edit, as a multiple of its size")
    ap.add_argument("--min-side", type=int, default=384, help="upscale small crops to at least this size")
    a = ap.parse_args()
    out = Path(a.out)

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))   # repo root
    from synthgen.config import JobConfig, _build
    from synthgen.run import _source

    cfg = _build(JobConfig, json.loads((out / "job.json").read_text())["config"], "job.json")
    source = _source(cfg)
    verdicts = {json.loads(l)["task_id"]: json.loads(l) for l in (out / "verdicts.jsonl").read_text().splitlines() if l}
    tasks = [json.loads(l) for l in (out / "tasks.jsonl").read_text().splitlines() if l]
    (out / "closeups").mkdir(exist_ok=True)
    for t in tasks:
        tid = t["task_id"]
        v = verdicts.get(tid)
        if not (out / "images" / f"{tid}.jpg").exists() or (a.accepted_only and not (v and v["accepted"])):
            continue
        orig = source.load_image(t["image_id"])
        res = Image.open(out / "images" / f"{tid}.jpg").convert("RGB")
        m = np.asarray(Image.open(out / "masks" / f"{tid}.png")) > 127
        ys, xs = np.where(m)
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        pw, ph = (x1 - x0) * a.context, (y1 - y0) * a.context
        box = (int(max(0, x0 - pw)), int(max(0, y0 - ph)), int(min(orig.width, x1 + pw)), int(min(orig.height, y1 + ph)))
        crops = [orig.crop(box), res.crop(box)]
        s = max(1.0, a.min_side / min(crops[0].size))
        crops = [c.resize((round(c.width * s), round(c.height * s)), Image.LANCZOS) for c in crops]
        W, H = crops[0].size
        sheet = Image.new("RGB", (2 * W + 8, H + 24), "black")
        sheet.paste(crops[0], (0, 24))
        sheet.paste(crops[1], (W + 8, 24))
        status = "no verdict" if not v else ("ACCEPT" if v["accepted"] else f"REJECT: {v['reason']}")
        ImageDraw.Draw(sheet).text((4, 5), f"{tid}  before | after  {status}", fill="white")
        path = out / "closeups" / f"{tid}.jpg"
        sheet.save(path, quality=92)
        print(path)


if __name__ == "__main__":
    main()
