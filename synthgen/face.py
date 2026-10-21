"""Face jobs: part masks from the face parser, re-parsing for label recovery, parse-map export.

Labels produced per accepted image:
  * labels/<task>.png  full re-parsed label map (CelebAMask-HQ ids, see face_labels.json)
  * masks/<task>.png   the edited region (manipulation-localisation ground truth)
Identity change is not measured yet (needs a face-recognition model; the common ones carry
non-commercial licences).
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import numpy as np
from PIL import Image

from synthgen import registry
from synthgen.config import JobConfig
from synthgen.data.folder import ImageFolderSource
from synthgen.labelers.face_parsing import PARTS
from synthgen.masks import shape_mask
from synthgen.types import Backend, Conditioning
from synthgen.verify import Verdict


def _parser(cfg: JobConfig):
    lab = registry.get(cfg.labeler.adapter)(backend=Backend(cfg.labeler.backend) if cfg.labeler.backend else None)
    if not hasattr(lab, "parse"):
        raise ValueError(f"face jobs need a face-parsing labeler, got {cfg.labeler.adapter!r}")
    lab.load()
    return lab


def plan_face(cfg: JobConfig, source: ImageFolderSource, out: Path) -> list[dict]:
    """Parse each image, build the generation mask for a sampled part, save it to masks/."""
    t = cfg.targets
    unknown = [p for p in t.classes if p not in PARTS]
    if unknown or not t.classes:
        raise ValueError(f"face targets.classes must be parts from {sorted(PARTS)}; got {t.classes}")
    rng = random.Random(cfg.seed)
    ids = list(source.images)
    rng.shuffle(ids)
    cond = Conditioning(cfg.conditioning.type)
    lab = _parser(cfg)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    tasks, skipped = [], {"part_too_small": 0}
    for image_id in ids * t.max_per_image:
        if len(tasks) >= cfg.n_samples:
            break
        part = rng.choice(t.classes)
        image = source.load_image(image_id)
        det = next(iter(lab.detect(image, [part])), None)
        if det is None or (np.asarray(det.mask) > 127).sum() < t.min_area:
            skipped["part_too_small"] += 1
            continue
        mask = shape_mask(det.mask, cfg.mask.shape, cfg.mask.dilate_px)
        prompt = ""
        if cond is not Conditioning.NONE:
            prompt = re.sub(r"\s+", " ", cfg.conditioning.prompt.format(
                attr=rng.choice(cfg.conditioning.attributes), cls=part)).strip()
        ref = None
        if cond is Conditioning.REFERENCE_OBJECT:
            ref = {"image_id": rng.choice([i for i in source.images if i != image_id])}
        idx = len(tasks)
        task = {"task_id": f"{cfg.name}-{idx:05d}", "seed": cfg.seed * 100_000 + idx, "image_id": image_id,
                "ann_id": None, "original_class": part, "target_class": part, "prompt": prompt, "reference": ref,
                "original_part_area": int((np.asarray(det.mask) > 127).sum())}
        mask.save(out / "masks" / f"{task['task_id']}.png")
        tasks.append(task)
    lab.unload()
    print(f"[plan] selection: {json.dumps({'images': len(ids), 'skipped': skipped})}")
    return tasks


def verify_face(lab, image: Image.Image, region: Image.Image, task: dict, min_score: float, min_inside: float,
                area_ratio: tuple[float, float]) -> tuple[Verdict, np.ndarray]:
    """The edited part must still be parsed as that part, inside the edited region, at a plausible size."""
    labels, conf = lab.parse(image)
    part = task["target_class"]
    ids = [lab.label_ids[n] for n in PARTS[part]]
    m = np.isin(labels, ids)
    reg = np.asarray(region.convert("L")) > 127
    if not m.any():
        return Verdict(False, f"no '{part}' parsed in the output"), labels
    inside = (m & reg).sum() / m.sum()
    score = float(conf[m].mean())
    ratio = (m & reg).sum() / max(task["original_part_area"], 1)
    v = Verdict(True, "ok", inside_frac=float(inside), fill_frac=float(ratio))
    if score < min_score:
        v = Verdict(False, f"'{part}' parsed with low confidence {score:.2f}", inside_frac=inside, fill_frac=ratio)
    elif inside < min_inside:
        v = Verdict(False, f"'{part}' mostly outside the edited region", inside_frac=inside, fill_frac=ratio)
    elif not area_ratio[0] <= ratio <= area_ratio[1]:
        v = Verdict(False, f"'{part}' area changed x{ratio:.2f}", inside_frac=inside, fill_frac=ratio)
    v.score = score
    return v, labels


def export_face(cfg: JobConfig, source: ImageFolderSource, tasks: list[dict], verdicts: dict, gens: dict,
                label_names: dict, out: Path) -> list[dict]:
    records = []
    for t in tasks:
        v = verdicts.get(t["task_id"])
        if not v or not v["accepted"]:
            continue
        records.append({
            "file_name": f"images/{t['task_id']}.jpg", "parse_map": f"labels/{t['task_id']}.png",
            "manipulation_mask": f"masks/{t['task_id']}.png", "synthetic": True,
            "source_image": source.images[t["image_id"]]["file_name"], "task_id": t["task_id"],
            "edited_part": t["target_class"], "operation": cfg.operation, "prompt": t["prompt"],
            "reference_image": source.images[t["reference"]["image_id"]]["file_name"] if t["reference"] else None,
            "adapter": cfg.generator.adapter, "verdict": v, "identity_checked": False,
        })
    (out / "face_labels.json").write_text(json.dumps({
        "info": {"job": cfg.name, "split": cfg.source.split, "generator": cfg.generator.adapter,
                 "licence": registry.get(cfg.generator.adapter).caps.licence},
        "label_names": label_names, "images": records}, indent=1))
    return records
