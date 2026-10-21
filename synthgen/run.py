"""Run a job: plan -> generate -> verify -> export, one model in memory at a time.

    python -m synthgen.run plan configs/coco_car_regenerate.yaml     # task list + time/memory estimate
    python -m synthgen.run run  configs/coco_car_regenerate.yaml     # all stages; resumes where it stopped
    python -m synthgen.run run  configs/... --stage verify           # one stage

Output directory:
    job.json            resolved config + hash (a changed config refuses to reuse old tasks)
    tasks.jsonl         deterministic task list
    masks/<task>.png    generation region (also the manipulation-localisation label)
    images/<task>.jpg   synthetic image
    labels/<task>.png   recovered object mask (from the labeller, not the inpainting mask)
    generation.jsonl    per-task generator metadata
    verdicts.jsonl      per-task accept/reject with reason and scores
    annotations.json    COCO file of accepted samples, with provenance fields
    manifest.jsonl      everything above joined per task
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil

import numpy as np
import yaml
import sys
import time
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

from synthgen import registry
from synthgen.config import JobConfig, load_job, to_dict
from synthgen.data.coco import CocoSource, CocoWriter, mask_area_bbox, mask_to_rle
from synthgen.data.folder import ImageFolderSource
from synthgen import degrade, face
from synthgen.masks import shape_mask
from synthgen.plan import PlanError, build_tasks, contrast_classes, estimate, validate
from synthgen.refbank import reference_crop_and_mask
from synthgen.types import Backend, Domain, InpaintRequest, Operation
from synthgen.verify import color_change, recover_label, unlabelled_instances, verify_removal


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def _append_jsonl(path: Path, rec: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def _hash(d: dict) -> str:
    d = {k: v for k, v in d.items() if k != "output"}
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]


def _source(cfg: JobConfig):
    hint = 'datasets are not included; see "Source data" in README.md, or point source: in the job file at your copy'
    if cfg.source.format == "coco":
        if not cfg.source.annotations:
            raise PlanError("source.format coco needs source.annotations (path to the COCO json)")
        if not Path(cfg.source.annotations).is_file():
            raise PlanError(f"annotations not found: {cfg.source.annotations} ({hint})")
        if not cfg.source.download_missing and not Path(cfg.source.images).is_dir():
            raise PlanError(f"image folder not found: {cfg.source.images} ({hint})")
        return CocoSource(cfg.source.annotations, cfg.source.images, cfg.source.download_missing)
    if cfg.source.format == "images":
        if not Path(cfg.source.images).is_dir():
            raise PlanError(f"image folder not found: {cfg.source.images} ({hint})")
        return ImageFolderSource(cfg.source.images)
    raise PlanError(f"source format {cfg.source.format!r} not supported (coco | images)")


def _is_face(cfg: JobConfig) -> bool:
    return cfg.domain == Domain.FACE.value


def _ann(source: CocoSource, task: dict) -> dict:
    return next(a for a in source.anns_by_image[task["image_id"]] if a["id"] == task["ann_id"])


def _region(cfg: JobConfig, source, t: dict, size: tuple[int, int]) -> Image.Image:
    """The generation region for a scene task (face masks are written at plan time instead)."""
    if t.get("insert_box"):                          # insert: the planned free-space box
        x, y, w, h = t["insert_box"]
        s = cfg.placement.slack
        m = Image.new("L", size, 0)
        ImageDraw.Draw(m).rectangle((x - s * w, y - s * h, x + w * (1 + s), y + h * (1 + min(s, 0.1))), fill=255)
        return shape_mask(m, "bbox", cfg.mask.dilate_px)
    return shape_mask(CocoSource.ann_mask(_ann(source, t), size), cfg.mask.shape, cfg.mask.dilate_px)


def _keep_original(cfg: JobConfig, source, t: dict, out: Path, size, region: np.ndarray) -> np.ndarray | None:
    """Pixels of the region that come back from the original after generation, or None."""
    occ_path = out / "occluders" / f"{t['task_id']}.png"
    keep = np.asarray(Image.open(occ_path).convert("L")) > 127 if occ_path.exists() else np.zeros(region.shape, bool)
    oc = cfg.occlusion
    # only for occluded objects: an unoccluded new object may extend past the old outline
    if (oc.enabled and oc.composite == "silhouette" and occ_path.exists() and t.get("ann_id") is not None
            and not _is_face(cfg) and cfg.operation in (Operation.REGENERATE.value, Operation.REPLACE.value)):
        from scipy import ndimage
        obj = np.asarray(CocoSource.ann_mask(_ann(source, t), size)) > 127
        if oc.silhouette_dilate_px > 0:
            obj = ndimage.binary_dilation(obj, iterations=oc.silhouette_dilate_px)
        keep = keep | (region & ~obj)
    return keep if keep.any() else None


def plan_occluders(cfg: JobConfig, source, tasks: list[dict], out: Path, depth: dict | None = None) -> None:
    """Plan-time occlusion pass: save occluders/<task>.png with whatever is in front of the edited
    object (annotated objects, outline gaps, depth), and record the evidence in the task."""
    from synthgen.occlusion import DepthEstimator, find_occluders, ground_row, reference_disparity

    (out / "occluders").mkdir(parents=True, exist_ok=True)
    depth = depth if depth is not None else {}
    if "est" not in depth:
        depth["est"] = DepthEstimator()
    found = 0
    oc = cfg.occlusion
    for t in tasks:
        image = source.load_image(t["image_id"])
        if t["image_id"] not in depth:
            depth[t["image_id"]] = depth["est"].disparity(image)
        disp = depth[t["image_id"]]
        region = np.asarray(_region(cfg, source, t, image.size)) > 127
        obj = None if t.get("insert_box") else np.asarray(CocoSource.ann_mask(_ann(source, t), image.size)) > 127
        others = []
        for a in source.anns_by_image[t["image_id"]]:
            if a["id"] != t["ann_id"]:
                m = CocoSource.ann_mask(a, image.size)
                if m is not None:
                    others.append((a["id"], np.asarray(m) > 127))
        ref = reference_disparity(disp, obj, t.get("insert_box"))
        r = find_occluders(disp, region, ref, obj, others, oc.tol, oc.gap_tol, oc.min_frac,
                           ground_from_row=ground_row(obj, t.get("insert_box")))
        t["occlusion"] = r.info(int(region.sum()))
        t["occluder_frac"] = t["occlusion"]["frac"]
        if r.mask.any():
            Image.fromarray(r.mask.astype(np.uint8) * 255).save(out / "occluders" / f"{t['task_id']}.png")
            found += 1
    print(f"[plan] occlusion: occluders found in {found}/{len(tasks)} tasks")


def plan(cfg: JobConfig, source, out: Path, replan: bool = False, raw: dict | None = None) -> list[dict]:
    """`raw` is the job file as written. Its hash guards reuse of an output dir; a change in framework
    defaults alone (resolved config differs, job file doesn't) only warns."""
    validate(cfg, source)
    out.mkdir(parents=True, exist_ok=True)
    meta, tasks_path = out / "job.json", out / "tasks.jsonl"
    h = _hash(raw if raw is not None else to_dict(cfg))
    if tasks_path.exists() and not replan:
        old = json.loads(meta.read_text()) if meta.exists() else {}
        if old.get("hash") != h:
            raise PlanError(f"{out} was planned with a different job file; use a new output dir or --replan")
        if old.get("config") != to_dict(cfg):
            print(f"[plan] warning: framework defaults changed since {out} was planned; "
                  "existing outputs were made with the settings in job.json")
        return _read_jsonl(tasks_path)
    if replan:
        # task ids are positional: outputs from the old plan would be silently attached to new tasks
        stale = [p for p in (out / d for d in ("images", "masks", "labels", "raw", "occluders")) if p.exists()]
        stale += [p for p in (out / f for f in ("generation.jsonl", "verdicts.jsonl", "annotations.json",
                                                 "manifest.jsonl", "summary.json", "review.jpg", "face_labels.json")) if p.exists()]
        for p in stale:
            shutil.rmtree(p) if p.is_dir() else p.unlink()
        if stale:
            print(f"[plan] --replan removed outputs of the previous plan: {', '.join(p.name for p in stale)}")
    if _is_face(cfg):
        tasks = face.plan_face(cfg, source, out)
    else:
        tasks = build_tasks(cfg, source)
        print(f"[plan] selection: {json.dumps(build_tasks.last_stats)}")
        if cfg.occlusion.enabled:
            plan_occluders(cfg, source, tasks, out)
    tasks_path.write_text("".join(json.dumps(t) + "\n" for t in tasks))
    meta.write_text(json.dumps({"hash": h, "config": to_dict(cfg)}, indent=2))
    return tasks


def stage_generate(cfg: JobConfig, source: CocoSource, tasks: list[dict], out: Path) -> None:
    (out / "images").mkdir(exist_ok=True)
    (out / "masks").mkdir(exist_ok=True)
    todo = [t for t in tasks if not (out / "images" / f"{t['task_id']}.jpg").exists()]
    if not todo:
        return
    g = cfg.generator
    gen = registry.get(g.adapter)(backend=Backend(g.backend) if g.backend else None, native_res=g.res)
    print(f"[generate] {len(todo)} tasks with {g.adapter} on {gen.backend.value} at {gen.res}px", flush=True)
    gen.prepare([t["prompt"] for t in todo])
    gen.load()
    for i, t in enumerate(todo, 1):
        image = source.load_image(t["image_id"])
        mask_path = out / "masks" / f"{t['task_id']}.png"
        if mask_path.exists():                       # face jobs write masks at plan time
            mask = Image.open(mask_path).convert("L")
        else:
            mask = _region(cfg, source, t, image.size)
        ref = ref_mask = None
        if t["reference"] and t["reference"].get("ann_id") is not None:
            rc = cfg.conditioning.reference
            ref, ref_mask = reference_crop_and_mask(source, t["reference"]["image_id"], t["reference"]["ann_id"],
                                                    rc.pad, rc.white_bg)
        elif t["reference"]:                         # whole-image reference (e.g. another face)
            ref = source.load_image(t["reference"]["image_id"])
        keep = _keep_original(cfg, source, t, out, image.size, np.asarray(mask) > 127)
        req = InpaintRequest(image=image, mask=mask, prompt=t["prompt"], reference=ref, reference_mask=ref_mask,
                             seed=t["seed"],
                             steps=g.steps, strength=g.strength, guidance=g.guidance,
                             context_scale=g.context_scale, feather_px=g.feather_px)
        t0 = time.perf_counter()
        res = gen.inpaint(req)
        dt = time.perf_counter() - t0
        if keep is not None:
            # the model painted the whole region; the original scene goes back on top where it must stay
            # (occluders in front, and with composite: silhouette everything outside the old outline)
            from synthgen.occlusion import restore
            res.image = restore(res.image, image, Image.fromarray(keep.astype(np.uint8) * 255))
            mask = Image.fromarray(((np.asarray(mask) > 127) & ~keep).astype(np.uint8) * 255)
        mask.save(out / "masks" / f"{t['task_id']}.png")
        final, applied = res.image, None
        if cfg.postprocess.ops:
            # keep the clean output so degradation settings can be studied without regenerating
            (out / "raw").mkdir(exist_ok=True)
            res.image.save(out / "raw" / f"{t['task_id']}.png")
            final, applied = degrade.apply(res.image, cfg.postprocess.ops, random.Random(t["seed"]), cfg.postprocess.repeat)
        final.save(out / "images" / f"{t['task_id']}.jpg", quality=cfg.output.jpeg_quality)
        _append_jsonl(out / "generation.jsonl", {"task_id": t["task_id"], **res.meta, "crop_box": res.crop_box,
                                                 "seconds": round(dt, 3), "licence": gen.caps.licence,
                                                 "postprocess": applied})
        print(f"[generate] {i}/{len(todo)} {t['task_id']} {dt:.1f}s", flush=True)
    gen.unload()


def stage_verify(cfg: JobConfig, source: CocoSource, tasks: list[dict], out: Path) -> None:
    (out / "labels").mkdir(exist_ok=True)
    done = {v["task_id"] for v in _read_jsonl(out / "verdicts.jsonl")}
    todo = [t for t in tasks if t["task_id"] not in done and (out / "images" / f"{t['task_id']}.jpg").exists()]
    if not todo:
        return
    lab = registry.get(cfg.labeler.adapter)(backend=Backend(cfg.labeler.backend) if cfg.labeler.backend else None)
    lab.load()
    v = cfg.verify
    print(f"[verify] {len(todo)} tasks with {cfg.labeler.adapter}", flush=True)
    for t in todo:
        image = Image.open(out / "images" / f"{t['task_id']}.jpg").convert("RGB")
        mask = Image.open(out / "masks" / f"{t['task_id']}.png")
        if _is_face(cfg):
            verdict, labels = face.verify_face(lab, image, mask, t, v.min_score, v.min_inside, tuple(v.face_area_ratio))
            if verdict.accepted:
                Image.fromarray(labels.astype(np.uint8)).save(out / "labels" / f"{t['task_id']}.png")
            _append_jsonl(out / "verdicts.jsonl", {
                "task_id": t["task_id"], "accepted": verdict.accepted, "reason": verdict.reason,
                "inside_frac": round(float(verdict.inside_frac), 3), "area_ratio": round(float(verdict.fill_frac), 3),
                "score": round(verdict.score, 3) if verdict.score is not None else None})
            continue
        if cfg.operation == Operation.REMOVE.value:
            verdict = verify_removal(lab, image, mask, t["original_class"], v.min_score, v.removal_max_inside, v.phrases)
        else:
            verdict = recover_label(lab, image, mask, t["target_class"], tuple(contrast_classes(cfg, t["original_class"])),
                                    v.min_score, v.min_inside, v.min_fill, margin=v.margin, phrases=v.phrases)
        rec = {"task_id": t["task_id"], "accepted": verdict.accepted, "reason": verdict.reason,
               "inside_frac": round(float(verdict.inside_frac), 3), "fill_frac": round(float(verdict.fill_frac), 3)}
        d = verdict.detection
        if d is not None:
            rec.update(score=round(d.score, 3), box=[round(x, 1) for x in d.box])
        if v.min_color_change > 0 and verdict.accepted:
            rec["color_change"] = round(color_change(source.load_image(t["image_id"]), image, mask), 1)
            if rec["color_change"] < v.min_color_change:
                rec["accepted"] = False
                rec["reason"] = f"colour barely changed (delta E {rec['color_change']} < {v.min_color_change})"
        if v.unlabelled != "off" and verdict.accepted:
            removing = cfg.operation == Operation.REMOVE.value
            cls = t["original_class"] if removing else t["target_class"]
            cid = source.cat_by_name[cls]
            known = [(a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3])
                     for a in source.anns_by_image[t["image_id"]] if a["category_id"] == cid and a["id"] != t["ann_id"]]
            if d is not None and not removing:
                known.append(tuple(d.box))                     # the edited object itself
            extra = unlabelled_instances(lab, image, cls, known, v.unlabelled_min_score, v.phrases)
            rec["unlabelled"] = [{"score": round(e.score, 3), "box": [round(x, 1) for x in e.box]} for e in extra]
            if extra and v.unlabelled == "reject":
                rec["accepted"] = False
                rec["reason"] = f"{len(extra)} unlabelled '{cls}' in the image (source annotation gap)"
        if rec["accepted"] and d is not None and d.mask is not None:
            label = d.mask
            if v.object_only and cfg.operation == Operation.INSERT.value:
                label = _object_only(cfg, source, t, out, np.asarray(d.mask) > 127)
                rec["object_only"] = True
            label.save(out / "labels" / f"{t['task_id']}.png")
        _append_jsonl(out / "verdicts.jsonl", rec)
    lab.unload()


def _object_only(cfg: JobConfig, source, t: dict, out: Path, obj: np.ndarray) -> Image.Image:
    """Insert: rebuild the image from the original plus only the new object and its shadow.

    The generator fills the whole box; where the object doesn't reach, it paints smears, motion blur or
    a second background. Keep the detected object (dilated 2 px), and below it a shadow band where the
    generated image is darker (applied as a luminance ratio to the original, so the ground texture stays).
    Occluders stay original. Re-applies the job's degradation with the task seed. Returns the new label."""
    from scipy import ndimage
    tid = t["task_id"]
    raw_path = out / "raw" / f"{tid}.png"
    gen = Image.open(raw_path if raw_path.exists() else out / "images" / f"{tid}.jpg").convert("RGB")
    orig = source.load_image(t["image_id"]).convert("RGB")
    region = np.asarray(Image.open(out / "masks" / f"{tid}.png")) > 127
    occ_path = out / "occluders" / f"{tid}.png"
    occ = np.asarray(Image.open(occ_path)) > 127 if occ_path.exists() else np.zeros_like(region)
    obj = obj & ~occ
    keep = ndimage.binary_dilation(obj, iterations=2) & region
    ys, xs = np.where(obj)
    shadow = np.zeros_like(region)
    if ys.size:
        h = ys.max() - ys.min() + 1
        shadow[int(ys.max() - 0.25 * h):int(ys.max() + 0.2 * h) + 1, xs.min():xs.max() + 1] = True
        shadow &= region & ~keep & ~occ
    g, o = np.asarray(gen, np.float32), np.asarray(orig, np.float32)
    ratio = np.clip((g.mean(2) + 1) / (o.mean(2) + 1), 0.35, 1.0)[..., None]
    a = np.asarray(Image.fromarray(keep.astype(np.uint8) * 255).filter(ImageFilter.GaussianBlur(1)), np.float32)[..., None] / 255
    s = np.asarray(Image.fromarray(shadow.astype(np.uint8) * 255).filter(ImageFilter.GaussianBlur(3)), np.float32)[..., None] / 255
    shaded = o * (1 - s * (1 - ratio))
    img = Image.fromarray(np.clip(a * g + (1 - a) * shaded, 0, 255).round().astype(np.uint8))
    changed = keep | (shadow & (ratio[..., 0] < 0.97))
    Image.fromarray(changed.astype(np.uint8) * 255).save(out / "masks" / f"{tid}.png")
    if cfg.postprocess.ops:
        img.save(raw_path)
        img, _ = degrade.apply(img, cfg.postprocess.ops, random.Random(t["seed"]), cfg.postprocess.repeat)
    img.save(out / "images" / f"{tid}.jpg", quality=cfg.output.jpeg_quality)
    return Image.fromarray(obj.astype(np.uint8) * 255)


def stage_export(cfg: JobConfig, source: CocoSource, tasks: list[dict], out: Path) -> dict:
    gens = {g["task_id"]: g for g in _read_jsonl(out / "generation.jsonl")}
    verdicts = {v["task_id"]: v for v in _read_jsonl(out / "verdicts.jsonl")}
    if _is_face(cfg):
        from synthgen.labelers.face_parsing import _REPO
        from transformers import AutoConfig
        names = AutoConfig.from_pretrained(_REPO).id2label
        face.export_face(cfg, source, tasks, verdicts, gens, names, out)
        reasons = Counter("accepted" if v["accepted"] else v["reason"] for v in verdicts.values())
        (out / "manifest.jsonl").write_text("".join(json.dumps({**t, "generation": gens.get(t["task_id"]),
                                                                "verdict": verdicts.get(t["task_id"])}) + "\n" for t in tasks))
        summary = {"tasks": len(tasks), "verified": sum(reasons.values()), "accepted": reasons.get("accepted", 0),
                   "reasons": dict(reasons)}
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        return summary
    writer = CocoWriter(list(source.categories.values()))
    manifest, reasons = [], Counter()
    for t in tasks:
        v = verdicts.get(t["task_id"])
        manifest.append({**t, "generation": gens.get(t["task_id"]), "verdict": v})
        if not v:
            continue
        reasons[v["reason"] if not v["accepted"] else "accepted"] += 1
        if not v["accepted"]:
            continue
        im = source.images[t["image_id"]]
        new_id = writer.add_image(f"images/{t['task_id']}.jpg", (im["width"], im["height"]), t["image_id"], t["task_id"])
        for a in source.anns_by_image[t["image_id"]]:
            if a["id"] != t["ann_id"]:
                writer.add_ann(new_id, a["category_id"], a["bbox"], a["area"], a["segmentation"], a["iscrowd"],
                               source_ann_id=a["id"])
        if cfg.operation != Operation.REMOVE.value:
            m = Image.open(out / "labels" / f"{t['task_id']}.png")
            area, bbox = mask_area_bbox(m)
            writer.add_ann(new_id, source.cat_by_name[t["target_class"]], bbox, area, mask_to_rle(m), synthetic=True,
                           edit={"operation": cfg.operation, "original_class": t["original_class"],
                                 "prompt": t["prompt"], "reference": t["reference"],
                                 "adapter": cfg.generator.adapter, "score": v.get("score")})
    info = {"job": cfg.name, "hash": json.loads((out / "job.json").read_text())["hash"], "split": cfg.source.split,
            "generator": cfg.generator.adapter, "labeler": cfg.labeler.adapter,
            "licence": registry.get(cfg.generator.adapter).caps.licence}
    writer.save(out / "annotations.json", info)
    (out / "manifest.jsonl").write_text("".join(json.dumps(m) + "\n" for m in manifest))
    summary = {"tasks": len(tasks), "verified": sum(reasons.values()), "accepted": reasons.get("accepted", 0),
               "reasons": dict(reasons)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(prog="synthgen.run")
    ap.add_argument("cmd", choices=["plan", "run"])
    ap.add_argument("job")
    ap.add_argument("--stage", choices=["generate", "verify", "export", "all"], default="all")
    ap.add_argument("--replan", action="store_true")
    args = ap.parse_args(argv)

    try:
        cfg = load_job(args.job)
        raw = yaml.safe_load(Path(args.job).read_text())
    except ValueError as e:
        sys.exit(f"config error: {e}")
    out = Path(cfg.output.dir)
    try:
        source = _source(cfg)
        tasks = plan(cfg, source, out, args.replan, raw)
    except (PlanError, ValueError) as e:
        sys.exit(f"plan error: {e}")
    classes = Counter(f"+{t['target_class']}" if t.get("insert_box") else f"{t['original_class']}->{t['target_class']}"
                      for t in tasks)
    print(f"[plan] {len(tasks)} tasks ({dict(classes)}) -> {out}")
    print(f"[plan] estimate: {json.dumps(estimate(cfg, len(tasks)))}")
    if args.cmd == "plan":
        return
    if args.stage in ("generate", "all"):
        stage_generate(cfg, source, tasks, out)
    if args.stage in ("verify", "all"):
        stage_verify(cfg, source, tasks, out)
    if args.stage in ("export", "all"):
        print(f"[export] {json.dumps(stage_export(cfg, source, tasks, out))}")


if __name__ == "__main__":
    main()
