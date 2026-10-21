"""Turn a job config into a deterministic task list, check it against adapter capabilities,
and estimate run time and memory from this machine's benchmark profile."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import numpy as np

from synthgen import registry
from synthgen.config import JobConfig
from synthgen.data.coco import CocoSource
from synthgen.device import available
from synthgen.hardware import machine_profile, profile_id
from synthgen.masks import shape_mask
from synthgen.refbank import ReferenceBank
from synthgen.types import Backend, Conditioning, Domain, Operation

SUPPORTED_OPS = {Operation.REMOVE, Operation.REGENERATE, Operation.REPLACE, Operation.CHANGE_CLASS, Operation.INSERT}
PROFILES = Path(__file__).resolve().parent.parent / "profiles"


class PlanError(ValueError):
    pass


def validate(cfg: JobConfig, source: CocoSource) -> None:
    domain, op, cond = Domain(cfg.domain), Operation(cfg.operation), Conditioning(cfg.conditioning.type)
    if domain is Domain.FACE and cfg.source.format != "images":
        raise PlanError("face jobs use source.format: images (labels come from the face parser)")
    if domain is Domain.FACE and op not in (Operation.REGENERATE, Operation.REPLACE):
        raise PlanError("face jobs support regenerate and replace")
    if op not in SUPPORTED_OPS:
        raise PlanError(f"operation {op.value!r} is not supported by the runner")
    if op is Operation.REMOVE and cond is not Conditioning.NONE:
        raise PlanError("remove uses conditioning.type: none")
    if op is not Operation.REMOVE and cond is Conditioning.NONE:
        raise PlanError(f"{op.value} needs text or reference conditioning")
    if op is Operation.CHANGE_CLASS and not cfg.change_to:
        raise PlanError("change_class needs change_to: [class, ...]")
    if op is Operation.INSERT and not cfg.targets.classes:
        raise PlanError("insert needs targets.classes (what to insert)")
    if cfg.targets.crowd not in ("exclude", "include", "only"):
        raise PlanError("targets.crowd must be exclude, include or only")
    if cfg.targets.crowd != "exclude" and op is not Operation.REMOVE:
        raise PlanError("targets.crowd: include/only works with remove only (a crowd region is not one object)")
    if cfg.verify.unlabelled not in ("off", "flag", "reject"):
        raise PlanError("verify.unlabelled must be off, flag or reject")
    for c in ([] if domain is Domain.FACE else [*cfg.targets.classes, *cfg.change_to, *cfg.placement.support_classes,
                                                *cfg.targets.exclude_near]):
        if c not in source.cat_by_name:
            raise PlanError(f"class {c!r} is not in the source categories")

    caps = registry.get(cfg.generator.adapter).caps
    problems = []
    if domain not in caps.domains:
        problems.append(f"domain {domain.value}")
    if op not in caps.operations:
        problems.append(f"operation {op.value}")
    if cond not in caps.conditioning:
        problems.append(f"conditioning {cond.value}")
    if cfg.generator.backend and cfg.generator.backend not in {b.value for b in caps.backends}:
        problems.append(f"backend {cfg.generator.backend}")
    if problems:
        raise PlanError(f"generator {cfg.generator.adapter!r} does not support: {', '.join(problems)}")
    for role, adapter, backend in (("generator", cfg.generator.adapter, cfg.generator.backend),
                                   ("labeler", cfg.labeler.adapter, cfg.labeler.backend)):
        want = [Backend(backend)] if backend else registry.get(adapter).caps.backends
        if not any(available(b) for b in want):
            raise PlanError(f"{role} {adapter!r} needs {'/'.join(b.value for b in want)}, which this machine doesn't have")
    from synthgen.degrade import validate_ops
    try:
        validate_ops(cfg.postprocess.ops)
    except ValueError as e:
        raise PlanError(str(e)) from None
    if cfg.occlusion.composite not in ("silhouette", "detected"):
        raise PlanError("occlusion.composite must be silhouette or detected")
    lab = registry.get(cfg.labeler.adapter).caps
    if domain not in lab.domains:
        raise PlanError(f"labeler {cfg.labeler.adapter!r} does not support domain {domain.value}")


def contrast_classes(cfg: JobConfig, original: str) -> list[str]:
    if cfg.verify.contrast == "auto":
        return [original] if cfg.operation == Operation.CHANGE_CLASS.value else []
    return list(cfg.verify.contrast)


def _eligible(source: CocoSource, classes: set[str] | None, min_area: float,
              crowd: str = "exclude") -> dict[str, list[tuple[int, dict]]]:
    out: dict[str, list[tuple[int, dict]]] = {}
    for image_id, anns in source.anns_by_image.items():
        for a in anns:
            c = source.class_name(a["category_id"])
            if (crowd == "exclude" and a["iscrowd"]) or (crowd == "only" and not a["iscrowd"]):
                continue
            if (classes is None or c in classes) and a["area"] >= min_area and a.get("segmentation"):
                out.setdefault(c, []).append((image_id, a))
    return out


def _overlap_ok(source: CocoSource, image_id: int, ann: dict, gen: np.ndarray, size, limit: float,
                protected: bool = False) -> bool:
    """The edit may cover at most `limit` of any other annotated object. With occlusion handling on,
    other objects' visible pixels are protected (kept or restored), so only crowd regions block: they
    hold unlabelled instances, so the edit may not touch them at all."""
    x0, y0 = np.where(gen.any(0))[0][[0, -1]], np.where(gen.any(1))[0][[0, -1]]
    for o in source.anns_by_image[image_id]:
        if o["id"] == ann["id"]:
            continue
        ox, oy, ow, oh = o["bbox"]
        if ox > x0[1] or ox + ow < x0[0] or oy > y0[1] or oy + oh < y0[0]:
            continue
        om = CocoSource.ann_mask(o, size)
        if om is None:          # no segmentation: can't tell what we'd damage
            return False
        om = np.asarray(om) > 127
        if o.get("iscrowd"):
            if (om & gen).any():
                return False
            continue
        if not protected and om.sum() and (om & gen).sum() / om.sum() > limit:
            return False
    return True


def build_insert_tasks(cfg: JobConfig, source: CocoSource) -> list[dict]:
    """One task per target image: a free-space box placed by synthgen/placement.py."""
    from synthgen.placement import class_aspect, place, sharpness

    rng = random.Random(cfg.seed)
    pc, t = cfg.placement, cfg.targets
    aspects = {c: class_aspect(source, c) for c in t.classes}
    cond = Conditioning(cfg.conditioning.type)
    bank = None
    if cond is Conditioning.REFERENCE_OBJECT:
        rc = cfg.conditioning.reference
        bank = ReferenceBank(source, t.classes, rc.min_area, rng, max_uses=rc.max_uses)
        if cfg.occlusion.enabled and cfg.occlusion.see_through_skip:
            from synthgen.occlusion import SeeThroughScorer
            scorer = SeeThroughScorer(cfg.occlusion.see_through_phrases)
            bank.reject = lambda e, cls: scorer.score(source.load_image(e.image_id), e.bbox, cls) >= \
                cfg.occlusion.see_through_threshold
    ids = list(source.images)
    rng.shuffle(ids)
    lab = None
    if pc.ground_phrases:      # ground masks come from the job's open-vocabulary labeller
        lab = registry.get(cfg.labeler.adapter)(backend=None)
        lab.load()
    tasks, skipped = [], {"no_support": 0, "no_free_space": 0, "out_of_focus": 0, "unsuitable_scene": 0,
                          "no_reference": 0}
    scene = None
    for image_id in ids:
        if len(tasks) >= cfg.n_samples:
            break
        cls = rng.choice(t.classes)
        support = pc.support_classes or [cls]
        if not any(source.class_name(a["category_id"]) in support for a in source.anns_by_image[image_id]):
            skipped["no_support"] += 1
            continue
        ground = None
        if lab is not None:
            image = source.load_image(image_id)
            ground = np.zeros((image.height, image.width), bool)
            for d in lab.detect(image, list(pc.ground_phrases)):
                if d.mask is not None:
                    ground |= np.asarray(d.mask) > 127
        p = place(source, image_id, cls, support, aspects[cls], rng,
                  pc.y_jitter, pc.size_jitter, pc.gap, pc.min_height_px, ground=ground, ground_min_frac=pc.ground_min_frac)
        if p is None:
            skipped["no_free_space"] += 1
            continue
        if pc.min_sharpness > 0:
            gray = np.asarray(source.load_image(image_id).convert("L"))
            if sharpness(gray, p.box) < pc.min_sharpness:    # inserted objects look pasted into bokeh
                skipped["out_of_focus"] += 1
                continue
        if pc.scene_skip_threshold > 0:
            if scene is None:
                from synthgen.occlusion import SeeThroughScorer
                scene = SeeThroughScorer(pc.scene_skip_phrases, negatives=[
                    "a sharp photo of a street", "a photo of a road", "a photo of a parking lot",
                    "a photo of a city street at night"])
            x, y, w, h = p.box
            if scene.score(source.load_image(image_id), [x - w, y - h, 3 * w, 3 * h], cls, margin=0) >= pc.scene_skip_threshold:
                skipped["unsuitable_scene"] += 1
                continue
        prompt = ""
        if cond is not Conditioning.NONE:
            prompt = re.sub(r"\s+", " ", cfg.conditioning.prompt.format(
                attr=rng.choice(cfg.conditioning.attributes), cls=cls)).strip()
        ref = None
        if bank is not None:
            e = bank.pick(cls, image_id, p.box, cfg.conditioning.reference.match)
            if e is None:
                skipped["no_reference"] += 1
                continue
            ref = {"image_id": e.image_id, "ann_id": e.ann_id}
        idx = len(tasks)
        tasks.append({"task_id": f"{cfg.name}-{idx:05d}", "seed": cfg.seed * 100_000 + idx, "image_id": image_id,
                      "ann_id": None, "original_class": None, "target_class": cls, "prompt": prompt, "reference": ref,
                      "insert_box": p.box, "placement": {"support_ann_ids": p.support_ann_ids, "size_model": p.size_model}})
    if lab is not None:
        lab.unload()
    build_tasks.last_stats = {"images_tried": ids.index(image_id) + 1 if tasks else len(ids), "skipped": skipped,
                              "ground_check": bool(pc.ground_phrases)}
    if scene is not None:
        scene.unload()
    if bank is not None:
        build_tasks.last_stats["reference_pool"] = {c: len(v) for c, v in bank.entries.items()}
        if bank.reject is not None:
            scorer.unload()
            build_tasks.last_stats["references_seen_through"] = sum(bank._rejected.values())
    return tasks


def build_tasks(cfg: JobConfig, source: CocoSource) -> list[dict]:
    if cfg.operation == Operation.INSERT.value:
        return build_insert_tasks(cfg, source)
    rng = random.Random(cfg.seed)
    t = cfg.targets
    pools = _eligible(source, set(t.classes) if t.classes else None, t.min_area, t.crowd)
    if t.image_ids:
        keep = set(t.image_ids)
        pools = {c: [x for x in p if x[0] in keep] for c, p in pools.items()}
        pools = {c: p for c, p in pools.items() if p}
    if not pools:
        raise PlanError("no eligible instances for the target classes")
    for p in pools.values():
        rng.shuffle(p)
    classes = sorted(pools)

    if t.policy == "specified":
        flat = [x for c in classes for x in pools[c]]
        rng.shuffle(flat)
        draw = iter(flat).__next__
    elif t.policy in ("random", "rare_weighted"):
        weights = [1.0 if t.policy == "random" else 1.0 / len(pools[c]) for c in classes]
        cursor = {c: 0 for c in classes}

        def draw():
            live = [i for i, c in enumerate(classes) if cursor[c] < len(pools[c])]
            if not live:
                raise StopIteration
            c = classes[rng.choices(live, [weights[i] for i in live])[0]]
            cursor[c] += 1
            return pools[c][cursor[c] - 1]
    else:
        raise PlanError(f"unknown targets.policy {t.policy!r}")

    cond = Conditioning(cfg.conditioning.type)
    target_classes = cfg.change_to or None
    bank = None
    if cond is Conditioning.REFERENCE_OBJECT:
        rc = cfg.conditioning.reference
        embedder = None
        if rc.match == "visual":
            from synthgen.embed import DinoEmbedder
            embedder = DinoEmbedder()
        bank = ReferenceBank(source, target_classes or classes, rc.min_area, rng, max_uses=rc.max_uses,
                             embedder=embedder)

    tasks, per_image, attempts = [], {}, 0
    skipped = {"per_image_limit": 0, "truncated": 0, "overlaps_other": 0, "near_excluded": 0, "see_through": 0,
               "no_reference": 0}
    near_ids = {source.cat_by_name[c] for c in t.exclude_near}
    oc = cfg.occlusion
    see_through, see_through_skipped = {}, []        # scorer loaded on first use, freed at the end

    def st_score_of(image_id, bbox, cls):
        if "scorer" not in see_through:
            from synthgen.occlusion import SeeThroughScorer
            see_through["scorer"] = SeeThroughScorer(oc.see_through_phrases)
        return round(see_through["scorer"].score(source.load_image(image_id), bbox, cls), 3)

    if bank is not None and oc.enabled and oc.see_through_skip:
        # a reference seen through a fence or window makes the generator copy the fence or the glare
        bank.reject = lambda e, cls: st_score_of(e.image_id, e.bbox, cls) >= oc.see_through_threshold
    while len(tasks) < cfg.n_samples and attempts < 50 * cfg.n_samples:
        attempts += 1
        try:
            image_id, ann = draw()
        except StopIteration:
            break
        if per_image.get(image_id, 0) >= t.max_per_image:
            skipped["per_image_limit"] += 1
            continue
        im = source.images[image_id]
        size = (im["width"], im["height"])
        x, y, w, h = ann["bbox"]
        edge = max(3, 0.01 * max(size))   # COCO boxes often stop a few px short of the border
        if t.skip_truncated and (x <= edge or y <= edge or x + w >= size[0] - edge or y + h >= size[1] - edge):
            skipped["truncated"] += 1
            continue
        gen = np.asarray(shape_mask(CocoSource.ann_mask(ann, size), cfg.mask.shape, cfg.mask.dilate_px)) > 127
        if not _overlap_ok(source, image_id, ann, gen, size, t.max_overlap_other, protected=cfg.occlusion.enabled):
            skipped["overlaps_other"] += 1
            continue
        if near_ids:
            m = t.exclude_near_margin
            tb = (x - m * w, y - m * h, x + w * (1 + m), y + h * (1 + m))
            if any(o["category_id"] in near_ids and o["id"] != ann["id"] and
                   tb[0] < o["bbox"][0] + o["bbox"][2] and o["bbox"][0] < tb[2] and
                   tb[1] < o["bbox"][1] + o["bbox"][3] and o["bbox"][1] < tb[3]
                   for o in source.anns_by_image[image_id]):
                skipped["near_excluded"] += 1
                continue

        original = source.class_name(ann["category_id"])
        st_score = None
        if oc.enabled and oc.see_through_skip:
            st_score = st_score_of(image_id, ann["bbox"], original)
            if st_score >= oc.see_through_threshold:
                skipped["see_through"] += 1
                see_through_skipped.append({"image_id": image_id, "ann_id": ann["id"], "score": st_score})
                continue
        target = rng.choice(target_classes) if target_classes else original
        prompt = ""
        if cond is not Conditioning.NONE:
            prompt = re.sub(r"\s+", " ", cfg.conditioning.prompt.format(
                attr=rng.choice(cfg.conditioning.attributes), cls=target)).strip()
        ref = None
        if bank is not None:
            e = bank.pick(target, image_id, ann["bbox"], cfg.conditioning.reference.match, target_ann_id=ann["id"])
            if e is None:
                skipped["no_reference"] += 1
                continue
            ref = {"image_id": e.image_id, "ann_id": e.ann_id}

        idx = len(tasks)
        tasks.append({"task_id": f"{cfg.name}-{idx:05d}", "seed": cfg.seed * 100_000 + idx,
                      "image_id": image_id, "ann_id": ann["id"], "original_class": original,
                      "target_class": target, "prompt": prompt, "reference": ref})
        if st_score is not None:
            tasks[-1]["see_through_score"] = st_score
        per_image[image_id] = per_image.get(image_id, 0) + 1
    if "scorer" in see_through:
        see_through["scorer"].unload()
    build_tasks.last_stats = {"candidates": sum(map(len, pools.values())), "drawn": attempts, "skipped": skipped}
    if see_through_skipped:
        build_tasks.last_stats["see_through_skipped"] = see_through_skipped
    if bank is not None:
        build_tasks.last_stats["reference_pool"] = {c: len(v) for c, v in bank.entries.items()}
        build_tasks.last_stats["references_used"] = len(bank.uses)
        if bank.reject is not None:
            build_tasks.last_stats["references_seen_through"] = sum(bank._rejected.values())
    return tasks


def _profile_record(results: dict, adapter: str, backend: str | None, res: int | None, steps: int | None):
    recs = [r for r in results.values() if r["adapter"] == adapter and (backend is None or r["backend"] == backend)]
    if not recs:
        return None, "no benchmark for this adapter"
    exact = [r for r in recs if (res is None or r["res"] == res) and (steps is None or r["steps"] == steps)]
    if exact:
        return exact[0], "measured"
    r = min(recs, key=lambda r: abs(r["res"] - (res or r["res"])))
    return r, f"nearest benchmark (res {r['res']}, steps {r['steps']})"


def estimate(cfg: JobConfig, n_tasks: int) -> dict:
    prof = machine_profile()
    path = PROFILES / f"{profile_id(prof)}.json"
    if not path.exists():
        return {"note": f"no profile at {path}; run `python -m synthgen.bench all`"}
    results = json.loads(path.read_text())["results"]
    g, gnote = _profile_record(results, cfg.generator.adapter, cfg.generator.backend, cfg.generator.res, cfg.generator.steps)
    lab, lnote = _profile_record(results, cfg.labeler.adapter, cfg.labeler.backend, None, None)
    out = {"generator_basis": gnote, "labeler_basis": lnote}
    calls = 1 + len(contrast_classes(cfg, "original"))   # one detector query per class
    if g:
        out["generate_min"] = round((g["load_s"] + g.get("prepare_s", 0) + n_tasks * g["s_per_item_median"]) / 60, 1)
        out["generator_peak_gb"] = max(g.get("peak_prepare_gb", 0), g.get("peak_warmup_gb", 0), g["peak_infer_gb"])
    if lab:
        # labeler benchmark detects 2 classes per call; per-class cost is roughly half
        out["verify_min"] = round((lab["load_s"] + n_tasks * calls * lab["s_per_item_median"] / 2) / 60, 1)
        out["labeler_peak_gb"] = max(lab.get("peak_warmup_gb", 0), lab["peak_infer_gb"])
    limit = prof.get("gpu_recommended_working_set_gb") or prof.get("gpu_memory_gb")   # Mac / NVIDIA
    if limit and out.get("generator_peak_gb", 0) > 0.95 * limit:
        out["warning"] = f"generator peak {out['generator_peak_gb']} GB is within 5% of the {limit} GB GPU memory limit"
    return out
