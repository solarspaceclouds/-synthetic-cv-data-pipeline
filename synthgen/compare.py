"""Compare generators on identical tasks.

    python -m synthgen.compare configs/compare_all.yaml                 # every job in the spec
    python -m synthgen.compare configs/compare_all.yaml --only coco_car_remove
    python -m synthgen.compare configs/compare_all.yaml --sheets-only   # rebuild the sheets and summary

For each job in the spec, the job's own config runs as the baseline in <root>/<job>/<adapter>/ (plan,
generate, verify, export, as `synthgen.run`). Each alternative generator then runs on the baseline's
tasks: same images, objects, masks, occluders, prompts, references and seeds, so differences come
from the model alone. Alternatives can override the prompt (`prompt`, e.g. a removal prompt for a text model, `{cls}` = original
class; or `prompt_sub`, regex substitutions on each task's prompt). Outputs per job: summary.json and sheet_<n>.jpg (original | each model,
with each verdict), plus <root>/summary.md across all jobs.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw

from synthgen.config import GeneratorCfg, load_job, to_dict
from synthgen.run import _hash, _is_face, _read_jsonl, _source, plan, stage_export, stage_generate, stage_verify


def _run(cfg, source, tasks, out: Path) -> dict:
    stage_generate(cfg, source, tasks, out)
    stage_verify(cfg, source, tasks, out)
    return stage_export(cfg, source, tasks, out)


def _prompt(alt: dict, t: dict) -> str:
    if "prompt" in alt:
        return alt["prompt"].format(cls=t.get("original_class") or t["target_class"])
    p = t["prompt"]
    for pattern, repl in alt.get("prompt_sub", []):
        p = re.sub(pattern, repl, p)
    return p


def _label(g: dict) -> str:
    return g.get("label") or g["adapter"]


def labels(job: dict) -> list[str]:
    return [job.get("baseline_label", load_job(job["config"]).generator.adapter)] + [_label(a) for a in job.get("alternatives", [])]


def run_job(job: dict, root: Path, model: str) -> None:
    """Run one model of one job: the baseline (plans the tasks) or an alternative (reuses them)."""
    cfg = load_job(job["config"])
    raw = yaml.safe_load(Path(job["config"]).read_text())
    name = job.get("name", cfg.name)
    base_label = job.get("baseline_label", cfg.generator.adapter)
    base = root / name / base_label
    cfg.output.dir = str(base)
    source = _source(cfg)
    if model == base_label:
        print(f"[compare] {name}: baseline {base_label}", flush=True)
        _run(cfg, source, plan(cfg, source, base, False, raw), base)
        return
    if not (base / "tasks.jsonl").exists():
        raise SystemExit(f"run the baseline {base_label} of {name} first")
    tasks = _read_jsonl(base / "tasks.jsonl")
    for alt in [a for a in job.get("alternatives", []) if _label(a) == model]:
        out = root / name / _label(alt)
        out.mkdir(parents=True, exist_ok=True)
        acfg = copy.deepcopy(cfg)
        # job-level crop settings carry over; model settings (res, steps, strength, guidance) don't
        acfg.generator = GeneratorCfg(context_scale=cfg.generator.context_scale, feather_px=cfg.generator.feather_px,
                                      **{k: v for k, v in alt.items() if k not in ("label", "prompt", "prompt_sub")})
        acfg.output.dir = str(out)
        atasks = [dict(t, prompt=_prompt(alt, t)) for t in tasks]
        for d in ("occluders",) + (("masks",) if _is_face(cfg) else ()):
            if (base / d).exists() and not (out / d).exists():
                shutil.copytree(base / d, out / d)
        (out / "tasks.jsonl").write_text("".join(json.dumps(t) + "\n" for t in atasks))
        meta = {"config": to_dict(acfg), "tasks_from": str(base)}
        (out / "job.json").write_text(json.dumps({"hash": _hash(meta), **meta}, indent=2))
        print(f"[compare] {name}: {_label(alt)}", flush=True)
        _run(acfg, source, atasks, out)


def _crop_box(mask: np.ndarray, w: int, h: int) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if not len(xs):
        return 0, 0, w, h
    p = int(0.5 * max(np.ptp(xs), np.ptp(ys))) + 24
    return max(0, xs.min() - p), max(0, ys.min() - p), min(w, xs.max() + p), min(h, ys.max() + p)


def sheets(job: dict, root: Path, per_sheet: int = 8, row_h: int = 200) -> dict:
    cfg = load_job(job["config"])
    name = job.get("name", cfg.name)
    labels = [job.get("baseline_label", cfg.generator.adapter)] + [_label(a) for a in job.get("alternatives", [])]
    dirs = [root / name / l for l in labels if (root / name / l / "summary.json").exists()]
    if not dirs:
        return {}
    cfg.output.dir = str(dirs[0])
    source = _source(cfg)
    tasks = _read_jsonl(dirs[0] / "tasks.jsonl")
    per = {}
    for d in dirs:
        gens = {g["task_id"]: g for g in _read_jsonl(d / "generation.jsonl")}
        ver = {v["task_id"]: v for v in _read_jsonl(d / "verdicts.jsonl")}
        secs = [g["seconds"] for g in gens.values()]
        per[d.name] = {"dir": d, "ver": ver, "accepted": sum(v["accepted"] for v in ver.values()), "tasks": len(tasks),
                       "s_per_item": round(float(np.median(secs)), 2) if secs else None}
    rows = []
    for t in tasks:
        tid = t["task_id"]
        orig = source.load_image(t["image_id"]).convert("RGB")
        mpath = dirs[0] / "masks" / f"{tid}.png"
        box = _crop_box(np.asarray(Image.open(mpath)) > 127, *orig.size) if mpath.exists() else (0, 0, *orig.size)
        cells = [(orig.crop(box), f"original  {t.get('prompt', '')[:40]}")]
        for lab, p in per.items():
            ip = p["dir"] / "images" / f"{tid}.jpg"
            v = p["ver"].get(tid)
            tag = "missing" if v is None else ("OK" if v["accepted"] else "REJ " + v["reason"][:34])
            cells.append((Image.open(ip).convert("RGB").crop(box) if ip.exists() else Image.new("RGB", (box[2] - box[0], box[3] - box[1]), "gray"),
                          f"{lab}: {tag}"))
        cw = max(int(c.width * row_h / c.height) for c, _ in cells)
        row = Image.new("RGB", ((cw + 8) * len(cells), row_h + 16), "white")
        dr = ImageDraw.Draw(row)
        for i, (c, txt) in enumerate(cells):
            c = c.resize((max(1, int(c.width * row_h / c.height)), row_h))
            row.paste(c, (i * (cw + 8), 16))
            dr.text((i * (cw + 8) + 2, 2), txt, fill="black")
        rows.append((tid, row))
    for k in range(0, len(rows), per_sheet):
        chunk = [r for _, r in rows[k:k + per_sheet]]
        sheet = Image.new("RGB", (max(r.width for r in chunk), sum(r.height + 6 for r in chunk)), (230, 230, 230))
        y = 0
        for r in chunk:
            sheet.paste(r, (0, y))
            y += r.height + 6
        sheet.save(root / name / f"sheet_{k // per_sheet + 1}.jpg", quality=88)
    summary = {lab: {k: v for k, v in p.items() if k not in ("dir", "ver")} for lab, p in per.items()}
    (root / name / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(prog="synthgen.compare")
    ap.add_argument("spec")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--sheets-only", action="store_true")
    ap.add_argument("--model", help="internal: run only this model of the --only job")
    args = ap.parse_args(argv)
    spec = yaml.safe_load(Path(args.spec).read_text())
    root = Path(spec.get("root", "out/compare"))
    jobs = [j for j in spec["jobs"] if not args.only or j.get("name", load_job(j["config"]).name) in args.only]
    if args.model:
        run_job(jobs[0], root, args.model)
        return
    for job in jobs:
        name = job.get("name", load_job(job["config"]).name)
        for lab in [] if args.sheets_only else labels(job):
            # one process per model, so each starts with all memory free (PyTorch-MPS and MLX caches
            # left by the previous model slowed the next one 2-4x through swapping)
            r = subprocess.run([sys.executable, "-m", "synthgen.compare", args.spec, "--only", name, "--model", lab])
            if r.returncode:
                print(f"[compare] {name}: {lab} failed (exit {r.returncode}); continuing", flush=True)
        print(f"[compare] {job.get('name', load_job(job['config']).name)}: {json.dumps(sheets(job, root))}", flush=True)
    lines = ["| job | model: accepted / tasks (median s per item) |", "|---|---|"]
    for job in spec["jobs"]:                       # every job with results, not only this invocation's
        name = job.get("name", load_job(job["config"]).name)
        if (root / name / "summary.json").exists():
            s = json.loads((root / name / "summary.json").read_text())
            lines.append(f"| {name} | " + "; ".join(f"{k}: {v['accepted']}/{v['tasks']} ({v['s_per_item']} s)"
                                                    for k, v in s.items()) + " |")
    (root / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
