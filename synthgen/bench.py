"""Benchmark adapters on this machine and record results in profiles/<machine>.json.

    python -m synthgen.bench list
    python -m synthgen.bench run lama --n 5
    python -m synthgen.bench run flux2_klein_inpaint --res 512
    python -m synthgen.bench run sd15_inpaint --duration 1200     # sustained / thermal check
    python -m synthgen.bench all                                  # every ungated adapter, one subprocess each

The planner reads the profile to estimate run time and reject configs that won't fit in memory.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

from synthgen import registry
from synthgen.adapters.base import GeneratorAdapter, LabelerAdapter
from synthgen.device import available
from synthgen.types import Conditioning
from synthgen.hardware import PeakSampler, lifetime_peak_bytes, machine_profile, profile_id
from synthgen.types import Backend, InpaintRequest

ROOT = Path(__file__).resolve().parent.parent
ASSET = ROOT / "assets" / "coco_000000039769.jpg"   # COCO val2017: two cats on a couch
MASK = ROOT / "assets" / "bench_mask_cat.png"       # written by the labeler benchmark (left cat, dilated)
OUT = ROOT / "bench_out"
PROFILES = ROOT / "profiles"
GB = 2**30


def bench_case() -> InpaintRequest:
    image = Image.open(ASSET).convert("RGB")
    if MASK.exists():
        mask = Image.open(MASK).convert("L")
    else:
        mask = Image.new("L", image.size, 0)
        ImageDraw.Draw(mask).ellipse((40, 80, 300, 440), fill=255)
    reference = image.crop((346, 24, 639, 373))   # the right-hand cat, for reference-conditioned adapters
    return InpaintRequest(image=image, mask=mask, prompt="a small brown dog lying on a pink couch",
                          reference=reference, seed=0)


def _run_once(adapter, req):
    t = time.perf_counter()
    if isinstance(adapter, LabelerAdapter):
        out = adapter.detect(req.image, ["cat", "remote control"])
    else:
        out = adapter.inpaint(req)
    return time.perf_counter() - t, out


def run(name: str, backend: str | None, res: int | None, steps: int | None, n: int, duration: float, quantize: int | None) -> dict:
    cls = registry.get(name)
    kwargs = {"backend": Backend(backend) if backend else None, "native_res": res}
    if quantize is not None:
        kwargs["quantize"] = quantize
    adapter = cls(**kwargs)
    req = bench_case()
    req.steps = steps
    if Conditioning.REFERENCE_OBJECT not in cls.caps.conditioning or Conditioning.TEXT in cls.caps.conditioning:
        req.reference = None    # text-capable adapters are benchmarked without a reference
    pids = getattr(adapter, "child_pids", lambda: ())

    rec = {"adapter": name, "backend": adapter.backend.value, "res": adapter.res, "steps": steps or "default",
           "licence": cls.caps.licence, "weights": cls.caps.weights}
    if quantize is not None:
        rec["quantize"] = quantize

    if isinstance(adapter, GeneratorAdapter):
        with PeakSampler(child_pids=pids) as p:
            t = time.perf_counter()
            adapter.prepare([req.prompt])
            rec["prepare_s"] = round(time.perf_counter() - t, 2)
        rec["peak_prepare_gb"] = round(p.peak / GB, 2)

    with PeakSampler(child_pids=pids) as p:
        t = time.perf_counter()
        adapter.load()
        rec["load_s"] = round(time.perf_counter() - t, 2)
    rec["peak_load_gb"] = round(p.peak / GB, 2)

    with PeakSampler(child_pids=pids) as p:   # first call: kernel compilation, and lazy weight loading/quantisation on MLX
        warmup, out = _run_once(adapter, req)
    rec["warmup_s"] = round(warmup, 2)
    rec["peak_warmup_gb"] = round(p.peak / GB, 2)

    with PeakSampler(child_pids=pids) as p:   # steady state: what a long generation run actually holds
        times = []
        t_end = time.perf_counter() + duration
        while len(times) < n or time.perf_counter() < t_end:
            req.seed += 1
            dt, _ = _run_once(adapter, req)
            times.append(dt)
    rec["peak_infer_gb"] = round(p.peak / GB, 2)
    rec["lifetime_peak_gb"] = round(lifetime_peak_bytes() / GB, 2)   # main process only
    if isinstance(adapter, GeneratorAdapter):
        adapter.unload()

    rec.update(
        n=len(times),
        s_per_item_mean=round(statistics.mean(times), 3),
        s_per_item_median=round(statistics.median(times), 3),
        s_per_item_std=round(statistics.stdev(times), 3) if len(times) > 1 else 0.0,
        s_per_item_min=round(min(times), 3),
        items_per_hour=round(3600 / statistics.median(times)),
    )
    if duration and len(times) >= 10:
        k = max(5, len(times) // 10)
        rec["sustained"] = {"seconds": round(sum(times)), "first_mean_s": round(statistics.mean(times[:k]), 3),
                            "last_mean_s": round(statistics.mean(times[-k:]), 3)}

    OUT.mkdir(exist_ok=True)
    tag = f"{name}_{rec['backend']}_{rec['res']}"
    if isinstance(adapter, LabelerAdapter):
        rec["detections"] = [{"label": d.label, "score": round(d.score, 3), "box": [round(v) for v in d.box]} for d in out]
        _draw_detections(req.image, out).save(OUT / f"{tag}.png")
        cats = sorted((d for d in out if d.label == "cat" and d.mask is not None), key=lambda d: d.box[0])
        if cats:
            cats[0].mask.filter(ImageFilter.MaxFilter(21)).save(MASK)
    else:
        out.image.save(OUT / f"{tag}.png")
    return rec


def _draw_detections(image, dets):
    im = image.copy()
    overlay = Image.new("RGB", im.size, (255, 0, 0))
    for d in dets:
        if d.mask is not None:
            im = Image.composite(overlay, im, d.mask.point(lambda v: 90 if v else 0))
    dr = ImageDraw.Draw(im)
    for d in dets:
        dr.rectangle(d.box, outline=(255, 255, 0), width=2)
        dr.text((d.box[0] + 3, d.box[1] + 3), f"{d.label} {d.score:.2f}", fill=(255, 255, 0))
    return im


def save(rec: dict) -> Path:
    prof = machine_profile()
    PROFILES.mkdir(exist_ok=True)
    path = PROFILES / f"{profile_id(prof)}.json"
    data = json.loads(path.read_text()) if path.exists() else {"machine": prof, "results": {}}
    data["machine"] = prof
    key = f"{rec['adapter']}@{rec['backend']}:{rec['res']}:{rec['steps']}" + (f":q{rec['quantize']}" if "quantize" in rec else "")
    rec["measured_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    data["results"][key] = rec
    path.write_text(json.dumps(data, indent=2))
    return path


DEFAULT_SUITE = [
    ["gdino_sam2"],          # first: writes the cat mask the generators use
    ["lama"],
    ["sd15_inpaint"],
    ["flux2_klein_inpaint", "--res", "512"],
    ["flux2_klein_inpaint", "--res", "1024"],
    ["copy_paste"],
    ["face_parsing"],
    ["mimicbrush", "--steps", "25"],
    ["anydoor", "--steps", "30"],
    # mflux img2img always skips >= 1 step: 5 steps at strength 0.85 = 4 denoising steps, matching MPS
    ["flux2_klein_mlx_refine", "--res", "512", "--steps", "5", "--quantize", "8"],
    ["flux2_klein_mlx_refine", "--res", "1024", "--steps", "5", "--quantize", "8"],
]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="synthgen.bench")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    r = sub.add_parser("run")
    r.add_argument("name", choices=registry.names())
    r.add_argument("--backend")
    r.add_argument("--res", type=int)
    r.add_argument("--steps", type=int)
    r.add_argument("--quantize", type=int)
    r.add_argument("--n", type=int, default=5)
    r.add_argument("--duration", type=float, default=0, help="keep running for at least this many seconds")
    a = sub.add_parser("all")
    a.add_argument("--n", type=int, default=5)
    args = ap.parse_args(argv)

    if args.cmd == "list":
        for n in registry.names():
            c = registry.get(n).caps
            print(f"{n:24s} {','.join(b.value for b in c.backends):9s} res={c.native_res:<5d} "
                  f"ops={','.join(sorted(o.value for o in c.operations)) or '-':40s} {c.licence}{'  [gated]' if c.gated else ''}")
    elif args.cmd == "run":
        rec = run(args.name, args.backend, args.res, args.steps, args.n, args.duration, args.quantize)
        print(json.dumps(rec, indent=2))
        print("saved to", save(rec))
    elif args.cmd == "all":
        failed = []
        for item in DEFAULT_SUITE:
            backends = registry.get(item[0]).caps.backends
            if not any(available(b) for b in backends):
                print(f"\n=== {' '.join(item)}: skipped, needs {'/'.join(b.value for b in backends)} ===", flush=True)
                continue
            print(f"\n=== {' '.join(item)} ===", flush=True)
            rc = subprocess.run([sys.executable, "-m", "synthgen.bench", "run", *item, "--n", str(args.n)], cwd=ROOT).returncode
            if rc:
                failed.append(" ".join(item))
        if failed:
            print("\nfailed:", *failed, sep="\n  ")
            sys.exit(1)


if __name__ == "__main__":
    main()
