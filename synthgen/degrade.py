"""Post-processing: realistic image degradations applied to the whole output image.

Two reasons, from the plan review:
  * domain matching: CCTV/body-cam/phone footage is low-resolution, noisy and heavily compressed;
    clean diffusion output is not
  * shortcut prevention: applying the same degradations everywhere masks the edited region's
    tell-tale texture, so a detector learns the object, not the inpainting artefact

Only photometric and resolution operations are used, never geometric ones, so masks and boxes
stay valid. Parameters are sampled per image from a seeded RNG and recorded.

Config (job YAML):
    postprocess:
      repeat: 1                      # 2 = second-order degradation (Real-ESRGAN style)
      ops:
        - {op: downscale, p: 0.5, factor: [0.4, 0.8]}
        - {op: blur, p: 0.3, sigma: [0.3, 1.2]}
        - {op: noise, p: 0.5, sigma: [1, 6]}
        - {op: color, p: 0.3, brightness: [0.9, 1.1], contrast: [0.9, 1.1], saturation: [0.85, 1.1]}
        - {op: jpeg, p: 0.8, quality: [40, 85]}

Apply the same pipeline to real images (so real and synthetic share one distribution):
    python -m synthgen.degrade configs/job.yaml <in_dir> <out_dir>
"""

from __future__ import annotations

import argparse
import hashlib
import io
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

_PARAMS = {
    "downscale": {"factor"},
    "blur": {"sigma"},
    "noise": {"sigma"},
    "color": {"brightness", "contrast", "saturation"},
    "jpeg": {"quality"},
}


def validate_ops(ops: list[dict]) -> None:
    for i, o in enumerate(ops):
        name = o.get("op")
        if name not in _PARAMS:
            raise ValueError(f"postprocess.ops[{i}]: unknown op {name!r}; allowed: {sorted(_PARAMS)}")
        extra = set(o) - _PARAMS[name] - {"op", "p"}
        if extra:
            raise ValueError(f"postprocess.ops[{i}] ({name}): unknown key(s) {sorted(extra)}; allowed: {sorted(_PARAMS[name])}")


def _u(rng: random.Random, r) -> float:
    return rng.uniform(*r) if isinstance(r, (list, tuple)) else float(r)


def apply(image: Image.Image, ops: list[dict], rng: random.Random, repeat: int = 1) -> tuple[Image.Image, list[dict]]:
    im = image.convert("RGB")
    W, H = im.size
    applied = []
    for _ in range(max(1, repeat)):
        for o in ops:
            if rng.random() >= o.get("p", 1.0):
                continue
            name = o["op"]
            if name == "downscale":
                f = _u(rng, o.get("factor", [0.5, 0.8]))
                small = im.resize((max(8, round(W * f)), max(8, round(H * f))), Image.BILINEAR)
                im = small.resize((W, H), Image.BICUBIC)
                applied.append({"op": name, "factor": round(f, 3)})
            elif name == "blur":
                s = _u(rng, o.get("sigma", [0.3, 1.2]))
                im = im.filter(ImageFilter.GaussianBlur(s))
                applied.append({"op": name, "sigma": round(s, 3)})
            elif name == "noise":
                s = _u(rng, o.get("sigma", [1, 6]))
                gen = np.random.default_rng(rng.getrandbits(32))
                a = np.asarray(im, dtype=np.float32) + gen.normal(0, s, (H, W, 1 if rng.random() < 0.3 else 3))
                im = Image.fromarray(np.clip(a, 0, 255).round().astype(np.uint8))
                applied.append({"op": name, "sigma": round(s, 3)})
            elif name == "color":
                rec = {"op": name}
                for k, cls in (("brightness", ImageEnhance.Brightness), ("contrast", ImageEnhance.Contrast),
                               ("saturation", ImageEnhance.Color)):
                    if k in o:
                        v = _u(rng, o[k])
                        im = cls(im).enhance(v)
                        rec[k] = round(v, 3)
                applied.append(rec)
            elif name == "jpeg":
                q = int(round(_u(rng, o.get("quality", [40, 85]))))
                buf = io.BytesIO()
                im.save(buf, "JPEG", quality=q)
                im = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
                applied.append({"op": name, "quality": q})
    return im, applied


def seed_for(name: str, base: int = 0) -> int:
    return base * 1_000_003 + int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)


def main(argv=None):
    from synthgen.config import load_job

    ap = argparse.ArgumentParser(prog="synthgen.degrade")
    ap.add_argument("job", help="job YAML whose postprocess pipeline to apply")
    ap.add_argument("src")
    ap.add_argument("dst")
    a = ap.parse_args(argv)
    cfg = load_job(a.job)
    pp = cfg.postprocess
    validate_ops(pp.ops)
    dst = Path(a.dst)
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(Path(a.src).iterdir()):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            continue
        im, _ = apply(Image.open(p), pp.ops, random.Random(seed_for(p.name, cfg.seed)), pp.repeat)
        im.save(dst / (p.stem + ".jpg"), quality=cfg.output.jpeg_quality)
        n += 1
    print(f"degraded {n} images -> {dst}")


if __name__ == "__main__":
    main()
