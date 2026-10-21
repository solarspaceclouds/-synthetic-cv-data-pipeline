"""Vet a candidate model before integrating it: licence, gating, weight formats and size (Hugging
Face), and Mac blockers / unsafe loading / hidden downloads in its code (a local clone).

    .venv/bin/python .claude/skills/synthgen-add-model/scripts/vet_model.py --hf black-forest-labs/FLUX.2-klein-4B
    .venv/bin/python .claude/skills/synthgen-add-model/scripts/vet_model.py --hf spaces/xichenhku/AnyDoor --repo third_party/AnyDoor
    .venv/bin/python .claude/skills/synthgen-add-model/scripts/vet_model.py --repo third_party/MimicBrush --all-files

Read-only: queries the Hugging Face API (no download) and greps the clone. Training code is skipped
by default (paths containing train/finetune/dataset/eval/test); --all-files includes it.
"""

import argparse
import json
import re
import urllib.request
from pathlib import Path

PICKLE = (".bin", ".ckpt", ".pth", ".pt", ".pkl")
CHECKS = [
    ("cuda", r"\.cuda\(|device\s*=\s*['\"]cuda|torch\.device\(['\"]cuda", "hard-coded CUDA: patch to the configured device"),
    ("triton", r"\bimport triton|from triton", "Triton has no macOS support: use another implementation (e.g. transformers port)"),
    ("xformers", r"xformers", "no xformers on macOS: check there's a fallback, or swap in scaled_dot_product_attention"),
    ("flash_attn", r"flash_attn", "no flash-attention on macOS"),
    ("bitsandbytes", r"bitsandbytes", "bitsandbytes quantisation doesn't run on MPS: use fp16/bf16 or GGUF/MLX"),
    ("torch.load", r"torch\.load\(", "pickle loading: force weights_only=True (and mmap=True for big files)"),
    ("downloads", r"hub\.load\(|load_state_dict_from_url|pretrained\s*=\s*True|urlopen\(|requests\.get\(|wget ",
     "downloads at runtime/construction: check whether the main checkpoint already contains those weights"),
    ("exec", r"\bsubprocess\.|os\.system\(|(?<![.\w])eval\(|(?<![.\w])exec\(", "process/exec calls: read them before running"),
]
PINS = ("torch", "diffusers", "transformers", "huggingface_hub", "accelerate", "pytorch_lightning", "xformers",
        "numpy", "timm", "open_clip", "omegaconf", "triton", "bitsandbytes")
TRAINISH = re.compile(r"(^|/)(train|training|finetune|fine_tune|dataset|datasets|eval|evaluation|test|tests|scripts/train)", re.I)


def hf(repo: str):
    kind = "models"
    if repo.startswith("spaces/"):
        kind, repo = "spaces", repo[len("spaces/"):]
    url = f"https://huggingface.co/api/{kind}/{repo}?blobs=true"
    try:
        d = json.load(urllib.request.urlopen(url, timeout=30))
    except Exception as e:
        print(f"[hf] {repo}: could not query ({e})")
        return
    card = d.get("cardData") or {}
    files = d.get("siblings", [])
    total = sum(f.get("size") or 0 for f in files)
    print(f"[hf] {kind}/{repo}")
    print(f"  licence  : {card.get('license') or 'NOT STATED on the card: check the repo LICENSE and base models'}")
    print(f"  gated    : {d.get('gated')}  (manual = needs approval + `hf auth login`)")
    print(f"  total    : {total / 1e9:.2f} GB in {len(files)} files")
    weights = [f for f in files if f["rfilename"].endswith(PICKLE + (".safetensors", ".gguf", ".onnx"))]
    by_fmt = {}
    for f in weights:
        ext = Path(f["rfilename"]).suffix
        by_fmt.setdefault(ext, []).append(f)
    for ext, fs in sorted(by_fmt.items()):
        size = sum(f.get("size") or 0 for f in fs) / 1e9
        flag = "  <- pickle: load with weights_only=True" if ext in PICKLE else ""
        print(f"  {ext:12s} {len(fs):3d} files {size:7.2f} GB{flag}")
    for f in sorted(weights, key=lambda f: -(f.get("size") or 0))[:8]:
        print(f"     {f['rfilename']}  {(f.get('size') or 0) / 1e9:.2f} GB")
    st = sum(f.get("size") or 0 for f in weights if f["rfilename"].endswith(".safetensors") and "fp16" not in f["rfilename"])
    if st:
        print(f"  memory   : safetensors (non-fp16 variants) = {st / 1e9:.1f} GB on disk; in fp16 roughly half that if stored fp32, "
              "plus 30-50% activations. Budget <= 12 GB per model on a 24 GB Mac.")


def repo(path: Path, all_files: bool):
    print(f"\n[repo] {path}")
    lic = next((p for p in path.iterdir() if p.name.upper().startswith(("LICENSE", "LICENCE", "COPYING"))), None)
    print(f"  licence  : {lic.read_text(errors='ignore').strip().splitlines()[0][:80] if lic else 'NO LICENSE FILE: treat as all rights reserved'}")
    for req in [path / "requirements.txt", path / "environment.yaml", path / "setup.py", path / "pyproject.toml"]:
        if req.exists():
            lines = [l.strip() for l in req.read_text(errors="ignore").splitlines()
                     if any(re.match(rf"[-\s]*{p}\b", l.strip(), re.I) for p in PINS)]
            if lines:
                print(f"  pins ({req.name}): " + "; ".join(lines[:14]))
    files = [p for p in path.rglob("*.py") if all_files or not TRAINISH.search(str(p.relative_to(path)))]
    print(f"  scanned  : {len(files)} .py files" + ("" if all_files else " (training/eval/test code skipped)"))
    for name, pat, advice in CHECKS:
        rx = re.compile(pat)
        hits = []
        for p in files:
            for i, line in enumerate(p.read_text(errors="ignore").splitlines(), 1):
                if rx.search(line) and not line.lstrip().startswith("#"):
                    if name == "torch.load" and "weights_only=True" in line:
                        continue
                    hits.append(f"{p.relative_to(path)}:{i}: {line.strip()[:100]}")
        if hits:
            print(f"  {name:12s} {len(hits):3d} hit(s): {advice}")
            for h in hits[:4]:
                print(f"      {h}")
            if len(hits) > 4:
                print(f"      ... {len(hits) - 4} more")
    demo = [p.name for p in path.glob("*.py") if re.search(r"demo|gradio|inference|predict|run_", p.name)]
    if demo:
        print(f"  entry points: {', '.join(sorted(demo))}  (demo scripts often load the model at import: "
              "copy their pre/post-processing into a worker instead of importing them)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", action="append", default=[], help="model id, or spaces/<org>/<name>; repeatable")
    ap.add_argument("--repo", action="append", default=[], help="path to a local clone; repeatable")
    ap.add_argument("--all-files", action="store_true", help="also scan training/eval/test code")
    a = ap.parse_args()
    if not a.hf and not a.repo:
        ap.error("give --hf and/or --repo")
    for r in a.hf:
        hf(r)
    for r in a.repo:
        repo(Path(r), a.all_files)


if __name__ == "__main__":
    main()
