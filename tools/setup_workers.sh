#!/usr/bin/env bash
# Set up the isolated model environments for the worker adapters (MimicBrush, AnyDoor).
# Each repo pins a 2023-era stack, so it gets its own Python 3.10 env in envs/<name>; its code is
# cloned unmodified into third_party/<Repo> at the commit this repo was tested with.
#   tools/setup_workers.sh mimicbrush          # ~5 GB env; weights download on first use
#   tools/setup_workers.sh anydoor             # ~3 GB env + 16.8 GB checkpoint download, converted to 4.6 GB
#   tools/setup_workers.sh all
set -euo pipefail
cd "$(dirname "$0")/.."
command -v uv >/dev/null || PATH="$(python3 -m site --user-base)/bin:$PATH"   # pip --user location
command -v uv >/dev/null || { echo "uv not found: python3 -m pip install --user uv"; exit 1; }

clone() {  # clone <url> <dir> <commit>
  if [ ! -d "$2/.git" ]; then git clone --quiet "$1" "$2"; fi
  git -C "$2" checkout --quiet "$3"
}

mimicbrush() {
  clone https://github.com/ali-vilab/MimicBrush third_party/MimicBrush 2ec7bc8b99c5dd8c8ae9aed0579470d81c74f3a0  # Apache-2.0
  [ -x envs/mimicbrush/bin/python ] || uv venv envs/mimicbrush --python 3.10
  VIRTUAL_ENV=envs/mimicbrush uv pip install "torch==2.14.1" "torchvision==0.29.1" "diffusers==0.23.0" "transformers==4.25.1" \
    "huggingface_hub==0.20.1" "accelerate==0.25.0" omegaconf einops opencv-python-headless "numpy<2" Pillow safetensors
  echo "[mimicbrush] ready (weights download on first use)"
}

anydoor() {
  clone https://github.com/ali-vilab/AnyDoor third_party/AnyDoor 44ca2b2a70ec2cf107f3d26a5b46def6670fb0a5  # MIT
  [ -x envs/anydoor/bin/python ] || uv venv envs/anydoor --python 3.10
  # torch 2.14 declares setuptools>=77 but pytorch_lightning 1.5 needs setuptools<70 (pkg_resources).
  # A resolver satisfying both falls back to torch 2.12, which runs AnyDoor 2.6x slower on MPS
  # (19 s vs 7.3 s per image), so install torch first, then the old setuptools without dependency checks.
  VIRTUAL_ENV=envs/anydoor uv pip install "torch==2.14.1" "torchvision==0.29.1" "pytorch_lightning==1.5.0" \
    "torchmetrics==0.6.0" "open_clip_torch==2.17.1" "timm==0.6.12" "transformers==4.25.1" "omegaconf==2.1.1" \
    einops scipy opencv-python-headless "numpy<2" safetensors "albumentations==1.3.0" Pillow tqdm
  VIRTUAL_ENV=envs/anydoor uv pip install --no-deps "setuptools==69.5.1"
  local out=~/.cache/synthgen/anydoor_fp16.safetensors
  if [ ! -f "$out" ]; then
    # the 16.8 GB Lightning checkpoint, converted once to fp16 safetensors (schedule buffers kept fp32)
    local ckpt
    ckpt=$(.venv/bin/python -c "from huggingface_hub import hf_hub_download as d; print(d('xichenhku/AnyDoor', 'epoch=1-step=8687.ckpt', repo_type='space'))")
    mkdir -p "$(dirname "$out")"
    .venv/bin/python tools/convert_anydoor_ckpt.py "$ckpt" "$out"
    echo "[anydoor] converted; you can delete the original checkpoint to free 16.8 GB: $ckpt"
  fi
  echo "[anydoor] ready"
}

[ -x .venv/bin/python ] || { echo "run 'uv sync' first"; exit 1; }
[ $# -gt 0 ] || { echo "usage: $0 mimicbrush|anydoor|all"; exit 1; }
for w in "$@"; do
  case "$w" in
    mimicbrush) mimicbrush ;;
    anydoor) anydoor ;;
    all) mimicbrush; anydoor ;;
    *) echo "unknown worker: $w"; exit 1 ;;
  esac
done
