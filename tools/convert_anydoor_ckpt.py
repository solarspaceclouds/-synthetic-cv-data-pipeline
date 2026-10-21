"""Convert the AnyDoor Lightning checkpoint (16.8 GB, pickle, includes optimizer state) into an
fp16 safetensors state dict (~5 GB) that the worker loads without unpickling anything.

The .ckpt is opened with torch.load(weights_only=True, mmap=True): only tensors and plain
containers are accepted, so code embedded in the pickle cannot run.

    .venv/bin/python tools/convert_anydoor_ckpt.py <path/to/epoch=1-step=8687.ckpt> ~/.cache/synthgen/anydoor_fp16.safetensors
"""

import sys
from pathlib import Path

import torch
from safetensors.torch import save_file


def main(src: str, dst: str):
    ckpt = torch.load(src, map_location="cpu", weights_only=True, mmap=True)
    sd = ckpt.get("state_dict", ckpt)
    # network weights -> fp16; diffusion schedule buffers (alphas_cumprod, ...) stay fp32 for sampling precision
    nets = ("model.", "control_model.", "cond_stage_model.", "first_stage_model.")
    out = {k: (v.half() if v.is_floating_point() and k.startswith(nets) else v).contiguous()
           for k, v in sd.items() if isinstance(v, torch.Tensor)}
    prefixes = sorted({k.split(".")[0] for k in out})
    print(f"{len(out)} tensors; top-level modules: {prefixes}")
    print("DINOv2 weights present:", any(k.startswith("cond_stage_model.model.") for k in out))
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    save_file(out, dst)
    print("wrote", dst, f"{Path(dst).stat().st_size / 2**30:.2f} GB")


if __name__ == "__main__":
    main(*sys.argv[1:3])
