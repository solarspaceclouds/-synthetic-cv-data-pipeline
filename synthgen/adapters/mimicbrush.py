"""MimicBrush (SD-1.5 + reference U-Net, Apache-2.0): fills the masked region by imitating a
reference image. Suited to part-level edits (eyes/mouth/hair from another face, a texture,
a car part). Runs in envs/mimicbrush via workers/mimicbrush_worker.py."""

from __future__ import annotations

import os

from huggingface_hub import snapshot_download

from synthgen.adapters.worker import WorkerAdapter
from synthgen.types import Backend, Capabilities, Conditioning, Domain, Operation


class MimicBrushAdapter(WorkerAdapter):
    name = "mimicbrush"
    env = "mimicbrush"
    script = "workers/mimicbrush_worker.py"
    caps = Capabilities(
        domains=frozenset({Domain.FACE, Domain.SCENE}),
        operations=frozenset({Operation.REGENERATE, Operation.REPLACE}),
        conditioning=frozenset({Conditioning.REFERENCE_OBJECT}),
        backends=(Backend.MPS, Backend.CUDA, Backend.CPU),
        native_res=512,
        licence="Apache-2.0 (MimicBrush); SD-1.5 bases CreativeML OpenRAIL-M",
        weights="xichenhku/MimicBrush + xichenhku/cleansd",
        approx_peak_gb=10.0,
        notes="Reference required. Ignores text prompts. 50 DDIM steps by default.",
    )

    def worker_config(self) -> dict:
        mb = snapshot_download("xichenhku/MimicBrush", allow_patterns=["mimicbrush/*", "image_encoder/*", "sd-vae-ft-mse/*", "depth_model/*"])
        sd = snapshot_download("xichenhku/cleansd", allow_patterns=["stable-diffusion-inpainting/*", "stable-diffusion-v1-5/*"])
        return {"weights": {
            "mimicbrush": os.path.join(mb, "mimicbrush", "mimicbrush.bin"),
            "image_encoder": os.path.join(mb, "image_encoder"),
            "vae": os.path.join(mb, "sd-vae-ft-mse"),
            "depth_model": os.path.join(mb, "depth_model", "depth_anything_vitb14.pth"),
            "sd_inpainting": os.path.join(sd, "stable-diffusion-inpainting"),
            "sd_reference": os.path.join(sd, "stable-diffusion-v1-5"),
        }}
