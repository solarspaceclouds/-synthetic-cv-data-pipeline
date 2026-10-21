"""Stable Diffusion 1.5 inpainting (UNet). Text-guided regeneration/insertion; stands in for the SD-1.5
family (BrushNet, PowerPaint, MimicBrush, AnyDoor) when estimating cost on this machine."""

from __future__ import annotations

import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image

from synthgen.adapters.base import GeneratorAdapter
from synthgen.types import Backend, Capabilities, Conditioning, Domain, InpaintRequest, Operation

_REPO = "stable-diffusion-v1-5/stable-diffusion-inpainting"


class SD15InpaintAdapter(GeneratorAdapter):
    name = "sd15_inpaint"
    caps = Capabilities(
        domains=frozenset({Domain.SCENE, Domain.FACE}),
        operations=frozenset({Operation.REGENERATE, Operation.REPLACE, Operation.CHANGE_CLASS, Operation.INSERT}),
        conditioning=frozenset({Conditioning.TEXT}),
        backends=(Backend.MPS, Backend.CUDA, Backend.CPU),
        native_res=512,
        licence="CreativeML OpenRAIL-M",
        weights=_REPO,
        approx_peak_gb=4.0,
    )

    def load(self):
        dtype = torch.float16 if self.backend in (Backend.MPS, Backend.CUDA) else torch.float32
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            _REPO, torch_dtype=dtype, variant="fp16", use_safetensors=True,
            safety_checker=None, requires_safety_checker=False,
        ).to(self.backend.value)
        self.pipe.set_progress_bar_config(disable=True)

    def _inpaint_crop(self, image: Image.Image, mask: Image.Image, req: InpaintRequest) -> Image.Image:
        g = torch.Generator("cpu").manual_seed(req.seed)
        return self.pipe(
            prompt=req.prompt, image=image, mask_image=mask, height=image.height, width=image.width,
            num_inference_steps=req.steps or 25, guidance_scale=req.guidance or 7.5,
            strength=req.strength or 1.0, generator=g,
        ).images[0]
