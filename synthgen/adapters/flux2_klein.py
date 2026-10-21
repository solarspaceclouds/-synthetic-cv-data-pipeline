"""FLUX.2 [klein] 4B masked inpainting via diffusers on MPS or CUDA (Apache-2.0, 4-step distilled).

bf16 weights are ~7.75 GB transformer + ~8 GB Qwen3 text encoder, too much to hold together
under the ~17.8 GB GPU working set. So prompts are encoded in `prepare()` with only the text
encoder loaded, which is then freed before `load()` brings in the transformer and VAE.
"""

from __future__ import annotations

import gc

import torch
from diffusers import Flux2KleinInpaintPipeline
from PIL import Image

from synthgen.adapters.base import GeneratorAdapter
from synthgen.device import empty_cache
from synthgen.types import Backend, Capabilities, Conditioning, Domain, InpaintRequest, Operation

_REPO = "black-forest-labs/FLUX.2-klein-4B"


class Flux2KleinInpaintAdapter(GeneratorAdapter):
    name = "flux2_klein_inpaint"
    caps = Capabilities(
        domains=frozenset({Domain.SCENE, Domain.FACE}),
        operations=frozenset({Operation.REGENERATE, Operation.REPLACE, Operation.CHANGE_CLASS, Operation.INSERT}),
        conditioning=frozenset({Conditioning.TEXT, Conditioning.REFERENCE_OBJECT}),
        backends=(Backend.MPS, Backend.CUDA),
        native_res=1024,
        licence="Apache-2.0",
        weights=_REPO,
        approx_peak_gb=12.0,
        notes="Also accepts image_reference for reference-guided inpainting.",
    )

    _keep = ("backend", "res", "embeds")

    def __init__(self, backend=None, native_res: int | None = None):
        super().__init__(backend, native_res)
        self.embeds: dict[str, torch.Tensor] = {}

    def prepare(self, prompts: list[str]) -> None:
        prompts = sorted(set(prompts) - set(self.embeds))
        if not prompts:
            return
        d = self.backend.value
        enc = Flux2KleinInpaintPipeline.from_pretrained(_REPO, transformer=None, vae=None, torch_dtype=torch.bfloat16).to(d)
        with torch.inference_mode():
            for p in prompts:
                self.embeds[p] = enc.encode_prompt(p, device=torch.device(d))[0]
        del enc
        gc.collect()
        empty_cache(d)

    def load(self):
        self.pipe = Flux2KleinInpaintPipeline.from_pretrained(
            _REPO, text_encoder=None, tokenizer=None, torch_dtype=torch.bfloat16
        ).to(self.backend.value)
        self.pipe.set_progress_bar_config(disable=True)

    def _inpaint_crop(self, image: Image.Image, mask: Image.Image, req: InpaintRequest) -> Image.Image:
        if req.prompt not in self.embeds:
            raise RuntimeError("call prepare() with this request before load()")
        g = torch.Generator("cpu").manual_seed(req.seed)
        return self.pipe(
            prompt_embeds=self.embeds[req.prompt], image=image, mask_image=mask,
            image_reference=req.reference, height=image.height, width=image.width,
            num_inference_steps=req.steps or 4, guidance_scale=req.guidance or 1.0,
            strength=req.strength or 1.0, generator=g,
        ).images[0]
