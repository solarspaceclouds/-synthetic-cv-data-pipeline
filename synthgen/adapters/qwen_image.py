"""Qwen-Image editing models.

qwen_image21_edit      Qwen/Qwen-Image-2.1 (7B DiT + Qwen3-VL-8B text encoder, 33 GB in bf16; Qwen RESEARCH
                       licence, non-commercial only) on MLX via mflux, quantised at load (4-bit default).
                       Native mask input: unmasked latents follow the source and are composited back exactly.
                       It is an instruction editor, so the job prompt is wrapped in an edit instruction.
                       M5 Pro 24 GB, 512 px, 20 steps: 22 s per image, 14.6 GB peak (bench).
qwen_image_edit_4bit   ovedrive/qwen-image-edit-4bit: Qwen-Image-Edit (20B DiT, Apache-2.0), pre-quantised with
                       bitsandbytes NF4, through diffusers' QwenImageEditInpaintPipeline. bitsandbytes has no Metal
                       kernels: on MPS every layer is dequantised in fp32 PyTorch on each call, which took 25 min
                       and ~28 GB for one 768 px image here (swapping), and the next image ran out of memory.
                       Use it on a CUDA GPU with 20 GB or more (backend: cuda).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from PIL import Image

from synthgen.adapters.base import GeneratorAdapter
from synthgen.types import Backend, Capabilities, Conditioning, Domain, InpaintRequest, Operation

_OPS = frozenset({Operation.REGENERATE, Operation.REPLACE, Operation.CHANGE_CLASS, Operation.INSERT})


class QwenImage21EditAdapter(GeneratorAdapter):
    name = "qwen_image21_edit"
    caps = Capabilities(
        domains=frozenset({Domain.SCENE, Domain.FACE}),
        operations=_OPS,
        conditioning=frozenset({Conditioning.TEXT}),
        backends=(Backend.MLX,),
        native_res=512,
        licence="Qwen Research (non-commercial)",
        weights="Qwen/Qwen-Image-2.1",
        approx_peak_gb=15.0,
        notes="Instruction editor with a native mask; 33 GB download, quantised at load. Non-commercial licence.",
    )

    # Job prompts are captions ("a photo of a red car"); the model follows edit instructions better.
    # A prompt that is already an instruction ("Remove the person ...") is used as it is.
    INSTRUCTION = "Replace the masked area with {subject}. Keep the same position, perspective, lighting and scale."
    VERBS = ("Remove ", "Replace ", "Change ", "Make ", "Add ", "Turn ")

    def __init__(self, backend=None, native_res: int | None = None, quantize: int = 4, cache_limit_gb: float = 2.0):
        super().__init__(backend, native_res)
        if self.res % 32:
            raise ValueError(f"qwen_image21_edit needs res to be a multiple of 32, got {self.res}")
        self.quantize = quantize
        self.cache_limit_gb = cache_limit_gb

    _keep = ("backend", "res", "quantize", "cache_limit_gb")

    def load(self):
        import mlx.core as mx
        from mflux.models.qwen21.variants.edit.qwen_image_21_edit import QwenImage21Edit

        mx.set_cache_limit(int(self.cache_limit_gb * 2**30))
        self.model = QwenImage21Edit(quantize=self.quantize)
        self.tmp = tempfile.TemporaryDirectory()

    def _instruction(self, prompt: str) -> str:
        if prompt.startswith(self.VERBS):
            return prompt
        return self.INSTRUCTION.format(subject=prompt.removeprefix("a photo of "))

    def _inpaint_crop(self, image: Image.Image, mask: Image.Image, req: InpaintRequest) -> Image.Image:
        src = Path(self.tmp.name) / "crop.png"
        image.save(src)
        out = self.model.generate_image(
            seed=req.seed, prompt=self._instruction(req.prompt), image_paths=[str(src)],
            mask_image=mask.convert("L"), width=image.width, height=image.height, output_resolution=self.res,
            num_inference_steps=req.steps or 20, guidance=req.guidance or 1.0, strength=req.strength or 1.0,
        )
        return out.image.convert("RGB").resize(image.size, Image.LANCZOS)


class QwenImageEdit4bitAdapter(GeneratorAdapter):
    name = "qwen_image_edit_4bit"
    caps = Capabilities(
        domains=frozenset({Domain.SCENE, Domain.FACE}),
        operations=_OPS,
        conditioning=frozenset({Conditioning.TEXT}),
        backends=(Backend.CUDA, Backend.MPS),
        native_res=1024,
        licence="Apache-2.0",
        weights="ovedrive/qwen-image-edit-4bit",
        approx_peak_gb=20.0,
        notes="bitsandbytes NF4: CUDA only in practice; on MPS ~25 min per image and over 24 GB. guidance = true_cfg_scale.",
    )

    def load(self):
        import torch
        from diffusers import QwenImageEditInpaintPipeline
        if self.backend is Backend.MPS:
            print("[qwen_image_edit_4bit] warning: bitsandbytes has no Metal kernels; expect ~25 min per image "
                  "and memory above 24 GB on Apple Silicon")
        self.pipe = QwenImageEditInpaintPipeline.from_pretrained(self.caps.weights, torch_dtype=torch.bfloat16)
        self.pipe.to(self.backend.value)
        self.pipe.set_progress_bar_config(disable=True)

    def _inpaint_crop(self, image: Image.Image, mask: Image.Image, req: InpaintRequest) -> Image.Image:
        import torch
        g = torch.Generator("cpu").manual_seed(req.seed)
        return self.pipe(
            image=image, mask_image=mask, prompt=req.prompt, negative_prompt=" ",
            height=image.height, width=image.width, num_inference_steps=req.steps or 20,
            true_cfg_scale=req.guidance or 4.0, strength=req.strength or 1.0, generator=g,
        ).images[0].resize(image.size, Image.LANCZOS)
