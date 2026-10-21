"""FLUX.2 [klein] 4B on MLX via mflux, quantized.

mflux has no masked FLUX.2 inpainting, so this runs img2img on the crop and composites the
result back through the feathered mask ("paste-then-refine"). Content inside the mask is
seeded from the original pixels, so it suits regenerate/refine better than insertion into
empty space. It is here mainly to compare the MLX runtime against PyTorch-MPS on the same model.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from PIL import Image

from synthgen.adapters.base import GeneratorAdapter
from synthgen.types import Backend, Capabilities, Conditioning, Domain, InpaintRequest, Operation


class Flux2KleinMLXRefineAdapter(GeneratorAdapter):
    name = "flux2_klein_mlx_refine"
    caps = Capabilities(
        domains=frozenset({Domain.SCENE, Domain.FACE}),
        operations=frozenset({Operation.REGENERATE}),
        conditioning=frozenset({Conditioning.TEXT}),
        backends=(Backend.MLX,),
        native_res=1024,
        licence="Apache-2.0",
        weights="black-forest-labs/FLUX.2-klein-4B",
        approx_peak_gb=6.0,
    )

    def __init__(self, backend=None, native_res: int | None = None, quantize: int = 8, cache_limit_gb: float = 2.0):
        super().__init__(backend, native_res)
        self.quantize = quantize
        self.cache_limit_gb = cache_limit_gb

    _keep = ("backend", "res", "quantize", "cache_limit_gb")

    def load(self):
        import mlx.core as mx
        from mflux.models.common.config.model_config import ModelConfig
        from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

        # MLX keeps freed buffers for reuse; uncapped, that cache alone reached 6-9 GB here.
        mx.set_cache_limit(int(self.cache_limit_gb * 2**30))
        self.model = Flux2Klein(quantize=self.quantize, model_config=ModelConfig.flux2_klein_4b())
        self.tmp = tempfile.TemporaryDirectory()

    def _inpaint_crop(self, image: Image.Image, mask: Image.Image, req: InpaintRequest) -> Image.Image:
        src = Path(self.tmp.name) / "crop.png"
        image.save(src)
        out = self.model.generate_image(
            seed=req.seed, prompt=req.prompt, num_inference_steps=req.steps or 4,
            height=image.height, width=image.width, guidance=req.guidance or 1.0,
            # mflux's image_strength is the share of the original kept; req.strength is the share
            # re-noised (diffusers convention). mflux starts at step max(1, int(steps * image_strength)),
            # so img2img always skips at least one step: steps=5, strength=0.85 -> 4 denoising steps.
            image_path=str(src), image_strength=1.0 - (req.strength or 0.85),
        )
        return out.image
