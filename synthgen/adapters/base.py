"""Adapter interfaces. A new model repo is integrated by writing one small subclass."""

from __future__ import annotations

import gc
from abc import ABC, abstractmethod
from typing import ClassVar

from PIL import Image

from synthgen.crop import context_box, crop_for_model, paste_back
from synthgen.device import empty_cache, pick
from synthgen.types import Backend, Capabilities, Detection, InpaintRequest, InpaintResult


class Adapter(ABC):
    name: ClassVar[str]
    caps: ClassVar[Capabilities]

    _keep = ("backend", "res")

    def __init__(self, backend: Backend | None = None, native_res: int | None = None):
        self.backend = backend or pick(self.caps.backends)
        self.res = native_res or self.caps.native_res
        if self.backend not in self.caps.backends:
            raise ValueError(f"{self.name} does not support backend {self.backend.value}")

    @abstractmethod
    def load(self) -> None: ...

    def unload(self) -> None:
        for k in list(vars(self)):
            if k not in self._keep:
                delattr(self, k)
        gc.collect()
        empty_cache(self.backend.value)


class GeneratorAdapter(Adapter):
    """Subclasses implement `_inpaint_crop` on a native-resolution crop; cropping and paste-back are shared."""

    def prepare(self, prompts: list[str]) -> None:
        """Batch-level work before `load()`, e.g. encoding all prompts with a text encoder that is then freed."""

    @abstractmethod
    def _inpaint_crop(self, image: Image.Image, mask: Image.Image, req: InpaintRequest) -> Image.Image: ...

    def inpaint(self, req: InpaintRequest) -> InpaintResult:
        box = context_box(req.mask, req.context_scale)
        img, m = crop_for_model(req.image, req.mask, box, self.res)
        gen = self._inpaint_crop(img, m, req)
        out = paste_back(req.image, req.mask, gen, box, req.feather_px)
        return InpaintResult(image=out, crop_box=box, meta={"adapter": self.name, "backend": self.backend.value, "res": self.res})


class LabelerAdapter(Adapter):
    @abstractmethod
    def detect(self, image: Image.Image, classes: list[str]) -> list[Detection]: ...
