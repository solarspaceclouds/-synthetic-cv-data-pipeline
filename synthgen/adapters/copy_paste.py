"""Copy-paste baseline (no generative model): cut the reference object out, scale it into the edit
region, and blend it in. The classic Cut-Paste-Learn / Simple Copy-Paste augmentation, and the
baseline every inpainting model here should be compared against: cheap, exact object appearance,
but no relighting, shadows or perspective change.

For `replace`, the old object must disappear first, so the region is cleared with LaMa (an
existing adapter, composed here) before pasting. This also shows how adapters can be combined.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

from synthgen.adapters.base import GeneratorAdapter
from synthgen.adapters.lama import LamaAdapter
from synthgen.types import Backend, Capabilities, Conditioning, Domain, InpaintRequest, Operation


class CopyPasteAdapter(GeneratorAdapter):
    name = "copy_paste"
    caps = Capabilities(
        domains=frozenset({Domain.SCENE}),
        operations=frozenset({Operation.REPLACE, Operation.INSERT, Operation.CHANGE_CLASS}),
        conditioning=frozenset({Conditioning.REFERENCE_OBJECT}),
        backends=(Backend.MPS, Backend.CUDA, Backend.CPU),
        native_res=512,
        licence="no model; background clearing uses LaMa (Apache-2.0)",
        weights="none (+ big-lama for clearing the region)",
        approx_peak_gb=1.6,
        notes="Reference object required (crop + mask). Baseline for comparing generative models.",
    )

    _keep = ("backend", "res", "clear_background", "harmonize", "fill")

    def __init__(self, backend=None, native_res=None, clear_background: bool = True, harmonize: float = 0.3,
                 fill: float = 0.9):
        super().__init__(backend, native_res)
        self.clear_background = clear_background   # erase what's in the region first (needed for replace)
        self.harmonize = harmonize                 # 0..1: pull the object's brightness toward its surroundings
        self.fill = fill                           # object size relative to the region's bounding box

    def load(self):
        if self.clear_background:
            self.lama = LamaAdapter(self.backend)
            self.lama.load()

    def _inpaint_crop(self, image: Image.Image, mask: Image.Image, req: InpaintRequest) -> Image.Image:
        if req.reference is None:
            raise ValueError("copy_paste needs a reference object")
        base = self.lama._inpaint_crop(image, mask, req) if self.clear_background else image.convert("RGB")

        ref = req.reference.convert("RGB")
        ref_mask = req.reference_mask.convert("L") if req.reference_mask is not None else \
            Image.fromarray(((np.asarray(ref).min(-1) < 245) * 255).astype(np.uint8))   # object on white
        rm = np.asarray(ref_mask) > 127
        if not rm.any():
            raise ValueError("empty reference mask")
        ys, xs = np.where(rm)
        box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
        obj, alpha = ref.crop(box), ref_mask.crop(box)

        # fit inside the region's bounding box, keep aspect ratio, stand on the region's bottom edge
        m = np.asarray(mask) > 127
        ys, xs = np.where(m)
        bx0, by0, bx1, by1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        s = self.fill * min((bx1 - bx0) / obj.width, (by1 - by0) / obj.height)
        w, h = max(1, round(obj.width * s)), max(1, round(obj.height * s))
        obj, alpha = obj.resize((w, h), Image.LANCZOS), alpha.resize((w, h), Image.LANCZOS)
        x = (bx0 + bx1 - w) // 2
        y = by1 - h - max(0, round((by1 - by0) * (1 - self.fill) / 4))

        if self.harmonize > 0:
            ring = np.asarray(mask.filter(ImageFilter.MaxFilter(15))) > 127
            ring &= ~m
            ctx = np.asarray(base.convert("L"), np.float32)[ring].mean() if ring.any() else None
            a = np.asarray(alpha) > 127
            arr = np.asarray(obj, np.float32)
            own = arr[a].mean() if a.any() else None
            if ctx is not None and own:
                gain = 1 + self.harmonize * (ctx / own - 1)
                obj = Image.fromarray(np.clip(arr * gain, 0, 255).astype(np.uint8))

        out = base.copy()
        out.paste(obj, (x, y), alpha.filter(ImageFilter.GaussianBlur(1.5)))
        return out
