"""LaMa (big-lama, Fourier-convolution CNN). Context-only fill: removal and background completion."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from synthgen.adapters.base import GeneratorAdapter
from synthgen.types import Backend, Capabilities, Conditioning, Domain, InpaintRequest, Operation

# TorchScript export used by IOPaint (Apache-2.0 weights from advimman/lama).
_URL = "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt"
_MD5 = "e3aa4aaa15225a33ec84f9f4bc47e500"
_CACHE = Path.home() / ".cache" / "synthgen" / "big-lama.pt"


def _fetch() -> Path:
    if not _CACHE.exists():
        _CACHE.parent.mkdir(parents=True, exist_ok=True)
        tmp = _CACHE.with_suffix(".part")
        urllib.request.urlretrieve(_URL, tmp)
        if hashlib.md5(tmp.read_bytes()).hexdigest() != _MD5:
            tmp.unlink()
            raise RuntimeError("big-lama.pt checksum mismatch")
        tmp.rename(_CACHE)
    return _CACHE


class LamaAdapter(GeneratorAdapter):
    name = "lama"
    caps = Capabilities(
        domains=frozenset({Domain.SCENE, Domain.FACE}),
        operations=frozenset({Operation.REMOVE}),
        conditioning=frozenset({Conditioning.NONE}),
        backends=(Backend.MPS, Backend.CUDA, Backend.CPU),
        native_res=512,
        licence="Apache-2.0",
        weights=_URL,
        approx_peak_gb=1.5,
        notes="Places2-trained; for faces a face-trained context model (MAT FFHQ) is the better choice.",
    )

    def load(self):
        self.device = torch.device(self.backend.value)
        self.model = torch.jit.load(str(_fetch()), map_location=self.device).eval()

    @torch.inference_mode()
    def _inpaint_crop(self, image: Image.Image, mask: Image.Image, req: InpaintRequest) -> Image.Image:
        img = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1)[None]
        m = torch.from_numpy((np.asarray(mask) > 127).astype(np.float32))[None, None]
        out = self.model(img.to(self.device), m.to(self.device))
        arr = (out[0].permute(1, 2, 0).clamp(0, 1) * 255).round().byte().cpu().numpy()
        return Image.fromarray(arr)
