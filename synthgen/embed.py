"""DINOv2 image embeddings (facebook/dinov2-small, Apache-2.0) for reference matching.

The CLS embedding of an object crop tracks viewpoint, shape and object type more than colour, so
nearest neighbours of a target car are cars seen from a similar angle.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from synthgen.device import torch_device

_REPO = "facebook/dinov2-small"


class DinoEmbedder:
    def __init__(self, device: str | None = None):
        self.device = device = device or torch_device()
        self.proc = AutoImageProcessor.from_pretrained(_REPO)
        self.model = AutoModel.from_pretrained(_REPO).to(device).eval()
        self.cache: dict = {}

    @torch.inference_mode()
    def embed(self, image: Image.Image, key=None) -> np.ndarray:
        if key is not None and key in self.cache:
            return self.cache[key]
        x = self.proc(images=image.convert("RGB"), return_tensors="pt").to(self.device)
        v = self.model(**x).last_hidden_state[0, 0].float().cpu().numpy()
        v /= np.linalg.norm(v) + 1e-8
        if key is not None:
            self.cache[key] = v
        return v
