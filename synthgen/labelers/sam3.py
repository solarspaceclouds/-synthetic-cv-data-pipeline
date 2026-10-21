"""SAM 3 via transformers (the official repo needs Triton, which has no Mac support).

facebook/sam3 is gated on Hugging Face with manual approval: request access on the model page,
then run `hf auth login`.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor

from synthgen.adapters.base import LabelerAdapter
from synthgen.types import Backend, Capabilities, Conditioning, Detection, Domain

_REPO = "facebook/sam3"


class Sam3Labeler(LabelerAdapter):
    name = "sam3"
    caps = Capabilities(
        domains=frozenset({Domain.SCENE, Domain.FACE}),
        operations=frozenset(),
        conditioning=frozenset({Conditioning.TEXT}),
        backends=(Backend.MPS, Backend.CUDA, Backend.CPU),
        native_res=1008,
        licence="SAM License (see model card)",
        weights=_REPO,
        approx_peak_gb=4.0,
        gated=True,
    )

    def __init__(self, backend=None, native_res=None, threshold: float = 0.5):
        super().__init__(backend, native_res)
        self.threshold = threshold

    _keep = ("backend", "res", "threshold")

    def load(self):
        self.proc = Sam3Processor.from_pretrained(_REPO)
        self.model = Sam3Model.from_pretrained(_REPO).to(self.backend.value).eval()

    @torch.inference_mode()
    def detect(self, image: Image.Image, classes: list[str]) -> list[Detection]:
        image = image.convert("RGB")
        dets = []
        for c in classes:
            inputs = self.proc(images=image, text=c, return_tensors="pt").to(self.backend.value)
            out = self.model(**inputs)
            r = self.proc.post_process_instance_segmentation(
                out, threshold=self.threshold, mask_threshold=0.5,
                target_sizes=inputs.get("original_sizes").tolist(),
            )[0]
            for box, score, m in zip(r["boxes"].cpu().tolist(), r["scores"].cpu().tolist(), r["masks"].cpu()):
                arr = (m.numpy() > 0).astype(np.uint8) * 255
                dets.append(Detection(label=c, score=score, box=tuple(box), mask=Image.fromarray(arr)))
        return dets
