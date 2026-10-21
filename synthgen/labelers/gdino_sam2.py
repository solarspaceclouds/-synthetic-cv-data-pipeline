"""Grounding DINO (open-vocabulary boxes) + SAM 2.1 (masks from boxes), both via transformers.

Default label checker: Apache-2.0, ungated, runs on MPS. Used to re-detect generated objects
instead of trusting the inpainting mask.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, Sam2Model, Sam2Processor

from synthgen.adapters.base import LabelerAdapter
from synthgen.types import Backend, Capabilities, Conditioning, Detection, Domain, Operation

_DINO = "IDEA-Research/grounding-dino-tiny"
_SAM = "facebook/sam2.1-hiera-base-plus"


class GDinoSam2Labeler(LabelerAdapter):
    name = "gdino_sam2"
    caps = Capabilities(
        domains=frozenset({Domain.SCENE}),
        operations=frozenset(),
        conditioning=frozenset({Conditioning.TEXT}),
        backends=(Backend.MPS, Backend.CUDA, Backend.CPU),
        native_res=800,
        licence="Apache-2.0",
        weights=f"{_DINO} + {_SAM}",
        approx_peak_gb=2.0,
    )

    def __init__(self, backend=None, native_res=None, box_threshold: float = 0.35, text_threshold: float = 0.25):
        super().__init__(backend, native_res)
        self.box_threshold, self.text_threshold = box_threshold, text_threshold

    _keep = ("backend", "res", "box_threshold", "text_threshold")

    def load(self):
        d = self.backend.value
        self.dino_proc = AutoProcessor.from_pretrained(_DINO)
        self.dino = AutoModelForZeroShotObjectDetection.from_pretrained(_DINO).to(d).eval()
        self.sam_proc = Sam2Processor.from_pretrained(_SAM)
        # the sam2.1 checkpoint is published as "sam2_video"; loading the image model from it is intended,
        # so silence transformers' type-mismatch warning
        from transformers.utils import logging as hf_logging
        level = hf_logging.get_verbosity()
        hf_logging.set_verbosity_error()
        try:
            self.sam = Sam2Model.from_pretrained(_SAM).to(d).eval()
        finally:
            hf_logging.set_verbosity(level)

    @torch.inference_mode()
    def detect(self, image: Image.Image, classes: list[str]) -> list[Detection]:
        d = self.backend.value
        image = image.convert("RGB")
        text = [[c for c in classes]]
        inputs = self.dino_proc(images=image, text=text, return_tensors="pt").to(d)
        out = self.dino(**inputs)
        res = self.dino_proc.post_process_grounded_object_detection(
            out, inputs.input_ids, threshold=self.box_threshold, text_threshold=self.text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]
        if len(res["boxes"]) == 0:
            return []
        boxes = res["boxes"].cpu().tolist()
        sin = self.sam_proc(images=image, input_boxes=[boxes], return_tensors="pt").to(d)
        sout = self.sam(**sin, multimask_output=False)
        masks = self.sam_proc.post_process_masks(sout.pred_masks.cpu(), sin["original_sizes"])[0]
        dets = []
        for box, score, label, m in zip(boxes, res["scores"].cpu().tolist(), res["text_labels"], masks):
            arr = (m[0].numpy() > 0).astype(np.uint8) * 255
            dets.append(Detection(label=label, score=score, box=tuple(box), mask=Image.fromarray(arr)))
        return dets
