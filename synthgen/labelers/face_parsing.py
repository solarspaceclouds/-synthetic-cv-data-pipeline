"""Face parsing (SegFormer-B5 fine-tuned on CelebAMask-HQ, 19 classes) via transformers.

Produces the part masks for face jobs and re-parses generated faces for label recovery.
Licence: the model card states none; base model NVIDIA SegFormer and CelebAMask-HQ are both
non-commercial research only. Works best on aligned, roughly frontal faces filling the frame.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

from synthgen.adapters.base import LabelerAdapter
from synthgen.types import Backend, Capabilities, Conditioning, Detection, Domain

_REPO = "jonathandinu/face-parsing"

# config-level part names -> CelebAMask-HQ labels
PARTS = {
    "eyes": ["l_eye", "r_eye"], "left_eye": ["l_eye"], "right_eye": ["r_eye"],
    "brows": ["l_brow", "r_brow"], "nose": ["nose"], "mouth": ["mouth", "u_lip", "l_lip"],
    "lips": ["u_lip", "l_lip"], "hair": ["hair"], "ears": ["l_ear", "r_ear"], "skin": ["skin"],
    "glasses": ["eye_g"], "hat": ["hat"], "neck": ["neck"], "earrings": ["ear_r"], "necklace": ["neck_l"],
    "cloth": ["cloth"],
}


class FaceParsingLabeler(LabelerAdapter):
    name = "face_parsing"
    caps = Capabilities(
        domains=frozenset({Domain.FACE}),
        operations=frozenset(),
        conditioning=frozenset({Conditioning.NONE}),
        backends=(Backend.MPS, Backend.CUDA, Backend.CPU),
        native_res=512,
        licence="unspecified; base SegFormer (NVIDIA) and CelebAMask-HQ are non-commercial",
        weights=_REPO,
        approx_peak_gb=1.5,
    )

    def load(self):
        self.proc = SegformerImageProcessor.from_pretrained(_REPO)
        self.model = SegformerForSemanticSegmentation.from_pretrained(_REPO).to(self.backend.value).eval()
        self.label_ids = {v: int(k) for k, v in self.model.config.id2label.items()}

    @torch.inference_mode()
    def parse(self, image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
        """Per-pixel label map and its softmax probability, at the input image size."""
        image = image.convert("RGB")
        inputs = self.proc(images=image, return_tensors="pt").to(self.backend.value)
        logits = self.model(**inputs).logits
        logits = F.interpolate(logits, size=image.size[::-1], mode="bilinear", align_corners=False)
        prob = logits.softmax(1)[0]
        conf, labels = prob.max(0)
        return labels.cpu().numpy(), conf.float().cpu().numpy()

    def detect(self, image: Image.Image, classes: list[str]) -> list[Detection]:
        labels, conf = self.parse(image)
        dets = []
        for part in classes:
            ids = [self.label_ids[n] for n in PARTS.get(part, [part]) if n in self.label_ids]
            m = np.isin(labels, ids)
            if not m.any():
                continue
            ys, xs = np.where(m)
            dets.append(Detection(label=part, score=float(conf[m].mean()),
                                  box=(float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)),
                                  mask=Image.fromarray(m.astype(np.uint8) * 255)))
        return dets
