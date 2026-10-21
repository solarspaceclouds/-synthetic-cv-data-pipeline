"""Occlusion-aware editing: whatever was in front of the edited object stays in front.

A car behind a pole, replaced or regenerated, must come out behind the pole; a person removed from
behind a fence must leave the fence. The pipeline finds the occluders inside the edit region at
plan time, keeps them out of the generator's mask (so the model paints around them), and pastes the
original pixels back afterwards. The manipulation mask excludes them.

Evidence, combined (depth from Depth Anything V2 Small, Apache-2.0; relative disparity, larger =
closer; `ref` = median disparity of the edited object, or of the ground under an inserted box):

  1. annotated   other annotated objects' visible pixels inside the region, with exact edges from the
                 dataset's masks. In front of the object (median disparity > ref * (1 + gap_tol)) they
                 are occluders; behind it they are protected, so the edit can't repaint them and their
                 labels stay valid (a new object then can't grow over them, which is the trade-off).
  2. outline gap pixels inside the object's convex hull but outside its mask that are closer than it:
                 the pole splitting a car's outline. Grown along connected closer pixels in the region,
                 so the pole above and below the car is kept too.
  3. depth       components of clearly closer pixels (disparity > ref * (1 + tol)) touching the
                 region border: occluders crossing the object.

The generator paints the whole region (a hole in its mask made models paint clutter into it), then the
original goes back on top: the occluders, and for an occluded regenerate/replace with composite:
silhouette, everything in the region outside the object's original outline (see run._keep_original).
The object's own pixels are never occluders (no fragments of the old object come back). Objects
seen *through* a mesh (chain-link fence, railing) can't be separated by depth: the planner skips them
with `SeeThroughScorer` (SigLIP 2 zero-shot on the target crop, Apache-2.0). On the 133 untruncated
COCO val2017 cars of 4000 px or more, threshold 0.8 caught 9 of 11 see-through cases with no false
positives (it missed a car seen through a wet window and one behind a tennis-court windscreen); the
"wet window" phrase added later catches the first. The same check screens references.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from PIL import Image, ImageFilter
from scipy import ndimage
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from synthgen.device import empty_cache, torch_device

_REPO = "depth-anything/Depth-Anything-V2-Small-hf"


class DepthEstimator:
    def __init__(self, device: str | None = None):
        self.device = device = device or torch_device()
        self.proc = AutoImageProcessor.from_pretrained(_REPO)
        self.model = AutoModelForDepthEstimation.from_pretrained(_REPO).to(device).eval()

    @torch.inference_mode()
    def disparity(self, image: Image.Image) -> np.ndarray:
        x = self.proc(images=image.convert("RGB"), return_tensors="pt").to(self.device)
        d = self.model(**x).predicted_depth[None]
        d = torch.nn.functional.interpolate(d, size=image.size[::-1], mode="bilinear", align_corners=False)
        return d[0, 0].float().cpu().numpy()


class SeeThroughScorer:
    """Probability that an object is seen through a fence, mesh or railing, from its crop."""

    _REPO = "google/siglip2-base-patch16-224"
    _NEG = ["a photo of a {cls}", "a photo of a {cls} on a street", "a photo of a {cls} behind a person",
            "a photo of a {cls} behind a pole"]

    def __init__(self, phrases: list[str], device: str | None = None, negatives: list[str] | None = None):
        from transformers import AutoModel, AutoProcessor
        self.device, self.phrases = device or torch_device(), phrases
        if negatives is not None:
            self._NEG = negatives
        self.proc = AutoProcessor.from_pretrained(self._REPO)
        self.model = AutoModel.from_pretrained(self._REPO).to(self.device).eval()
        self._text = {}

    @staticmethod
    def _feat(out) -> torch.Tensor:
        f = getattr(out, "pooler_output", out)
        return f / f.norm(dim=-1, keepdim=True)

    @torch.inference_mode()
    def score(self, image: Image.Image, box: list[float], cls: str, margin: float = 0.15) -> float:
        if cls not in self._text:
            texts = [p.format(cls=cls) for p in self.phrases + self._NEG]
            t = self.proc(text=texts, padding="max_length", max_length=64, return_tensors="pt").to(self.device)
            self._text[cls] = self._feat(self.model.get_text_features(**t))
        x, y, w, h = box
        crop = image.convert("RGB").crop((max(0, x - margin * w), max(0, y - margin * h),
                                          min(image.width, x + (1 + margin) * w), min(image.height, y + (1 + margin) * h)))
        f = self._feat(self.model.get_image_features(**self.proc(images=crop, return_tensors="pt").to(self.device)))
        logits = (f @ self._text[cls].T)[0] * self.model.logit_scale.exp() + self.model.logit_bias
        return float(torch.softmax(logits, 0)[:len(self.phrases)].sum())

    def unload(self):
        import gc
        del self.model
        gc.collect()
        empty_cache(self.device)


@dataclass
class Occluders:
    mask: np.ndarray                                   # bool, inside the region
    annotated_front: list[int] = field(default_factory=list)   # in front of the object: restored on top
    annotated_behind: list[int] = field(default_factory=list)  # behind it: visible pixels protected
    px: dict = field(default_factory=dict)             # pixels per evidence source

    def info(self, region_px: int) -> dict:
        return {"frac": round(float(self.mask.sum() / max(region_px, 1)), 4), "annotated_front": self.annotated_front,
                "annotated_behind": self.annotated_behind, "px": self.px}


def in_front(disp: np.ndarray, pixels: np.ndarray, ref: float, gap_tol: float) -> bool:
    return bool(pixels.any()) and float(np.median(disp[pixels])) > ref * (1 + gap_tol)


def find_occluders(disp: np.ndarray, region: np.ndarray, ref: float, obj: np.ndarray | None,
                   others: list[tuple[int, np.ndarray]], tol: float = 0.3, gap_tol: float = 0.05,
                   min_frac: float = 0.005, ground_from_row: int | None = None) -> Occluders:
    """`obj`: the edited object's mask (None for insert). `others`: (ann_id, mask) of other annotated objects.
    `ground_from_row`: rows from here down are the ground the object stands on. Ground in front of the
    contact line is closer than the object but is not an occluder: restoring it would clip the tyres."""
    min_px = max(16, min_frac * region.sum())
    free = region if obj is None else region & ~obj          # never restore the object itself
    depth_free = free.copy()                                  # depth evidence ignores the ground band;
    if ground_from_row is not None:                           # annotated occluders keep their full mask
        depth_free[max(0, ground_from_row):] = False
    occ = np.zeros_like(region)
    res = Occluders(occ)

    for ann_id, m in others:                                  # 1. annotated objects
        ov = m & free
        if ov.sum() < min_px:
            continue
        occ |= ov
        (res.annotated_front if in_front(disp, ov, ref, gap_tol) else res.annotated_behind).append(ann_id)
    res.px["annotated"] = int(occ.sum())

    if obj is not None and obj.any():                         # 2. gaps in the object's outline
        from synthgen.masks import convex_hull_mask
        hull = np.asarray(convex_hull_mask(Image.fromarray(obj.astype(np.uint8) * 255))) > 127
        closer = depth_free & (disp > ref * (1 + gap_tol))
        seeds = closer & hull
        if seeds.sum() >= min_px:
            labels, n = ndimage.label(closer)
            ids = np.unique(labels[seeds])
            gap = np.isin(labels, ids[ids > 0])
            gap = ndimage.binary_opening(gap, iterations=1)    # drop single-pixel depth noise
            res.px["outline_gap"] = int((gap & ~occ).sum())
            occ |= gap

    near = depth_free & (disp > ref * (1 + tol))              # 3. strong occluders crossing the border
    if near.any():
        border = region & ~ndimage.binary_erosion(region, iterations=2)
        labels, n = ndimage.label(near)
        before = occ.sum()
        for i in range(1, n + 1):
            comp = labels == i
            if comp.sum() >= min_px and (comp & border).any():
                occ |= comp
        res.px["depth"] = int(occ.sum() - before)

    res.mask = ndimage.binary_closing(occ, iterations=1) & free
    return res


def ground_row(obj: np.ndarray | None, insert_box: list[float] | None) -> int | None:
    """First row of the ground band: the bottom 15% of the object (or inserted box) and below."""
    if insert_box is not None:
        _, y, _, h = insert_box
        return int(y + 0.85 * h)
    if obj is None or not obj.any():
        return None
    ys = np.where(obj.any(1))[0]
    return int(ys[-1] - 0.15 * (ys[-1] - ys[0]))


def reference_disparity(disp: np.ndarray, object_mask: np.ndarray | None, insert_box: list[float] | None) -> float:
    if insert_box is not None:
        x, y, w, h = insert_box
        strip = disp[int(y + 0.85 * h):int(y + h) + 1, int(x):int(x + w) + 1]
        return float(np.median(strip))
    return float(np.median(disp[object_mask]))


def restore(generated: Image.Image, original: Image.Image, occluder: Image.Image, feather_px: int = 1) -> Image.Image:
    """Paste the original occluder pixels back over the generated image."""
    m = occluder.filter(ImageFilter.GaussianBlur(feather_px)) if feather_px else occluder
    return Image.composite(original.convert("RGB"), generated.convert("RGB"), m)
