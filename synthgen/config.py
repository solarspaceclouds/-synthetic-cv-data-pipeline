"""Job config (YAML). Unknown keys are errors, so a typo can't silently fall back to a default."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, get_type_hints

import yaml


@dataclass
class SourceCfg:
    images: str
    annotations: str = ""               # COCO json; not used by format: images
    format: str = "coco"                # coco | images
    split: str = "unspecified"          # recorded in outputs; references are drawn only from this source
    download_missing: bool = False


@dataclass
class TargetsCfg:
    classes: list[str] = field(default_factory=list)   # empty = all categories (random / rare_weighted)
    policy: str = "specified"                           # specified | random | rare_weighted
    min_area: float = 4000
    max_overlap_other: float = 0.05     # skip if the edit covers more than this share of another object
                                        # (only when occlusion is off; with it on, other objects are protected)
    max_per_image: int = 1
    skip_truncated: bool = True         # skip instances cut off by the image border
    exclude_near: list[str] = field(default_factory=list)   # skip targets touching these classes, e.g. what a person
                                        # holds or rides: removing the person would leave it floating
    exclude_near_margin: float = 0.1    # grow the target box by this fraction before the touch test
    image_ids: list[int] = field(default_factory=list)   # only these source images (empty = all)
    crowd: str = "exclude"              # exclude | include | only: iscrowd regions as targets (remove only)


@dataclass
class ReferenceCfg:
    match: str = "aspect"               # aspect | aspect_light (+ local brightness) | visual (+ DINOv2 viewpoint) | random
    white_bg: bool = True
    pad: float = 0.1
    min_area: float = 16000             # references should carry at least as much detail as the target
    max_uses: int = 2                   # cap reuse of one reference instance (avoids near-duplicate samples)


@dataclass
class ConditioningCfg:
    type: str = "text"                  # none | text | reference_object
    prompt: str = "a photo of a {attr} {cls}"
    attributes: list[str] = field(default_factory=lambda: [""])
    reference: ReferenceCfg = field(default_factory=ReferenceCfg)


@dataclass
class MaskCfg:
    shape: str = "exact"                # exact | bbox | convex_hull
    dilate_px: int = 12


@dataclass
class GeneratorCfg:
    adapter: str
    backend: str | None = None
    res: int | None = None
    steps: int | None = None
    strength: float | None = None
    guidance: float | None = None
    context_scale: float = 2.0
    feather_px: int = 8


@dataclass
class LabelerCfg:
    adapter: str = "gdino_sam2"
    backend: str | None = None


@dataclass
class VerifyCfg:
    min_score: float = 0.35
    min_inside: float = 0.7
    min_fill: float = 0.15
    margin: float = 0.1
    contrast: list[str] | str = "auto"  # "auto" = the original class for change_class, none otherwise
    removal_max_inside: float = 0.5     # remove: reject if a detection of the class lies this much inside the region
    phrases: dict[str, list[str]] = field(default_factory=dict)  # detector phrases per class, e.g. truck: [pickup truck, truck]
    face_area_ratio: list[float] = field(default_factory=lambda: [0.4, 2.5])  # face: new/original part area bounds
    unlabelled: str = "flag"            # off | flag | reject: instances of the class the source dataset didn't label
    unlabelled_min_score: float = 0.5
    min_color_change: float = 0.0       # reject if the edit's colour change (p75 CIELAB delta E) is below; 0 = off
    # insert: keep only the new object (its detected mask) and its shadow from the generated region; the rest
    # of the region comes back from the original (models fill unused box space with smears and blur)
    object_only: bool = True


@dataclass
class PlacementCfg:                     # insert only; see synthgen/placement.py
    support_classes: list[str] = field(default_factory=list)   # empty = the inserted classes
    y_jitter: float = 0.08              # bottom edge within +-8% of image height of a support's bottom
    size_jitter: float = 0.15
    gap: float = 0.05                   # keep this fraction of each object's size free around it
    min_height_px: float = 24
    ground_phrases: list[str] = field(default_factory=list)  # e.g. [road, street, parking lot]; empty = no ground check
    ground_min_frac: float = 0.6        # share of the box's bottom strip that must be on the ground mask
    # generation region = the planned box widened by slack * w on each side and slack * h on top (0.1 * h
    # below, for the shadow). A region exactly the box makes the model stretch the object to fill it or pad
    # the gap with motion blur; with verify.object_only only the detected object is kept from the region.
    slack: float = 0.25
    # Skip scenes an inserted object can't match (SigLIP 2 on the area around the box). On 20 COCO
    # inserts only a long-exposure light-trail street scored above 0.7; ordinary night streets up to 0.57.
    scene_skip_threshold: float = 0.7   # 0 = off
    scene_skip_phrases: list[str] = field(default_factory=lambda: [
        "a blurry out of focus photo", "a long exposure photo with light trails", "a photo with strong motion blur",
        "a bokeh background"])
    min_sharpness: float = 10.0         # mean |Laplacian| around the box; below = out of focus (bokeh), skip; 0 = off


@dataclass
class OcclusionCfg:                     # scene jobs; ignored for faces. See synthgen/occlusion.py
    enabled: bool = True                # keep whatever is in front of the edited object in front
    tol: float = 0.3                    # depth evidence: occluder if disparity > object's * (1 + tol)
    gap_tol: float = 0.05               # annotated objects / outline gaps: "in front" if > object's * (1 + gap_tol)
    min_frac: float = 0.005             # ignore occluder pieces smaller than this share of the region
    # How the original scene goes back over a regenerated / replaced object that has occluders. The generator
    # always paints the whole region (a hole where an occluder was makes models paint clutter into it); then:
    #   silhouette  new pixels only inside the object's original outline (+ silhouette_dilate_px); everything
    #               else in the region, poles and people in front included, comes back from the original
    #   detected    only the detected occluders come back (the new object may extend past the old outline)
    composite: str = "silhouette"
    silhouette_dilate_px: int = 4
    # Skip targets seen *through* a chain-link fence, mesh or railing: depth can't separate the mesh from
    # the object, so the edit would paint over it. Scored at plan time with SigLIP 2 (Apache-2.0).
    see_through_skip: bool = True       # also keeps such instances out of the reference bank
    see_through_threshold: float = 0.8  # share of SigLIP probability on the "seen through" phrases
    see_through_phrases: list[str] = field(default_factory=lambda: [
        "a photo of a {cls} seen through a chain-link fence", "a photo of a {cls} behind a wire mesh fence",
        "a photo of a {cls} behind a metal railing", "a photo of a {cls} seen through a net",
        "a photo of a {cls} seen through a wet window"])


@dataclass
class PostprocessCfg:
    ops: list[dict] = field(default_factory=list)   # see synthgen/degrade.py; empty = none
    repeat: int = 1


@dataclass
class OutputCfg:
    dir: str
    jpeg_quality: int = 95


@dataclass
class JobConfig:
    name: str
    domain: str
    source: SourceCfg
    operation: str
    generator: GeneratorCfg
    output: OutputCfg
    targets: TargetsCfg = field(default_factory=TargetsCfg)
    change_to: list[str] = field(default_factory=list)
    conditioning: ConditioningCfg = field(default_factory=ConditioningCfg)
    mask: MaskCfg = field(default_factory=MaskCfg)
    labeler: LabelerCfg = field(default_factory=LabelerCfg)
    verify: VerifyCfg = field(default_factory=VerifyCfg)
    postprocess: PostprocessCfg = field(default_factory=PostprocessCfg)
    placement: PlacementCfg = field(default_factory=PlacementCfg)
    occlusion: OcclusionCfg = field(default_factory=OcclusionCfg)
    n_samples: int = 10
    seed: int = 0


def _build(cls, data: Any, path: str):
    if not dataclasses.is_dataclass(cls):
        return data
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping")
    hints = get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ValueError(f"{path}: unknown key(s) {sorted(unknown)}; allowed: {sorted(names)}")
    kwargs = {}
    for k, v in data.items():
        t = hints[k]
        kwargs[k] = _build(t, v, f"{path}.{k}") if dataclasses.is_dataclass(t) else v
    try:
        return cls(**kwargs)
    except TypeError as e:
        raise ValueError(f"{path}: {e}") from None


def load_job(path: str | Path) -> JobConfig:
    return _build(JobConfig, yaml.safe_load(Path(path).read_text()), Path(path).name)


def to_dict(cfg: JobConfig) -> dict:
    return dataclasses.asdict(cfg)
