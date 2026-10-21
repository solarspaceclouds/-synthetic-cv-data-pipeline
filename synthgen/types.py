"""Core vocabulary shared by configs, adapters and the planner."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from PIL import Image


class Domain(str, Enum):
    FACE = "face"
    SCENE = "scene"


class Operation(str, Enum):
    REMOVE = "remove"              # fill with background
    REGENERATE = "regenerate"      # same part/class, new appearance
    REPLACE = "replace"            # another instance of the same class, usually from a reference
    CHANGE_CLASS = "change_class"
    INSERT = "insert"              # new object in empty space


class Conditioning(str, Enum):
    NONE = "none"                  # context-only fill
    TEXT = "text"
    REFERENCE_OBJECT = "reference_object"
    REFERENCE_IDENTITY = "reference_identity"
    STRUCTURE = "structure"


class Backend(str, Enum):
    MPS = "mps"
    MLX = "mlx"
    CPU = "cpu"
    CUDA = "cuda"


@dataclass(frozen=True)
class Capabilities:
    """What an adapter can do and what it needs. The planner matches job configs against this."""

    domains: frozenset[Domain]
    operations: frozenset[Operation]
    conditioning: frozenset[Conditioning]
    backends: tuple[Backend, ...]          # in order of preference
    native_res: int                        # working size of the crop sent to the model
    licence: str
    weights: str                           # HF repo id or URL
    approx_peak_gb: float                  # rough planning figure; `bench` replaces it with a measurement
    gated: bool = False
    notes: str = ""


@dataclass
class InpaintRequest:
    image: Image.Image                     # full RGB image
    mask: Image.Image                      # "L" mode, 255 = region to generate
    prompt: str = ""
    reference: Image.Image | None = None
    reference_mask: Image.Image | None = None   # object mask within `reference` (AnyDoor-style models)
    seed: int = 0
    steps: int | None = None               # None = adapter default
    strength: float | None = None
    guidance: float | None = None
    context_scale: float = 2.0             # crop side = mask bbox side * context_scale
    feather_px: int = 8


@dataclass
class InpaintResult:
    image: Image.Image                     # full image with the crop pasted back
    crop_box: tuple[int, int, int, int]
    meta: dict = field(default_factory=dict)


@dataclass
class Detection:
    label: str
    score: float
    box: tuple[float, float, float, float]  # x0, y0, x1, y1 in pixels
    mask: Image.Image | None = None
