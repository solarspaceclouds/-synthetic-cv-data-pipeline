"""Adapter registry. Adapters are imported lazily so one broken repo can't break the others."""

from __future__ import annotations

import importlib

from synthgen.adapters.base import Adapter

_ADAPTERS = {
    "lama": "synthgen.adapters.lama:LamaAdapter",
    "sd15_inpaint": "synthgen.adapters.sd15_inpaint:SD15InpaintAdapter",
    "flux2_klein_inpaint": "synthgen.adapters.flux2_klein:Flux2KleinInpaintAdapter",
    "flux2_klein_mlx_refine": "synthgen.adapters.flux2_klein_mlx:Flux2KleinMLXRefineAdapter",
    "qwen_image_edit_4bit": "synthgen.adapters.qwen_image:QwenImageEdit4bitAdapter",
    "qwen_image21_edit": "synthgen.adapters.qwen_image:QwenImage21EditAdapter",
    "copy_paste": "synthgen.adapters.copy_paste:CopyPasteAdapter",
    "mimicbrush": "synthgen.adapters.mimicbrush:MimicBrushAdapter",
    "anydoor": "synthgen.adapters.anydoor:AnyDoorAdapter",
    "face_parsing": "synthgen.labelers.face_parsing:FaceParsingLabeler",
    "gdino_sam2": "synthgen.labelers.gdino_sam2:GDinoSam2Labeler",
    "sam3": "synthgen.labelers.sam3:Sam3Labeler",
}


def names() -> list[str]:
    return list(_ADAPTERS)


def get(name: str) -> type[Adapter]:
    if name not in _ADAPTERS:
        raise ValueError(f"unknown adapter {name!r}; registered: {', '.join(_ADAPTERS)} "
                         "(add new ones to synthgen/registry.py)")
    module, cls = _ADAPTERS[name].split(":")
    return getattr(importlib.import_module(module), cls)
