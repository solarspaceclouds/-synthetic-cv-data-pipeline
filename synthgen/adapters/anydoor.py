"""AnyDoor (SD-2.1 + ControlNet + DINOv2-g identity encoder, MIT): inserts or replaces an object
in the masked region from a reference image of that object (object-level identity transfer).
Runs in envs/anydoor via workers/anydoor_worker.py; needs the converted fp16 checkpoint."""

from __future__ import annotations

from pathlib import Path

from synthgen.adapters.worker import WorkerAdapter, WorkerError
from synthgen.types import Backend, Capabilities, Conditioning, Domain, Operation

WEIGHTS = Path.home() / ".cache" / "synthgen" / "anydoor_fp16.safetensors"


class AnyDoorAdapter(WorkerAdapter):
    name = "anydoor"
    env = "anydoor"
    script = "workers/anydoor_worker.py"
    caps = Capabilities(
        domains=frozenset({Domain.SCENE}),
        operations=frozenset({Operation.REPLACE, Operation.CHANGE_CLASS, Operation.INSERT}),
        conditioning=frozenset({Conditioning.REFERENCE_OBJECT}),
        backends=(Backend.MPS, Backend.CUDA, Backend.CPU),
        native_res=512,
        licence="MIT (AnyDoor code and weights); built on SD-2.1 (CreativeML OpenRAIL++-M) and DINOv2 (Apache-2.0)",
        weights="xichenhku/AnyDoor (space) epoch=1-step=8687.ckpt -> converted fp16 safetensors",
        approx_peak_gb=12.0,
        notes="Reference required (object crop + mask). Ignores text prompts. 50 DDIM steps by default.",
    )

    def worker_config(self) -> dict:
        if not WEIGHTS.exists():
            raise WorkerError(f"{WEIGHTS} missing; run tools/convert_anydoor_ckpt.py (see README)")
        return {"weights": str(WEIGHTS)}
