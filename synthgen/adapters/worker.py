"""Adapters whose model runs in its own Python environment, as a persistent worker process.

Research repos pin old stacks (e.g. diffusers 0.23, transformers 4.25) that can't share the main
environment. The worker loads the model once, then serves one JSON request per line on stdin and
answers one JSON line on stdout. Images go through a temp directory as PNG files.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import ClassVar

from PIL import Image

from synthgen.adapters.base import GeneratorAdapter
from synthgen.types import InpaintRequest

ROOT = Path(__file__).resolve().parents[2]


class WorkerError(RuntimeError):
    pass


class WorkerAdapter(GeneratorAdapter):
    env: ClassVar[str]          # envs/<env>/bin/python
    script: ClassVar[str]       # path relative to the repo root

    def worker_config(self) -> dict:
        """Startup config sent to the worker (weight paths etc.), resolved in the main environment."""
        return {}

    def load(self):
        python = ROOT / "envs" / self.env / "bin" / "python"
        if not python.exists():
            raise WorkerError(f"{python} missing; see README for creating the {self.env} environment")
        self.tmp = tempfile.TemporaryDirectory()
        env = {**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "1", "PYTHONUNBUFFERED": "1"}
        self.proc = subprocess.Popen(
            [str(python), str(ROOT / self.script), json.dumps({"device": self.backend.value, **self.worker_config()})],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None, text=True, env=env, cwd=ROOT,
        )
        msg = self._read()
        if not msg.get("ready"):
            raise WorkerError(f"{self.name} worker failed to start: {msg}")

    def child_pids(self) -> list[int]:
        p = getattr(self, "proc", None)
        return [p.pid] if p and p.poll() is None else []

    def _read(self) -> dict:
        line = self.proc.stdout.readline()
        if not line:
            raise WorkerError(f"{self.name} worker exited (code {self.proc.poll()}); see stderr above")
        return json.loads(line)

    def call(self, **payload) -> dict:
        self.proc.stdin.write(json.dumps(payload) + "\n")
        self.proc.stdin.flush()
        msg = self._read()
        if not msg.get("ok"):
            raise WorkerError(f"{self.name} worker error: {msg.get('error')}")
        return msg

    def _paths(self, image: Image.Image, mask: Image.Image, reference: Image.Image | None,
               reference_mask: Image.Image | None = None):
        d = Path(self.tmp.name)
        image.save(d / "image.png")
        mask.save(d / "mask.png")
        ref = ref_mask = None
        if reference is not None:
            reference.convert("RGB").save(d / "reference.png")
            ref = str(d / "reference.png")
        if reference_mask is not None:
            reference_mask.convert("L").save(d / "reference_mask.png")
            ref_mask = str(d / "reference_mask.png")
        return str(d / "image.png"), str(d / "mask.png"), ref, ref_mask, str(d / "out.png")

    def _inpaint_crop(self, image: Image.Image, mask: Image.Image, req: InpaintRequest) -> Image.Image:
        img, m, ref, ref_mask, out = self._paths(image, mask, req.reference, req.reference_mask)
        self.call(image=img, mask=m, reference=ref, reference_mask=ref_mask, out=out, seed=req.seed,
                  steps=req.steps, guidance=req.guidance, prompt=req.prompt)
        return Image.open(out).convert("RGB").copy()

    def unload(self):
        p = getattr(self, "proc", None)
        if p and p.poll() is None:
            p.stdin.close()
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill()
        super().unload()
