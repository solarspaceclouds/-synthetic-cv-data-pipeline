# Adding inpainting / generation models

A model is added by writing one **adapter**: a small class that says what the model can do and
how to run it on one image crop. Cropping, pasting back, planning, verification, export and
benchmarking are shared, so an adapter is usually 40-80 lines.

- [1. Before you start: is the model usable here?](#1-before-you-start-is-the-model-usable-here)
- [2. Pick a pattern](#2-pick-a-pattern)
- [3. Pattern A: in-process model (diffusers / transformers / plain PyTorch)](#3-pattern-a-in-process-model)
- [4. Pattern B: research repo in its own environment (worker)](#4-pattern-b-research-repo-in-its-own-environment-worker)
- [5. Pattern C: composing adapters, or no model at all](#5-pattern-c-composing-adapters-or-no-model-at-all)
- [6. Pattern D: MLX models](#6-pattern-d-mlx-models)
- [7. Register, benchmark, test, compare](#7-register-benchmark-test-compare)
- [8. Adding a labeller](#8-adding-a-labeller)
- [9. Mac porting problems and fixes](#9-mac-porting-problems-and-fixes)
- [Reference: Capabilities fields](#reference-capabilities-fields)

---

## 1. Before you start: is the model usable here?

Check these first. Each has stopped a model in this project.

| Check | How | Why |
|---|---|---|
| **Licence** of code *and* weights | Repo LICENSE, Hugging Face model card (`curl -s https://huggingface.co/api/models/<org>/<name> \| python3 -m json.tool \| grep -i licen`) | FLUX.1-Fill-dev, Depth Anything V2 Base/Large, insightface and the CelebAMask-HQ face parser are non-commercial |
| **Gated** weights | Same API call: `"gated"` | SAM 3 needs manual approval on Hugging Face |
| **Weight format** | `.safetensors` is safe; `.bin` / `.ckpt` / `.pth` are pickles | Pickles can run code when loaded; load them only with `torch.load(..., weights_only=True)` (section 9) |
| **Memory** | Parameters x 2 bytes (fp16/bf16) + 30-50% for activations | ~17.8 GB is usable by the GPU on a 24 GB Mac; plan for <= 12 GB per model |
| **CUDA-only code** | `grep -rnE "cuda\|triton\|xformers\|flash_attn\|bitsandbytes" <repo>` | Triton, xformers, flash-attention and bitsandbytes don't run on Apple Silicon |
| **Dependency pins** | `requirements.txt` | Old pins (diffusers 0.2x, transformers 4.2x, pytorch_lightning 1.x) won't share the main environment: use pattern B |

---

## 2. Pick a pattern

| Situation | Pattern | Examples here |
|---|---|---|
| Model loads with the current `diffusers` / `transformers` / PyTorch | **A: in-process** | `sd15_inpaint.py`, `flux2_klein.py`, `lama.py` |
| Research repo with its own code and old pinned dependencies | **B: worker** (own env, subprocess) | `mimicbrush.py` + `workers/mimicbrush_worker.py`, `anydoor.py` + `workers/anydoor_worker.py` |
| Combination of existing adapters, or a non-learned method | **C: compose** | `copy_paste.py` (LaMa clears, then paste) |
| Apple-native MLX port exists (e.g. in mflux) | **D: MLX** | `flux2_klein_mlx.py` |

---

## 3. Pattern A: in-process model

### The contract

```python
class MyAdapter(GeneratorAdapter):
    name = "my_model"          # used in job YAML: generator: {adapter: my_model}
    caps = Capabilities(...)   # what it can do (planner checks jobs against this)

    def load(self): ...        # load weights onto self.backend.value ("mps" / "cpu")

    def _inpaint_crop(self, image, mask, req) -> Image.Image:
        # image: RGB crop, self.res x self.res, centred on the edit region with context around it
        # mask:  "L" crop, same size, 255 = generate here
        # req:   InpaintRequest: prompt, reference, reference_mask, seed, steps, strength, guidance
        # return an RGB image of the same size; only the masked part is used (with soft edges)
```

Optional:
- `prepare(self, prompts)`: runs **before** `load()` with every prompt of the job. Use it to
  encode all prompts with a large text encoder, then free the encoder (see `flux2_klein.py`; this
  is what lets FLUX.2 fit in 24 GB).
- `__init__` extra arguments: add their names to `_keep` so `unload()` doesn't delete them.

### Worked template: a diffusers inpainting pipeline

Most Hugging Face inpainting models look like this. It mirrors `sd15_inpaint.py`, which is
tested. Swap the pipeline class, repo and defaults.

```python
# synthgen/adapters/sdxl_inpaint.py
"""SDXL inpainting (diffusers). Text-guided regeneration/insertion at 1024."""

from __future__ import annotations

import torch
from diffusers import StableDiffusionXLInpaintPipeline
from PIL import Image

from synthgen.adapters.base import GeneratorAdapter
from synthgen.types import Backend, Capabilities, Conditioning, Domain, InpaintRequest, Operation

_REPO = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"


class SDXLInpaintAdapter(GeneratorAdapter):
    name = "sdxl_inpaint"
    caps = Capabilities(
        domains=frozenset({Domain.SCENE, Domain.FACE}),
        operations=frozenset({Operation.REGENERATE, Operation.REPLACE, Operation.CHANGE_CLASS, Operation.INSERT}),
        conditioning=frozenset({Conditioning.TEXT}),
        backends=(Backend.MPS, Backend.CUDA, Backend.CPU),   # default: the first this machine has
        native_res=1024,
        licence="CreativeML OpenRAIL++-M",
        weights=_REPO,
        approx_peak_gb=10.0,          # rough; `bench` replaces it with a measurement
    )

    def load(self):
        dtype = torch.float16 if self.backend in (Backend.MPS, Backend.CUDA) else torch.float32
        self.pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            _REPO, torch_dtype=dtype, variant="fp16", use_safetensors=True,
        ).to(self.backend.value)
        self.pipe.set_progress_bar_config(disable=True)

    def _inpaint_crop(self, image: Image.Image, mask: Image.Image, req: InpaintRequest) -> Image.Image:
        g = torch.Generator("cpu").manual_seed(req.seed)   # CPU generator: reproducible across devices
        return self.pipe(
            prompt=req.prompt, image=image, mask_image=mask, height=image.height, width=image.width,
            num_inference_steps=req.steps or 30, guidance_scale=req.guidance or 7.0,
            strength=req.strength or 0.99, generator=g,
        ).images[0]
```

Notes:
- Always use `use_safetensors=True` where the repo has safetensors.
- Use `torch.Generator("cpu")` for seeds, so outputs are reproducible.
- If the model takes a reference image, add `Conditioning.REFERENCE_OBJECT` to `conditioning`
  and read `req.reference` (an RGB crop of the reference object) and `req.reference_mask`.
- Don't resize or paste back yourself; the base class does it.

---

## 4. Pattern B: research repo in its own environment (worker)

For repos that pin old libraries or ship their own model code. The model runs in a separate Python
environment as a long-lived subprocess. The main process sends one JSON request per image over
stdin and reads one JSON reply on stdout.

### Step 1: clone and review

```bash
git clone --depth 1 https://github.com/<org>/<repo> third_party/<Repo>
cd third_party/<Repo> && git log -1 --format='%H %cd' && cat LICENSE | head -3
grep -rnE "cuda|torch\.load|hub\.load|urlopen|requests\.|subprocess|os\.system|eval\(|exec\(" --include='*.py' . | grep -v train
```

Read the inference path (usually `inference.py`, `run_*.py`, a gradio demo) and note:
- where weights load (`torch.load` without `weights_only`?)
- hard-coded `"cuda"` / `.cuda()`
- anything downloaded at import or construction time (e.g. `pretrained=True` backbones)
- code that runs at import time (demo scripts often load the model when imported: don't import those; copy the pre/post-processing functions you need into the worker instead)

Keep `third_party/` unmodified and apply fixes inside the worker (monkey-patching), so updating the
repo stays a `git pull`.

### Step 2: environment

```bash
export PATH=~/Library/Python/3.9/bin:$PATH
uv venv envs/<name> --python 3.10
VIRTUAL_ENV=envs/<name> uv pip install torch torchvision <the repo's pins, minus CUDA-only packages>
envs/<name>/bin/python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
```

Use a current `torch` (MPS support and safe `torch.load` defaults) with the repo's other pins.
Drop `xformers`, `triton`, `bitsandbytes` and `flash-attn`. Common missing pieces: `setuptools<70`
(old libraries import `pkg_resources`), `numpy<2`, and `huggingface_hub` versions old enough
for `cached_download`.

### Step 3: the worker script

```python
# workers/<name>_worker.py   (runs inside envs/<name>)
import json, os, sys, time, traceback

REPO = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party", "<Repo>")
_proto = sys.stdout          # protocol channel
sys.stdout = sys.stderr      # library prints must not corrupt the protocol


def reply(**msg):
    _proto.write(json.dumps(msg) + "\n")
    _proto.flush()


def main():
    cfg = json.loads(sys.argv[1])          # {"device": "mps", ...whatever worker_config() returned}
    sys.path.insert(0, REPO)
    os.chdir(REPO)
    import torch
    from PIL import Image

    _load = torch.load                     # tensors only: code inside pickles can't run
    torch.load = lambda f, *a, **kw: _load(f, *a, **{**kw, "weights_only": True})

    # ... apply Mac patches (section 9), build the model on cfg["device"], load cfg["weights"] ...

    reply(ready=True)
    for line in sys.stdin:                 # one request per line until stdin closes
        try:
            r = json.loads(line)           # image, mask, reference, reference_mask, out, seed, steps, guidance, prompt
            image = Image.open(r["image"]).convert("RGB")
            mask = Image.open(r["mask"]).convert("L")
            # ... run the model ...
            result.save(r["out"])          # same size as `image`
            reply(ok=True, out=r["out"])
        except Exception as e:
            traceback.print_exc()
            reply(ok=False, error=f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        traceback.print_exc()
        reply(ready=False, error=f"{type(e).__name__}: {e}")
```

### Step 4: the adapter (main environment)

```python
# synthgen/adapters/<name>.py
from huggingface_hub import snapshot_download
from synthgen.adapters.worker import WorkerAdapter
from synthgen.types import Backend, Capabilities, Conditioning, Domain, Operation


class MyRepoAdapter(WorkerAdapter):
    name = "my_repo_model"
    env = "<name>"                         # envs/<name>/bin/python
    script = "workers/<name>_worker.py"
    caps = Capabilities(...)

    def worker_config(self) -> dict:       # resolved in the main env, sent to the worker at start
        path = snapshot_download("<org>/<weights>", allow_patterns=["*.safetensors", "*.json"])
        return {"weights": path}
```

`WorkerAdapter` handles starting the process, temp files, the request/reply loop, shutdown, and
counting the worker's memory in benchmarks.

### Step 5: debug the worker on its own

```bash
.venv/bin/python - <<'EOF'
from synthgen.adapters.<name> import MyRepoAdapter
from synthgen.bench import bench_case
a = MyRepoAdapter(); a.load()                      # prints the worker's stderr: watch for errors here
print(a.inpaint(bench_case()).meta); a.unload()
EOF
```

---

## 5. Pattern C: composing adapters, or no model at all

An adapter can use other adapters. `copy_paste.py` clears the region with LaMa, then pastes the
reference object:

```python
def load(self):
    self.lama = LamaAdapter(self.backend)
    self.lama.load()

def _inpaint_crop(self, image, mask, req):
    base = self.lama._inpaint_crop(image, mask, req)   # erase what was there
    ...                                                # paste req.reference using req.reference_mask
```

Other compositions worth adding the same way:
- **paste then refine:** copy-paste, then a low-strength diffusion pass (`strength` 0.3-0.5) for
  lighting and edges
- **remove then insert:** LaMa clears, a text model generates a different object
- **generate then restore:** add a restoration model (e.g. SeedVR2) after generation

---

## 6. Pattern D: MLX models

If an MLX port exists (e.g. in `mflux`), call it from `load()` / `_inpaint_crop()` and declare
`backends=(Backend.MLX,)`. See `flux2_klein_mlx.py`. Two things learned:
- Cap MLX's buffer cache in `load()` (`mx.set_cache_limit(...)`), or it holds several GB of freed memory.
- Check the meaning of each argument against the source. mflux's `image_strength` is the
  opposite of diffusers' `strength`.

Benchmark against the PyTorch-MPS version before preferring MLX. For FLUX.2-klein, MLX was
slower at 512 and only ~11% faster at 1024.

---

## 7. Register, benchmark, test, compare

```bash
# 1. register: add one line to synthgen/registry.py
#    "my_model": "synthgen.adapters.my_model:MyAdapter",
.venv/bin/python -m synthgen.bench list                 # appears with its operations and licence

# 2. measure on this machine (writes profiles/<machine>.json; the planner uses it for estimates)
.venv/bin/python -m synthgen.bench run my_model --n 5
.venv/bin/python -m synthgen.bench run my_model --res 512 --duration 1200   # optional: 20-min sustained run

# 3. small real job: copy an example config, change generator.adapter and output.dir
sed 's/adapter: anydoor, steps: 30, guidance: 4.5/adapter: my_model/; s/coco_train_car_replace_anydoor/coco_train_car_replace_my_model/g' \
    configs/coco_train_car_replace_anydoor.yaml > configs/coco_train_car_replace_my_model.yaml
.venv/bin/python -m synthgen.run run configs/coco_train_car_replace_my_model.yaml
.venv/bin/python -m synthgen.review out/coco_train_car_replace_my_model --max 8
```

Also add it to `DEFAULT_SUITE` in `synthgen/bench.py` so `bench all` includes it, and add a line to
the model table in `docs/USAGE.md`.

**Compare on the same tasks.** Jobs with the same `seed`, source and targets plan the same
tasks and references, so a new model can be compared side by side with existing ones. Always
include the **copy-paste baseline** (`configs/coco_train_car_replace_copypaste.yaml`): if a
generative model doesn't beat copy-paste, it isn't worth its cost.

| Job (train2017, 8 cars, same tasks and references) | Accepted | Time per image |
|---|---|---|
| `copy_paste` | 8 / 8 | 0.19 s |
| `anydoor` | 8 / 8 | ~7 s |

Both pass verification: **verification checks that the label is right, not that the image is
good training data.** Use the review sheets for realism, and a downstream test (train a detector
on real + each synthetic set, evaluate on held-out *real* images) to choose between models.

---

## 8. Adding a labeller

Labellers re-detect objects after generation (`gdino_sam2.py`, `sam3.py`, `face_parsing.py`).

```python
class MyLabeler(LabelerAdapter):
    name = "my_labeler"
    caps = Capabilities(domains=frozenset({Domain.SCENE}), operations=frozenset(),
                        conditioning=frozenset({Conditioning.TEXT}), backends=(Backend.MPS, Backend.CUDA, Backend.CPU),
                        native_res=800, licence="...", weights="...", approx_peak_gb=2.0)

    def load(self): ...

    def detect(self, image, classes: list[str]) -> list[Detection]:
        # one Detection(label, score, box=(x0, y0, x1, y1), mask="L" image or None) per instance
        ...
```

Return masks where possible: verification compares the detected mask with the edit region.
Detect each requested class separately or keep per-class scores. Detectors given several class
names at once often assign one label per box and hide near-ties (see the car/truck finding in the README).

---

## 9. Mac porting problems and fixes

All of these came up while adding MimicBrush and AnyDoor.

| Problem | Symptom | Fix (in the worker or adapter) |
|---|---|---|
| Hard-coded CUDA | `AssertionError: Torch not compiled with CUDA` | Build the model yourself on `cfg["device"]`; patch defaults: `Cls.__init__.__defaults__ = ("cpu", ...)`; replace `register_buffer`-style helpers that force `cuda` |
| Downloads at construction | Long hang, network traffic | Patch the constructor (`functools.partial(fn, pretrained=False)`) when the checkpoint already contains those weights |
| Unsafe pickle loading | `torch.load(path)` in repo code | Wrap `torch.load` with `weights_only=True`; convert large checkpoints once to safetensors (`tools/convert_anydoor_ckpt.py`) |
| No xformers | Slow fallback attention (einsum) | Replace the attention `forward` with `torch.nn.functional.scaled_dot_product_attention` |
| Unsupported MPS op | `NotImplementedError ... MPS` | `PYTORCH_ENABLE_MPS_FALLBACK=1` (set by `WorkerAdapter`): that op runs on CPU |
| Huge RAM while loading | Swap, everything slow | `torch.load(..., mmap=True)`; cast to fp16 before moving to the GPU; `gc.collect()` after building |
| Long attention sequences | 10+ GB jumps during generation, swap | Chunk queries (see `chunked_sdpa` in `workers/mimicbrush_worker.py`) |
| fp32 where fp16 is expected | Very slow attention | Check dtypes; `torch.autocast("mps", dtype=torch.float16)` |
| Precision-sensitive buffers | Washed-out or noisy samples after fp16 conversion | Keep diffusion schedule buffers (`alphas_cumprod`, ...) in fp32 |
| `pkg_resources` / `cached_download` errors | ImportError in old libraries | `setuptools<70`; pin `huggingface_hub` to the repo's version inside the worker env |
| Library output on stdout | `JSONDecodeError` in the main process | `sys.stdout = sys.stderr` at the top of the worker, protocol on the saved stdout |

If a model is unexpectedly slow, check swap first: `sysctl vm.swapusage`. To see where the time
goes, time stages in the worker with `torch.mps.synchronize()` around them. Plain profilers
attribute GPU time to whichever line waits for the GPU.

---

## Reference: Capabilities fields

| Field | Meaning | Used by |
|---|---|---|
| `domains` | `Domain.SCENE`, `Domain.FACE` | planner validation |
| `operations` | `REMOVE`, `REGENERATE`, `REPLACE`, `CHANGE_CLASS`, `INSERT` | planner validation |
| `conditioning` | `NONE`, `TEXT`, `REFERENCE_OBJECT`, `REFERENCE_IDENTITY`, `STRUCTURE` | planner validation; `bench` passes a reference only to reference-only models |
| `backends` | Preferred first: `MPS`, `MLX`, `CPU`, `CUDA` | default device, `generator.backend` check |
| `native_res` | Crop size sent to `_inpaint_crop` (override per job with `generator.res`) | cropping |
| `licence`, `weights` | Recorded in every output's metadata and `bench list` | provenance |
| `approx_peak_gb` | Rough memory before a benchmark exists | documentation |
| `gated`, `notes` | Shown in `bench list` | documentation |
