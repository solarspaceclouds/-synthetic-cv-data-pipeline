---
name: synthgen-add-model
description: Integrate a new inpainting, image-editing, object-insertion or generation model (from a GitHub repo, a Hugging Face model id, a paper's code, a diffusers pipeline or an MLX port) into this repo's synthgen pipeline as an adapter - vet its licence, gating, weights and Mac compatibility, pick in-process vs separate-environment worker vs composition, write and register the adapter, benchmark it, and compare it against existing models on identical tasks. Use whenever the user wants to add, try, plug in, evaluate or swap in a model for synthetic data generation (e.g. "can we use BrushNet / PowerPaint / SDXL inpainting / Insert Anything / OmniPaint here", "add this repo as a generator", "try a new labeller"), even if they don't say "adapter".
---

# Adding a model to synthgen

A model becomes usable by writing one adapter class. The shared code does cropping, paste-back,
planning, verification, export and benchmarking. The full guide with templates is
`docs/ADDING_MODELS.md`. This skill is the working procedure and the judgement calls; open the
guide's sections for code templates (§3 in-process, §4 worker, §5 compose, §6 MLX, §8 labeller,
§9 Mac porting fixes).

## 1. Vet before writing code
```bash
.venv/bin/python .claude/skills/synthgen-add-model/scripts/vet_model.py --hf <org>/<model>   # weights
git clone --depth 1 <github-url> third_party/<Repo>                                          # if there's code
.venv/bin/python .claude/skills/synthgen-add-model/scripts/vet_model.py --repo third_party/<Repo>
```
It reports licence, gating, weight formats and size, dependency pins, and CUDA-only, unsafe-loading,
download and exec calls in the inference code. Stop and tell the user before investing time if:
- **the licence is non-commercial or missing** (e.g. FLUX.1-Fill-dev, Depth Anything V2 Base/Large,
  insightface, the CelebAMask-HQ face parser). It may still be fine for research; the user decides.
- **the weights are gated**: the user has to request access and run `.venv/bin/hf auth login`.
- **it can't fit**: fp16 weights plus 30-50% activations must stay under ~12 GB per model on a 24 GB
  Mac (~17.8 GB usable by the GPU, shared with the OS).
- **it depends on Triton, flash-attention or bitsandbytes with no fallback** (no macOS support).

When something blocks the model (licence, gating, size), look for an alternative of the same kind
that doesn't have the problem: another size, an earlier version, or a different repo. Run the
vetting script on it too. Depth Anything V2 Base/Large are non-commercial, but V1 Large is
Apache-2.0 and loads through the same class.

**Ask whether the new model fixes the actual problem.** Find out what prompted the request, then check
the README's findings sections for the documented failure. A stronger model of the same kind
doesn't help if the failure has another cause: a more accurate depth model still can't separate a
mesh fence from the car behind it. Say so; the cheaper fix may be a setting or a filter.

Also look for a ready-made port first: a `diffusers` pipeline (`python -c "import diffusers; print([n for n in dir(diffusers) if 'Inpaint' in n])"`),
a `transformers` class, or an MLX port in `mflux`. A maintained port beats patching research code.

## 2. Pick the pattern
| If | Pattern | Model files to copy |
|---|---|---|
| loads with the main env's diffusers / transformers / torch | in-process | `synthgen/adapters/sd15_inpaint.py`, `flux2_klein.py` |
| repo pins old libraries or ships its own model code | worker in `envs/<name>` | `synthgen/adapters/mimicbrush.py` + `workers/mimicbrush_worker.py` |
| a combination of existing adapters, or no learned model | compose | `synthgen/adapters/copy_paste.py` |
| an MLX port exists | MLX | `synthgen/adapters/flux2_klein_mlx.py` |

Reading the closest existing adapter is faster than starting from the template: it shows the
conventions (CPU `torch.Generator` for seeds, `use_safetensors=True`, `prepare()` for text encoders,
`_keep` for extra init arguments).

## 3. Write the adapter
- Declare `caps` honestly. The planner rejects jobs the adapter can't do. Operations and
  conditioning are what it can do *well*: a context-only model like LaMa can `remove` but can't
  insert a requested class.
- `_inpaint_crop(image, mask, req)` gets a square RGB crop at `native_res` plus a mask (255 =
  generate) and returns the same size. Don't paste back or resize; the base class does that.
- Large text encoder? Encode every prompt in `prepare()`, free the encoder, then `load()` the
  rest. That pattern is what makes FLUX.2-klein fit in 24 GB.
- Reference models read `req.reference` (object crop, white background) and `req.reference_mask`.
- **Swapping a model that's already in use** (e.g. the depth model behind `occlusion`, the labeller,
  the DINOv2 embedder): make it a config option with the old model as default, not an edited
  constant. The model id then lands in `job.json` and the job hash, so outputs record which model
  made them and old output folders aren't silently mixed. Record the licence when it differs.
- **Auxiliary models** (depth, embeddings, parsers loaded during planning or verification) should be
  freed when their stage ends (`del`, `gc.collect()`, `torch.mps.empty_cache()`). A 100 MB model can be
  left loaded; a 1.3 GB one competes with the generator for the ~17.8 GB.

For a **worker**: keep `third_party/<Repo>` unmodified and patch at runtime inside the worker. Don't
import the repo's demo scripts, since they often load the model at import time; copy their
pre/post-processing instead. Put the protocol on the saved stdout and library prints on stderr.
Load every pickle with `torch.load(..., weights_only=True)`, plus `mmap=True` for multi-GB files.
Convert a huge training checkpoint once to fp16 safetensors, keeping diffusion schedule buffers
(`alphas_cumprod`, ...) in fp32 (see `tools/convert_anydoor_ckpt.py`).

## 4. Register, smoke-test, benchmark
```bash
# synthgen/registry.py:  "my_model": "synthgen.adapters.my_model:MyAdapter",
.venv/bin/python -m synthgen.bench list
.venv/bin/python - <<'EOF'
from synthgen.adapters.my_model import MyAdapter
from synthgen.bench import bench_case
a = MyAdapter(); a.load(); print(a.inpaint(bench_case()).meta); a.unload()
EOF
.venv/bin/python -m synthgen.bench run my_model --n 5        # records time and peak memory for the planner
```
Look at `bench_out/my_model_*.png`. If the first call is far slower than expected or memory climbs
toward the limit, switch to the `apple-silicon-ml-perf` skill before tuning anything else: swap
was behind the worst slowdown in this repo (MimicBrush ran 5x slower until two memory fixes).

Add the model to `DEFAULT_SUITE` in `synthgen/bench.py` and to the model table in `docs/USAGE.md` §3.

## 5. Compare on identical tasks
Jobs with the same `seed`, source and targets get the same tasks and references, so copy an existing
comparison config and change only the generator and output dir:
```bash
sed 's/adapter: anydoor, steps: 30, guidance: 4.5/adapter: my_model/; s/coco_train_car_replace_anydoor/coco_train_car_replace_my_model/g' \
    configs/coco_train_car_replace_anydoor.yaml > configs/coco_train_car_replace_my_model.yaml
.venv/bin/python -m synthgen.run run configs/coco_train_car_replace_my_model.yaml
.venv/bin/python -m synthgen.review out/coco_train_car_replace_my_model --max 8
```
Then view its `review.jpg` next to `out/coco_train_car_replace_anydoor/` and
`out/coco_train_car_replace_copypaste/`. Report speed, memory, acceptance and visible quality.
Copy-paste and AnyDoor both pass verification 8/8, so acceptance alone can't rank models. Recommend
a downstream test (train on real + each synthetic set, evaluate on real) before adopting a model for
data generation at scale.

## 6. Finish
Tell the user what was added (pattern, licence, weights location, disk used), the benchmark numbers,
the comparison result, and any limits found (gated, non-commercial, slow, memory-bound). Record
porting problems you had to solve in `docs/ADDING_MODELS.md` §9 so the next integration is faster.
