# synthgen

Synthetic labelled image datasets by inpainting: remove, regenerate, replace, change the class of,
or insert objects in COCO-format datasets, or edit eyes / mouth / hair in face images. Each edit
is re-detected to recover its label, and the result is exported as standard COCO. Everything runs
locally, on an Apple Silicon Mac (developed and tested there) or a Linux machine with an NVIDIA GPU.

- **scene datasets (COCO):** remove / regenerate / replace (text or reference) / change_class / insert
- **face datasets (image folder):** regenerate / replace of eyes, brows, nose, mouth, lips, hair, ...
- **generators:** LaMa, SD-1.5, FLUX.2-klein (MPS and MLX), Qwen-Image-2.1 (MLX), MimicBrush, AnyDoor,
  copy-paste baseline; Qwen-Image-Edit 4-bit (CUDA only in practice)
- **occlusion handling (on by default):** whatever is in front of the edited object stays in front;
  targets seen through a fence or mesh are skipped
- **post-processing:** seeded, recorded degradations (resolution loss, blur, noise, colour, JPEG)

Not yet: identity check for faces, video.

**Docs:** user guide with job templates and sample commands: [docs/USAGE.md](docs/USAGE.md).
Adding inpainting / generation models: [docs/ADDING_MODELS.md](docs/ADDING_MODELS.md).
**Claude Code skills** (in `.claude/skills/`, picked up automatically when you open the repo in
Claude Code): `synthgen-dataset` (design, run and review a job), `synthgen-add-model` (vet and
integrate a new model), `apple-silicon-ml-perf` (diagnose slow or memory-bound models on a Mac).

## Requirements

- **Apple Silicon Mac** with a recent macOS. The models run on the GPU through PyTorch-MPS and
  MLX.
- **Or Linux with an NVIDIA GPU (untested so far).** The PyTorch models run on CUDA (`backend: cuda`, picked
  automatically when there is no MPS); the MLX adapters (`flux2_klein_mlx_refine`, `qwen_image21_edit`)
  are Mac-only and the planner rejects them elsewhere. Use `flux2_klein_inpaint` or
  `qwen_image_edit_4bit` instead. Peak memory is GPU memory (NVML), so run `bench all` there to get
  a profile; the Mac numbers below don't carry over. The worker environments install the default
  PyPI torch, which is a CUDA build on Linux.
- **24 GB unified memory recommended.** Developed and tested only on an M5 Pro (16-core GPU, 24 GB).
  Peak memory per model is in the benchmark table below (FLUX.2 at 512: 9.7 GB, AnyDoor: 10 GB,
  MimicBrush: 11.2 GB); on a smaller Mac, run `bench all` and the planner will refuse what doesn't fit.
- **Disk:** ~20 GB for the main model weights (in `~/.cache/huggingface`), plus ~13 GB if you set up
  MimicBrush and AnyDoor, ~33 GB if you use Qwen-Image-2.1, plus your datasets.
- Python 3.12 via [uv](https://docs.astral.sh/uv/), and git.

## Setup

```bash
git clone <this repo> synthgen && cd synthgen
python3 -m pip install --user uv                           # if uv is missing
export PATH="$(python3 -m site --user-base)/bin:$PATH"     # where pip --user puts uv
uv sync                                                    # creates .venv (Python 3.12)
.venv/bin/python -m synthgen.bench list                    # adapters and their licences
.venv/bin/python -m synthgen.bench all                     # ~15 min; downloads ~20 GB of weights
```

`bench all` measures each model's speed and peak memory on your machine and saves them to
`profiles/<chip>-<gpu>gpu-<mem>gb.json` (on NVIDIA: `profiles/<gpu>-<vram>gb.json`). The planner uses that profile to estimate run time and to
refuse jobs that won't fit in memory. Without a profile for your machine, jobs still run, without
estimates. The committed `profiles/m5-pro-16gpu-24gb.json` is the reference machine.

**Optional:**

| You want | Do this |
|---|---|
| MimicBrush (face parts or textures from a reference) | `tools/setup_workers.sh mimicbrush` (~5 GB env; weights download on first use) |
| AnyDoor (object from a reference) | `tools/setup_workers.sh anydoor` (~3 GB env; downloads a 16.8 GB checkpoint and converts it to 4.6 GB in `~/.cache/synthgen/`) |
| SAM 3 as labeller | Request access on the `facebook/sam3` Hugging Face page, then `.venv/bin/hf auth login` |

MimicBrush and AnyDoor pin 2023-era library versions, so each runs in its own environment
(`envs/<name>`) as a worker process (`synthgen/adapters/worker.py`, `workers/*.py`). Their code is
cloned unmodified into `third_party/` at a tested commit; Mac/MPS fixes are applied at runtime
inside the worker. Pickle weights (`.bin`, `.ckpt`, `.pth`) are only ever loaded with
`torch.load(weights_only=True)`, so code embedded in them cannot run.

## Source data (bring your own)

No datasets are included. Download the ones you need and put them where the configs expect them,
or edit `source:` in your job file to point anywhere on disk.

| Dataset | Used by | Put it at |
|---|---|---|
| COCO 2017 annotations ([cocodataset.org](https://cocodataset.org/#download), `annotations_trainval2017.zip`) | all `configs/coco_*` | `data/coco/annotations/instances_val2017.json` (and `instances_train2017.json` for `coco_train_*`) |
| COCO 2017 images | same | `data/coco/val2017/`, `data/coco/train2017/` |
| A folder of roughly frontal face images | `configs/faces_*` | `data/faces_synth/` (or generate synthetic portraits: `.venv/bin/python tools/make_synthetic_faces.py`) |

```bash
mkdir -p data/coco && cd data/coco
curl -LO http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip 'annotations/instances_*2017.json' && cd -
```

With `source.download_missing: true` (set in the example configs), COCO images that a job needs
are fetched one by one from images.cocodataset.org instead of downloading the whole image set. Set it to
`false` to use only local files. Any COCO-format dataset with instance masks (polygon or RLE)
works as a scene source. `tools/download_datasets.sh` fetches some other public sets (LVIS,
MOT17/20, VisDrone, ADE20K, Open Images, CrowdHuman); check each dataset's licence and terms first.

## Quick recipes

Each line runs a complete job: plan, generate, verify, export. Outputs go to `out/<config name>/`
(`annotations.json`, `images/`, `masks/`). Runs are resumable; rerun the same command to continue.
The COCO recipes need the COCO annotations (see "Source data"); images download as needed.

```bash
R=".venv/bin/python -m synthgen.run run"

# Hard negatives: remove objects, background filled in (LaMa, ~1 min)
$R configs/coco_person_remove.yaml            # people
$R configs/coco_car_remove.yaml               # cars

# Appearance diversity: same class, new look from text (FLUX.2, ~4 s per image)
$R configs/coco_car_regenerate.yaml           # a new car of another colour/model in the same place
$R configs/coco_vehicle_recolor.yaml          # same car/truck/bus, new paint colour
$R configs/coco_car_regenerate_qwen21.yaml    # Qwen-Image-2.1 instead (~25 s/car, 33 GB download, non-commercial)

# Swap in another real instance of the class (needs tools/setup_workers.sh anydoor)
$R configs/coco_car_replace_anydoor.yaml      # val2017 references
$R configs/coco_train_car_replace_anydoor.yaml   # train2017: many more references, better matches
$R configs/coco_train_car_replace_copypaste.yaml # copy-paste baseline on the same tasks

# More objects per image: insert new cars where they fit, with camera-like degradation
$R configs/coco_car_insert.yaml

# Change class (car -> truck), rejecting outputs that still look like a car
$R configs/coco_car_to_truck.yaml

# Faces: new mouths from text, or eyes from another face (needs tools/setup_workers.sh mimicbrush)
.venv/bin/python tools/make_synthetic_faces.py data/faces_synth --n 12   # test faces, no real people
$R configs/faces_mouth_flux_text.yaml
$R configs/faces_eyes_mimicbrush.yaml

# Give your real training images the same degradation as a job's synthetic ones
.venv/bin/python -m synthgen.degrade configs/coco_car_insert.yaml <real_images_dir> <out_dir>
```

**Check the results before using them.** Verification only checks that the right class is (or
isn't) detected in the edited area; it misses shadows left behind, smears and odd placements.
```bash
.venv/bin/python -m synthgen.review out/coco_car_remove --max 12      # contact sheet: out/<job>/review.jpg
.venv/bin/python .claude/skills/synthgen-dataset/scripts/edit_closeups.py out/coco_car_remove --accepted-only
```

**Your own dataset or classes:** copy the closest config and change `name`, `source`,
`targets.classes` and `output.dir`. Any COCO-format dataset with instance masks works.
```bash
cp configs/coco_car_remove.yaml configs/my_job.yaml   # then edit those four keys
.venv/bin/python -m synthgen.run plan configs/my_job.yaml   # check targets found, time and memory first
.venv/bin/python -m synthgen.run run  configs/my_job.yaml
```
All job templates and options: [docs/USAGE.md](docs/USAGE.md).

## Licences

This repo's code is licensed under the Apache License 2.0 ([LICENSE.md](LICENSE.md)). The models it downloads have their own
licences, shown by `python -m synthgen.bench list`. Most are Apache-2.0 or MIT. Exceptions:
- **face_parsing** (the face labeller): licence unspecified; the base model and training data
  (CelebAMask-HQ) are non-commercial. Treat face outputs as research-only.
- **SD-1.5 / SD-2.1 based models** (sd15_inpaint, MimicBrush, AnyDoor): CreativeML OpenRAIL-M /
  OpenRAIL++-M use restrictions.
- **SAM 3:** gated, SAM licence.
- **Qwen-Image-2.1** (qwen_image21_edit): Qwen Research Licence, **non-commercial only**; a model
  trained on its outputs and made available must state "Built with Qwen". The 4-bit Qwen-Image-Edit
  (qwen_image_edit_4bit) is Apache-2.0.
- **Datasets** keep their own licences: COCO images are Flickr images under various Creative Commons
  licences. Synthetic images derived from them inherit those terms.

## Running a job

```bash
.venv/bin/python -m synthgen.run plan configs/coco_car_regenerate.yaml   # tasks + time/memory estimate
.venv/bin/python -m synthgen.run run  configs/coco_car_regenerate.yaml   # generate -> verify -> export (resumable)
.venv/bin/python -m synthgen.review out/coco_car_regenerate               # contact sheet: original | region | result
```

Example configs in `configs/`: `coco_car_remove` (LaMa), `coco_person_remove` / `coco_crowd_remove` (LaMa; individuals,
or COCO `iscrowd` regions with `targets.crowd: only`), `coco_car_regenerate` (FLUX.2, text),
`coco_car_replace_ref` (FLUX.2 with another car from the same split as reference),
`coco_car_to_truck` (change_class with contrast check), `coco_car_replace_anydoor` (AnyDoor,
lighting-matched references). They expect COCO val2017 annotations at
`data/coco/annotations/instances_val2017.json`; images are downloaded on demand.

`coco_car_insert`: new cars placed into free space by `synthgen/placement.py` (perspective from
the cars already in the scene, free space from the annotations, and a ground check: the new car's
bottom must sit on "road / street / parking lot" as segmented by the labeller), with post-processing.
`coco_train_car_replace_anydoor`: AnyDoor on train2017 with `match: visual` references (aspect,
local lighting and DINOv2 viewpoint similarity). Needs `data/coco/annotations/instances_train2017.json`.

`coco_car_insert_occlusion`: the insert job with occlusion handling (now on for all scene jobs). At plan time a depth
pass marks pixels in the edit region that are clearly closer than the edited object (connected to the
region's border, i.e. passing in front of it); after generation the original pixels are restored
there and the manipulation mask excludes them.

Post-processing (`postprocess:` in a job, see `synthgen/degrade.py`) applies the same degradation
family to the whole output image; only photometric and resolution operations, so labels stay valid.
The clean output is kept in `raw/`. To give real images the same treatment:
`python -m synthgen.degrade <job.yaml> <in_dir> <out_dir>`.

Face configs: `faces_eyes_mimicbrush` (eyes from another face as reference) and
`faces_mouth_flux_text` (mouth from text). Development data is synthetic portraits from
`tools/make_synthetic_faces.py`, so no real people's faces are needed for testing. Face jobs
export `face_labels.json` with, per image, the re-parsed label map (`labels/`, CelebAMask-HQ ids)
and the manipulation mask (`masks/`). The face parser's licence is unspecified and its base model
and training data are non-commercial; treat face outputs as research-only until replaced.

Each job runs one model at a time: plan -> generate (generator loaded) -> verify (labeller loaded)
-> export. Outputs per task: the image, the generation mask (also a manipulation-localisation
label), the recovered object mask, generator metadata and the verdict. `annotations.json` is
standard COCO (checked with pycocotools) with extra fields: `synthetic`, `source_image_id`,
`task_id` on images; `synthetic` and `edit` (operation, prompt, reference, adapter, score) on
the edited annotation. Unchanged annotations are copied with `source_ann_id`. A changed config
refuses to reuse an old output dir; `--replan` clears that dir's derived outputs.

## Layout

```
synthgen/
  types.py            Domain / Operation / Conditioning / Backend enums, Capabilities, request/result types
  crop.py             crop-around-mask at the model's native size + feathered paste-back
  adapters/base.py    Adapter, GeneratorAdapter (prepare -> load -> inpaint -> unload), LabelerAdapter
  adapters/*.py       lama, sd15_inpaint, flux2_klein (diffusers/MPS), flux2_klein_mlx (mflux/MLX),
                      mimicbrush, anydoor (worker adapters, see adapters/worker.py)
  labelers/*.py       gdino_sam2 (default), sam3 (gated), face_parsing (SegFormer, CelebAMask-HQ)
  face.py             face planning (part masks), face verification, face_labels.json export
  data/folder.py      unannotated image-folder source
  verify.py           label recovery: re-detect the generated object, accept/reject with a reason
  config.py           job YAML -> dataclasses (unknown keys are errors)
  plan.py             validation against adapter capabilities, task selection, time/memory estimate
  run.py              stage runner + COCO export;  review.py: contact sheets
  data/coco.py        COCO source (polygons via PIL, RLE decoded) and writer (uncompressed RLE)
  masks.py            exact / bbox / convex_hull + dilation;  refbank.py: same-split reference crops
  placement.py        insert: perspective-aware free-space boxes, optional ground check
  degrade.py          post-processing pipeline (+ CLI for real images)
  embed.py            DINOv2 embeddings for viewpoint-matched references
  occlusion.py        occluder detection (annotations, outline gaps, depth) and restoration
  hardware.py         machine profile; peak memory from phys_footprint on a Mac (counts MPS and MLX GPU memory), NVML on NVIDIA
  device.py           which backends this machine has; cache clearing per backend
  bench.py            per-adapter benchmark, one subprocess each -> profiles/<machine>.json
  registry.py         name -> adapter class (lazy imports)
workers/              MimicBrush and AnyDoor worker scripts (run inside envs/<name>)
tools/                setup_workers.sh, download_datasets.sh, make_synthetic_faces.py, convert_anydoor_ckpt.py
configs/              example jobs;  profiles/: benchmark results per machine
docs/                 USAGE.md (user guide), ADDING_MODELS.md
.claude/skills/       Claude Code skills for this repo
```

### Adding a model

See [docs/ADDING_MODELS.md](docs/ADDING_MODELS.md): licence/memory/Mac checks, in-process,
worker-process (separate environment) and MLX patterns with templates, composing adapters,
registering, benchmarking, comparing against the copy-paste baseline, and Mac porting fixes.

## Benchmarks on the reference machine (M5 Pro, 16-core GPU, 24 GB)

M5 Pro, 16-core GPU, 24 GB, macOS 26.5.1, torch 2.14.1, MLX 0.32.3. Default GPU working set
17.8 GB. Test case: COCO val2017 #39769, left cat (SAM 2.1 mask, dilated 10 px) regenerated as
"a small brown dog". Median of 5 runs after one warmup. Peak = process phys_footprint, steady state.

| adapter | backend | crop | steps | s / item | items / h | peak GB | notes |
|---|---|---|---|---|---|---|---|
| gdino_sam2 (labeller) | MPS | 800 | - | 0.58 | 6,194 | 6.4 | per image, 2 classes |
| lama | MPS | 512 | 1 | 0.13 | 27,566 | 1.6 | removal only |
| sd15_inpaint | MPS fp16 | 512 | 25 | 8.54 | 422 | 6.2 | |
| flux2_klein_inpaint | MPS bf16 | 512 | 4 | 4.00 | 900 | 9.7 | + 9.3 GB one-off prompt-encoding phase |
| flux2_klein_inpaint | MPS bf16 | 1024 | 4 | 19.84 | 181 | 16.6 | near the 17.8 GB limit; higher variance |
| flux2_klein_mlx_refine | MLX q8 | 512 | 4 | 5.11 | 704 | 14.3 | img2img + composite; re-encodes prompt per call |
| flux2_klein_mlx_refine | MLX q8 | 1024 | 4 | 17.60 | 205 | 16.5 | same |
| face_parsing (labeller) | MPS | 512 | - | 0.06 | 57,644 | 1.8 | |
| mimicbrush | MPS fp16 (worker) | 512 | 25 | 33.9 | 106 | 11.2 | ~35 s per face in the eyes job |
| anydoor | MPS fp16 (worker) | 512 | 30 | 11.4 | 316 | 10.0 | 6.6 s per car in the COCO job |
| qwen_image21_edit | MLX q4 | 512 | 20 | 22.4 | 161 | 14.6 | 20-26 s per car in the COCO job |
| qwen_image_edit_4bit | MPS NF4 (bitsandbytes) | 768 | 20 | ~1,480 | ~2 | ~28 (swaps) | not usable on Apple Silicon; second image ran out of memory |

Worker peaks include the worker process (`hardware.total_footprint_bytes`).

What this means:

- **MLX is not automatically faster.** For FLUX.2-klein-4B, MLX (8-bit) was slower than
  PyTorch-MPS (bf16) at 512 and about 11% faster at 1024. The MLX figures include re-encoding the
  prompt on every call, which the MPS adapter does once in `prepare()`. Default to PyTorch-MPS.
  Use MLX where it buys memory headroom (9B/20B models that don't fit in bf16), and benchmark each case.
- **Plan on sustained throughput, not the burst figure.** In a 20-minute continuous FLUX.2 512 run
  (242 images), the mean time per image rose from 4.38 s (first 10%) to 5.53 s (last 10%), about 26%
  slower, as the machine heated up. Median over the run was 4.99 s, about 720 images/h against 900
  in the short burst. `profiles/` now holds this sustained record for that key.
- **Use 512 crops for FLUX.2 in bulk runs.** 1024 is ~5x slower and runs at 16.6 of 17.8 GB.
- **Watch total memory, not just the GPU limit.** MimicBrush first ran at ~7 s/step because its
  worker reached 19 GB and macOS swapped. Two fixes made it ~5x faster (25 steps: 179 s -> 34 s,
  peak 18.4 -> 11.2 GB): memory-mapping the 9.4 GB checkpoint while it is copied into fp16 models,
  and chunking 8192-token reference attention so MPS doesn't cache multi-GB attention matrices.
  Check `sysctl vm.swapusage` if a model is unexpectedly slow.
- **Few-step distilled models beat older many-step ones.** FLUX.2-klein (4 steps) is about 2x faster
  than SD-1.5 (25 steps) at 512, and its output is better.

## Label recovery findings (from the benchmark outputs)

- The inpainting mask is not the label. SD-1.5's dog occupied 36% of the mask, so its tight box
  (40,125)-(218,358) was far smaller than the mask box. SD-1.5 also painted an extra object outside
  the intended shape.
- With the exact cat outline as the mask, FLUX.2 at 512 produced a cat-shaped animal that Grounding
  DINO scored 0.74 "dog" vs 0.70 "cat". `recover_label(..., contrast=("cat",))` rejects it as
  ambiguous. Use dilated/bbox masks for `replace`, and always pass the original class as `contrast`.
- Open-vocabulary detectors given several phrases at once assign one label per box and hide
  near-ties. `verify.py` therefore queries each class separately.

## Findings from the example jobs (COCO val2017, cars)

| job | tasks | accepted | rejections |
|---|---|---|---|
| remove (LaMa) | 8 | 8 | |
| regenerate (FLUX.2, text) | 6 | 5 | 1 no car generated (blurry background car) |
| replace (FLUX.2, reference) | 6 | 5 | 1 no car generated (same background case) |
| car -> truck (FLUX.2) | 2 | 1 | 1 ambiguous: truck 0.35 vs car 0.57 (distant night scene) |

- **Few eligible targets.** Of 311 non-crowd cars >= 4000 px, 178 touch the image border and
  ~125 overlap another annotated object. Street scenes are dense. Loosen `max_overlap_other`
  or use train2017 for volume. `plan` prints these counts.
- **The detector's vocabulary is not the dataset's taxonomy.** Grounding DINO-tiny scored clear
  pickups as "car" above "truck", so every car -> truck sample was rejected until
  `verify.phrases: {truck: [pickup truck, truck]}` was added. Check phrases per class on a few
  samples before a large change_class run.
- **The labeller can't catch physics errors.** In one replace, the original car was half
  submerged in floodwater and the new car sits on the water surface. Keep human spot checks
  (`synthgen.review`).
- **FLUX.2-klein's `image_reference` gives working reference-based replacement** with no extra repo.
- **Reference quality decides reference-based output quality** (AnyDoor, COCO cars). In turn:
  - Occluded references are copied with their occluders: a car behind umbrellas came out as a
    car fragment with a ribbon, and the verifier accepted it. References now pass the same
    filters as targets (not truncated, no overlap).
  - Small or black-and-white references give blurry or grey objects. References need
    `min_area` >= 16000 px, and greyscale images are skipped.
  - AnyDoor copies the reference's lighting (a night car became a dark blob in daylight).
    `match: aspect_light` compares brightness around the target and reference objects.
  - After these filters COCO val2017 has only **4** usable car references. `max_uses: 2` stops
    one car appearing in many samples, but volume needs train2017 or a dedicated object bank.
  - Viewpoint is not matched yet (a car photographed from above went into a street-level scene).
    DINOv2 similarity between the target crop and candidate references is the next step.

## Findings from the face jobs (synthetic portraits)

| job | tasks | accepted | s / face |
|---|---|---|---|
| eyes from another face (MimicBrush, 25 steps) | 6 | 6 | ~35 |
| mouth from text (FLUX.2-klein) | 6 | 6 | ~5 |

- Both give seamless edits. MimicBrush changes are subtle (eye colour and shape follow the
  reference), which makes for hard examples in manipulation-detection data. FLUX.2 follows
  prompts closely (smiling with teeth, lipstick, thin or full lips).
- Face verification checks that the edited part is still parsed as that part, inside the edited
  region, at 0.4-2.5x its original area. It does not check identity: a recognition model
  (ArcFace-class) is needed, and the common ones carry non-commercial licences.

## Findings: insert, post-processing, reference matching

- **Insert needs a ground check.** Placement from the scene's own cars alone (same "row", perspective
  fit, free space) put a car on a pier among pigeons, and verification accepted it. With the
  labeller's "road / street / parking lot" mask under the new car's bottom edge, all 8 placements
  landed on roads or car parks and 7/8 passed verification (1 too small to detect). Free space only
  knows annotated objects, so unannotated signs or hydrants can still be painted over.
- **COCO val2017 is too small for references; train2017 isn't.** Clean reference cars (>= 16000 px,
  not truncated, not overlapped): 4 in val2017, 185 in train2017.
- **DINOv2 matching finds the viewpoint on clean crops.** Nearest neighbours of a rear 3/4 SUV are
  rear views, of a side-on classic car side views, of a front 3/4 car front 3/4 views (similarity
  0.5-0.73 vs ~0 for the farthest). On val2017's occluded crops the signal was weak (~0.2 spread).
  With `match: visual` on train2017, AnyDoor replacements matched the original viewpoint in 8/8,
  all accepted, no reference reused.
- **Occlusion: solid occluders yes, see-through ones no.** With `occlusion.enabled`, weed stems in
  front of an inserted car and a wall corner beside one were detected and restored on top (5 of 12
  insert tasks had occluders); an unoccluded van got zero occluder pixels, so an object's own near
  parts are not mistaken for occluders. A car seen *through* a mesh fence can't be handled: depth
  puts the fence and the car at the same distance. Such targets need excluding upstream.
  Depth Anything V2 Small is used because it is the only Apache-2.0 size (Base/Large are CC-BY-NC).

## Findings: occlusion for every operation

Test set: 5 COCO val2017 cars partly hidden behind a pole, a tree trunk (+ dog and frisbee), a fire
hydrant, a skateboarder and galloping horses (`configs/coco_car_occluded_*.yaml`).

- **Before:** depth-only occluders were opt-in, and the planner skipped 3 of the 5 because annotated
  objects overlapped the edit region.
- **After:** all 5 are planned. Regenerate kept every occluder in front in all 5 (a green car behind
  the hydrant, a black sedan behind the pole, a white car behind the tree, dog and frisbee). Remove kept
  every occluder. Its two rejections were real failures: LaMa left a dark blob where it had to invent a
  large hidden area.
- **The pole was found by the outline-gap rule:** COCO's polygon for the car excludes the pole, and depth
  confirms the gap is closer than the car. Annotated occluders (hydrant, person, horses) get exact edges
  from their masks.
- **Objects seen through a mesh are skipped at plan time.** Depth finds a fence's posts and rails but not
  the mesh, so a regenerated car came out in front of a chain-link fence. Grounding DINO couldn't tell
  fences apart: "fence" fired on 60 of 133 val2017 cars, including ladders, mirrors and horses, and
  scores overlapped. SigLIP 2 zero-shot on the target crop ("a car seen through a chain-link fence"
  vs "a car") separated them: at 0.8 it caught 9 of 11 see-through cars (fences, an iron railing),
  with 0 false positives and about 10 s added to planning. With it, `coco_car_regenerate` skips the fence car,
  draws another target, and all 8 regenerated cars stay behind their occluders (8/8 accepted). One
  shows a separate problem: a blurred background car came back sharp (depth-of-field mismatch).

## Findings: fixing occluded edits and inserts

The occluded replace and insert outputs looked wrong. Causes and fixes, checked on the same tasks:
- **Occluded replace (AnyDoor): clutter and tilted, ghosted cars.** Two causes. The occluder was cut out
  of the generation mask, and AnyDoor painted clutter into the hole: a crouching figure beside a
  skateboarder, fragments between horses' legs. And 2 of the 8 val2017 references were bad: a car
  photographed from above through a wet window (copied as a tilted car) and a van behind a chain-link
  fence (copied with fence lines). Fixes:
  - The generator now paints the whole region, then the original scene goes back on top.
  - With `occlusion.composite: silhouette` (the default, as suggested), new pixels are kept only
    inside the old car's outline, and everything else in the region comes back from the original.
  - The see-through check now also screens references, and "seen through a wet window" was added.
  - Result: 5 of 5 accepted and plausible, against 3 of 5 before. Compared with restoring only the
    detected occluders, silhouette was better in 4 of 5 (a car behind a hedge no longer floats over
    it) and tied in the fifth. Regenerate looked the same in both modes.
- **Inserts: motion-blur trails, smears, translucent cars.** The region was exactly the planned box, so
  FLUX.2 stretched the car to fill it or padded the gap with motion blur. A prompt asking for "a
  stationary car, no motion blur" did not help. Fixes:
  - `placement.slack` widens the region so the car can take its natural shape.
  - `verify.object_only` keeps only the detected car and a shadow band; the rest of the region
    stays original.
  - Scenes the car can't match are skipped: an out-of-focus alley (`min_sharpness`) and a
    long-exposure light-trail street (`scene_skip_threshold`).
  - Result on `coco_car_insert_occlusion`: 8 of 12 accepted. 7 are sharp and complete and sit behind
    poles and plants; one is an odd bubble-shaped car.
- Remaining: blur from the job's own `downscale` degradation is intended. Placement can still put a
  car close against a parked one.

## Findings: recolouring vehicles

`configs/coco_vehicle_recolor.yaml` (FLUX.2-klein, exact outline, partial strength) on COCO val2017
cars, trucks and buses:
- **`strength` decides whether the vehicle stays the same.** At 8 steps: 0.5 mostly changed nothing;
  0.65 kept the make, body, livery and lettering (a removal truck turned green with its text intact,
  a shuttle bus turned yellow); 0.8 started to change the vehicle (a hatchback became a different SUV).
- **With the default 4 steps, strength is quantised:** 0.6 and 0.75 both run 3 of 4 steps and gave
  byte-identical images.
- **Verification needed a colour check.** Detection alone accepted a silver car "recoloured" to white
  with no visible change. `verify.min_color_change` (p75 CIELAB delta E, threshold 12) rejected it.
  Final run: 9 of 12 accepted, all 9 correct on close-up inspection.


## Findings: Qwen-Image models

- **Qwen-Image-2.1 (`qwen_image21_edit`, `configs/coco_car_regenerate_qwen21.yaml`)** runs natively on
  MLX through mflux, quantised to 4-bit at load: 22 s per car at 512 px and 20 steps, 14.6 GB peak.
  It has a real mask input: everything outside the mask stays exactly the same.
- **It needs an edit instruction, not a caption.** Given "Inside the masked region, show a photo of
  a red car", 2 of 4 cars barely changed (colour change 3 and 7) yet passed detection. With "Replace
  the masked area with a red car. Keep the same position, perspective, lighting and scale." all four
  changed; 3 were accepted and correct (black to yellow with the skateboarder and rope in front kept,
  a yellow Peugeot to red with its number plate kept, a car seen in a mirror to green). The fourth,
  a floating dark van shape, was rightly rejected. The adapter now wraps job prompts this way.
- **Against FLUX.2 on the same 8 cars** (`coco_car_regenerate_qwen21` vs the FLUX.2 run with the same
  seed; 7 cars in common): Qwen accepted 6 of 8, all correct on close-up. Its two rejections were right:
  the same floating van, and a near-invisible car in a dark blurred corner. Qwen tends to keep the
  original car's body and number plate and change the paint, where FLUX.2 draws a new car. On a blurred
  background car behind footballers, Qwen matched the blur; FLUX.2's car came out sharp. FLUX.2 is
  about 5x faster (4 s against 22 s) and Apache-2.0, so it stays the default.
- **Qwen-Image-Edit 4-bit (`qwen_image_edit_4bit`, ovedrive/qwen-image-edit-4bit)** is a
  bitsandbytes NF4 checkpoint. bitsandbytes has no Metal kernels, so on a Mac every layer is
  dequantised in fp32 on each call: one 768 px image took 25 minutes at ~28 GB (swapping), and the
  second ran out of memory. The adapter is kept for a CUDA GPU with 20 GB or more (`backend: cuda`).

## Findings: model comparison

`python -m synthgen.compare configs/compare_all.yaml` re-ran every example job and ran every other
model that can do the same task on identical tasks (same images, objects, masks, prompts, references
and seeds). Below, "good" counts samples that look right on close-up inspection, as opposed to
"acc", which counts samples the automatic check accepted. Sheets: `out/compare/<job>/sheet_*.jpg`.

| Task | Best | Others (good / tasks) |
|---|---|---|
| remove cars (8 + 5 occluded) | `qwen_image21_edit` 7/8, 4/5 | LaMa ~5/8, 3/5; SD-1.5 4/8; FLUX.2 2/8 (flat white blocks) |
| remove people (24) | `lama` 17 acc, smears but no new content | SD-1.5 paints other people in; FLUX.2 blocks; Qwen leaves people in place (5/24, 4/24 with "Remove the person") |
| remove crowds (16) | none usable | every model fails; skip crowd regions |
| regenerate cars (8) | `flux2_klein_inpaint` ~6-7 | Qwen 6 (keeps body and plate, matches blur); SD-1.5 ~1; MLX refine 0 |
| regenerate occluded cars (5) | `qwen_image21_edit` 5/5 | FLUX.2 4; MLX refine ~2 (halo seams); SD-1.5 ~1 |
| recolour vehicles (12) | `flux2_klein_inpaint` 9 acc, all right | Qwen ~8 at strength 1 (0 at strength 0.65: no change); SD-1.5 often the wrong colour |
| replace from reference (8 train2017) | `anydoor` ~6-7 | FLUX.2 ~5-6 (twice returned the original car); copy-paste ~3; MimicBrush ~2 |
| car to pickup truck (8) | none usable | all mostly paint cars; real pickups rejected as "ambiguous truck vs car" |
| insert cars (8 + 12) | `flux2_klein_inpaint` ~5/8 | Qwen ~6/8 look right but are rejected by `min_fill` (smaller, correctly scaled cars); SD-1.5 paints nothing |
| face: mouth from text (6) | `flux2_klein_inpaint` 6/6 | Qwen ~2 (smiles not applied); SD-1.5 ~1; MLX refine ~1 |
| face: eyes from another face (6) | `mimicbrush` ~5/6 | FLUX.2 with the face as reference barely changes the eyes |

The automatic check misses these failures (accepted but wrong):
- flat fills after removal;
- output identical to the input (FLUX.2 replace and eyes, Qwen mouths);
- the wrong colour on recolour;
- grotesque teeth;
- halo seams;
- the word "EMPTY" painted from the prompt.

It also rejects some good outputs: small inserted cars, and real pickups.

Speeds during the comparison ran up to 2x the benchmark table: the machine was swapping with other
apps open. The first in-process version of the tool, which kept every model in one process, ran 2-4x
slower than the benchmarks, so `compare` now runs each model in its own process.
