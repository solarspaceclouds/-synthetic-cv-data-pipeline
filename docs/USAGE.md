# Using synthgen

How to generate a synthetic labelled dataset: from setup to a checked COCO file. Commands are
run from the repository root on a Mac (Apple Silicon); they work the same on Linux with an NVIDIA GPU,
except that the MLX adapters are Mac-only (see the README). Everything runs locally.

- [1. Setup](#1-setup)
- [2. The workflow in five commands](#2-the-workflow-in-five-commands)
- [3. Choosing an operation and a model](#3-choosing-an-operation-and-a-model)
- [4. Job templates](#4-job-templates)
- [5. Reading the outputs](#5-reading-the-outputs)
- [6. Tuning and troubleshooting](#6-tuning-and-troubleshooting)
- [7. Config reference](#7-config-reference)
- [8. Command reference](#8-command-reference)

---

## 1. Setup

One-time, about 15 minutes plus downloads.

```bash
python3 -m pip install --user uv                         # if `uv` isn't installed
export PATH="$(python3 -m site --user-base)/bin:$PATH"   # where pip --user puts uv on macOS
uv sync                                                  # creates .venv (Python 3.12) with all main dependencies
```

Measure the models on your machine once. The planner uses these numbers to estimate run time and
to refuse jobs that won't fit in memory.

```bash
.venv/bin/python -m synthgen.bench all      # ~15 min; downloads ~20 GB of weights on first run
```

**Optional extras:**

| You want | Do this |
|---|---|
| MimicBrush (face parts / textures from a reference) | `tools/setup_workers.sh mimicbrush` |
| AnyDoor (object from a reference) | `tools/setup_workers.sh anydoor` (downloads and converts a 16.8 GB checkpoint) |
| SAM 3 as labeller | Request access on the `facebook/sam3` Hugging Face page, then `.venv/bin/hf auth login` |
| The COCO examples | `mkdir -p data/coco && cd data/coco && curl -LO http://images.cocodataset.org/annotations/annotations_trainval2017.zip && unzip annotations_trainval2017.zip 'annotations/instances_*2017.json' && cd -` (images download on demand; see "Source data" in the README for other datasets) |
| Face examples without real people | `.venv/bin/python tools/make_synthetic_faces.py data/faces_synth --n 12` |

---

## 2. The workflow in five commands

```bash
# 1. write or copy a job file (see section 4), then check it: task count, time and memory estimate
.venv/bin/python -m synthgen.run plan configs/coco_car_regenerate.yaml

# 2. run it: generate -> verify -> export. Safe to interrupt; rerunning resumes where it stopped
.venv/bin/python -m synthgen.run run configs/coco_car_regenerate.yaml

# 3. look at the results before using them: original | edit region | result (+ recovered box)
.venv/bin/python -m synthgen.review out/coco_car_regenerate --max 12
open out/coco_car_regenerate/review.jpg

# 4. see what was accepted and why the rest was rejected
cat out/coco_car_regenerate/summary.json

# 5. use the dataset: out/coco_car_regenerate/annotations.json + images/
```

What `plan` prints, and what to check:

```
[plan] selection: {"candidates": 311, "drawn": 312, "skipped": {"truncated": 178, "overlaps_other": 125, ...}}
[plan] 6 tasks ({'car->car': 6}) -> out/coco_car_regenerate
[plan] estimate: {"generate_min": 0.8, "generator_peak_gb": 9.69, "verify_min": 0.1, ...}
```

- **Fewer tasks than `n_samples`?** `skipped` says which filter removed the candidates (section 6).
- **`warning` in the estimate?** The model is close to the Mac's GPU memory limit (~17.8 GB on a 24 GB machine).
  Use a smaller `generator.res`.

**Always look at `review.jpg`.** Verification catches missing, misplaced and ambiguous objects. It
does not catch physically wrong scenes (a car floating on water, wrong lighting).

---

## 3. Choosing an operation and a model

| Operation | What it makes | Label effect | Good models |
|---|---|---|---|
| `remove` | Object erased, background filled in | Annotation dropped (hard negative) | `lama` (people); `qwen_image21_edit` with a removal prompt (cars, objects) |
| `regenerate` | Same class, new appearance from text | Box/mask re-detected | `flux2_klein_inpaint`; `qwen_image21_edit` (occluded or blurred targets) |
| `replace` | Same class, appearance copied from a reference object | Box/mask re-detected | `anydoor`; `flux2_klein_inpaint`; `copy_paste` (baseline) |
| `change_class` | Object becomes another class (car -> truck) | New class, checked against the old one | weak with every model (see the comparison) |
| `insert` | New object in free space | New annotation | `flux2_klein_inpaint`; `qwen_image21_edit` with a lower `verify.min_fill` |
| face `regenerate` / `replace` | Eyes, mouth, hair... from text or another face | Face re-parsed | `flux2_klein_inpaint` (text), `mimicbrush` (reference) |

These picks come from running every model on identical tasks (`configs/compare_all.yaml`; results
and per-task judgements in the README, "Findings: model comparison").

Measured on an M5 Pro (16-core GPU, 24 GB):

| Model | Conditioning | Time per image | Peak memory | Licence |
|---|---|---|---|---|
| `lama` | none (fills from context) | 0.13 s | 1.6 GB | Apache-2.0 |
| `copy_paste` | reference object (no model: LaMa clears, object pasted) | 0.19 s | 1.6 GB | n/a (LaMa: Apache-2.0) |
| `flux2_klein_inpaint` | text, reference | 4 s at 512 (~5 s sustained), 20 s at 1024 | 9.7 / 16.6 GB | Apache-2.0 |
| `sd15_inpaint` | text | 8.5 s | 6.2 GB | OpenRAIL-M |
| `anydoor` | reference object | 7-11 s | 10 GB | MIT |
| `mimicbrush` | reference image | 34 s | 11 GB | Apache-2.0 |
| `flux2_klein_mlx_refine` | text (img2img, not true inpainting) | 5 s | 14 GB | Apache-2.0 |
| `qwen_image21_edit` | text (instruction editor, native mask) | 22 s at 512 | 14.6 GB | Qwen Research (non-commercial) |
| `qwen_image_edit_4bit` | text | CUDA GPU with 20 GB+ only; ~25 min on a Mac | over 24 GB on MPS | Apache-2.0 |

`copy_paste` is the baseline to beat: jobs with the same seed get the same tasks and references, so
compare any generative model against it (`configs/coco_train_car_replace_copypaste.yaml`).

To add a model, see [ADDING_MODELS.md](ADDING_MODELS.md).

List what's installed, with each model's operations and licence:

```bash
.venv/bin/python -m synthgen.bench list
```

---

## 4. Job templates

Copy one into `configs/`, change the paths and classes, then `plan` it. The source dataset must be
COCO-format (polygon or RLE masks) for scene jobs, or a plain image folder for face jobs.

### A. Remove objects (hard negatives)

```yaml
name: mydata_person_remove
domain: scene
source: {annotations: data/mydata/annotations.json, images: data/mydata/images, split: train}
targets:
  classes: [person]
  min_area: 3000
  # skip people touching what they hold or ride: removing them would leave it floating
  exclude_near: [tennis racket, skis, snowboard, surfboard, skateboard, baseball bat, bicycle, motorcycle, horse, umbrella, kite]
operation: remove
conditioning: {type: none}
mask: {shape: exact, dilate_px: 16}        # dilate to cover edges and some shadow
generator: {adapter: lama}
verify: {unlabelled: reject}               # drop images with people the source dataset never labelled
output: {dir: out/mydata_person_remove}
n_samples: 200
```

Removal needs a closer look than other operations. Verification only checks that no person is
detected in the edited area. It doesn't see a leftover **shadow or reflection** (a woman removed
from a river left her orange reflection in the water), **fragments** (part of an arm), or **smears**.
Check full-resolution close-ups of every edit
(`.claude/skills/synthgen-dataset/scripts/edit_closeups.py out/<job>`). Expect to keep roughly
a quarter to a half after human review, so generate 2-4x the number you need.

To clear whole crowds instead of individuals, set `targets.crowd: only` (and usually
`skip_truncated: false`): see `configs/coco_crowd_remove.yaml`.

### B. New appearance from text

```yaml
name: mydata_car_recolour
domain: scene
source: {annotations: data/mydata/annotations.json, images: data/mydata/images, split: train}
targets: {classes: [car], min_area: 4000}
operation: regenerate
conditioning:
  type: text
  prompt: "a photo of a {attr} car"           # {attr} is drawn from attributes, {cls} is the class
  attributes: [red, white, black, silver, blue]
mask: {shape: convex_hull, dilate_px: 16}    # not the exact outline: avoids copying the old silhouette
generator: {adapter: flux2_klein_inpaint, res: 512}
output: {dir: out/mydata_car_recolour}
n_samples: 200
```

### C. Replace with another instance (reference-based)

```yaml
name: mydata_car_swap
domain: scene
source: {annotations: data/mydata/annotations.json, images: data/mydata/images, split: train}
targets: {classes: [car], min_area: 6000}
operation: replace
conditioning:
  type: reference_object
  reference: {match: visual, min_area: 16000, max_uses: 2}   # viewpoint + lighting matched, same split only
mask: {shape: convex_hull, dilate_px: 16}
generator: {adapter: anydoor, steps: 30}
output: {dir: out/mydata_car_swap}
n_samples: 200
```

### D. Insert new objects into the scene

```yaml
name: mydata_car_insert
domain: scene
source: {annotations: data/mydata/annotations.json, images: data/mydata/images, split: train}
targets: {classes: [car]}
operation: insert
conditioning:
  type: text
  prompt: "a photo of a {attr} car parked on the street"
  attributes: [red, white, black, silver]
placement:
  ground_phrases: [road, street, parking lot]   # the new car's bottom must stand on these
  ground_min_frac: 0.6
mask: {dilate_px: 8}
generator: {adapter: flux2_klein_inpaint, res: 512, context_scale: 2.5}
verify: {min_fill: 0.25}
output: {dir: out/mydata_car_insert}
n_samples: 200
```

Insert needs at least one existing instance of the class in an image (it is used to judge position
and size). Images without one are skipped.

### E. Change class

```yaml
name: mydata_car_to_truck
domain: scene
source: {annotations: data/mydata/annotations.json, images: data/mydata/images, split: train}
targets: {classes: [car], min_area: 6000}
operation: change_class
change_to: [truck]
conditioning:
  type: text
  prompt: "a photo of a {attr} pickup truck"
  attributes: [red, white, black]
mask: {shape: bbox, dilate_px: 24}           # the new class may be bigger: give it room
generator: {adapter: flux2_klein_inpaint, res: 512}
verify:
  contrast: auto                             # reject outputs that still read as the old class
  phrases: {truck: [pickup truck, truck]}    # detector wording for the class (see section 6)
output: {dir: out/mydata_car_to_truck}
n_samples: 200
```

### E2. Recolour vehicles (same vehicle, new paint)

Regenerate inside the exact outline at partial strength: the model keeps the vehicle's shape,
livery and lettering and changes mostly the colour. Example: `configs/coco_vehicle_recolor.yaml`.

```yaml
name: mydata_vehicle_recolor
domain: scene
source: {annotations: data/mydata/annotations.json, images: data/mydata/images, split: train}
targets: {classes: [car, truck, bus], policy: random, min_area: 8000}
operation: regenerate
conditioning:
  type: text
  prompt: "a photo of a {attr} {cls}"
  attributes: [bright red, bright blue, yellow, green, white, black, orange]
mask: {shape: exact, dilate_px: 4}           # exact outline: the shape is meant to stay
generator: {adapter: flux2_klein_inpaint, res: 512, steps: 8, strength: 0.65}
verify: {min_color_change: 12}               # reject outputs whose colour barely changed
output: {dir: out/mydata_vehicle_recolor}
n_samples: 200
```

`strength` sets how much may change. With 8 steps, 0.5 often changes nothing, 0.65 recolours the
same vehicle, and 0.8 starts to change its shape. Keep `steps: 8`: with FLUX.2-klein's default of
4 steps, strength only acts in quarter steps (0.6 and 0.75 give identical results).
`verify.min_color_change` measures the 75th percentile of the per-pixel colour difference (CIELAB
delta E) in the edited area: about 3 for no visible change, 10 for a faint tint, 19-61 for clear
recolours. In testing, 9 of 12 COCO vehicles were accepted and all 9 looked right; the 3 rejections
were 2 near-unchanged outputs and 1 vehicle not re-detected. Small or blurred vehicles often get
only a partial tint and are rejected. The check doesn't verify that the new colour is the requested
one, only that the colour changed.

### F. Face parts

```yaml
name: myfaces_mouth
domain: face
source: {format: images, images: data/myfaces, split: train}
targets: {classes: [mouth], min_area: 300}   # eyes, brows, nose, mouth, lips, hair, ears, skin, glasses, ...
operation: regenerate
conditioning:
  type: text
  prompt: "close-up photo of a face, {attr} mouth"
  attributes: [smiling, "slightly open", "with red lipstick"]
mask: {shape: exact, dilate_px: 14}
generator: {adapter: flux2_klein_inpaint, res: 512, context_scale: 2.5}
labeler: {adapter: face_parsing}
output: {dir: out/myfaces_mouth}
n_samples: 100
```

For eyes or other parts copied from another face, use `operation: replace`,
`conditioning: {type: reference_object}` and `generator: {adapter: mimicbrush, steps: 25}`
(see `configs/faces_eyes_mimicbrush.yaml`). Faces should be roughly frontal and fill most of the
frame.

### Occlusion (on by default)

Every scene operation keeps whatever is in front of the edited object in front. A car behind a pole,
regenerated, replaced or removed, stays behind the pole. At plan time (`synthgen/occlusion.py`,
Depth Anything V2 Small) the pipeline marks pixels in the edit region to keep, from three kinds of
evidence:
1. **other annotated objects:** their visible pixels, with the dataset's exact edges. Those in
   front of the target are occluders; those behind are protected, so their labels stay valid.
2. **gaps in the target's outline that are closer than it:** the pole splitting a car's
   outline, followed above and below the car.
3. **clearly closer regions crossing the region border.**

The generator paints the whole region, occluders included: a hole in its mask made models paint
clutter into the hole (a crouching figure next to a skateboarder, fragments between horses' legs).
Afterwards the original scene goes back on top (`occlusion.composite`):
- `silhouette` (default): for a regenerated or replaced object with occluders, new pixels are kept only
  inside the object's original outline (+ `silhouette_dilate_px`); everything else in the region, the
  pole, hedge or person in front included, comes back from the original, even where depth found nothing.
- `detected`: only the detected occluders come back, so the new object may extend past the old outline.

`masks/` (the manipulation label) excludes the restored pixels. The target's own
pixels are never kept, and ground just in front of its contact line is ignored. Because other
objects are protected, the planner no longer skips targets for overlapping them (crowd regions still
do). `occluders/<task>.png` shows what was kept. A new object can't grow over a protected object
behind it.

Objects seen *through* a chain-link fence, mesh or railing can't be separated by depth, so the planner
skips them: SigLIP 2 (Apache-2.0) scores the target's crop against "a {cls} seen through a chain-link
fence / behind a wire mesh fence / behind a metal railing / seen through a net". Targets scoring
`see_through_threshold` (0.8) or more are skipped, and the next candidate is drawn. `plan` prints them
under `see_through_skipped`, and each task records its `see_through_score`. On COCO val2017 cars this
caught 9 of 11 such cases with no false positives; adding "seen through a wet window" caught the
wet-window car too (plus one car behind a dark window and one borderline rainy windscreen). It can still
miss a dark windscreen fence, so look at the close-ups. The same check keeps such instances out of the
reference bank (`references_seen_through` in the plan output): a reference behind a fence or a wet
window made AnyDoor draw a ghosted car with fence lines, or a car tilted to the window's viewpoint.
Examples: `configs/coco_car_occluded_regenerate.yaml`, `configs/coco_car_occluded_remove.yaml`.

### Add camera-like degradation to any job

```yaml
postprocess:
  ops:
    - {op: downscale, p: 0.5, factor: [0.4, 0.8]}
    - {op: blur, p: 0.3, sigma: [0.3, 1.2]}
    - {op: noise, p: 0.5, sigma: [1, 6]}
    - {op: jpeg, p: 0.8, quality: [40, 85]}
```

Then give your real training images the same treatment, so the detector can't tell real from
synthetic by image quality:

```bash
.venv/bin/python -m synthgen.degrade configs/mydata_car_insert.yaml data/mydata/images data/mydata/images_degraded
```

---

## 5. Reading the outputs

```
out/<job>/
  annotations.json   COCO file of accepted samples (scene jobs)
  face_labels.json   per-image parse maps and edit masks (face jobs)
  images/            synthetic images (JPEG)
  masks/             region that was changed per image: ground truth for manipulation detection
  labels/            recovered object mask (scene) or full face parse map (face)
  raw/               clean output before post-processing (only if postprocess is set)
  occluders/         pixels kept from the original: things in front of the edit and other objects' visible parts
  review.jpg         contact sheet from synthgen.review
  summary.json       accepted / rejected counts with reasons
  manifest.jsonl     everything per task: target, prompt, reference, model, seed, timings, verdict
  tasks.jsonl, job.json, generation.jsonl, verdicts.jsonl   plan and stage records
```

In `annotations.json`, images carry `synthetic: true`, `source_image_id` and `task_id`.
Unchanged objects are copied from the source with `source_ann_id`. The edited object has
`synthetic: true` and an `edit` record (operation, prompt, reference, model, detector score), and
its box and mask come from re-detection, not from the inpainting mask.

Load it like any COCO file:

```python
from pycocotools.coco import COCO
coco = COCO("out/mydata_car_swap/annotations.json")
edited = [a for a in coco.dataset["annotations"] if a["synthetic"]]
```

Merge with real data by concatenating COCO files (renumber image and annotation ids). **Keep the
test set real:** evaluate a model trained on real + synthetic against real images only.

---

## 6. Tuning and troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `plan` makes fewer tasks than `n_samples` | Filters: `truncated` (object touches border), `overlaps_other` | Raise `targets.max_overlap_other`, set `targets.skip_truncated: false`, or use a bigger source |
| `no_reference` skips, or the same reference used repeatedly | Small reference pool | Bigger source split; lower `conditioning.reference.min_area`; raise `max_uses` |
| Many `no 'X' detected` rejections | Object too small or blurry in context, or generator failed | Raise `targets.min_area`; try another generator; check `review.jpg` |
| Many `ambiguous: 'truck' vs 'car'` rejections | Detector wording differs from your class names | Add `verify.phrases: {truck: [pickup truck, truck]}`; or lower `verify.margin` |
| Verification accepts things that look wrong | It checks presence and position, not realism | Human spot checks with `review.jpg` and full-resolution close-ups |
| Removed object left a floating racket, skis or bike | The person was holding or riding it | `targets.exclude_near` (see template A) |
| Removed object left a shadow or reflection | The edit region covers the object only | Larger `mask.dilate_px`; reject by eye; avoid water and low-sun scenes for removal |
| `unlabelled` entries in verdicts | The source dataset missed instances of the class | `verify.unlabelled: reject` for hard negatives; otherwise they are just flagged |
| Edited object drawn over a pole, fence post or person | Occluder not found | Check `occluders/<task>.png`; lower `occlusion.gap_tol` for faint depth differences; keep `occlusion.composite: silhouette` |
| Inserted car with a motion-blur trail, smear or second background in its box | Region too tight, or `verify.object_only` off | Keep `placement.slack` (0.25) and `verify.object_only: true` |
| Inserted car looks pasted into a blurred or long-exposure scene | Unsuitable scene | Keep `placement.min_sharpness` and `scene_skip_threshold` on; raise them to skip more |
| Recolour changes the vehicle's shape or model | `strength` too high | Lower to ~0.65 with `steps: 8`; keep `mask.shape: exact` |
| Recolour does nothing, or many `colour barely changed` rejections | `strength` too low, or only 4 steps | `steps: 8`, raise `strength` towards 0.7 |
| Object seen through a mesh fence comes out in front of the mesh | The see-through check missed it | Lower `occlusion.see_through_threshold` (e.g. 0.6; more false skips), or add a phrase to `occlusion.see_through_phrases` |
| Planner skips targets that aren't behind a fence | See-through false positive | Check `see_through_skipped` in the plan output; raise `occlusion.see_through_threshold` |
| Reference-based output has wrong lighting or angle | Poor reference match | `reference.match: visual` |
| A model is suddenly very slow | macOS is swapping | Check `sysctl vm.swapusage`; close other apps; one job at a time |
| `was planned with a different job file` | Job file changed after outputs were made | New `output.dir`, or `--replan` (deletes that dir's previous outputs) |
| `unknown key(s)` | Typo in the job file | Use the key names in section 7 |

Thresholds to tune (defaults in brackets): `verify.min_score` [0.35] detector confidence,
`verify.min_inside` [0.7] share of the detected object inside the edit region, `verify.min_fill`
[0.15] how much of the region the object must fill, `verify.margin` [0.1] how far the target class
must beat the contrast class.

---

## 7. Config reference

Unknown keys are errors. Only `name`, `domain`, `source`, `operation`, `generator` and `output`
are required.

| Key | Default | Meaning |
|---|---|---|
| `name` | required | Job name; prefixes task ids |
| `domain` | required | `scene` or `face` |
| `operation` | required | `remove`, `regenerate`, `replace`, `change_class`, `insert` (face: `regenerate`, `replace`) |
| `n_samples` | 10 | Number of tasks to plan |
| `seed` | 0 | Controls target selection, prompts, references and generation |
| `change_to` | [] | Target classes for `change_class` |
| **source** | | |
| `source.images` | required | Image folder |
| `source.annotations` | "" | COCO json (required for `format: coco`) |
| `source.format` | coco | `coco` or `images` |
| `source.split` | unspecified | Recorded in outputs; references only come from this source |
| `source.download_missing` | false | Fetch missing images from `coco_url` (COCO itself) |
| **targets** | | |
| `targets.classes` | [] | Classes (scene) or face parts to edit; empty = all (scene) |
| `targets.policy` | specified | `specified`, `random`, `rare_weighted` (favours rare classes) |
| `targets.min_area` | 4000 | Minimum object area in pixels |
| `targets.max_overlap_other` | 0.05 | Max share of another object the edit may cover (only with occlusion off; with it on, other objects are protected) |
| `targets.max_per_image` | 1 | Tasks per source image |
| `targets.skip_truncated` | true | Skip objects cut off by the image border |
| `targets.image_ids` | [] | Only edit these source images (empty = all) |
| `targets.crowd` | exclude | `exclude`, `include`, `only`: whether `iscrowd` regions (one mask over a group of instances) can be targets. Remove only. A crowd region touching any other target's edit always skips that target |
| `targets.exclude_near` | [] | Skip targets whose box (grown by `exclude_near_margin`, 0.1) touches these classes. For removal: things a person holds or rides |
| **conditioning** | | |
| `conditioning.type` | text | `none`, `text`, `reference_object` |
| `conditioning.prompt` | "a photo of a {attr} {cls}" | Text prompt template |
| `conditioning.attributes` | [""] | Values sampled for `{attr}` |
| `conditioning.reference.match` | aspect | `random`, `aspect`, `aspect_light`, `visual` |
| `conditioning.reference.min_area` | 16000 | Minimum reference object area |
| `conditioning.reference.max_uses` | 2 | Max tasks per reference instance |
| `conditioning.reference.white_bg` | true | Cut the reference object out onto white |
| `conditioning.reference.pad` | 0.1 | Padding around the reference crop |
| **mask** | | |
| `mask.shape` | exact | `exact`, `bbox`, `convex_hull` |
| `mask.dilate_px` | 12 | Grow the edit region by this many pixels |
| **generator** | | |
| `generator.adapter` | required | Model name (`bench list`) |
| `generator.res` | model default | Working resolution of the crop |
| `generator.steps`, `.strength`, `.guidance` | model defaults | Sampling settings |
| `generator.context_scale` | 2.0 | Crop side = edit region size x this |
| `generator.feather_px` | 8 | Soft edge when pasting the result back |
| `generator.backend` | model default | `mps`, `mlx`, `cpu` |
| **labeler** | | |
| `labeler.adapter` | gdino_sam2 | `gdino_sam2`, `sam3`, `face_parsing` |
| **verify** | | |
| `verify.min_score` / `min_inside` / `min_fill` / `margin` | 0.35 / 0.7 / 0.15 / 0.1 | See section 6 |
| `verify.contrast` | auto | Classes the result must not look like; auto = old class for `change_class` |
| `verify.phrases` | {} | Detector wording per class |
| `verify.removal_max_inside` | 0.5 | `remove`: object counts as still present above this |
| `verify.face_area_ratio` | [0.4, 2.5] | Face: allowed size change of the edited part |
| `verify.min_color_change` | 0 (off) | Reject when the edit's colour change (p75 CIELAB delta E in the edited area) is below this; 12 for recolouring |
| `verify.unlabelled` | flag | `off`, `flag`, `reject`: detect the class over the whole image and flag/reject instances the source dataset didn't annotate (score >= `unlabelled_min_score`, 0.5) |
| **placement** (insert) | | |
| `placement.support_classes` | [] | Classes used to judge position and size; empty = the inserted class |
| `placement.ground_phrases` | [] | Ground the object must stand on; empty = no check |
| `placement.ground_min_frac` | 0.6 | Share of the object's bottom strip on the ground |
| `placement.y_jitter` / `size_jitter` / `gap` / `min_height_px` | 0.08 / 0.15 / 0.05 / 24 | Placement spread, free-space margin, minimum size |
| `placement.slack` | 0.25 | Generation region = the box widened by this share of its width each side and height on top |
| `placement.min_sharpness` | 10 | Skip placements whose surroundings are out of focus (mean abs. Laplacian); 0 = off |
| `placement.scene_skip_threshold` / `scene_skip_phrases` | 0.7 / 4 phrases | Skip scenes SigLIP 2 rates as blurred, long-exposure or bokeh; 0 = off |
| `verify.object_only` | true | Insert: keep only the detected new object and its shadow; the rest of the region stays original |
| **occlusion** (scene) | | |
| `occlusion.enabled` | true | Keep what is in front of the edited object in front; protect other annotated objects' visible pixels |
| `occlusion.gap_tol` | 0.05 | Relative depth margin for annotated objects and outline gaps to count as in front |
| `occlusion.tol` | 0.3 | Larger margin for depth-only occluders crossing the region border |
| `occlusion.min_frac` | 0.005 | Ignore occluder pieces smaller than this share of the edit region |
| `occlusion.composite` | silhouette | Occluded regenerate/replace: `silhouette` keeps new pixels only inside the old outline; `detected` restores only detected occluders |
| `occlusion.silhouette_dilate_px` | 4 | Growth of the old outline for `silhouette` |
| `occlusion.see_through_skip` | true | Skip targets, and references, seen through a fence, mesh, railing or wet window (SigLIP 2) |
| `occlusion.see_through_threshold` | 0.8 | Skip when the "seen through" phrases get this share of probability |
| `occlusion.see_through_phrases` | 5 phrases | Prompts with `{cls}`; the planner compares them with plain "a photo of a {cls}" prompts |
| **postprocess** | | |
| `postprocess.ops` | [] | List of `{op, p, ...}`: `downscale` (factor), `blur` (sigma), `noise` (sigma), `color` (brightness, contrast, saturation), `jpeg` (quality) |
| `postprocess.repeat` | 1 | 2 = apply the chain twice (stronger, second-order degradation) |
| **output** | | |
| `output.dir` | required | Output folder |
| `output.jpeg_quality` | 95 | Quality of saved images |

---

## 8. Command reference

```bash
# jobs
.venv/bin/python -m synthgen.run plan  <job.yaml>                   # validate + task list + estimate
.venv/bin/python -m synthgen.run run   <job.yaml>                   # all stages, resumable
.venv/bin/python -m synthgen.run run   <job.yaml> --stage generate  # or verify / export
.venv/bin/python -m synthgen.run run   <job.yaml> --replan          # re-plan; deletes previous outputs in output.dir
.venv/bin/python -m synthgen.review    out/<job> --max 12           # contact sheet -> out/<job>/review.jpg

# models
.venv/bin/python -m synthgen.bench list                              # models, operations, licences
.venv/bin/python -m synthgen.bench run <model> --n 5                 # measure one model
.venv/bin/python -m synthgen.bench run flux2_klein_inpaint --res 512 --duration 1200   # sustained (thermal) check
.venv/bin/python -m synthgen.bench all                               # measure everything

# compare models on identical tasks (same images, objects, masks, prompts, references, seeds)
.venv/bin/python -m synthgen.compare configs/compare_all.yaml                 # every job, ~3 h
.venv/bin/python -m synthgen.compare configs/compare_all.yaml --only coco_car_remove
.venv/bin/python -m synthgen.compare configs/compare_all.yaml --sheets-only   # rebuild sheets + summary
# -> out/compare/<job>/<model>/ (a normal job output), out/compare/<job>/sheet_*.jpg, out/compare/summary.md

# data utilities
.venv/bin/python -m synthgen.degrade <job.yaml> <in_dir> <out_dir>   # job's degradation on real images
.venv/bin/python tools/make_synthetic_faces.py data/faces_synth --n 12
.venv/bin/python tools/convert_anydoor_ckpt.py <ckpt> ~/.cache/synthgen/anydoor_fp16.safetensors
```

Worked examples you can run as-is (COCO annotations required, images download on demand):

```bash
.venv/bin/python -m synthgen.run run configs/coco_car_remove.yaml                 # LaMa, ~1 min
.venv/bin/python -m synthgen.run run configs/coco_car_regenerate.yaml             # FLUX.2, ~2 min
.venv/bin/python -m synthgen.run run configs/coco_vehicle_recolor.yaml           # FLUX.2 recolour, ~2 min
.venv/bin/python -m synthgen.run run configs/coco_car_regenerate_qwen21.yaml      # Qwen-Image-2.1, ~4 min, non-commercial
.venv/bin/python -m synthgen.run run configs/coco_car_insert_occlusion.yaml       # insert + depth + degradation
.venv/bin/python -m synthgen.run run configs/coco_train_car_replace_anydoor.yaml  # AnyDoor, train2017, visual references
.venv/bin/python -m synthgen.run run configs/faces_mouth_flux_text.yaml           # needs data/faces_synth
```
