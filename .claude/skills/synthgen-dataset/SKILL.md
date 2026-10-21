---
name: synthgen-dataset
description: Create, run and quality-check synthetic labelled image datasets with this repo's synthgen pipeline (inpainting-based remove / regenerate / replace / change-class / insert of objects in COCO datasets, and eyes/mouth/hair edits on face image folders), ending with a verified COCO file or face label maps. Use this whenever the user wants synthetic or augmented training/benchmark data, wants objects or face parts removed, inserted, swapped, recoloured or changed in a dataset, mentions a synthgen job, config, YAML or output folder, asks why samples were rejected or how to get more/better samples, or wants to compare generators on the same tasks - even if they don't say "synthgen".
---

# Synthetic datasets with synthgen

synthgen turns a source dataset into a synthetic labelled dataset: it picks targets, edits them
with an inpainting model, re-detects the result to recover labels (the inpainting mask is never
trusted as the label), and exports COCO. The user guide `docs/USAGE.md` has the templates (§4),
outputs (§5), troubleshooting (§6) and every config key (§7); read the section you need instead of
guessing key names, because unknown keys are rejected.

Run everything from the repo root with `.venv/bin/python`. The machine is typically an Apple
Silicon Mac with ~17.8 GB usable GPU memory, so run **one job at a time**: two generation jobs at
once push macOS into swap and everything slows down 5x or more.

## Workflow

### 1. Pin down the goal
Translate the request into: domain (`scene` COCO dataset / `face` image folder), operation, target
classes or face parts, conditioning (none / text / reference), and how many samples. Pick a model
from the table in `docs/USAGE.md` §3. Defaults that worked well here:

| Goal | Operation | Generator | Notes |
|---|---|---|---|
| hard negatives | `remove` | `lama` | `targets.exclude_near`, `verify.unlabelled: reject` (template A) |
| new appearance from text | `regenerate` | `flux2_klein_inpaint`, `res: 512` | `mask.shape: convex_hull` |
| same vehicle, new colour | `regenerate` | `flux2_klein_inpaint`, `steps: 8, strength: 0.65` | `mask.shape: exact`, `verify.min_color_change: 12` |
| swap for another real instance | `replace` | `anydoor` | `reference.match: visual` |
| new object in free space | `insert` | `flux2_klein_inpaint` | `placement.ground_phrases` |
| class change | `change_class` | `flux2_klein_inpaint` | `mask.shape: bbox`, `verify.phrases` |
| face parts from text / another face | `regenerate` / `replace` | `flux2_klein_inpaint` / `mimicbrush` | `labeler: face_parsing` |
| baseline for any comparison | `replace` / `insert` | `copy_paste` | same seed = same tasks |
| second text model to compare | `regenerate` | `qwen_image21_edit`, `res: 512` (MLX, ~22 s, non-commercial licence) | `verify.min_color_change: 12`; never `qwen_image_edit_4bit` on a Mac (CUDA only) |

If the user's source data isn't COCO-format with instance masks (polygon or RLE), say so early. Scene jobs need it;
face jobs only need a folder of roughly frontal faces.

### 2. Look at the source data before writing the job
```bash
.venv/bin/python .claude/skills/synthgen-dataset/scripts/inspect_coco.py <annotations.json> --classes car truck
```
This shows how many instances can actually be targets (non-crowd, big enough, not cut off by the
border) and how many are clean enough to be references. Use it to set `n_samples` and
`targets.min_area` realistically. Reference-based jobs need a pool of clean references: COCO
val2017 has 4 usable cars, train2017 185. A tiny pool means repeated references and
near-duplicate samples.

### 3. Write the job YAML
Start from the closest template in `docs/USAGE.md` §4 or `configs/`, and save to
`configs/<name>.yaml` with its own `output.dir`. Ask for more samples than the user wants when
quality will be judged by eye. For removal, typically a quarter to a half survive a careful review,
so plan 2-4x. Keep jobs with the same `seed` and targets when
comparing generators: they then get identical tasks and references.

### 4. Plan, and read what it says
```bash
.venv/bin/python -m synthgen.run plan configs/<name>.yaml
```
- `selection.skipped` explains a shortfall (`truncated`, `overlaps_other`, `no_reference`,
  `no_support`, `no_free_space`). Fix with `docs/USAGE.md` §6 rather than lowering every threshold.
- `estimate` gives minutes and peak memory from this machine's benchmarks. A `warning` means it is
  close to the GPU limit: lower `generator.res`.
- `was planned with a different job file` means the YAML changed after outputs were made. Use a new
  `output.dir`, or `--replan` (it deletes that folder's previous outputs, so ask first if they matter).

### 5. Run
```bash
.venv/bin/python -m synthgen.run run configs/<name>.yaml
```
Resumable: rerunning continues where it stopped. For runs longer than a few minutes, run in the
background and check back. Speed is roughly: LaMa 0.1 s, copy_paste 0.2 s, FLUX.2 at 512 4-5 s,
AnyDoor 7-11 s, MimicBrush 35 s per image.

### 6. Review: look at every edit up close
```bash
.venv/bin/python .claude/skills/synthgen-dataset/scripts/job_report.py out/<name>
.venv/bin/python -m synthgen.review out/<name> --max 12                                   # overview
.venv/bin/python .claude/skills/synthgen-dataset/scripts/edit_closeups.py out/<name> --accepted-only
```
Use the contact sheet for an overview, but judge each sample on its full-resolution close-up
(`out/<name>/closeups/*.jpg`, before | after; Read each image). Thumbnails hide small defects:
in testing, a removed woman's orange reflection was still in the river, easy to miss on the sheet
and obvious in the close-up. Verification only checks that an object of the right class is (or,
for removal, isn't) detected in the edit region. It accepted all of these real failures:

- object placed somewhere physically wrong (car on a pier, car floating on floodwater)
- lighting mismatch (night reference in a day scene: a dark blob)
- wrong viewpoint (car photographed from above in a street-level scene)
- object painted over something that should be in front (fence, pole, hydrant)
- fragments copied from an occluded reference (umbrella ribbon on a car)
- removal: the object's **shadow or reflection** left behind; **fragments** (part of an arm) or smears
- removal: **something the person held or rode left floating** (tennis racket, ski poles, a riderless
  motocross bike in mid-air). That is a physically impossible scene, not a useful negative. Prevent
  it with `targets.exclude_near`
- removal: **another instance the dataset never labelled** (a person behind horses). In a hard-negative
  set this teaches the detector to ignore a real person. `verify.unlabelled: reject` catches most;
  look anyway

Be strict. A training image with an artefact teaches the artefact, and a borderline sample costs
more than it adds. Discard when unsure and say how many you discarded and why.

It also rejects some good samples, usually small or distant objects. Name the patterns you see and
map each to a setting: `reference.match: visual`, `occlusion` (`occluders/<task>.png` shows what was
kept in front; `gap_tol`), `placement.ground_phrases`,
`reference.min_area`, `targets.min_area`, `verify.phrases`.

### 7. Report back
Tell the user: what was generated (counts, verification acceptance, top rejection reasons), how many
you kept after looking at the close-ups and why the rest were discarded (with examples), where the outputs are (`annotations.json` or
`face_labels.json`, `images/`, `masks/`), and the next change you recommend. Two points to make
whenever the data is meant for training:
- keep the evaluation set real: compare a detector trained on real data vs real + synthetic, tested on held-out real images
- include the `copy_paste` baseline: passing verification doesn't show a generator beats cheap copy-paste

## Things that matter but aren't obvious
- **Detector vocabulary differs from dataset taxonomy.** COCO calls a pickup a "truck"; the
  open-vocabulary detector scores it as "car". For `change_class`, set
  `verify.phrases: {truck: [pickup truck, truck]}` and check a few samples before a large run.
- **Occlusion is handled by default** for scene jobs: poles, people, hydrants and other annotated
  objects in front of the target stay in front, and other objects' visible pixels are protected.
  Targets seen through a fence, mesh, railing or wet window are skipped at plan time
  (`see_through_skipped` in the plan output), and kept out of the reference bank. For occluded
  regenerate/replace, `occlusion.composite: silhouette` (default) keeps new pixels only inside the old
  outline. Inserts keep only the detected new object and its shadow (`verify.object_only`) from a
  region widened by `placement.slack`, and skip out-of-focus or long-exposure scenes. Check on the close-ups that it worked: the check can miss a car behind a wet window or
  a dark windscreen fence.
- **Exact object masks leak the old silhouette.** Use `convex_hull` or `bbox` with dilation for
  regenerate/replace/change_class.
- **Faces:** identity is not checked, and the face parser's licence is non-commercial. Say so if
  the user plans commercial use or face-recognition training.
- **Degradation:** if the target domain is CCTV or phone footage, add a `postprocess` block, and
  give the real training images the same treatment with
  `python -m synthgen.degrade <job.yaml> <in_dir> <out_dir>`, so real and synthetic can't be told
  apart by image quality.
- **New models** belong to the `synthgen-add-model` skill. **Slow runs or memory problems** belong to
  `apple-silicon-ml-perf`.
