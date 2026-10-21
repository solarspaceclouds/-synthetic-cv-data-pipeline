"""Generate synthetic portrait images (FLUX.2-klein-4B, Apache-2.0) as development data for face
jobs, so testing does not need real people's faces. The identities are not real people.

    .venv/bin/python tools/make_synthetic_faces.py data/faces_synth --n 12
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import random
from pathlib import Path

import torch
from diffusers import Flux2KleinPipeline

REPO = "black-forest-labs/FLUX.2-klein-4B"
AGES = ["young adult", "middle-aged", "elderly"]
PEOPLE = ["East Asian woman", "East Asian man", "South Asian woman", "South Asian man", "Malay woman",
          "Malay man", "white woman", "white man", "Black woman", "Black man"]
HAIR = ["short black hair", "long wavy hair", "curly hair", "a buzz cut", "shoulder-length straight hair", "grey hair"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    combos = list(itertools.product(AGES, PEOPLE, HAIR))
    rng.shuffle(combos)
    prompts = [f"a frontal head-and-shoulders passport-style photo of a {age} {p} with {h}, neutral expression, "
               f"looking at the camera, plain light background, even lighting, sharp focus"
               for age, p, h in combos[: a.n]]

    # encode prompts with only the text encoder resident, then free it (same pattern as the adapter)
    enc = Flux2KleinPipeline.from_pretrained(REPO, transformer=None, vae=None, torch_dtype=torch.bfloat16).to("mps")
    with torch.inference_mode():
        embeds = [enc.encode_prompt(p, device=torch.device("mps"))[0] for p in prompts]
    del enc
    gc.collect()
    torch.mps.empty_cache()

    pipe = Flux2KleinPipeline.from_pretrained(REPO, text_encoder=None, tokenizer=None, torch_dtype=torch.bfloat16).to("mps")
    pipe.set_progress_bar_config(disable=True)
    meta = []
    for i, (p, e) in enumerate(zip(prompts, embeds)):
        img = pipe(prompt_embeds=e, height=a.res, width=a.res, num_inference_steps=4, guidance_scale=1.0,
                   generator=torch.Generator("cpu").manual_seed(a.seed * 1000 + i)).images[0]
        name = f"face_{i:03d}.png"
        img.save(out / name)
        meta.append({"file_name": name, "prompt": p, "seed": a.seed * 1000 + i, "generator": REPO})
        print(name, flush=True)
    (out / "generation.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
