"""MimicBrush worker. Runs in envs/mimicbrush (diffusers 0.23, transformers 4.25), not the main env.

Mirrors the model setup of third_party/MimicBrush/run_gradio3_demo.py with these changes:
  * device is configurable (mps or cuda), not hard-coded "cuda"
  * the depth model is built here instead of via depthanything/fast_import.py (which calls .cuda())
  * weights load with torch.load(weights_only=True): tensors only, no pickle code execution
  * nothing loads at import time; protocol output goes to the real stdout, library prints to stderr

Protocol: argv[1] = JSON config; then one JSON request per stdin line, one JSON reply per stdout line.
"""

import functools
import json
import os
import sys
import time
import traceback

REPO = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party", "MimicBrush")

_proto = sys.stdout
sys.stdout = sys.stderr


_t0 = time.perf_counter()


def log(msg):
    print(f"[mimicbrush {time.perf_counter() - _t0:7.1f}s] {msg}", file=sys.stderr, flush=True)


def reply(**msg):
    _proto.write(json.dumps(msg) + "\n")
    _proto.flush()


def main():
    cfg = json.loads(sys.argv[1])
    sys.path[:0] = [REPO, os.path.join(REPO, "depthanything")]
    os.chdir(REPO)

    import numpy as np
    import torch
    import torch.nn.functional as F
    import cv2
    from PIL import Image

    _torch_load = torch.load

    def safe_load(f, *a, **kw):
        """Tensors only (no pickle code), memory-mapped where the file format allows, so a 9.4 GB
        fp32 checkpoint is paged in from disk instead of being held in RAM while copied to fp16."""
        kw["weights_only"] = True
        try:
            return _torch_load(f, *a, mmap=True, **kw)
        except RuntimeError:     # legacy (non-zip) format can't be memory-mapped
            return _torch_load(f, *a, **kw)

    torch.load = safe_load

    # The reference attention runs over image+reference tokens (8192 at 64x64 latents). MPS builds the
    # full attention matrix for such calls and keeps it cached, which pushed this process to 18 GB and
    # into swap on a 24 GB Mac. Chunking the queries bounds each temporary to ~0.5 GB at equal speed.
    _sdpa = F.scaled_dot_product_attention

    def chunked_sdpa(q, k, v, *a, **kw):
        if q.shape[-2] * k.shape[-2] <= 4096 * 4096 or a or kw.get("attn_mask") is not None:
            return _sdpa(q, k, v, *a, **kw)
        return torch.cat([_sdpa(q[..., i:i + 2048, :], k, v, **kw) for i in range(0, q.shape[-2], 2048)], dim=-2)

    F.scaled_dot_product_attention = chunked_sdpa

    from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
    from diffusers.image_processor import VaeImageProcessor
    from torchvision.transforms import Compose
    from depth_anything.dpt import DepthAnything
    from depth_anything.util.transform import NormalizeImage, PrepareForNet, Resize
    from models.depth_guider import DepthGuider
    from models.pipeline_mimicbrush import MimicBrushPipeline
    from models.ReferenceNet import ReferenceNet
    from mimicbrush import MimicBrush_RefNet

    device, dtype = cfg["device"], torch.float16
    w = cfg["weights"]

    log("imports done")
    depth_model = DepthAnything({"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]})
    depth_model.load_state_dict(torch.load(w["depth_model"], map_location="cpu"))
    depth_model = depth_model.eval()
    transform = Compose([
        Resize(width=518, height=518, resize_target=False, keep_aspect_ratio=True, ensure_multiple_of=14,
               resize_method="lower_bound", image_interpolation_method=cv2.INTER_CUBIC),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

    scheduler = DDIMScheduler(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012,
                              beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False, steps_offset=1)
    log("depth model loaded")
    vae = AutoencoderKL.from_pretrained(w["vae"]).to(dtype=dtype)
    unet = UNet2DConditionModel.from_pretrained(w["sd_inpainting"], subfolder="unet", in_channels=13,
                                                low_cpu_mem_usage=False, ignore_mismatched_sizes=True).to(dtype=dtype)
    log("inpainting unet loaded")
    pipe = MimicBrushPipeline.from_pretrained(w["sd_inpainting"], torch_dtype=dtype, scheduler=scheduler, vae=vae,
                                              unet=unet, feature_extractor=None, safety_checker=None)
    pipe.set_progress_bar_config(disable=True)
    log("pipeline loaded")
    referencenet = ReferenceNet.from_pretrained(w["sd_reference"], subfolder="unet").to(dtype=dtype)
    log("reference net loaded; loading mimicbrush checkpoint")
    model = MimicBrush_RefNet(pipe, w["image_encoder"], w["mimicbrush"], depth_model, DepthGuider(), referencenet, device)
    import gc
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()
    mask_processor = VaeImageProcessor(vae_scale_factor=1, do_normalize=False, do_binarize=True, do_convert_grayscale=True)

    def pad_square(im, fill):
        W, H = im.size
        if W == H:
            return im, (0, 0)
        s = max(W, H)
        out = Image.new("RGB", (s, s), fill)
        off = ((s - W) // 2, (s - H) // 2)
        out.paste(im, off)
        return out, off

    log("ready")
    reply(ready=True)
    for line in sys.stdin:
        try:
            r = json.loads(line)
            t0 = time.perf_counter()
            image = Image.open(r["image"]).convert("RGB")
            mask = Image.open(r["mask"]).convert("L").point(lambda v: 255 if v > 127 else 0)
            ref, _ = pad_square(Image.open(r["reference"]).convert("RGB"), "white")
            target, off = pad_square(image, "white")
            mask_sq, _ = pad_square(mask.convert("RGB"), "black")
            # masked region blacked out, as in the demo's collage_region
            m = (np.asarray(mask_sq) > 128).astype(np.uint8)
            collage = Image.fromarray(np.asarray(target) * (1 - m))
            depth = transform({"image": np.asarray(target)})["image"]
            depth = torch.from_numpy(depth).unsqueeze(0) / 255 * (1.0 if r.get("shape_control") else 0.0)
            mask_pt = mask_processor.preprocess(mask_sq, height=512, width=512)
            with torch.inference_mode():
                pred, _ = model.generate(pil_image=ref, depth_image=depth, num_samples=1,
                                         num_inference_steps=r.get("steps") or 50, seed=int(r["seed"]),
                                         image=collage, mask_image=mask_pt, strength=1.0,
                                         guidance_scale=r.get("guidance") or 5.0)
            out = pred[0].resize(target.size, Image.LANCZOS).crop((off[0], off[1], off[0] + image.width, off[1] + image.height))
            out.save(r["out"])
            reply(ok=True, out=r["out"], seconds=round(time.perf_counter() - t0, 3))
        except Exception as e:
            traceback.print_exc()
            reply(ok=False, error=f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        traceback.print_exc()
        reply(ready=False, error=f"{type(e).__name__}: {e}")
