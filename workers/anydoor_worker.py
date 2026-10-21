"""AnyDoor worker. Runs in envs/anydoor (pytorch_lightning 1.5, transformers 4.25), not the main env.

Uses third_party/AnyDoor unmodified, patched at runtime:
  * DINOv2-g is built with pretrained=False and the separate DINOv2 weight file is skipped: its
    weights are inside the AnyDoor checkpoint (otherwise the constructor downloads 4.5 GB)
  * encoder/sampler device defaults are "cuda"; they are moved to the configured device instead
  * CrossAttention uses torch's fused scaled_dot_product_attention (no xformers on macOS)
  * weights come from a safetensors file converted once from the Lightning .ckpt
    (tools/convert_anydoor_ckpt.py), so no pickle is loaded here
process_pairs / crop_back are re-implemented from run_inference.py (MIT, DAMO Vision Intelligence
Lab), because importing that module loads the model and moves it to CUDA.

Protocol: argv[1] = JSON config; one JSON request per stdin line, one JSON reply per stdout line.
"""

import json
import os
import sys
import tempfile
import time
import traceback

REPO = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party", "AnyDoor")

_proto = sys.stdout
sys.stdout = sys.stderr
_t0 = time.perf_counter()


def log(msg):
    print(f"[anydoor {time.perf_counter() - _t0:7.1f}s] {msg}", file=sys.stderr, flush=True)


def reply(**msg):
    _proto.write(json.dumps(msg) + "\n")
    _proto.flush()


def main():
    cfg = json.loads(sys.argv[1])
    sys.path.insert(0, REPO)
    os.chdir(REPO)

    import functools
    import cv2
    import einops
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from safetensors.torch import load_file

    device = cfg["device"]

    import ldm.modules.attention as attn
    import ldm.modules.encoders.modules as enc

    def sdpa_forward(self, x, context=None, mask=None):
        h = self.heads
        context = x if context is None else context
        q, k, v = (t.view(t.shape[0], t.shape[1], h, -1).transpose(1, 2)
                   for t in (self.to_q(x), self.to_k(context), self.to_v(context)))
        out = F.scaled_dot_product_attention(q, k, v)
        return self.to_out(out.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1))

    attn.CrossAttention.forward = sdpa_forward

    enc.hubconf.dinov2_vitg14 = functools.partial(enc.hubconf.dinov2_vitg14, pretrained=False)
    empty = os.path.join(tempfile.mkdtemp(), "empty.pth")
    torch.save({}, empty)
    enc.DINOv2_weight_path = empty
    enc.FrozenDinoV2Encoder.__init__.__defaults__ = ("cpu", True)

    from cldm.model import create_model
    from cldm.ddim_hacked import DDIMSampler
    from datasets.data_utils import box2squre, box_in_box, expand_bbox, expand_image_mask, get_bbox_from_mask, pad_to_square, sobel

    def register_buffer(self, name, value):
        if isinstance(value, torch.Tensor):
            value = value.to(device, dtype=torch.float32)
        setattr(self, name, value)

    DDIMSampler.register_buffer = register_buffer
    log("patched; building model")

    model = create_model("./configs/anydoor.yaml").cpu()
    # networks in fp16 (~4.8 GB instead of 9.6); the top-level diffusion schedule buffers stay fp32
    for sub in (model.model, model.control_model, model.cond_stage_model, model.first_stage_model):
        sub.half()
    sd = load_file(cfg["weights"])
    missing, unexpected = model.load_state_dict(sd, strict=False)
    del sd
    import gc
    gc.collect()
    log(f"weights loaded: {len(missing)} missing, {len(unexpected)} unexpected keys")
    if missing:
        log(f"first missing: {missing[:5]}")
    model = model.to(device).eval()
    model.cond_stage_model.device = device
    sampler = DDIMSampler(model)
    model.control_scales = [1.0] * 13

    def process_pairs(ref_image, ref_mask, tar_image, tar_mask):
        y1, y2, x1, x2 = get_bbox_from_mask(ref_mask)
        m3 = np.stack([ref_mask] * 3, -1)
        masked_ref = (ref_image * m3 + np.ones_like(ref_image) * 255 * (1 - m3))[y1:y2, x1:x2]
        ref_mask = ref_mask[y1:y2, x1:x2]
        masked_ref, ref_mask = expand_image_mask(masked_ref, ref_mask, ratio=1.2)
        m3 = np.stack([ref_mask] * 3, -1)
        masked_ref = cv2.resize(pad_to_square(masked_ref, pad_value=255, random=False), (224, 224)).astype(np.uint8)
        m3 = cv2.resize(pad_to_square(m3 * 255, pad_value=0, random=False), (224, 224)).astype(np.uint8)
        ref_mask = m3[:, :, 0]
        collage_ref = sobel(masked_ref, ref_mask / 255)

        tar_box = expand_bbox(tar_mask, get_bbox_from_mask(tar_mask), ratio=[1.1, 1.2])
        crop_box = box2squre(tar_image, expand_bbox(tar_image, tar_box, ratio=[1.5, 3]))
        y1, y2, x1, x2 = crop_box
        cropped = tar_image[y1:y2, x1:x2, :]
        y1, y2, x1, x2 = box_in_box(tar_box, crop_box)
        collage = cropped.copy()
        collage[y1:y2, x1:x2, :] = cv2.resize(collage_ref, (x2 - x1, y2 - y1))
        collage_mask = cropped.copy() * 0.0
        collage_mask[y1:y2, x1:x2, :] = 1.0
        H1, W1 = collage.shape[:2]
        collage = pad_to_square(collage, pad_value=0, random=False).astype(np.uint8)
        collage_mask = pad_to_square(collage_mask, pad_value=-1, random=False).astype(np.uint8)
        H2, W2 = collage.shape[:2]
        collage = cv2.resize(collage, (512, 512)).astype(np.float32) / 127.5 - 1.0
        collage_mask = (cv2.resize(collage_mask, (512, 512)).astype(np.float32) > 0.5).astype(np.float32)
        hint = np.concatenate([collage, collage_mask[:, :, :1]], -1)
        return masked_ref / 255, hint, (H1, W1, H2, W2), crop_box

    def crop_back(pred, tar_image, sizes, crop_box):
        H1, W1, H2, W2 = sizes
        y1, y2, x1, x2 = crop_box
        pred = cv2.resize(pred, (W2, H2))
        m = 5
        if W1 != H1:
            if W1 < W2:
                p1 = (W2 - W1) // 2
                pred = pred[:, p1:p1 + W1, :]
            else:
                p1 = (H2 - H1) // 2
                pred = pred[p1:p1 + H1, :, :]
        out = tar_image.copy()
        out[y1 + m:y2 - m, x1 + m:x2 - m, :] = pred[m:-m, m:-m]
        return out

    log("ready")
    reply(ready=True)
    for line in sys.stdin:
        try:
            r = json.loads(line)
            t0 = time.perf_counter()
            tar = np.asarray(Image.open(r["image"]).convert("RGB"))
            tar_mask = (np.asarray(Image.open(r["mask"]).convert("L")) > 127).astype(np.uint8)
            ref = np.asarray(Image.open(r["reference"]).convert("RGB"))
            if r.get("reference_mask"):
                ref_mask = (np.asarray(Image.open(r["reference_mask"]).convert("L")) > 127).astype(np.uint8)
            else:   # object on white background
                ref_mask = (ref.min(-1) < 245).astype(np.uint8)
            ref_in, hint, sizes, crop_box = process_pairs(ref, ref_mask, tar, tar_mask)
            torch.manual_seed(int(r["seed"]))
            control = einops.rearrange(torch.from_numpy(hint).float()[None].to(device), "b h w c -> b c h w")
            clip_in = einops.rearrange(torch.from_numpy(ref_in.copy()).float()[None].to(device), "b h w c -> b c h w")
            with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.float16, enabled=device in ("mps", "cuda")):
                cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning(clip_in)]}
                uncond = {"c_concat": [control],
                          "c_crossattn": [model.get_learned_conditioning([torch.zeros((1, 3, 224, 224))])]}
                samples, _ = sampler.sample(r.get("steps") or 50, 1, (4, 64, 64), cond, verbose=False, eta=0.0,
                                            unconditional_guidance_scale=r.get("guidance") or 4.5,
                                            unconditional_conditioning=uncond)
                x = model.decode_first_stage(samples)
            pred = np.clip((einops.rearrange(x, "b c h w -> b h w c") * 127.5 + 127.5).float().cpu().numpy()[0], 0, 255)
            out = crop_back(pred, tar.astype(np.float32), sizes, crop_box)
            Image.fromarray(out.astype(np.uint8)).save(r["out"])
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
