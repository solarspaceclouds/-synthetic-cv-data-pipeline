---
name: apple-silicon-ml-perf
description: Diagnose and fix slow, memory-hungry or crashing PyTorch-MPS / MLX model inference on Apple Silicon Macs (M1-M5, unified memory) - measure real memory including GPU buffers and worker processes, detect swap, time GPU work correctly, and apply the fixes that matter (memory-mapped checkpoint loading, fp16, chunked attention, MLX cache limits, CPU fallbacks, MPS vs MLX choice). Use whenever a model or pipeline on a Mac is unexpectedly slow, gets slower over time, runs out of memory, makes the machine sluggish, shows "MPS backend out of memory", or when deciding whether a model fits or whether MLX or PyTorch-MPS is faster - even if the user just says "it's really slow on my Mac".
---

# ML performance on Apple Silicon

On a Mac the GPU shares memory with everything else. A 24 GB machine gives the GPU about 17.8 GB
by default, and macOS needs room too. Most "slow model" problems on these machines are memory
problems in disguise. So **measure memory and swap first, optimise kernels last.**

Real case from this repo: a reference-attention inpainting model took 7 s per step. A profiler
blamed one Python function and attention looked 3x slower than in isolation. The actual cause was
19 GB of process memory pushing 8 GB into swap. Two memory fixes made it 5x faster (179 s -> 34 s per
image) with no kernel changes.

## 1. Measure (2 minutes)
```bash
sysctl vm.swapusage                                   # swap "used" growing during a run = trouble
sysctl -n hw.memsize iogpu.wired_limit_mb             # RAM; GPU limit (0 = default ~2/3-3/4 of RAM)
python3 .claude/skills/apple-silicon-ml-perf/scripts/mem_watch.py --match "<part of the command line>" -i 2
```
`mem_watch.py` reports *phys_footprint* (what Activity Monitor shows) for the process and its
children. That figure includes MPS and MLX GPU buffers; RSS and `torch.mps.current_allocated_memory()`
miss parts of it. Children matter when a model runs in a worker subprocess. Read the result:

| Observation | Meaning | Go to |
|---|---|---|
| footprint near or above the GPU limit, swap rising | doesn't fit | §2 memory fixes |
| big jump during *loading*, then drops | checkpoint read into RAM before conversion | mmap / fp16 loading |
| big jump on the *first generation call* | activations (often long attention) or lazy weight loading (MLX) | chunked attention, MLX cache |
| memory flat, still slow | compute | §3 compute checks |
| slows down over 10-30 min | thermal throttling (expect ~25% on a MacBook) | plan with sustained numbers |

## 2. Memory fixes, in order of payoff
1. **One heavy job at a time.** Two diffusion jobs on 24 GB will swap.
2. **Load checkpoints without holding them in RAM:** `torch.load(path, map_location="cpu", weights_only=True, mmap=True)`.
   `mmap` pages tensors in from disk. `weights_only` also blocks code execution from pickles. For
   huge training checkpoints, convert once to fp16 safetensors (keep precision-sensitive buffers,
   like diffusion schedules, in fp32).
3. **fp16/bf16 weights** (half of fp32). Cast before `.to("mps")`, and `gc.collect()` after building.
4. **Free stage-specific parts:** encode all prompts with the text encoder, delete it, then load the
   denoiser. That is how a 4B transformer plus an 8 GB text encoder fit in 24 GB.
5. **Chunk long attention.** MPS can hold a full attention matrix per call and keep it cached. Split
   queries into chunks of ~2048 when `q_len * k_len > 4096²`: same speed, bounded memory (example:
   `chunked_sdpa` in `workers/mimicbrush_worker.py`).
6. **MLX:** cap the buffer cache: `mx.set_cache_limit(2 * 2**30)`. Uncapped, it held 6-9 GB of freed
   buffers here. MLX loads lazily, so the first call carries the load and quantisation peak.
7. **Smaller working resolution:** 512 instead of 1024 cut one model from 16.6 GB to 9.7 GB and was 5x faster.
8. Raising the GPU limit (`sudo sysctl iogpu.wired_limit_mb=...`) is a last resort; it takes memory from macOS.

## 3. Compute checks
- **Time the GPU properly.** MPS is asynchronous: wrap a stage in `torch.mps.synchronize()` before
  and after timing. Python profilers blame whichever line happens to wait for the GPU (a boolean
  index, a `.cpu()`), not the work that is slow.
- **Attention cost at your sequence length and dtype:**
  `<env>/bin/python .claude/skills/apple-silicon-ml-perf/scripts/attn_bench.py --lengths 4096 8192`.
  On an M5 Pro, 8192 tokens took 0.17 s in fp16 and 4.4 s in fp32. Make sure long attention runs in fp16/bf16.
- **No xformers / flash-attention on macOS:** replace custom einsum attention with
  `torch.nn.functional.scaled_dot_product_attention`.
- **Silent CPU fallback:** `PYTORCH_ENABLE_MPS_FALLBACK=1` runs unsupported ops on the CPU. If CPU
  usage is high during GPU work, find the op (run without the flag to get the error).
- **Few-step models beat many-step ones:** a 4-step distilled model was 2x faster than a 25-step
  model with better output.

## 4. MPS or MLX?
Benchmark both on the real workload; don't assume. For FLUX.2-klein-4B, MLX 8-bit was slower than
PyTorch-MPS bf16 at 512 and ~11% faster at 1024. MLX earns its place when quantisation is what
makes a model fit (9B/20B models). Check argument semantics when porting between them (mflux's
`image_strength` is the inverse of diffusers' `strength`).

## 5. Report
Give the user the numbers before and after (seconds per item, peak GB, swap), the cause, and the
change made. Note what to watch next time (`mem_watch.py` during the first run of any new model).
