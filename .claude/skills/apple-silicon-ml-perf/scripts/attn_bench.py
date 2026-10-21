"""Microbenchmark scaled_dot_product_attention on MPS: sequence length x dtype, plain vs query-chunked.

Run it with the same Python environment as the slow model. Use it to check whether attention at the
model's real sequence length is the bottleneck, and whether chunking or fp16 helps.

    <env>/bin/python attn_bench.py                          # defaults: 4096 and 8192 tokens, 8 heads x 40 dims
    <env>/bin/python attn_bench.py --lengths 1024 4096 16384 --heads 8 --dim 64 --batch 2

Measured on an M5 Pro: 8192 tokens took 0.17 s in fp16 but 3.0 s in fp32, so an fp32 path at long
sequence lengths is a cliff. Inside a real model the same call ran 3x slower when the process was
swapping: check swap before optimising kernels.
"""

import argparse
import time

import torch
import torch.nn.functional as F


def bench(fn, n=5):
    fn()
    torch.mps.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.mps.synchronize()
    return (time.perf_counter() - t) / n * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", type=int, nargs="+", default=[4096, 8192])
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--dim", type=int, default=40)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=2048)
    a = ap.parse_args()
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS not available in this environment")
    print(f"torch {torch.__version__}; batch {a.batch}, heads {a.heads}, head dim {a.dim}; times in ms")
    print(f"{'tokens':>7s} {'dtype':>8s} {'plain':>9s} {'chunked':>9s}")
    for L in a.lengths:
        for dt in (torch.float16, torch.bfloat16, torch.float32):
            q = torch.randn(a.batch, a.heads, L, a.dim, device="mps", dtype=dt)
            plain = bench(lambda: F.scaled_dot_product_attention(q, q, q))
            chunked = bench(lambda: torch.cat([F.scaled_dot_product_attention(q[:, :, i:i + a.chunk], q, q)
                                               for i in range(0, L, a.chunk)], 2))
            print(f"{L:7d} {str(dt).split('.')[-1]:>8s} {plain:9.1f} {chunked:9.1f}", flush=True)
            del q
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
