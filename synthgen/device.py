"""Which backends this machine has, and the few calls that differ between PyTorch-MPS, CUDA and MLX."""

from __future__ import annotations

import platform

from synthgen.types import Backend


def available(backend: Backend) -> bool:
    if backend is Backend.CPU:
        return True
    if backend is Backend.MLX:
        if platform.system() != "Darwin":
            return False   # the MLX adapters use mflux, which is only installed on macOS
        try:
            import mlx.core  # noqa: F401
            return True
        except ImportError:
            return False
    import torch
    if backend is Backend.MPS:
        return torch.backends.mps.is_available()
    return torch.cuda.is_available()


def pick(backends: tuple[Backend, ...]) -> Backend:
    """The first of an adapter's backends that this machine has (the first listed if it has none)."""
    return next((b for b in backends if available(b)), backends[0])


def torch_device() -> str:
    """Device for the helper models outside adapters (depth, embedder, see-through scorer)."""
    return pick((Backend.CUDA, Backend.MPS, Backend.CPU)).value


def empty_cache(device: str) -> None:
    """Return the framework's cached GPU buffers, so the next model starts with that memory free."""
    if device == "mps":
        import torch
        torch.mps.empty_cache()
    elif device == "cuda":
        import torch
        torch.cuda.empty_cache()
    elif device == "mlx":
        import mlx.core as mx
        mx.clear_cache()
