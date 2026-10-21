"""Machine profile and memory measurement.

"Memory" is what a model is limited by, which differs by machine:

- Apple Silicon: GPU buffers live in unified memory and count towards the process's
  `phys_footprint` (what Activity Monitor shows as "Memory"), not towards RSS. So it is measured
  from `proc_pid_rusage`, which works the same for PyTorch-MPS and MLX.
- NVIDIA (Linux): the process's GPU memory from NVML, which includes the CUDA context and the
  allocator's cache. When NVML can't attribute GPU memory to the job's PIDs (in a container it
  often lists none, or host PIDs), the whole GPU's used memory is taken instead, which is right
  on a GPU the job has to itself.
- Anything else: resident set size.
"""

from __future__ import annotations

import ctypes
import os
import platform
import re
import resource
import subprocess
import sys
import threading
import time

import psutil

_RUSAGE_INFO_V4 = 4


class _RusageInfoV4(ctypes.Structure):
    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [
        (n, ctypes.c_uint64)
        for n in (
            "ri_user_time ri_system_time ri_pkg_idle_wkups ri_interrupt_wkups ri_pageins ri_wired_size "
            "ri_resident_size ri_phys_footprint ri_proc_start_abstime ri_proc_exit_abstime ri_child_user_time "
            "ri_child_system_time ri_child_pkg_idle_wkups ri_child_interrupt_wkups ri_child_pageins "
            "ri_child_elapsed_abstime ri_diskio_bytesread ri_diskio_byteswritten ri_cpu_time_qos_default "
            "ri_cpu_time_qos_maintenance ri_cpu_time_qos_background ri_cpu_time_qos_utility "
            "ri_cpu_time_qos_legacy ri_cpu_time_qos_user_initiated ri_cpu_time_qos_user_interactive "
            "ri_billed_system_time ri_serviced_system_time ri_logical_writes ri_lifetime_max_phys_footprint "
            "ri_instructions ri_cycles ri_billed_energy ri_serviced_energy ri_interval_max_phys_footprint "
            "ri_runnable_time"
        ).split()
    ]


_DARWIN = platform.system() == "Darwin"
_libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib") if _DARWIN else None
_nvml_mod = None


def _nvml():
    """The initialised pynvml module, or None without an NVIDIA driver."""
    global _nvml_mod
    if _nvml_mod is None:
        try:
            import pynvml
            pynvml.nvmlInit()
            _nvml_mod = pynvml
        except Exception:
            _nvml_mod = False
    return _nvml_mod or None


def _gpu_bytes(pids: list[int]) -> int:
    nv = _nvml()
    used, total_used = {}, 0
    for i in range(nv.nvmlDeviceGetCount()):
        h = nv.nvmlDeviceGetHandleByIndex(i)
        total_used += nv.nvmlDeviceGetMemoryInfo(h).used
        for p in nv.nvmlDeviceGetComputeRunningProcesses(h):
            used[p.pid] = used.get(p.pid, 0) + (p.usedGpuMemory or 0)
    return sum(used.get(p, 0) for p in pids) or total_used


def _rusage(pid: int | None = None) -> _RusageInfoV4:
    info = _RusageInfoV4()
    if _libc.proc_pid_rusage(pid or os.getpid(), _RUSAGE_INFO_V4, ctypes.byref(info)) != 0:
        raise OSError("proc_pid_rusage failed")
    return info


def footprint_bytes(pid: int | None = None) -> int:
    if _DARWIN:
        return _rusage(pid).ri_phys_footprint
    if _nvml():
        return _gpu_bytes([pid or os.getpid()])
    return psutil.Process(pid or os.getpid()).memory_info().rss


def total_footprint_bytes(child_pids=()) -> int:
    """This process plus worker subprocesses (adapters that run a model in another environment)."""
    if not _DARWIN and _nvml():
        return _gpu_bytes([os.getpid(), *child_pids])
    total = footprint_bytes()
    for pid in child_pids:
        try:
            total += footprint_bytes(pid)
        except (OSError, psutil.Error):
            pass
    return total


def lifetime_peak_bytes() -> int:
    """Main process only. On CUDA, the allocator's peak (without the ~0.5 GB CUDA context)."""
    if _DARWIN:
        return _rusage().ri_lifetime_max_phys_footprint
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        return torch.cuda.max_memory_reserved()
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024   # KiB on Linux


class PeakSampler:
    """Samples footprint_bytes in a background thread; use as a context manager around one phase."""

    def __init__(self, interval_s: float = 0.02, child_pids=lambda: ()):
        self.interval_s = interval_s
        self.child_pids = child_pids
        self.peak = 0
        self._stop = threading.Event()

    def _sample(self) -> int:
        return total_footprint_bytes(self.child_pids())

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self._sample())
            time.sleep(self.interval_s)

    def __enter__(self):
        self.peak = self._sample()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()
        self.peak = max(self.peak, self._sample())


def _sysctl(key: str) -> str:
    return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True).stdout.strip()


def machine_profile() -> dict:
    if not _DARWIN:
        return _linux_profile()
    gpu_cores = None
    sp = subprocess.run(["system_profiler", "SPDisplaysDataType"], capture_output=True, text=True).stdout
    if m := re.search(r"Total Number of Cores:\s*(\d+)", sp):
        gpu_cores = int(m.group(1))
    prof = {
        "chip": _sysctl("machdep.cpu.brand_string"),
        "cpu_cores": int(_sysctl("hw.ncpu") or 0),
        "gpu_cores": gpu_cores,
        "memory_gb": round(int(_sysctl("hw.memsize")) / 2**30, 1),
        "wired_limit_mb": int(_sysctl("iogpu.wired_limit_mb") or 0),  # 0 = macOS default
        "macos": platform.mac_ver()[0],
    }
    try:
        import mlx.core as mx
        info = mx.device_info()
        prof["gpu_recommended_working_set_gb"] = round(info["max_recommended_working_set_size"] / 2**30, 1)
        prof["mlx"] = mx.__version__
    except Exception:
        pass
    try:
        import torch
        prof["torch"] = torch.__version__
        prof["mps_available"] = torch.backends.mps.is_available()
    except Exception:
        pass
    return prof


def _linux_profile() -> dict:
    cpu = platform.processor()
    try:
        cpu = re.search(r"model name\s*:\s*(.+)", open("/proc/cpuinfo").read()).group(1)
    except (OSError, AttributeError):
        pass
    prof = {
        "cpu": cpu,
        "cpu_cores": os.cpu_count(),
        "memory_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "os": platform.platform(),
    }
    if nv := _nvml():
        h = nv.nvmlDeviceGetHandleByIndex(0)
        name = nv.nvmlDeviceGetName(h)
        prof["gpu"] = name.decode() if isinstance(name, bytes) else name
        prof["gpu_count"] = nv.nvmlDeviceGetCount()
        prof["gpu_memory_gb"] = round(nv.nvmlDeviceGetMemoryInfo(h).total / 2**30, 1)   # models run on GPU 0
        driver = nv.nvmlSystemGetDriverVersion()
        prof["nvidia_driver"] = driver.decode() if isinstance(driver, bytes) else driver
    try:
        import torch
        prof["torch"] = torch.__version__
        prof["cuda"] = torch.version.cuda
        prof["cuda_available"] = torch.cuda.is_available()
    except Exception:
        pass
    return prof


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def profile_id(prof: dict) -> str:
    if "chip" in prof:
        chip = prof["chip"].replace("Apple ", "").replace(" ", "-").lower()
        return f"{chip}-{prof['gpu_cores']}gpu-{int(prof['memory_gb'])}gb"
    if "gpu" in prof:
        gpu = _slug(prof["gpu"].replace("NVIDIA ", "").replace("GeForce ", ""))
        return f"{gpu}-{round(prof['gpu_memory_gb'])}gb"
    return f"{_slug(prof['cpu'])}-cpu-{int(prof['memory_gb'])}gb"
