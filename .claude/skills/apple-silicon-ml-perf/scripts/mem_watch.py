"""Watch the memory macOS charges a process (and its children) plus system swap, over time.

Uses phys_footprint (what Activity Monitor calls "Memory"). On Apple Silicon that includes GPU
buffers allocated by PyTorch-MPS and MLX, which RSS does not. Standard library only.

    python3 mem_watch.py --pid 12345                 # until the process exits, every 2 s
    python3 mem_watch.py --match "synthgen.run" -i 5  # first process whose command line contains this
    python3 mem_watch.py --pid 12345 --duration 120

Swap growing while a model runs means the job no longer fits in memory: expect large slowdowns.
"""

import argparse
import ctypes
import os
import subprocess
import sys
import time


class _Rusage(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(f"f{i}", ctypes.c_uint64) for i in range(40)]


_libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
_PHYS_FOOTPRINT = 7          # index of ri_phys_footprint after uuid in rusage_info_v4
_LIFETIME_MAX = 28           # ri_lifetime_max_phys_footprint


def rusage(pid: int):
    r = _Rusage()
    if _libc.proc_pid_rusage(pid, 4, ctypes.byref(r)) != 0:
        return None
    return getattr(r, f"f{_PHYS_FOOTPRINT}"), getattr(r, f"f{_LIFETIME_MAX}")


def children(pid: int) -> list[int]:
    out = subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True).stdout.split()
    kids = [int(p) for p in out]
    for k in list(kids):
        kids += children(k)
    return kids


def swap_used_gb() -> float:
    s = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    used = s.split("used = ")[1].split("M")[0]
    return float(used) / 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int)
    ap.add_argument("--match", help="substring of the command line")
    ap.add_argument("-i", "--interval", type=float, default=2.0)
    ap.add_argument("--duration", type=float, default=0, help="seconds; 0 = until the process exits")
    a = ap.parse_args()
    pid = a.pid
    if a.match and not pid:
        # plain substring match (pgrep -f would treat the text as a regex)
        me = os.getpid()
        rows = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True).stdout.splitlines()
        found = [int(r.split(None, 1)[0]) for r in rows
                 if a.match in r.split(None, 1)[-1] and int(r.split(None, 1)[0]) != me and "mem_watch.py" not in r]
        if not found:
            sys.exit(f"no process matches {a.match!r}")
        pid = found[0]
    if not pid:
        ap.error("give --pid or --match")

    gb = 2**30
    swap0 = swap_used_gb()
    gpu = subprocess.run(["sysctl", "-n", "iogpu.wired_limit_mb"], capture_output=True, text=True).stdout.strip()
    mem = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout) / gb
    print(f"pid {pid}; machine {mem:.0f} GB, GPU wired limit {'default (~2/3-3/4 of RAM)' if gpu in ('', '0') else gpu + ' MB'}; "
          f"swap used at start {swap0:.2f} GB")
    print(f"{'t(s)':>6s} {'main GB':>8s} {'children GB':>12s} {'total GB':>9s} {'peak GB':>8s} {'swap +GB':>9s}")
    t0, peak = time.time(), 0.0
    while True:
        r = rusage(pid)
        if r is None:
            print("process exited")
            break
        kids = [(k, rusage(k)) for k in children(pid)]
        kid_gb = sum(x[0] for _, x in kids if x) / gb
        total = r[0] / gb + kid_gb
        peak = max(peak, total)
        print(f"{time.time() - t0:6.0f} {r[0] / gb:8.2f} {kid_gb:12.2f} {total:9.2f} {peak:8.2f} {swap_used_gb() - swap0:+9.2f}",
              flush=True)
        if a.duration and time.time() - t0 >= a.duration:
            break
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
