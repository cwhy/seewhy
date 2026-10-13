"""Run this project's experiments on whichever GPUs are free, as they become free (same rules as projects/dust/scripts/gpu_queue.py).

The box is shared, so a GPU can be busy with someone else's job. Before every launch this asks `nvidia-smi` whether the
GPU is idle: under 1.5 GiB of memory in use, under 10% utilisation, and no compute process on it. It never starts two of its
own jobs on one GPU (two JAX processes on one GPU is how XLA runs out of memory), polls every 20 seconds, and starts the next
experiment on the first GPU that passes. Experiments run in the order given, so put the longest first.

    ~/.local/bin/uv run --no-sync python projects/dust-history/scripts/gpu_queue.py exp1

Detach it with `setsid -f`; progress goes to logs/gpu_queue.log, each experiment's own output to logs/<exp>.log.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
LOGS = PROJECT / "logs"
RUNNER = None
MAX_MEM_MIB, MAX_UTIL = 1500, 10
POLL_S = 20


def say(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)


def smi(*query: str) -> list[list[str]]:
    out = subprocess.run(["nvidia-smi", *query, "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    return [[c.strip() for c in line.split(",")] for line in out.strip().splitlines() if line.strip()]


def idle_gpus() -> list[int]:
    gpus = smi("--query-gpu=index,uuid,memory.used,utilization.gpu")
    busy_uuids = {row[0] for row in smi("--query-compute-apps=gpu_uuid")}
    return [int(i) for i, uuid, mem, util in gpus if int(mem) < MAX_MEM_MIB and int(util) < MAX_UTIL and uuid not in busy_uuids]


def main(experiments: list[str]) -> None:
    LOGS.mkdir(exist_ok=True)
    pending, running = list(experiments), {}          # running: gpu -> (name, Popen)
    say(f"queue: {pending}")
    while pending or running:
        for gpu, (name, proc) in list(running.items()):
            if proc.poll() is not None:
                say(f"{name} finished on GPU {gpu} (exit {proc.returncode})")
                del running[gpu]
                time.sleep(10)                         # let the driver release the memory before the next check
        free = [g for g in idle_gpus() if g not in running]
        while pending and free:
            gpu, name = free.pop(0), pending.pop(0)
            log = open(LOGS / f"queue_{name}.log", "w")
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
            script = PROJECT / f"experiments{name.removeprefix('exp')}.py"
            proc = subprocess.Popen([sys.executable, str(script)], stdout=log, stderr=subprocess.STDOUT, env=env,
                                    cwd=PROJECT.parents[1])
            running[gpu] = (name, proc)
            say(f"started {name} on GPU {gpu}")
            time.sleep(60)                             # a fresh process claims its memory slowly; do not hand its GPU to anyone
        if pending and not free:
            say(f"waiting: {len(pending)} queued, GPUs busy ({sorted(running)} mine)")
            time.sleep(POLL_S * 3)
        else:
            time.sleep(POLL_S)
    say("queue empty")


if __name__ == "__main__":
    main(sys.argv[1:])
