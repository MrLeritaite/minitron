"""Day 1 — distributed hello world.

Run (CPU, gloo backend, 4 processes; macOS-safe wrapper):
    scripts/run.sh 4 scripts/hello_dist.py

Each rank prints its rank/world size, then all ranks do one all_reduce so you can see
a collective actually move data: every rank contributes (rank + 1), so the result is
1 + 2 + ... + world_size on every rank.
"""
import os

import torch
import torch.distributed as dist


def main() -> None:
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}") if backend == "nccl" else torch.device("cpu")

    print(f"[rank {rank}/{world}] hello from pid={os.getpid()} backend={backend} device={device}", flush=True)

    t = torch.tensor([float(rank + 1)], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    expected = world * (world + 1) / 2
    assert t.item() == expected, (t.item(), expected)
    dist.barrier()
    if rank == 0:
        print(f"all_reduce OK: sum = {t.item():.0f} (expected {expected:.0f})", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
