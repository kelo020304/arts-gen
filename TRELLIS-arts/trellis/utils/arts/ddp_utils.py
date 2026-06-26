"""Distributed-training setup wrapper. Wraps trellis.utils.dist_utils.setup_dist
with the env-var conventions used by the 4 stage trainers (RANK / WORLD_SIZE /
MASTER_ADDR / MASTER_PORT / LOCAL_RANK)."""
import os
import torch
import torch.distributed as dist

from trellis.utils.dist_utils import setup_dist as _setup_dist


def setup_ddp() -> tuple[int, int, int]:
    """Initialize torch.distributed from env vars. Returns (rank, local_rank, world_size).

    Reads from torchrun-injected env vars:
      RANK, WORLD_SIZE, LOCAL_RANK, MASTER_ADDR (default 127.0.0.1),
      MASTER_PORT (default 29500).

    After this call, torch.cuda.set_device(local_rank) is set and dist is initialized.
    """
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    master_port = os.environ.get("MASTER_PORT", "29500")

    _setup_dist(rank, local_rank, world_size, master_addr, master_port)
    return rank, local_rank, world_size


def is_main_process() -> bool:
    return int(os.environ.get("RANK", 0)) == 0
