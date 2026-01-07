import os

import torch
import torch.distributed as dist


def setup(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12399"
    # https://discuss.pytorch.org/t/should-local-rank-be-equal-to-torch-cuda-current-device/150873/2
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        local_rank = None
        if device_count > 0:
            local_rank = rank % device_count
            torch.cuda.set_device(local_rank)
        else:
            raise ValueError("Unable to find CUDA devices.")
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"
    # initialize the process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    return device    # initialize the process group


import torch.distributed as dist


class ProcessGroupManager:
    def __init__(
        self,
        tp_size: int = 1,
        pp_size: int = 1
    ):
        world_size = dist.get_world_size()
        global_rank = dist.get_rank()
        self.local_rank = int(os.environ.get("LOCAL_RANK", global_rank % world_size))

        assert world_size == tp_size * pp_size
        
        self.grid = torch.arange(world_size).view(pp_size, tp_size)
        self.dp_rank, self.pp_rank, self.cp_rank, self.tp_rank = \
            (self.grid == global_rank).nonzero().flatten().tolist()

        self.tp_group = \
            dist.new_subgroups_by_enumeration([self.grid[p, :].tolist() for p in range(pp_size)])[0]
        
        self.pp_group = \
            dist.new_subgroups_by_enumeration([self.grid[:, t].tolist() for t in range(tp_size)])[0]
        
        tp_group_id = rank // tp_size
        tp_ranks = list(range(
            tp_group_id * tp_size,
            (tp_group_id + 1) * tp_size
        ))

        self.tp_group = dist.new_group(tp_ranks)
        
        pp_ranks = []
        for i in range(world_size // tp_size):
            pp_ranks.append(i * tp_size + (rank % tp_size))
        self.pp_group = dist.new_group(pp_ranks)
