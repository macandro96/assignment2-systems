import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
import time

def setup(rank, world_size, backend="gloo"):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def distributed_demo(rank, world_size, backend, size_mb=1, time_dict=None):
    setup(rank, world_size)
    
    device = torch.device("cpu")
    if backend == "nccl":
        device = torch.device(f"cuda:{rank}")
            
    num_elements = size_mb * 1024**2 // 4  # number of float32 elements to make up size_mb MB
    tensor = torch.randn(num_elements, dtype=torch.float32, device=device)  # approximately size_mb MB

    print(f"Rank {rank} data before `all_reduce` {tensor}")
    if backend == 'nccl':
        torch.cuda.synchronize()
    start = time.perf_counter()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=False)
    if backend == 'nccl':
        torch.cuda.synchronize()
    end = time.perf_counter()
    print(f"Rank {rank} data after `all_reduce` {tensor}")
    
    time_dict[rank] = (end - start) * 1000  # return time in milliseconds

def run(backend="gloo", size_mb=1, num_processes=4, warmup_iters=2):
    world_size = num_processes
    manager = mp.Manager()
    return_dict = manager.dict()

    print("Running DDP with backend:", backend)
    
    print("Warming up...")
    # Warmup runs
    for _ in range(warmup_iters):
        mp.spawn(
            distributed_demo,
            args=(world_size, backend, size_mb, manager.dict()),
            nprocs=world_size,
            join=True
        )
    
    mp.spawn(
        distributed_demo,  # fn to do ddp on
        args=(world_size, backend, size_mb, return_dict),  # fn is called as fn(rank, *args), 0 <= rank < world_size - 1
        nprocs=world_size,
        join=True
    )
    print(return_dict)
    return return_dict
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, choices=["gloo", "nccl"], default="gloo", help="Distributed backend")
    parser.add_argument("--data-size", type=int, default=1, help="size of data tensor in mb")
    parser.add_argument("--num-processes", type=int, default=4, help="number of processes to spawn")
    args = parser.parse_args()
    
    run(backend=args.backend, size_mb=args.data_size, num_processes=args.num_processes)