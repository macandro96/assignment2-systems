import argparse
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.data import get_batch
import numpy as np
import torch
import time
import torch.distributed as dist
import os
import torch.multiprocessing as mp
from cs336_systems.ddp.ddp_overlap_individual_parameters import DDPIndividualParameters

NUM_LAYERS = 12
D_MODEL = 768
D_FF = 3072
NUM_HEADS = 12
CONTEXT_LENGTH = 512
VOCAB_SIZE = 10_000
ROPE_THETA = 10000.0

BATCH_SIZE = 8

CONFIGS = {
    "small": {
        "num_layers": 12,
        "d_model": 768,
        "d_ff": 3072,
        "num_heads": 12,
    },
    "medium": {
        "num_layers": 24,
        "d_model": 1024,
        "d_ff": 4096,
        "num_heads": 16,
    },
    "large": {
        "num_layers": 36,
        "d_model": 1280,
        "d_ff": 5120,
        "num_heads": 20,
    },
    "xl": {
        "num_layers": 48,
        "d_model": 1600,
        "d_ff": 6400,
        "num_heads": 25,
    }
}
ARTIFACTS_DIR = "artifacts/naive_ddp_benchmarking"

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
    return device

def initialize_model(
    num_layers,
    d_model,
    d_ff,
    num_heads,
    context_window,
    vocab_size,
    rope_theta=ROPE_THETA,
):
    print(f"Initializing model with {num_layers} layers, d_model={d_model}, d_ff={d_ff}, num_heads={num_heads}")
    model = BasicsTransformerLM(
        num_layers=num_layers,
        d_model=d_model,
        d_ff=d_ff,
        num_heads=num_heads,
        context_length=context_window,
        vocab_size=vocab_size,
        rope_theta=rope_theta,
    )
    return model

def create_dummy_data(batch_size, seq_length, vocab_size, n_steps):
    # Ensure the dataset length is strictly greater than `seq_length` so
    # `get_batch` can sample valid starting indices (it needs at least
    # `seq_length + 1` items to form x and y sequences).
    total_len = max(n_steps * batch_size * seq_length, seq_length + 1)
    data = np.random.randint(0, vocab_size, size=(total_len,))

    X, Y = get_batch(data, batch_size, seq_length, device='cpu')
    print(f"Created dummy data with shape X: {X.shape}, Y: {Y.shape}")
    
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    
    with open(os.path.join(ARTIFACTS_DIR, "X.pt"), "wb") as f:
        torch.save(X, f)
    with open(os.path.join(ARTIFACTS_DIR, "Y.pt"), "wb") as f:
        torch.save(Y, f)

def load_data():
    """Loads data for training."""
    inputs = torch.load(open(os.path.join(ARTIFACTS_DIR, "X.pt"), "rb"))
    outputs = torch.load(open(os.path.join(ARTIFACTS_DIR, "Y.pt"), "rb"))
    
    return inputs, outputs

def broadcast_parameters(model):
    for param in model.parameters():
        dist.broadcast(param.data, src=0)
        
def train(rank, world_size, backend, model_cfg, vocab_size, return_dict):
    """This will run distributed and sequential training"""
    # setup ddp
    device = setup(rank, world_size, backend)
    
    dist.barrier()
    torch.manual_seed(rank)
    
    # init model
    ddp_model = initialize_model(
        **CONFIGS[model_cfg],
        context_window=CONTEXT_LENGTH, vocab_size=vocab_size
    ).to(device)
    ddp_model = DDPIndividualParameters(ddp_model)
    
    ddp_optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.01)
    
    loss_fn = torch.nn.CrossEntropyLoss()
    inputs, outputs = load_data()
    
    batch_size = inputs.shape[0]
    
    num_batches_ddp = batch_size // world_size  # if 4 gpus, effective batch size is 8 per gpu to make total batch size 32
    
    ddp_step_times = []
    ddp_comm_times = []

    # train ddp model
    print(f"Rank {rank}: Starting DDP training")
    for _ in range(5):
        for i in range(0, inputs.size(0), batch_size):
            # rank = 0: 
            #   start = 0, 32, 64, ..
            #   end = 8, 40, ..
            # rank = 1:
            #   start = 8, 40, ..
            #   end = 16, 48, ..
            
            # need to synchronize before and after to get accurate timing
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            start = i + rank * num_batches_ddp
            end = i + (rank + 1) * num_batches_ddp
            
            input_batch = inputs[start: end].to(device)
            output_batch = outputs[start: end].to(device)
            
            ddp_optimizer.zero_grad()
            predictions = ddp_model(input_batch)
            loss = loss_fn(predictions.view(-1, predictions.size(-1)), output_batch.view(-1))
            loss.backward()
            
            comm_time = ddp_model.finish_gradient_synchronization()
            
            ddp_comm_times.append(comm_time * 1000)  # in ms
            ddp_optimizer.step()
            
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            ddp_step_times.append((end_time - start_time) * 1000)  # in ms
            
    print(f"Rank {rank}: Finished DDP training")
    
    # cleanup
    dist.barrier()
    dist.destroy_process_group()
    
    return_dict[rank] = {
        "ddp_step_times": ddp_step_times,
        "ddp_comm_times": ddp_comm_times,
    }
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_cfg",
        type=str,
        choices=CONFIGS.keys(),
        default="small",
        help="Model configuration to use.",
    )
    parser.add_argument(
        "--world_size",
        type=int,
        default=2,
        help="Number of processes for DDP.",
    )
    parser.add_argument(
        "--context_length",
        type=int,
        default=CONTEXT_LENGTH,
        help="Context length for the model.",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=VOCAB_SIZE,
        help="Vocabulary size for the model.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=100,
        help="Number of steps for dummy data creation.",
    )
    args = parser.parse_args()
    
    # prepare
    create_dummy_data(
        batch_size=args.batch_size,
        seq_length=args.context_length,
        vocab_size=args.vocab_size,
        n_steps=args.n_steps,
    )
    print("Dummy data created for benchmarking.")
    
    manager = mp.Manager()
    return_dict = manager.dict()
    
    mp.spawn(
        train,
        args=(args.world_size, 'nccl', args.model_cfg, args.vocab_size, return_dict),
        nprocs=args.world_size,
        join=True,
    )
    
    print(return_dict)
    
    # get min, average, median p90, max times across all ranks
    all_broadcast_times = []
    all_ddp_step_times = []
    all_ddp_comm_times = []
    for rank in range(args.world_size):
        result = return_dict[rank]
        all_ddp_step_times.extend(result["ddp_step_times"])
        all_ddp_comm_times.extend(result["ddp_comm_times"])
    
    def compute_stats(times):
        times_sorted = sorted(times)
        n = len(times_sorted)
        avg_time = sum(times_sorted) / n
        median_time = times_sorted[n // 2] if n % 2 == 1 else (times_sorted[n // 2 - 1] + times_sorted[n // 2]) / 2
        p90_time = times_sorted[int(0.9 * n) - 1]
        max_time = times_sorted[-1]
        min_time = times_sorted[0]
        return {
            "min": min_time,
            "avg": avg_time,
            "median": median_time,
            "p90": p90_time,
            "max": max_time,
        }
    
    ddp_step_stats = compute_stats(all_ddp_step_times)
    ddp_comm_stats = compute_stats(all_ddp_comm_times)
    
    print("DDP Step Time Stats (ms):", ddp_step_stats)
    print("DDP Communication Time Stats (ms):", ddp_comm_stats)
