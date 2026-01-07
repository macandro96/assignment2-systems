import argparse
import time

import numpy as np
import torch
from cs336_basics.data import get_batch
from cs336_basics.model import BasicsTransformerLM

NUM_LAYERS = 12
D_MODEL = 768
D_FF = 3072
NUM_HEADS = 12
CONTEXT_LENGTH = 1024
VOCAB_SIZE = 10_000
ROPE_THETA = 10000.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

def _dtype_from_str(dtype_str):
    if dtype_str == "float32":
        return torch.float32
    elif dtype_str == "float16":
        return torch.float16
    elif dtype_str == "bfloat16":
        return torch.bfloat16
    else:
        raise ValueError(f"Unsupported dtype string: {dtype_str}")

def initialize_model(
    num_layers,
    d_model,
    d_ff,
    num_heads,
    context_window,
    vocab_size,
    rope_theta,
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

    return get_batch(data, batch_size, seq_length, device=DEVICE)

def benchmark_model(
    model, x, y, n_steps, backward=False, warmup=2, dtype=torch.float32,
    profile_memory=True, memory_snapshot_path="memory_snapshot.pickle"
):
    model.eval()
    device_is_cuda = x.is_cuda
    
    # 1. Warmup
    # We use a separate loop for warmup to ensure caches are hot
    print("Warming up...")
    ctx = torch.enable_grad() if backward else torch.no_grad()
    with ctx:
        for _ in range(warmup):
            with torch.autocast(device_type="cuda", dtype=dtype):
                out = model(x)
                if backward:
                    loss = torch.nn.functional.cross_entropy(
                        out.view(-1, out.size(-1)), y.view(-1)
                    )
                    loss.backward()
                    model.zero_grad()
    
    if device_is_cuda:
        torch.cuda.synchronize()

    # 2. Memory Profiling (Optional - run ONLY if requested)
    # We run this separately because the profiler adds overhead that ruins timing benchmarks
    if profile_memory:
        print("Capturing memory snapshot...")
        torch.cuda.memory._record_memory_history(max_entries=100000)
        try:
            # Run one step for memory profiling
            with ctx:
                with torch.autocast(device_type="cuda", dtype=dtype, cache_enabled=False):
                    out = model(x)
                    if backward:
                        loss = torch.nn.functional.cross_entropy(
                            out.view(-1, out.size(-1)), y.view(-1)
                        )
                        loss.backward()
                        model.zero_grad()
            torch.cuda.memory._dump_snapshot(memory_snapshot_path)
            print(f"Memory snapshot saved to {memory_snapshot_path}")
        finally:
            torch.cuda.memory._record_memory_history(enabled=None)
        
        # If we just wanted memory, we might want to return early or not count this time
        # For this script, let's proceed to timing, but note that the memory test is done.

    # 3. Timing Benchmark
    print(f"Benchmarking for {n_steps} steps...")
    fwd_times = []
    bwd_times = []

    for step in range(n_steps):
        # Explicit synchronization before T0 is crucial to ensure 
        # previous async GPU work is done.
        if device_is_cuda:
            torch.cuda.synchronize()
        
        t0 = time.perf_counter()
        
        # Explicit context manager for the benchmark loop
        with ctx:
            with torch.autocast(device_type="cuda", dtype=dtype, cache_enabled=False):
                outputs = model(x)
        
        if device_is_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        fwd_times.append(t1 - t0)

        if backward:
            if device_is_cuda:
                torch.cuda.synchronize()
            tb0 = time.perf_counter()
            
            with ctx: # Re-enter context (enable_grad)
                with torch.autocast(device_type="cuda", dtype=dtype, cache_enabled=False):
                    loss = torch.nn.functional.cross_entropy(
                        outputs.view(-1, outputs.size(-1)), y.view(-1)
                    )
                loss.backward()
            
            if device_is_cuda:
                torch.cuda.synchronize()
            tb1 = time.perf_counter()
            bwd_times.append(tb1 - tb0)
            
            model.zero_grad()

    fwd_avg = np.mean(fwd_times)
    bwd_avg = np.mean(bwd_times) if backward else 0.0
    print(f"Average Forward Time per Step: {fwd_avg:.6f} seconds")
    if backward:
        print(f"Average Backward Time per Step: {bwd_avg:.6f} seconds")
    # ... print results ...
    return { "fwd_avg": fwd_avg, "bwd_avg": bwd_avg }
def benchmark(
    args
) -> dict:
    model = initialize_model(
        num_layers=CONFIGS[args.model_cfg]["num_layers"],
        d_model=CONFIGS[args.model_cfg]["d_model"],
        d_ff=CONFIGS[args.model_cfg]["d_ff"],
        num_heads=CONFIGS[args.model_cfg]["num_heads"],
        context_window=args.context_length,
        vocab_size=VOCAB_SIZE,
        rope_theta=args.rope_theta,
    ).to(DEVICE)
    print(f"Model dtype: {next(model.parameters()).dtype}")  # Reveals truth
    print("Model initialized.", model)
    
    x, y = create_dummy_data(
        batch_size=1,
        seq_length=args.context_length,
        vocab_size=VOCAB_SIZE,
        n_steps=args.n_steps,
    )
    print("Dummy data created.")
    print(f"Input shape: {x.shape}, Target shape: {y.shape}")
    
    times = benchmark_model(
        model,
        x,
        y,
        n_steps=args.n_steps,
        backward=args.backward,
        warmup=args.warmup_steps,
        dtype=_dtype_from_str(args.dtype),
        memory_snapshot_path=args.memory_snapshot_path,
    )
    
    return times
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmarking Script")
    # model config
    parser.add_argument(
        "--model_cfg",
        type=str,
        choices=CONFIGS.keys(),
        default="small",
        help="Model configuration to use.",
    )
    
    parser.add_argument(
        "--context_length",
        type=int,
        default=CONTEXT_LENGTH,
        help="Context length for the model.",
    )
    
    parser.add_argument(
        "--rope_theta",
        type=float,
        default=ROPE_THETA,
        help="RoPE theta parameter.",
    )

    # benchmarking config
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=2,
        help="Number of warmup steps before benchmarking (default: 2)",
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=1,
        help="num of steps to run the benchmark",
    )
    parser.add_argument(
        "--backward",
        action="store_true",
        help="Whether to run backward pass during benchmarking",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        default="float32",
        help="Data type for model parameters and computations.",
    )
    parser.add_argument(
        "--memory_snapshot_path",
        type=str,
        default="memory_snapshot.pickle",
        help="path to memory snapshot"
    )

    args = parser.parse_args()

    benchmark(
        args
    )
