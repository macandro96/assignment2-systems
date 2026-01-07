"""Runs benchmarking script with diff. configs"""
import os
from argparse import Namespace
from itertools import product

from tqdm import tqdm

from cs336_systems.benchmarking.benchmarking import benchmark


def grid_run():
    model_size = "xl"
    context_lengths = [128, 256, 512]
    backward = [False, True]
    precision = ["float32", "float16", "bfloat16"]
    memory_snapshot_dir = "cs336_systems/benchmarking/artifacts/"
    
    context_len_backward = list(product(context_lengths, backward, precision))
    for config in tqdm(context_len_backward):
        context_length, backward, precision = config
        args = Namespace(
            model_cfg=model_size,
            context_length=context_length,
            rope_theta=10000.0,
            warmup_steps=2,
            n_steps=10,
            backward=backward,
            dtype=precision,
            memory_snapshot_path=os.path.join(memory_snapshot_dir, f"ms_{context_length}_{backward}_{precision}.pickle")
        )
        times = benchmark(args)
        print(f"context={context_length}, backward={backward}, precision={precision}: {times}")
    

if __name__ == "__main__":
    grid_run()