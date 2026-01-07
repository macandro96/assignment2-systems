from ddp_test import run
import torch

if __name__ == "__main__":
    backend = ["nccl", "gloo"]
    data_sizes = [1, 10, 100, 1000]  # in MB
    num_processes = [2, 4, 6]
    
    times_dict = {}
    
    for b in backend:
        for size in data_sizes:
            for nprocs in num_processes:
                if b == "nccl" and nprocs > torch.cuda.device_count():
                    print(f"Skipping test with backend={b}, data_size={size}MB, num_processes={nprocs} due to insufficient GPUs.")
                    continue
                print(f"\nRunning test with backend={b}, data_size={size}MB, num_processes={nprocs}")
                times = run(backend=b, size_mb=size, num_processes=nprocs)
                times_dict[(b, size, nprocs)] = times
                avg_time = sum(times.values()) / len(times)
                print(f"Average time for all_reduce with {size}MB data across {nprocs} processes using {b}: {avg_time:.2f} ms")

    print("\nSummary of all tests:")
    for config, times in times_dict.items():
        b, size, nprocs = config
        avg_time = sum(times.values()) / len(times)
        print(f"Backend: {b}, Data Size: {size}MB, Num Processes: {nprocs} => Average Time: {avg_time:.2f} ms")
    
    pickle_file = "ddp_performance_results.pkl"
    with open(pickle_file, "wb") as f:
        import pickle
        pickle.dump(times_dict, f)
    print(f"\nAll results have been saved to {pickle_file}")