import torch
import torch.nn as nn
import os
import torch.distributed as dist
from copy import deepcopy
import torch.multiprocessing as mp

BATCH_SIZE = 32


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 10, bias=False)
        self.fc2 = nn.Linear(10, 50, bias=False)
        self.fc3 = nn.Linear(50, 5, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

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

def get_data():
    """Generates random data for training."""
    inputs = torch.load(open("artifacts/ddp_train/X.pt", "rb"))
    outputs = torch.load(open("artifacts/ddp_train/Y.pt", "rb"))
    
    return inputs, outputs

def broadcast_parameters(model):
    for param in model.parameters():
        dist.broadcast(param.data, src=0)
    
def ddp_train(rank, world_size, backend='nccl'):
    """This will run both distributed and data parallel training and we will validate parameters are in fact the same."""
    # setup ddp
    device = setup(rank, world_size, backend)
    
    dist.barrier()
    torch.manual_seed(rank)
    
    # init model
    model = ToyModel()
    model = model.to(device)

    ddp_model = deepcopy(model).to(device)  # clone so that the weights are the same across models
    broadcast_parameters(ddp_model)
        
    if rank == 0:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    ddp_optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.01)
    
    loss_fn = nn.MSELoss()
    inputs, outputs = get_data()
    
    num_batches_ddp = BATCH_SIZE // world_size  # if 4 gpus, effective batch size is 8 per gpu to make total batch size 32
    
    # train non ddp model
    if rank == 0:
        print(f"Rank {rank}: Starting non-DDP training")
        for i in range(0, inputs.size(0), BATCH_SIZE):
            input_batch = inputs[i:i+BATCH_SIZE].to(device)
            output_batch = outputs[i:i+BATCH_SIZE].to(device)
            
            optimizer.zero_grad()
            predictions = model(input_batch)
            loss = loss_fn(predictions, output_batch)
            loss.backward()
            optimizer.step()

    torch.cuda.synchronize()
    dist.barrier()
    
    print(f"Rank {rank}: Starting DDP training")
    # train ddp model
    for i in range(0, inputs.size(0), BATCH_SIZE):
        start, end = i + rank * num_batches_ddp, i + (rank + 1) * num_batches_ddp
        input_batch = inputs[start: end].to(device)
        output_batch = outputs[start: end].to(device)
        
        preds = ddp_model(input_batch)
        loss = loss_fn(preds, output_batch)
        ddp_optimizer.zero_grad()
        loss.backward()
        
        torch.cuda.synchronize()
        for param in ddp_model.parameters():
            dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, async_op=False)
            
        torch.cuda.synchronize()    
        ddp_optimizer.step()
    
    # compare parameters
    torch.cuda.synchronize()
    if rank == 0:
        match_param = True
        for param, ddp_param in zip(model.parameters(), ddp_model.parameters()):
            if not torch.allclose(param.data, ddp_param.data, atol=1e-6):
                print(param.data, ddp_param.data)
                match_param = False
                print(f"Rank {rank}: Parameters do not match!")
                break
        if match_param:
            print(f"Rank {rank}: Parameters match!")
    torch.cuda.synchronize()
    dist.barrier()
    dist.destroy_process_group()
    
if __name__ == "__main__":
    world_size = 4
    backend = 'nccl'
    
    mp.spawn(
        ddp_train,
        args=(world_size, backend),
        nprocs=world_size,
        join=True
    )
    