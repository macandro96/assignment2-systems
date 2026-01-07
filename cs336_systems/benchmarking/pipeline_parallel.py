import argparse
import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from cs336_basics.model import BasicsTransformerLM

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

VOCAB_SIZE = 10_000
CONTEXT_LENGTH = 1024
ROPE_THETA = 10000.0


class PipelineParallel(nn.Module):
    def __init__(
        self,
        model: nn.Module
    ):
        super().__init__()
        
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        
        # pipeline of layers
        self.layers = self._pipeline(model.layers)
        
        # embeddings should be in first stage
        if self.rank == 0:
            self.emb_layer =  model.token_embeddings
        
        # last 2 layers in last stage
        if self.rank == dist.get_world_size() - 1:
            self.ln_final = model.ln_final
            self.lm_head = model.lm_head
    
    def _pipeline(self, layers):
        # assumption: num_layers is divisible by world_size
        num_layers_per_device = [
            len(layers) // self.world_size
            for _ in range(self.world_size)
        ]
        start_index = sum(num_layers_per_device[: self.rank])
        end_index = start_index + num_layers_per_device[self.rank]
        
        return layers[start_index: end_index]

    def forward(self, x):
        if self.rank == 0:
            x = self.emb_layer(x)

        for layer in self.layers:
            x = layer(x)
        
        if self.rank == dist.get_world_size() - 1:
            x = self.lm_head(self.ln_final(x))
        
        return x

    def backward(self, input_tensor, output_tensor, output_tensor_grad):
        if input_tensor is not None:
            # torch does not store grad for non-leaf tensors
            # `retain_grad` makes it do so; we can send grad to prev stage
            input_tensor.retain_grad()
        
        if output_tensor_grad is None:
            # in the last stage this var is None
            output_tensor_grad = torch.ones_like(
                output_tensor, memory_format=torch.preserve_format
            )
        
        torch.autograd.backward(
            output_tensor,
            grad_tensors=output_tensor_grad,
            retain_graph=False,  # graph freed; not reused later
            create_graph=False,  # no need to store higher order gradients
        )
        return input_tensor.grad if input_tensor else None

def pipeline_comm(
    comm_type: str,
    device,
    dtype,
    tensor_shape=None,
    tensor=None,
    
):
    rank = dist.get_rank()
    if comm_type == 'send':
        assert tensor is not None
        dist.send(tensor, dst=rank + 1)
        return None

    elif comm_type == 'recv':
        assert tensor_shape is not None
        tensor = torch.empty(tensor_shape, device=device, dtype=dtype, requires_grad=True)
        dist.recv(tensor, src=rank-1)
        return tensor

def train(
    model,
    device,
    tensor_shape,
    dtype=torch.float32,
):
    if dist.get_rank() == 0:
        input = torch.randint(
            low=0,
            high=VOCAB_SIZE,
            size=(tensor_shape[0], CONTEXT_LENGTH)
        ).to(device)
    else:
        input = pipeline_comm(comm_type="recv", device=device, dtype=dtype, tensor_shape=tensor_shape)
    
    out = model(input)
    
    if dist.get_rank() < dist.get_world_size() - 1:
        pipeline_comm(comm_type="send", device=device, dtype=dtype, tensor=out)

    # backward pass
    # compute loss if last stage
    if dist.get_rank() == dist.get_world_size() - 1:
        labels = torch.randint(
            low=0,
            high=VOCAB_SIZE,
            size=(tensor_shape[0], CONTEXT_LENGTH),
        ).to(device)
        out = F.cross_entropy(out.view(-1, VOCAB_SIZE), labels.view(-1))
        output_tensor_grad = None
    
    else:
        # receive gradients if not last stage
        output_tensor_grad = pipeline_comm(
            comm_type="recv",
            device=device,
            dtype=dtype,
            tensor_shape=tensor_shape,
        )
    
    # do backward pass: compute gradients
    input_tensor_grad = model.backward(
        input,
        out,
        output_tensor_grad
    )
    
    # send gradients if not first stage
    if dist.get_rank() > 0:
        pipeline_comm(
            comm_type="send",
            device=device,
            dtype=dtype,
            tensor=input_tensor_grad
        )

def run(args: argparse.Namespace, device: torch.device):
    model_size = args.model_cfg
    
    config = CONFIGS[model_size]
    
    model = BasicsTransformerLM(
        num_layers=config["num_layers"],
        d_model=config["d_model"],
        d_ff=config["d_ff"],
        num_heads=config["num_heads"],
        context_length=CONTEXT_LENGTH,
        vocab_size=VOCAB_SIZE,
        rope_theta=ROPE_THETA,
    ).to(device)
    
    model = PipelineParallel(model)
    
    train(
        model=model,
        device=device,
        tensor_shape=(2, CONTEXT_LENGTH, config["d_model"]),
        dtype=torch.float32,
    )
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_cfg",
        type=str,
        choices=CONFIGS.keys(),
        default="small",
        help="Model configuration to use.",
    )
    args = parser.parse_args()
    
    dist.init_process_group(backend="nccl", init_method="env://")
    
    local_rank = int(os.environ.get("LOCAL_RANK"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    
    run(args, device)