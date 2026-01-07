"""Convert model to tensor parallel."""

import argparse
import math
import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from cs336_basics.model import BasicsTransformerLM
from einops import einsum

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

class ColParallel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        """In col parallel forward activations flows through as is on both devices."""
        return input
    
    @staticmethod
    def backward(ctx, grad_output):
        """In backward pass, gradients get summed across both devices"""
        dist.all_reduce(grad_output, op=dist.ReduceOp.SUM)
        return grad_output

class RowParallel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        """In row parallel forward activations get all reduced across devices"""
        dist.all_reduce(input, op=dist.ReduceOp.SUM)
        return input
    
    @staticmethod
    def backward(ctx, grad_output):
        """In backward pass, gradients flow through as is across both devices"""
        return grad_output


def col_parallel_forward(input):
    return ColParallel.apply(input)

def row_parallel_forward(input):
    return RowParallel.apply(input)


def convert_to_tensor_parallel(
    model
):
    """Model architectures comprises of transformer blocks in layers.
    
    We convert linear layers in attention to tensor parallel layers.
    """
    
    def convert_layer(
        layer,
        weight_name,
        p_type
    ):
        lay_to_convert = getattr(layer, weight_name)
        if p_type == 'col':
            new_layer = ColParallelLinear(
                in_features=lay_to_convert.weight.shape[-1],
                out_features=lay_to_convert.weight.shape[-2],
            )
        elif p_type == 'row':
            new_layer = RowParallelLinear(
                in_features=lay_to_convert.weight.shape[-1],
                out_features=lay_to_convert.weight.shape[-2],
            )
        elif p_type == 'embedding':
            new_layer = EmbeddingParallel(
                vocab_size=lay_to_convert.weight.shape[0],
                d_model=lay_to_convert.weight.shape[1],
            )
        
        setattr(layer, weight_name, new_layer)
        
    _convert_layer = [
        ("attn", "q_proj", "col"),
        ("attn", "k_proj", "col"),
        ("attn", "v_proj", "col"),
        ("attn", "output_proj", "row"),
    ]
    layers = model.layers
    for layer in layers:
        for layer_type, weight_name, p_type in _convert_layer:
            att_layer = getattr(layer, layer_type)
            convert_layer(att_layer, weight_name, p_type)
    
    convert_layer(model, "token_embeddings", "embedding")
    
    return model


class ColParallelLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,   
    ):
        super().__init__()
        
        world_size = dist.get_world_size()
        
        
        self.out_parallel_size = out_features // world_size
        self.d_in, self.d_out = \
            in_features, out_features
        
        self.weight = nn.Parameter(
            torch.empty(self.out_parallel_size, in_features)
        )
        self._set_weights()
    
    def _set_weights(self):
        std = math.sqrt(2 / (self.d_in + self.d_out))
        weights = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(self.d_out, self.d_in), std=std, a=-3*std, b=3*std),
        )
        rank = dist.get_rank()
        start, end = rank * self.out_parallel_size, (rank + 1) * self.out_parallel_size
        self.weight.data = weights.data[start: end, :]
    
    
    def forward(self, x):
        x = col_parallel_forward(x)
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


class RowParallelLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,   
    ):
        super().__init__()
        
        world_size = dist.get_world_size()
        
        self.in_parallel_size = in_features // world_size
        self.d_in, self.d_out = \
            in_features, out_features
        
        self.weight = nn.Parameter(
            torch.empty(out_features, self.in_parallel_size)
        )
        self._set_weights()
    
    def _set_weights(self):
        std = math.sqrt(2 / (self.d_in + self.d_out))
        weights = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(self.d_out, self.d_in), std=std, a=-3*std, b=3*std),
        )
        rank = dist.get_rank()
        start, end = rank * self.in_parallel_size, (rank + 1) * self.in_parallel_size
        self.weight.data = weights.data[:, start: end]
    
    
    def forward(self, x):
        out = einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
        return row_parallel_forward(out)


class EmbeddingParallel(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        
        # vocab_size is a multiple of world_size; if not then what to do?
        vocab_per_rank = vocab_size // world_size
        
        self.vocab_start = rank * vocab_per_rank
        self.vocab_end = (rank + 1) * vocab_per_rank
        
        self.weight = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(vocab_per_rank, d_model), std=1, a=-3, b=3),
            requires_grad=True
        )
    
    def forward(self, input_ids):
        # scale input_ids by vocab_start
        mask = (input_ids >= self.vocab_start) & (input_ids < self.vocab_end)
        input_ids_scaled = input_ids.clone() - self.vocab_start
        input_ids_scaled[~mask] = 0

        # do forward wrt weight matrices
        out = self.weight[input_ids_scaled, :]
        out[~mask] = 0
        
        return row_parallel_forward(out)

def run(args: argparse.Namespace):
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
    )
    
    model = convert_to_tensor_parallel(model)
    model = model.to(torch.cuda.current_device())
    
    
    batch = torch.randint(
        low=0,
        high=VOCAB_SIZE,
        size=(8, CONTEXT_LENGTH)
    )
    labels = torch.randint(
        low=0,
        high=VOCAB_SIZE,
        size=(8, CONTEXT_LENGTH)
    ).to(torch.cuda.current_device())
    
    out = model(batch.to(torch.cuda.current_device()))
    loss = F.cross_entropy(
        out.view(-1, VOCAB_SIZE), labels.view(-1)
    )
    print(f"{loss.item()=}")
    loss.backward()

    
if __name__ == "__main__":
    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    parser = argparse.ArgumentParser("arguments for testing tensor parallelism")
    parser.add_argument(
        "--model_cfg",
        type=str,
        choices=CONFIGS.keys(),
        default="small",
        help="Model configuration to use.",
    )
    
    args = parser.parse_args()
    
    run(args)
    
    dist.destroy_process_group()