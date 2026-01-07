import torch.distributed as dist
import torch
import torch.nn as nn
import time


class DDPIndividualParameters(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.handles = []
        # returns True if `init_process_group` has been called
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            # broadcast parameters
            for param in self.module.parameters():
                dist.broadcast(param.data, src=0)
                if param.requires_grad:
                    param.register_post_accumulate_grad_hook(self._hook)
                
    
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
    
    def _hook(self, param: torch.Tensor):
        if param.grad is not None:
            param.grad /= self.world_size  # average the gradients
            # `all_reduce` gradients once they are ready
            handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
            self.handles.append(handle)
    
    def finish_gradient_synchronization(self):
        start = time.perf_counter()
        # wait for all `all_reduce` operations to complete
        for handle in self.handles:
            handle.wait()
        self.handles.clear()
        end = time.perf_counter()
        return end - start
        

        
        