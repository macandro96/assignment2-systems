from collections import defaultdict

import torch
import torch.distributed as dist


class ShardedOptimizer(torch.optim.Optimizer):
    def __init__(self, params, optimizer_cls, **kwargs):
        self.params = list(params)
        super().__init__(self.params, kwargs)
        self.world_size = dist.get_world_size()
        
        # assign each param to device
        numels = [(p.numel(), idx) for idx, p in enumerate(self.params)]
        all_numels = sum([numel for numel, _ in numels])
        device_numel = all_numels / self.world_size
        
        numels = sorted(numels)
        curr_numel_sum = 0
        self.devices = [0 for _ in range(len(numels))]
        curr_device = 0

        for numel, idx in numels:
            if curr_numel_sum + numel > device_numel:
                if curr_device + 1 < dist.get_world_size():
                    curr_device += 1
                curr_numel_sum = numel
            else:
                curr_numel_sum += numel
            
            self.devices[idx] = curr_device
        
        local_params = list(self.params[idx] for idx, i in enumerate(self.devices) if i == dist.get_rank())

        self.opt = optimizer_cls(
            local_params,
            **kwargs
        )
    
    def reduce_gradients(self):
        for dev_id, param in zip(self.devices, self.params):
            if param.grad is None:
                continue
            dist.reduce(param.grad, dst=dev_id, op=dist.ReduceOp.SUM)
            if dev_id != dist.get_rank():
                param.grad = None

            
            # if dev_id != dist.get_rank():
            #     # free if not the owner
            #     param.grad = None
    
    def step(self, closure=None, **kwargs):
        self.reduce_gradients()
        self.opt.step(closure=closure, **kwargs)
        
        # sync params across
        for dev_id, param in zip(self.devices, self.params):

            if dev_id == dist.get_rank():
                # broadcast
                buffer = param.data.detach()
            else:
                buffer = torch.empty_like(param.data)
            dist.broadcast(buffer, src=dev_id)
            
            if dev_id != dist.get_rank():
                with torch.no_grad():
                    param.data.copy_(buffer)

# class ShardedOptimizer(torch.optim.Optimizer):
#     def __init__(self, params, optimizer_cls, **kwargs):
#         params = list(params)
#         super().__init__(params, kwargs)
#         assert dist.is_initialized(), "Distributed package is not initialized"

#         self.world_size = dist.get_world_size()
#         # split params into chunks across world_size
#         chunk_size = len(params) // self.world_size
#         balance = len(params) - self.world_size * chunk_size

#         self.rank = dist.get_rank()
#         self.rank2param = defaultdict(list)
#         for rank in range(self.world_size):
#             self.rank2param[rank] = params[rank * chunk_size: (rank + 1) * chunk_size]

#         # add balance to rank 0
#         if balance > 0:
#             self.rank2param[0].extend(params[-balance:])
        
#         self.optimizer = optimizer_cls(self.rank2param[self.rank], **kwargs)

#     def step(self, closure=None, **kwargs):
#         self.optimizer.step(closure=closure, **kwargs)
#         # sync params
#         for rank in range(self.world_size):
#             params = self.rank2param[rank]
            
#             for p in params:
#                 if rank == self.rank:
#                     buf = p.detach()
#                 else:
#                     buf = torch.empty_like(p)
#                 dist.broadcast(buf, src=rank)
#                 if rank != self.rank:
#                     with torch.no_grad():
#                         p.copy_(buf)