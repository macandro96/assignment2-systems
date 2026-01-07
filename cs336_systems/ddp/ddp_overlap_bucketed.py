import torch.distributed as dist
import torch
import torch.nn as nn
import time
from collections import defaultdict

# b1 = [pn, pn-1, pn-2, .. p1]  (bucket 1)
# b2 = [pm, pm-1, .. pk]       (bucket 2)
# ..
class DDPOverlapBucketed(nn.Module):
    def __init__(self, module: nn.Module, bucket_size_mb: float):
        """Initializes the DDP module with bucketed gradient synchronization.
        
        This buckets gradients into buckets of size `bucket_size_mb` megabytes
        and asynchronously communicates each bucket as it is ready in the backward pass.
        If all the parameters in the bucket are ready, the bucket is communicated.
        Each bucket comprises of multiple parameters whose total size is less than or equal to `bucket_size_mb`.
        """
        super().__init__()
        self.module = module
        self.handles = []
        self.bucket_size = bucket_size_mb * 1024 * 1024  # convert to bytes
        current_bucket_size = 0
        
        self.name2bucket = {}
        bucket_idx = 0
        
        # returns True if `init_process_group` has been called
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            
            # broadcast parameters
            for name, param in list(self.module.named_parameters())[::-1]:
                # first broadcast
                dist.broadcast(param.data, src=0)
                
                if param.requires_grad:
                    # calculate size of parameter
                    tensor_size = param.numel() * param.element_size()  # in bytes

                    if current_bucket_size + tensor_size > self.bucket_size:
                        # reset bucket_size and increment bucket_idx
                        current_bucket_size = 0
                        bucket_idx += 1
                    
                    # keep track of current bucket_size
                    current_bucket_size += tensor_size
                    # update bucket dictionary and mapping
                    self.name2bucket[name] = bucket_idx
                    
                    param.register_post_accumulate_grad_hook(self._hook)
            
            # bucket2names: mapping from bucket_idx to list of parameter names
            self.bucket2names = defaultdict(list)
            for name, parameter in list(self.module.named_parameters())[::-1]:
                if parameter.requires_grad:
                    b_idx = self.name2bucket[name]
                    self.bucket2names[b_idx].append(name)
    
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
    
    def _hook(self, param: torch.Tensor):
        # whenever the gradient is populated, update that gradient is accumulated for the parameter
        if param.grad is not None:
            # get name of the param
            for name, p in self.module.named_parameters():
                if p is param:
                    param_name = name
                    break
            bucket_id = self.name2bucket[param_name]

            # check if all parameters in the bucket have their gradients ready
            if param_name == self.bucket2names[bucket_id][-1]:
                # last parameter in the bucket
                # average gradients in the bucket
                grads = []
                for pname in self.bucket2names[bucket_id]:
                    p = dict(self.module.named_parameters())[pname]
                    if p.grad is not None:
                        p.grad /= self.world_size
                        grads.append(p.grad)
                # `all_reduce` gradients in the bucket
                flat_grads = torch._utils._flatten_dense_tensors(grads)
                
                handle = dist.all_reduce(flat_grads, async_op=True)
                
                def unflatten(_):
                    # unflatten
                    grads_reduced = torch._utils._unflatten_dense_tensors(flat_grads, grads)
                    for pname, grad_reduced in zip(self.bucket2names[bucket_id], grads_reduced):
                        p = dict(self.module.named_parameters())[pname]
                        if p.grad is not None:
                            p.grad.data.copy_(grad_reduced)

                handle.get_future().then(unflatten)
                self.handles.append(handle)   

            else:
                return  # not the last parameter in the bucket, return

    
    def finish_gradient_synchronization(self):
        start = time.perf_counter()
        # wait for all `all_reduce` operations to complete
        for handle in self.handles:
            handle.wait()
        self.handles.clear()
        end = time.perf_counter()
        return end - start