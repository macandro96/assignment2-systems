import math

import torch
import triton
import triton.language as tl
from torch.autograd import Function


class FlashAttentionPT(Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=True):
        # tile sizes
        Bq, Bk = 16, 16
        Nq, Nk = Q.shape[-2], K.shape[-2]
        d = Q.shape[-1]
        bsz = Q.shape[0]
        
        O = torch.empty_like(Q)
        L = torch.empty((bsz, Nq,), dtype=torch.float32, device=Q.device)
        # split Q into Tq
        Tq = math.ceil(Nq / Bq)
        Qs = torch.chunk(Q, chunks=Tq, dim=-2)  # list[Q1, Q2 .. QTq] of size bsz x Bq x d
        
        # split K, V into Tr
        Tk = math.ceil(Nk / Bk)
        Ks = torch.chunk(K, chunks=Tk, dim=-2)  # # list[K1, K2 .. KTr] of size bsz x Br x d
        Vs = torch.chunk(V, chunks=Tk, dim=-2)  # # list[V1, V2 .. VTr] of size bsz x Br x d
        
        if is_causal:
            neg_inf = torch.full((Bq, Bk), fill_value= -float("inf"), dtype=torch.float32)
            zeros = torch.full((Bq, Bk), fill_value=0, dtype=torch.float32)
        
        for i in range(Tq):
            Qi = Qs[i]  # bsz x Bq x d
            Oi = torch.zeros_like(Qi)  # bsz x Bq x d
            li = torch.zeros((bsz, Bq,), dtype=torch.float32, device=Qi.device)  # bsz x Bq
            mi = torch.full((bsz, Bq,), fill_value=float('-inf'), dtype=torch.float32, device=Qi.device)  # bsz x Bq
            
            for j in range(Tk):
                Kj, Vj = Ks[j], Vs[j]  # bsz x Bk x d
                Sij = torch.matmul(Qi, Kj.transpose(-1, -2)) / math.sqrt(d)  # bsz x Bq x Bk
                
                if is_causal:
                    rows = i * Bq + torch.arange(0, Bq)[:, None]  # Q_TILE_SIZE x 1
                    columns = j * Bk + torch.arange(0, Bk)[None, :]  # 1 x K_TILE_SIZE
                    causal_mask = (columns > rows)
                    scores = torch.where(causal_mask, neg_inf, zeros)
                    Sij = Sij + scores
                
                # compute rowmax
                rowmax = torch.max(Sij, dim=-1)[0]
                mij = torch.maximum(mi, rowmax)  # bsz x Bq

                Pij = torch.exp(Sij - mij.unsqueeze(-1))  # bsz x Bq x Bk
                
                Pij_rowsum = torch.sum(Pij, dim=-1)  # bsz x Bq
                lij = torch.exp(mi - mij) * li + Pij_rowsum  # bsz x Bq

                Oi = torch.exp(mi - mij).unsqueeze(-1) * Oi + torch.matmul(Pij, Vj)  # bsz x Bq x d
                
                # update vals
                li = lij
                mi = mij
            Oi = (1 / li).unsqueeze(-1) * Oi  # bsz x Bq x d
            Li = mi + torch.log(li)  # bsz x Bq
            O[:, i * Bq: (i + 1) * Bq, :] = Oi
            L[:, i * Bq: (i + 1) * Bq] = Li

        ctx.save_for_backward(L, Q, K, V, O)
        return O
    
    @staticmethod
    def backward(ctx, grad_Q, grad_K, grad_v):
        raise NotImplementedError("backward pass not implemented yet..")

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, 
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,  # batch_stride, q_stride, d_stride
    stride_kb, stride_kk, stride_kd,  # batch_stride, k_stride, d_stride
    stride_vb, stride_vq, stride_vd,  # batch_stride, v_stride, d_stride
    stride_lb, stride_lq,  # batch_stride, lqstride
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,  # Bq
    K_TILE_SIZE: tl.constexpr,  # Bk
    is_causal: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    
    # Since we need to deal with batch_sizes too; set base_ptr to start of corresp. batch_index
    # offset will be then start of block in that batch_index; q_ptr + batch_index * stride_qb + Q_TILE_SIZE * query_tile_index
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    Q_block = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    
    # transpose K
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vq, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    
    output_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),        
        order=(1, 0),
    )
    
    # print(tl.device_print("hello"))
    offs = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
    L_ptrs = L_ptr + batch_index * stride_lb + offs * stride_lq
    
    
    # buffers
    Oi = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    li = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    mi = tl.full((Q_TILE_SIZE,), value=float("-inf"), dtype=tl.float32)
    
    Tk = tl.cdiv(N_KEYS, K_TILE_SIZE)
    
    # for mask
    if is_causal:
        neg_inf = tl.full([Q_TILE_SIZE, K_TILE_SIZE], -float("inf"), tl.float32)
        zeros = tl.full([Q_TILE_SIZE, K_TILE_SIZE], 0, tl.float32)
        
    for j in range(Tk):
        K_block = tl.load(K_block_ptr, boundary_check=(1, 0), padding_option="zero")
        V_block = tl.load(V_block_ptr, boundary_check=(1, 0), padding_option="zero")
        scores = None
        if is_causal:
            # create causal mask
            rows = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)[:, None]  # Q_TILE_SIZE x 1
            columns = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)[None, :]  # 1 x K_TILE_SIZE
            causal_mask = (columns > rows)
            scores = tl.where(causal_mask, neg_inf, zeros)
        
        Sij = tl.dot(Q_block, K_block.T, acc=scores) / scale  # Bq x Bk
        
        rowmax = tl.max(Sij, axis=-1)
        mij = tl.maximum(mi, rowmax)
        
        Pij = tl.exp(Sij - mij[:, None])
        Pij_rowsum = tl.sum(Pij, axis=-1)
        lij = tl.exp(mi - mij) * li + Pij_rowsum
        
        Oi = tl.exp(mi - mij)[:, None] * Oi + tl.dot(Pij, V_block)
        
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
        
        mi = mij
        li = lij
    
    Oi = (1 / li)[:, None] * Oi
    Li = mi + tl.log(li)
    
    # store output block and Li ptrs
    tl.store(output_block_ptr, Oi, boundary_check=(1, 0))
    tl.store(L_ptrs, Li, mask=offs < N_QUERIES)

    
class FlashAttentionTriton(Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        """FlashAttention forward pass using Triton
        
        Q: bsz x Nq x d
        K: bsz x Nk x d
        V: bsz x Nv x d
        """
        bsz, Nq, d = Q.shape
        Nk = K.shape[-2]

        O = torch.empty_like(Q)
        L = torch.empty((bsz, Nq,), dtype=torch.float32, device=Q.device)
        
        Q_TILE_SIZE, K_TILE_SIZE = 16, 16
        Tq, Tk = math.ceil(Nq / Q_TILE_SIZE), math.ceil(Nk / K_TILE_SIZE)
        
        
        grid = (Tq, bsz)
        flash_fwd_kernel[grid](
            Q, K, V, O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            L.stride(0), L.stride(1),
            Nq, Nk, math.sqrt(d),
            D=d,
            Q_TILE_SIZE=Q_TILE_SIZE, K_TILE_SIZE=K_TILE_SIZE,
            is_causal=is_causal,
        )
        
        ctx.save_for_backward(L, Q, K, V, O)

        return O
    

@triton.jit
def hello_kernel(out_ptr, block_size: tl.constexpr):
    pid = tl.program_id(0)
    # tl.device_print(f"Hello, Triton!")
    ids = pid * block_size + tl.arange(0, block_size)
    tl.store(out_ptr + ids, pid, mask=ids < 4)

def test():
    grid = (2,)
    out = torch.empty((4,), dtype=torch.int32, device='cuda')
    hello_kernel[grid](out, block_size=2)
    print(out)

if __name__ == "__main__":
    test()