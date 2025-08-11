from re import L
import socket
import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from einops import rearrange, einsum

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super(Linear, self).__init__()
        self.weights = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        std = np.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.weights, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # use einsum
        return einsum(x, self.weights, "... d_in, d_out d_in -> ... d_out")

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None,  dtype=None):
        """
        num_embeddings: vocab size
        embedding_dim: dimension of the embeddings, i.e., d_model
        """
        super(Embedding, self).__init__()
        self.embedding = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        nn.init.trunc_normal_(self.embedding, mean=0.0, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding[token_ids]
    

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super(RMSNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.dim = d_model
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return (x * self.weight / rms).to(in_dtype)
    

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super(SwiGLU, self).__init__()
        self.d_model = d_model
        self.d_ff =  d_ff
        self.w1 = Linear(self.d_model, self.d_ff, device=device, dtype=dtype)
        self.w2 = Linear(self.d_ff, self.d_model, device=device, dtype=dtype)
        self.w3 = Linear(self.d_model, self.d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.w1(x)
        x3 = self.w3(x)
        return self.w2(self.swilu(x1) * x3)

    def swilu(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)
    
class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super(RoPE, self).__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        freq_seq = torch.arange(d_k // 2, dtype=torch.float32, device=device) # for k, [0, 1,..., d_k // 2 - 1]
        inv_seq = 1.0 / (theta ** (freq_seq / (d_k // 2))) # theta for every k, which equals 1 / (theta ** (k / (d_k // 2)))
        seq = torch.arange(max_seq_len, dtype=torch.float32, device=device)

        freqs = einsum(seq, inv_seq, "i, j -> i j") # (seq_len, d_k // 2)
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)

        self.register_buffer("cos_", cos, persistent=False)
        self.register_buffer("sin_", sin, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        x: (..., seq_len, d_k)
        position: (..., seq_len)
        return: (..., seq_len, d_k)
        """
        # d_k = x last dimension
        d_k = x.size(-1)
        
        assert d_k == self.d_k, f"x's last dim {d_k} != d_k {self.d_k}"
        
        # convert the embedding elements as 2d vectors
        x = rearrange(x, "... seq_len (d_pair pair) -> ... seq_len d_pair pair", pair=2)

        # rotary matrix is [[cos, -sin], [sin, cos]]
        cos = self.cos_[token_positions]
        sin = self.sin_[token_positions]

        # reshape cos and sin to match x
        cos = rearrange(cos, "... s d -> ... 1 s d") # (..., seq_len, d_k // 2) -> (..., 1, seq_len, d_k // 2)
        sin = rearrange(sin, "... s d -> ... 1 s d") # (..., seq_len, d_k // 2) -> (..., 1, seq_len, d_k // 2)

        # rotary matrix is [[cos, -sin],[sin, cos]]
        x0, x1 = x[..., 0], x[..., 1] # (..., seq_len, d_k // 2)
        rotated = torch.stack([
            x0 * cos - x1 * sin,
            x0 * sin + x1 * cos
        ], dim=-1)

        out = rearrange(rotated, "... seq_len d_pair pair-> ... seq_len (d_pair pair)")
        return out
        

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    v_max = torch.max(x, dim=dim, keepdim=True).values
    return torch.exp(x - v_max) / torch.sum(torch.exp(x - v_max), dim=dim, keepdim=True)

def scaled_dot_product_attention(
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
    """
        query: (..., queries, d_k)
        key: (..., keys, d_k)
        value: (..., values, d_v)
        mask: (..., queries, keys)
        return: (..., queries, d_v)
    """
    d_k = query.size(-1)
    q_k = einsum(query, key, "... queries d_k, ... keys d_k -> ... queries keys")
    q_k = q_k / (d_k ** 0.5)
    if mask is not None:
        q_k = torch.where(
            mask,
            q_k,
            torch.tensor(float("-inf"), device=q_k.device, dtype=q_k.dtype),
        )
    scores = einsum(softmax(q_k, dim=-1), value, "... queries keys, ... keys d_v -> ... queries d_v")
    return scores

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, theta: float = 10000.0, max_seq_len: int = 512, use_rope: bool = False):
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.head_dim = d_model // num_heads
        self.num_heads = num_heads
        self.d_model = d_model
        self.use_rope = use_rope
        
        self.w_q = Linear(d_model, d_model)
        self.w_k = Linear(d_model, d_model)
        self.w_v = Linear(d_model, d_model)
        self.w_o = Linear(d_model, d_model)
        
        if self.use_rope:
            self.rope = RoPE(theta, d_k=self.head_dim, max_seq_len=max_seq_len)

    def forward(self, x: torch.Tensor, token_positions: Optional[torch.Tensor] = None):
        Q = self.w_q(x) # (..., seq_len, d_model)
        K = self.w_k(x) # (..., seq_len, d_model)
        V = self.w_v(x) # (..., seq_len, d_model)

        Q = rearrange(Q, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", d_k=self.head_dim)
        K = rearrange(K, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", d_k=self.head_dim)
        V = rearrange(V, "... seq_len (num_heads d_v) -> ... num_heads seq_len d_v", d_v=self.head_dim)

        # Apply RoPE if enabled
        if self.use_rope:
            if token_positions is None:
                raise ValueError("token_positions must be provided when using RoPE")
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        seq_len = Q.size(-2)
        
        # use the lower triangular matrix with diagonal
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=Q.device, dtype=torch.bool), diagonal=0)
        attention = scaled_dot_product_attention(Q, K, V, mask=causal_mask)
        attention = rearrange(attention, "... num_heads seq_len d_v -> ... seq_len (num_heads d_v)")
        output = self.w_o(attention)
        return output