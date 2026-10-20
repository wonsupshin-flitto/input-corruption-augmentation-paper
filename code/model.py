"""Custom decoder-only LLM implementation (~320M).

Standard recipe: GQA + RoPE + RMSNorm(pre-norm) + SwiGLU + tied embeddings.
Attention kernel uses F.scaled_dot_product_attention (FlashAttention automatic).
"""
from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint


@dataclass
class GPTConfig:
    vocab_size: int = 64000
    n_layer: int = 24
    n_head: int = 16
    n_kv_head: int = 4
    n_embd: int = 1024
    ffn_hidden: int = 2816
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    tie_embeddings: bool = True
    grad_checkpoint: bool = False

    @property
    def head_dim(self):
        return self.n_embd // self.n_head


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dt)


def precompute_rope(head_dim, max_seq_len, theta, device):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(t, inv)              # (T, head_dim/2)
    return torch.cos(freqs), torch.sin(freqs)  # each (T, head_dim/2)


def apply_rope(x, cos, sin):
    # x: (B, n_head, T, head_dim)
    T = x.size(2)
    cos = cos[:T].view(1, 1, T, -1)
    sin = sin[:T].view(1, 1, T, -1)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    xr1 = x1 * cos - x2 * sin
    xr2 = x1 * sin + x2 * cos
    out = torch.empty_like(x)
    out[..., 0::2] = xr1
    out[..., 1::2] = xr2
    return out


class Attention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.nh, self.nkv, self.hd = cfg.n_head, cfg.n_kv_head, cfg.head_dim
        self.q = nn.Linear(cfg.n_embd, cfg.n_head * cfg.head_dim, bias=False)
        self.k = nn.Linear(cfg.n_embd, cfg.n_kv_head * cfg.head_dim, bias=False)
        self.v = nn.Linear(cfg.n_embd, cfg.n_kv_head * cfg.head_dim, bias=False)
        self.o = nn.Linear(cfg.n_head * cfg.head_dim, cfg.n_embd, bias=False)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.nh, self.hd).transpose(1, 2)
        k = self.k(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
        v = self.v(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if self.nkv != self.nh:                      # GQA: repeat KV heads
            rep = self.nh // self.nkv
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, self.nh * self.hd)
        return self.o(y)


class SwiGLU(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.n_embd, cfg.ffn_hidden, bias=False)
        self.up = nn.Linear(cfg.n_embd, cfg.ffn_hidden, bias=False)
        self.down = nn.Linear(cfg.ffn_hidden, cfg.n_embd, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n1 = RMSNorm(cfg.n_embd)
        self.attn = Attention(cfg)
        self.n2 = RMSNorm(cfg.n_embd)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin)
        x = x + self.mlp(self.n2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        self._rope_cache = {}
        self.apply(self._init)
        # residual proj scaled init (GPT-2 style)
        for n, p in self.named_parameters():
            if n.endswith("o.weight") or n.endswith("down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _rope(self, device):
        if device not in self._rope_cache:
            self._rope_cache[device] = precompute_rope(
                self.cfg.head_dim, self.cfg.max_seq_len, self.cfg.rope_theta, device)
        return self._rope_cache[device]

    def forward(self, idx, targets=None):
        cos, sin = self._rope(idx.device)
        x = self.tok_emb(idx)
        for b in self.blocks:
            if self.cfg.grad_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(b, x, cos, sin, use_reentrant=False)
            else:
                x = b(x, cos, sin)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1), ignore_index=-1)
        return logits, loss

    def num_params(self, non_embedding=False):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding and self.cfg.tie_embeddings:
            n -= self.tok_emb.weight.numel()  # tied lm_head is not counted separately
        return n


if __name__ == "__main__":
    cfg = GPTConfig()
    m = GPT(cfg)
    print(f"params: {m.num_params()/1e6:.1f}M (non-emb {m.num_params(True)/1e6:.1f}M)")
    x = torch.randint(0, cfg.vocab_size, (2, 64))
    logits, loss = m(x, x)
    print("logits", tuple(logits.shape), "loss", float(loss))
