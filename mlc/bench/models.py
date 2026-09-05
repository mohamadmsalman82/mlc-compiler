"""Benchmark models.

Written out in plain PyTorch rather than pulled from a model hub, so the
graphs the compiler sees are exactly what is on this page and the benchmarks
have no dependency that can drift. The shapes are the real ones: BERT-base is
12 layers of 768 with 12 heads and a 3072 intermediate, GPT-2 small is 12
layers of 768 with a causal mask.

Both are post- and pre-norm transformers respectively, which is deliberate:
the two put layer norm in different places relative to the residual add, and
that changes what reduction fusion can reach.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BertConfig:
    vocab_size: int = 30522
    max_position: int = 512
    n_layer: int = 12
    n_head: int = 12
    d_model: int = 768
    d_ff: int = 3072
    eps: float = 1e-12


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    d_model: int = 768
    eps: float = 1e-5


# --------------------------------------------------------------------------
# BERT
# --------------------------------------------------------------------------

class BertSelfAttention(nn.Module):
    def __init__(self, cfg: BertConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.d_head = cfg.d_model // cfg.n_head
        self.query = nn.Linear(cfg.d_model, cfg.d_model)
        self.key = nn.Linear(cfg.d_model, cfg.d_model)
        self.value = nn.Linear(cfg.d_model, cfg.d_model)

    def _heads(self, x):
        b, t, _ = x.shape
        return x.view(b, t, self.n_head, self.d_head).transpose(1, 2)

    def forward(self, x, mask):
        q, k, v = self._heads(self.query(x)), self._heads(self.key(x)), self._heads(self.value(x))
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head) + mask
        probs = scores.softmax(dim=-1)
        b, t, _ = x.shape
        return (probs @ v).transpose(1, 2).reshape(b, t, -1)


class BertLayer(nn.Module):
    """Post-norm: the layer norm sits after the residual add."""

    def __init__(self, cfg: BertConfig):
        super().__init__()
        self.attn = BertSelfAttention(cfg)
        self.attn_out = nn.Linear(cfg.d_model, cfg.d_model)
        self.ln1 = nn.LayerNorm(cfg.d_model, eps=cfg.eps)
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff)
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model, eps=cfg.eps)

    def forward(self, x, mask):
        x = self.ln1(x + self.attn_out(self.attn(x, mask)))
        return self.ln2(x + self.fc2(F.gelu(self.fc1(x))))


class Bert(nn.Module):
    def __init__(self, cfg: BertConfig = BertConfig()):
        super().__init__()
        self.cfg = cfg
        self.word_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_position, cfg.d_model)
        self.type_emb = nn.Embedding(2, cfg.d_model)
        self.emb_ln = nn.LayerNorm(cfg.d_model, eps=cfg.eps)
        self.layers = nn.ModuleList(BertLayer(cfg) for _ in range(cfg.n_layer))
        self.pooler = nn.Linear(cfg.d_model, cfg.d_model)
        self.register_buffer("positions", torch.arange(cfg.max_position), persistent=False)

    def forward(self, input_ids, attention_mask):
        b, t = input_ids.shape
        pos = self.positions[:t]
        x = self.word_emb(input_ids) + self.pos_emb(pos) + self.type_emb(
            torch.zeros_like(input_ids)
        )
        x = self.emb_ln(x)
        # Additive mask, the shape a real encoder gets from padding.
        mask = (1.0 - attention_mask[:, None, None, :]) * torch.finfo(torch.float32).min
        for layer in self.layers:
            x = layer(x, mask)
        return torch.tanh(self.pooler(x[:, 0]))


# --------------------------------------------------------------------------
# GPT
# --------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.d_head = cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)

    def forward(self, x, causal):
        b, t, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        shape = (b, t, self.n_head, self.d_head)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head) + causal[:, :, :t, :t]
        probs = scores.softmax(dim=-1)
        return self.proj((probs @ v).transpose(1, 2).reshape(b, t, d))


class GPTBlock(nn.Module):
    """Pre-norm: the layer norm sits before the sublayer, on the residual
    branch. Its consumers are the projection's inputs rather than an add, so
    reduction fusion reaches a different set of ops than in BERT."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model, eps=cfg.eps)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model, eps=cfg.eps)
        self.fc1 = nn.Linear(cfg.d_model, 4 * cfg.d_model)
        self.fc2 = nn.Linear(4 * cfg.d_model, cfg.d_model)

    def forward(self, x, causal):
        x = x + self.attn(self.ln1(x), causal)
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig = GPTConfig()):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)
        self.blocks = nn.ModuleList(GPTBlock(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.d_model, eps=cfg.eps)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        causal = torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        mask = (1.0 - causal) * torch.finfo(torch.float32).min
        self.register_buffer("causal", mask.view(1, 1, cfg.block_size, cfg.block_size),
                             persistent=False)
        self.register_buffer("positions", torch.arange(cfg.block_size), persistent=False)

    def forward(self, input_ids):
        _, t = input_ids.shape
        x = self.tok_emb(input_ids) + self.pos_emb(self.positions[:t])
        for block in self.blocks:
            x = block(x, self.causal)
        return self.head(self.ln_f(x))


# --------------------------------------------------------------------------

#: name -> (builder, input builder). Sizes chosen so a full sweep finishes in
#: a few minutes; ``--full`` in the runner selects the real configurations.
SUITE = {
    "bert-base": (
        lambda: Bert(BertConfig()),
        lambda b, t, dev: (torch.randint(0, 30522, (b, t), device=dev),
                           torch.ones(b, t, device=dev)),
        128,
    ),
    "bert-small": (
        lambda: Bert(BertConfig(n_layer=4, n_head=8, d_model=512, d_ff=2048, vocab_size=8192)),
        lambda b, t, dev: (torch.randint(0, 8192, (b, t), device=dev),
                           torch.ones(b, t, device=dev)),
        128,
    ),
    "gpt2-small": (
        lambda: GPT(GPTConfig()),
        lambda b, t, dev: (torch.randint(0, 50257, (b, t), device=dev),),
        128,
    ),
    "gpt-small": (
        lambda: GPT(GPTConfig(n_layer=4, n_head=8, d_model=512, vocab_size=8192,
                              block_size=256)),
        lambda b, t, dev: (torch.randint(0, 8192, (b, t), device=dev),),
        128,
    ),
}


def build(name: str, batch: int, seq: int | None = None,
          device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32):
    """Build a model and its inputs.

    ``dtype`` halves the weights and float inputs. On a consumer card fp32 is
    the weak path, so fp16 is the realistic configuration there; index tensors
    stay integral either way.
    """
    builder, inputs, default_seq = SUITE[name]
    torch.manual_seed(0)
    model = builder().eval().to(device)
    args = inputs(batch, seq or default_seq, device)
    if dtype != torch.float32:
        model = model.to(dtype)
        args = tuple(a.to(dtype) if a.is_floating_point() else a for a in args)
    return model, args
