"""Small models used across the test suite.

Deliberately tiny so the tests run in seconds on CPU, but structurally
identical to the benchmark models: the same layer norm, softmax, gelu and
residual patterns, which is what the fusion passes key off.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Pointwise(nn.Module):
    """A pure elementwise chain: one fused kernel is the whole graph."""

    def forward(self, x, y):
        z = (x + y) * 2.0
        z = torch.relu(z) - x.sigmoid()
        return z * z + 1.0


class Broadcasting(nn.Module):
    def forward(self, x, bias, scale):
        return (x * scale + bias).tanh()


class MLP(nn.Module):
    def __init__(self, d=32, h=64):
        super().__init__()
        self.fc1 = nn.Linear(d, h)
        self.fc2 = nn.Linear(h, d)
        self.ln = nn.LayerNorm(d)

    def forward(self, x):
        return x + self.fc2(F.gelu(self.fc1(self.ln(x))))


class Softmax(nn.Module):
    def forward(self, x, mask):
        return ((x * 0.125) + mask).softmax(dim=-1)


class LayerNormScale(nn.Module):
    """Reduction feeding pointwise consumers through a broadcast."""

    def __init__(self, d=32):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.w = nn.Parameter(torch.randn(d))
        self.b = nn.Parameter(torch.randn(d))

    def forward(self, x):
        return self.ln(x) * self.w + self.b


class Attention(nn.Module):
    def __init__(self, d=32, h=4):
        super().__init__()
        self.h, self.d = h, d
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        shape = (B, T, self.h, D // self.h)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        att = ((q @ k.transpose(-2, -1)) * (D // self.h) ** -0.5).softmax(dim=-1)
        y = (att @ v).transpose(1, 2).reshape(B, T, D)
        return self.proj(y)


class Block(nn.Module):
    """A full pre-norm transformer block: the unit both benchmark models are
    made of."""

    def __init__(self, d=32, h=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = Attention(d, h)
        self.ln2 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, 4 * d)
        self.fc2 = nn.Linear(4 * d, d)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


class Masked(nn.Module):
    """Comparisons, boolean masks and where: the paths dtype rules get wrong."""

    def forward(self, x, mask):
        keep = mask > 0.5
        y = torch.where(keep, x, torch.full_like(x, -1e4))
        return (y.clamp(min=-1.0, max=1.0) * (x < 0).to(x.dtype)).sum(-1)


class ResidualChain(nn.Module):
    """One producer feeding several consumers: the recompute decision."""

    def forward(self, x):
        shared = torch.tanh(x * 0.5)
        a = shared + 1.0
        b = shared * 2.0
        c = shared - 3.0
        return a * b + c


def all_models():
    """(name, module, example_inputs) for every model the suite exercises."""
    torch.manual_seed(0)
    return [
        ("pointwise", Pointwise(), (torch.randn(4, 16), torch.randn(4, 16))),
        ("broadcasting", Broadcasting(), (torch.randn(4, 8, 16), torch.randn(16), torch.randn(8, 1))),
        ("mlp", MLP(), (torch.randn(4, 32),)),
        ("softmax", Softmax(), (torch.randn(2, 4, 8, 8), torch.randn(2, 1, 1, 8))),
        ("layernorm_scale", LayerNormScale(), (torch.randn(6, 32),)),
        ("attention", Attention(), (torch.randn(2, 8, 32),)),
        ("block", Block(), (torch.randn(2, 8, 32),)),
        ("block_b8", Block(), (torch.randn(8, 16, 32),)),
        ("masked", Masked(), (torch.randn(4, 8), torch.rand(4, 8))),
        ("residual_chain", ResidualChain(), (torch.randn(6, 12),)),
    ]
