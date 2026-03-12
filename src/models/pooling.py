"""
pooling.py — Temporal pooling strategies over clip-level feature sequences.

Both poolers accept a tensor of shape (B, T, D) — a batch of T clip embeddings
each of dimension D — and return a single video-level feature of shape (B, D).

Choices
-------
- MeanPooling      : Simple average over the time dimension.
- AttentionPooling : Learns a query vector that attends over clip features.
"""

import torch
import torch.nn as nn


class MeanPooling(nn.Module):
    """
    Average all clip features into a single video embedding.

    Input:  (B, T, D)
    Output: (B, D)
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1)


class AttentionPooling(nn.Module):
    """
    Soft-attention pooling with a learned query vector.

    A single linear layer produces scalar attention weights over the T clips;
    the output is the weighted sum of clip features.

    Input:  (B, T, D)
    Output: (B, D)

    Parameters
    ----------
    feature_dim : int
        Dimensionality D of the input clip features.
    """

    def __init__(self, feature_dim: int):
        super().__init__()
        self.attn = nn.Linear(feature_dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        weights = self.attn(x)          # (B, T, 1)
        weights = torch.softmax(weights, dim=1)   # normalise over T
        pooled  = (x * weights).sum(dim=1)        # (B, D)
        return pooled
