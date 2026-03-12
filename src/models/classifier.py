"""
classifier.py — Final classification heads.

ClassificationHead
------------------
A simple dropout → linear head that maps a video embedding vector to class
logits.  Swap in anything more complex (e.g. MLP, cross-attention) here.

ActionRecognitionModel
---------------------
End-to-end model that wires backbone → pooling → classification head.
Use this for clip-level or video-level action *classification*.

For temporal action *detection* (localisation + classification) see
the TemporalDetectionHead kept in this file as a starting point.
"""

import torch
import torch.nn as nn
from typing import Literal

from src.models.backbone import build_backbone
from src.models.pooling import MeanPooling, AttentionPooling


# ---------------------------------------------------------------------------
# Simple linear head
# ---------------------------------------------------------------------------

class ClassificationHead(nn.Module):
    """
    Dropout → Linear classification head.

    Input:  (B, feature_dim)
    Output: (B, num_classes)  — raw logits
    """

    def __init__(self, feature_dim: int, num_classes: int, dropout: float = 0.5):
        super().__init__()
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


# ---------------------------------------------------------------------------
# End-to-end action recognition model
# ---------------------------------------------------------------------------

class ActionRecognitionModel(nn.Module):
    """
    backbone → temporal pooling → classification head.

    Parameters
    ----------
    backbone_name : str
        Passed to ``build_backbone``.
    num_classes : int
        Number of action categories.
    pooling : {"mean", "attention"}
        How to aggregate clip features into a video embedding.
    pretrained : bool
        Use pretrained backbone weights.
    dropout : float
        Dropout probability before the linear classifier.
    """

    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        pooling: Literal["mean", "attention"] = "mean",
        pretrained: bool = True,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.backbone, feature_dim = build_backbone(backbone_name, pretrained)

        if pooling == "mean":
            self.pooling = MeanPooling()
        elif pooling == "attention":
            self.pooling = AttentionPooling(feature_dim)
        else:
            raise ValueError(f"Unknown pooling: '{pooling}'. Choose 'mean' or 'attention'.")

        self.head = ClassificationHead(feature_dim, num_classes, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, C, T, H, W)
            Raw video clip tensor.
        """
        features = self.backbone(x)   # (B, D) or (B, T, D) depending on backbone

        # If backbone returns (B, D), unsqueeze so pooling always gets (B, T, D)
        if features.dim() == 2:
            features = features.unsqueeze(1)

        pooled = self.pooling(features)   # (B, D)
        return self.head(pooled)          # (B, num_classes)


# ---------------------------------------------------------------------------
# Temporal detection head (clip-sequence → per-clip logits)
# ---------------------------------------------------------------------------

class TemporalDetectionHead(nn.Module):
    """
    Lightweight transformer-based temporal detection head.
    Takes a sequence of pre-extracted clip features and predicts per-clip
    class scores.  This is a simplified ActionFormer-style head — a good
    starting point before wiring in a full proposal network.

    Input:  (B, W, feature_dim)   — W clip features per window
    Output: (B, W, num_classes)   — per-clip class logits
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        num_heads: int = 8,
        num_layers: int = 4,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 256,
    ):
        super().__init__()

        self.input_proj = nn.Linear(feature_dim, ff_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=ff_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.cls_head = nn.Sequential(
            nn.LayerNorm(ff_dim),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, num_classes),
        )

        # Learnable positional encoding
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, ff_dim))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, W, feature_dim)
        x = self.input_proj(x)                            # (B, W, ff_dim)
        x = x + self.pos_embedding[:, :x.size(1), :]     # positional encoding
        x = self.transformer(x)                           # (B, W, ff_dim)
        return self.cls_head(x)                           # (B, W, num_classes)
