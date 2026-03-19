import torch
import torch.nn as nn
from typing import Dict

# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_temporal_model(config: Dict, feature_dim: int) -> nn.Module:
    model_name = config["model"].get("model", "transformer")

    if model_name == "transformer":
        return TransformerModel(
            feature_dim=feature_dim,
            num_classes=config["model"]["num_classes"],
            num_heads=config["model"].get("num_heads", 8),
            num_layers=config["model"].get("num_layers", 4),
            ff_dim=config["model"].get("ff_dim", 1024),
            dropout=config["model"].get("dropout", 0.1),
            max_seq_len=config["model"].get(
                "max_seq_len",
                config["data"].get("window_size", 128),
            ),
        )
    elif model_name == "cnn":
        return TemporalCNNModel(
            feature_dim=feature_dim,
            num_classes=config["model"]["num_classes"],
            hidden_dim=config["model"].get("hidden_dim", 512),
            num_layers=config["model"].get("num_layers", 4),
            kernel_size=config["model"].get("kernel_size", 3),
            dropout=config["model"].get("dropout", 0.1),
        )

    raise ValueError(f"Unknown temporal model: {model_name}")

# ---------------------------------------------------------------------------
# Temporal detection head (clip-sequence → per-clip logits)
# ---------------------------------------------------------------------------

class TransformerModel(nn.Module):
    """
    Lightweight transformer-based temporal detection head.
    Takes a sequence of pre-extracted clip features and predicts per-clip
    class scores.

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


class TemporalConvBlock(nn.Module):
    """
    Residual temporal conv block.

    Input:  (B, C, W)
    Output: (B, C, W)
    """

    def __init__(self, channels: int, kernel_size: int, dropout: float):
        super().__init__()

        padding = kernel_size // 2

        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
        )

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.block(x)
        x = x + residual
        return self.relu(x)


class TemporalCNNModel(nn.Module):
    """
    Temporal CNN for per-clip classification.

    Input:  (B, W, feature_dim)
    Output: (B, W, num_classes)
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        hidden_dim: int = 512,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.temporal_blocks = nn.Sequential(
            *[
                TemporalConvBlock(
                    channels=hidden_dim,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.cls_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)            # (B, W, hidden_dim)
        x = x.transpose(1, 2)             # (B, hidden_dim, W)
        x = self.temporal_blocks(x)       # (B, hidden_dim, W)
        x = self.cls_head(x)              # (B, num_classes, W)
        x = x.transpose(1, 2)             # (B, W, num_classes)
        return x