from typing import Tuple

import pytorchvideo.models as pv_models
import torch
import torch.nn as nn


class VideoMAEBackbone(nn.Module):
    """Wrap Hugging Face VideoMAE so it matches this repo's input shape."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # VideoMAE expects (B, T, C, H, W), but the repo uses (B, C, T, H, W).
        pixel_values = x.permute(0, 2, 1, 3, 4).contiguous()
        outputs = self.model(pixel_values=pixel_values)
        return outputs.last_hidden_state.mean(dim=1)

def build_backbone(backbone_name: str, pretrained: bool = True) -> Tuple[nn.Module, int]:
    """
    Returns a video backbone. Add more options here as needed.
    """
    if backbone_name == "x3d_m":
        model = pv_models.x3d.create_x3d(
            input_clip_length=16,
            input_crop_size=224,
            model_num_class=400,  # Kinetics pre-trained output size
            pretrained=pretrained,
        )
        # Strip the classification head so we get features out
        feature_dim = model.blocks[-1].proj.in_features
        model.blocks[-1].proj = nn.Identity()
        return model, feature_dim
    elif backbone_name == "videomae_base":
        try:
            from transformers import VideoMAEConfig, VideoMAEModel
        except ImportError as exc:
            raise ImportError(
                "videomae_base requires transformers. Add it to requirements first."
            ) from exc

        if pretrained:
            hf_model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
        else:
            config = VideoMAEConfig()
            hf_model = VideoMAEModel(config)

        return VideoMAEBackbone(hf_model), hf_model.config.hidden_size
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")


class ActionRecognitionModel(nn.Module):
    def __init__(self, backbone_name: str, num_classes: int, pretrained: bool = True):
        super().__init__()
        self.backbone, feature_dim = build_backbone(backbone_name, pretrained)
        # Simple classification head — swap this for ActionFormer etc. later
        self.head = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, x):
        # x: (B, C, T, H, W)
        features = self.backbone(x)
        return self.head(features)


class TemporalDetectionHead(nn.Module):
    """
    Lightweight transformer-based temporal detection head.
    Takes a sequence of clip features and predicts per-clip class scores.
    This is a simplified version of ActionFormer-style heads — a good
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
        x = self.input_proj(x)                           # (B, W, ff_dim)
        x = x + self.pos_embedding[:, :x.size(1), :]    # add positional encoding
        x = self.transformer(x)                          # (B, W, ff_dim)
        return self.cls_head(x)                          # (B, W, num_classes)