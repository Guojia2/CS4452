import torch
import torch.nn as nn
from typing import Tuple


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def build_backbone(backbone_name: str, pretrained: bool = True) -> Tuple[nn.Module, int]:
    """
    Build a video backbone and return (model, feature_dim).

    The returned model accepts input of shape (B, C, T, H, W) and outputs
    a flat feature vector of shape (B, feature_dim).

    Parameters
    ----------
    backbone_name : str
        One of "videomae_base", "videomae_large", "x3d_m".
    pretrained : bool
        Load ImageNet / Kinetics pretrained weights when True.
    """
    if backbone_name in ("videomae_base", "videomae_large"):
        return _build_videomae(backbone_name, pretrained)
    elif backbone_name == "x3d_m":
        return _build_x3d_m(pretrained)
    else:
        raise ValueError(
            f"Unknown backbone: '{backbone_name}'. "
            "Choose from: videomae_base, videomae_large, x3d_m."
        )


# ---------------------------------------------------------------------------
# VideoMAE  (Tong et al., 2022)
# ---------------------------------------------------------------------------

class _VideoMAEWrapper(nn.Module):
    """
    Wraps a HuggingFace VideoMAEModel so it accepts (B, C, T, H, W) tensors
    and returns a (B, hidden_size) feature vector via [CLS] token pooling.
    """

    def __init__(self, hf_model):
        super().__init__()
        self.model = hf_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        # HuggingFace VideoMAE expects pixel_values: (B, T, C, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        outputs = self.model(pixel_values=x)
        # last_hidden_state: (B, seq_len, hidden_size) — use mean over patches
        return outputs.last_hidden_state.mean(dim=1)   # (B, hidden_size)


def _build_videomae(name: str, pretrained: bool) -> Tuple[nn.Module, int]:
    try:
        from transformers import VideoMAEModel
    except ImportError:
        raise ImportError(
            "transformers is required for VideoMAE backbones. "
            "Install it with: pip install transformers"
        )

    hf_name = (
        "MCG-NJU/videomae-base-finetuned-kinetics"
        if name == "videomae_base"
        else "MCG-NJU/videomae-large-finetuned-kinetics"
    )

    if pretrained:
        hf_model = VideoMAEModel.from_pretrained(hf_name)
    else:
        from transformers import VideoMAEConfig
        config_name = (
            "MCG-NJU/videomae-base"
            if name == "videomae_base"
            else "MCG-NJU/videomae-large"
        )
        cfg = VideoMAEConfig.from_pretrained(config_name)
        hf_model = VideoMAEModel(cfg)

    feature_dim = hf_model.config.hidden_size  # 768 for base, 1024 for large
    return _VideoMAEWrapper(hf_model), feature_dim


# ---------------------------------------------------------------------------
# X3D-M  (Feichtenhofer, 2020) — lightweight baseline via pytorchvideo
# ---------------------------------------------------------------------------

def _build_x3d_m(pretrained: bool) -> Tuple[nn.Module, int]:
    try:
        from pytorchvideo.models.x3d import create_x3d
    except ImportError as e:
        raise ImportError(
            "pytorchvideo is required for X3D. Install it with: pip install pytorchvideo"
        ) from e

    model = create_x3d(
        input_clip_length=16,
        input_crop_size=224,
        model_num_class=400,
    )

    if pretrained:
        pretrained_model = torch.hub.load(
            "facebookresearch/pytorchvideo",
            "x3d_m",
            pretrained=True,
        )
        model.load_state_dict(pretrained_model.state_dict(), strict=False)

    feature_dim = model.blocks[-1].proj.in_features
    model.blocks[-1].proj = nn.Identity()

    return model, feature_dim
