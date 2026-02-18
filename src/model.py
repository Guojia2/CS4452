import torch
import torch.nn as nn
import pytorchvideo.models as pv_models


def build_backbone(backbone_name: str, pretrained: bool = True) -> nn.Module:
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

    elif backbone_name == "video_swin_tiny":
        # pytorchvideo wraps Swin as well; alternatively use the
        # official Video Swin repo or timm
        raise NotImplementedError(
            "Wire in Video Swin from the official repo or timm here."
        )
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