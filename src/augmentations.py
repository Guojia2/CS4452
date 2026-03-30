"""
Feature-level data augmentations for temporal action detection.

All functions operate on pre-extracted feature tensors (W, D) and
label tensors (W, C). They are applied in THUMOSFeatureDataset.__getitem__
during training only.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Dict, Tuple


def gaussian_noise(
    features: torch.Tensor, std: float = 0.01
) -> torch.Tensor:
    """Add Gaussian noise to feature vectors."""
    return features + torch.randn_like(features) * std


def time_mask(
    features: torch.Tensor, labels: torch.Tensor, max_len: int = 10
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Zero out a random contiguous block of clips (rows)."""
    W = features.shape[0]
    mask_len = torch.randint(1, max_len + 1, (1,)).item()
    start = torch.randint(0, max(1, W - mask_len), (1,)).item()
    features[start : start + mask_len] = 0.0
    # Labels stay — model must learn from surrounding context
    return features, labels


def feature_mask(
    features: torch.Tensor, max_len: int = 256
) -> torch.Tensor:
    """Zero out a random contiguous block of feature dimensions (columns)."""
    D = features.shape[1]
    mask_len = torch.randint(1, max_len + 1, (1,)).item()
    start = torch.randint(0, max(1, D - mask_len), (1,)).item()
    features[:, start : start + mask_len] = 0.0
    return features


def feature_dropout(
    features: torch.Tensor, p: float = 0.1
) -> torch.Tensor:
    """Randomly zero individual feature dimensions."""
    mask = torch.bernoulli(torch.full_like(features, 1.0 - p))
    return features * mask


def speed_perturb(
    features: torch.Tensor,
    labels: torch.Tensor,
    speed_range: Tuple[float, float] = (0.8, 1.2),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Resample the temporal axis at a random speed factor.

    speed < 1 → slow-motion (stretch fewer clips to fill window)
    speed > 1 → fast-forward (compress more clips into window)
    """
    W, D = features.shape
    C = labels.shape[1]
    lo, hi = speed_range
    speed = lo + torch.rand(1).item() * (hi - lo)

    # Number of source clips to sample
    src_len = int(round(W * speed))
    src_len = max(1, min(src_len, W))

    # Resample features: (1, 1, src_len, D) → (1, 1, W, D)
    feat_in = features[:src_len].unsqueeze(0).unsqueeze(0)  # (1, 1, src_len, D)
    feat_out = F.interpolate(feat_in, size=(W, D), mode="bilinear", align_corners=False)
    features = feat_out.squeeze(0).squeeze(0)  # (W, D)

    # Resample labels with nearest neighbor to keep binary values
    lab_in = labels[:src_len].unsqueeze(0).permute(0, 2, 1)  # (1, C, src_len)
    lab_out = F.interpolate(lab_in, size=W, mode="nearest")
    labels = lab_out.permute(0, 2, 1).squeeze(0)  # (W, C)

    return features, labels


def apply_augmentations(
    features: torch.Tensor,
    labels: torch.Tensor,
    aug_config: Dict,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply all enabled augmentations in sequence.

    Called from dataset.__getitem__ during training only.
    Mixup is handled separately in the training loop.
    """
    # Speed perturbation (changes temporal structure, apply first)
    sp_cfg = aug_config.get("speed_perturb", {})
    if sp_cfg.get("enabled", False):
        features, labels = speed_perturb(
            features, labels,
            speed_range=tuple(sp_cfg.get("range", [0.8, 1.2])),
        )

    # Gaussian noise
    gn_cfg = aug_config.get("gaussian_noise", {})
    if gn_cfg.get("enabled", False):
        features = gaussian_noise(features, std=gn_cfg.get("std", 0.01))

    # Time masking
    tm_cfg = aug_config.get("time_mask", {})
    if tm_cfg.get("enabled", False):
        features, labels = time_mask(
            features, labels, max_len=tm_cfg.get("max_len", 10),
        )

    # Feature masking
    fm_cfg = aug_config.get("feature_mask", {})
    if fm_cfg.get("enabled", False):
        features = feature_mask(features, max_len=fm_cfg.get("max_len", 256))

    # Feature dropout
    fd_cfg = aug_config.get("feature_dropout", {})
    if fd_cfg.get("enabled", False):
        features = feature_dropout(features, p=fd_cfg.get("p", 0.1))

    return features, labels
