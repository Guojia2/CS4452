import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
from typing import Dict, Tuple
import torch.nn.functional as F

from src.dataset import THUMOSFeatureDataset
from src.models.temporal_model import build_temporal_model
from src.utils import get_logger, save_checkpoint


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce    = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt     = torch.exp(-bce)
        focal  = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


def build_feature_dataset(config: Dict, training: bool = False) -> THUMOSFeatureDataset:
    aug_config = config.get("augmentation", {}) if training else {}
    return THUMOSFeatureDataset(
        feature_dir=config["paths"]["feature_dir"],
        ann_path=config["paths"].get(
            "ann_path",
            os.path.join(config["paths"]["data_root"], "annotations", "thumos_14_anno.json"),
        ),
        window_size=config["data"].get("window_size", 128),
        stride=config["data"].get("stride", 64),
        subset=config["data"].get("subset", "training"),
        aug_config=aug_config,
        training=training,
    )


def split_dataset(dataset, val_ratio: float, seed: int):
    val_size = max(1, int(len(dataset) * val_ratio))
    train_size = len(dataset) - val_size
    if train_size <= 0:
        raise ValueError("Dataset is too small for the requested validation split.")
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=generator)


def build_balanced_sampler(train_ds) -> WeightedRandomSampler:
    """
    Per-sample weights inversely proportional to the rarest action class
    present in that window.  Windows with no action all get equal low weight.
    """
    # Access the underlying dataset through the Subset wrapper
    full_ds = train_ds.dataset
    indices = train_ds.indices

    # Class frequency: number of windows that contain each class
    num_classes = len(full_ds.samples[0]["gt_segs"]) if full_ds.samples else 20
    class_counts = torch.zeros(20)

    sample_labels = []
    for idx in indices:
        s = full_ds.samples[idx]
        # Determine which classes appear in this window
        present = torch.zeros(20)
        clips_per_sec = s["n_clips"] / s["duration"]
        for ann in s["gt_segs"]:
            from src.dataset import CLASS_TO_IDX
            cls_idx = CLASS_TO_IDX.get(ann["label"], -1)
            if cls_idx == -1:
                continue
            seg_start, seg_end = ann["segment"]
            c_start = int(seg_start * clips_per_sec) - s["start"]
            c_end   = int(seg_end * clips_per_sec) - s["start"]
            if max(0, c_start) < min(full_ds.window_size, c_end):
                present[cls_idx] = 1.0
        sample_labels.append(present)
        class_counts += present

    # Per-sample weight = 1 / (frequency of the rarest class it contains)
    weights = []
    for present in sample_labels:
        active = class_counts[present.bool()]
        if len(active) == 0:
            # Background window — give baseline weight of 1 / median_class_count
            w = 1.0 / (class_counts[class_counts > 0].median().item() + 1e-6)
        else:
            w = 1.0 / (active.min().item() + 1e-6)
        weights.append(w)

    weights_t = torch.tensor(weights, dtype=torch.float)
    return WeightedRandomSampler(weights_t, num_samples=len(weights_t), replacement=True)


def build_dataloaders(config: Dict, train_ds, val_ds) -> Tuple[DataLoader, DataLoader]:
    batch_size  = config["training"]["batch_size"]
    num_workers = config["data"].get("num_workers", 4)
    aug_config  = config.get("augmentation", {})

    if aug_config.get("class_balanced_sampling", False):
        sampler = build_balanced_sampler(train_ds)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True,
        )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader


def build_optimizer(config: Dict, model: nn.Module):
    return AdamW(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"].get("weight_decay", 0.0),
    )


def build_scheduler(config: Dict, optimizer):
    epochs        = config["training"]["epochs"]
    warmup_epochs = config["training"].get("warmup_epochs", 5)
    cosine_epochs = epochs - warmup_epochs

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cosine_epochs,
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )


def apply_mixup(
    features: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Feature Mixup: mix pairs within the batch.
    lambda ~ Beta(alpha, alpha); shuffle batch to get the second pairing.
    """
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    B = features.size(0)
    idx = torch.randperm(B, device=features.device)
    mixed_features = lam * features + (1 - lam) * features[idx]
    mixed_labels   = lam * labels   + (1 - lam) * labels[idx]
    return mixed_features, mixed_labels


def train_one_epoch(
    model, loader, optimizer, criterion, device, logger, epoch,
    grad_clip_norm, aug_config: Dict,
):
    model.train()
    running_loss = 0.0

    mixup_cfg  = aug_config.get("mixup", {})
    do_mixup   = mixup_cfg.get("enabled", False)
    mixup_prob = mixup_cfg.get("prob", 0.5)
    mixup_alpha = mixup_cfg.get("alpha", 0.2)

    for step, (features, labels) in enumerate(loader):
        features = features.to(device, non_blocking=True)
        labels   = labels.to(device, non_blocking=True)

        # Mixup: applied to the batch after moving to device
        if do_mixup and torch.rand(1).item() < mixup_prob:
            features, labels = apply_mixup(features, labels, mixup_alpha)

        optimizer.zero_grad()
        logits = model(features)
        loss   = criterion(logits, labels)
        loss.backward()
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        running_loss += loss.item()
        if step % 20 == 0:
            logger.info(f"Epoch {epoch} | Step {step}/{len(loader)} | Loss: {loss.item():.4f}")
    return running_loss / len(loader)


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    for features, labels in loader:
        features = features.to(device, non_blocking=True)
        labels   = labels.to(device, non_blocking=True)
        logits   = model(features)
        loss     = criterion(logits, labels)
        running_loss += loss.item()
    return running_loss / len(loader)


def train(config: dict):
    logger = get_logger("train", log_dir=config["paths"]["log_dir"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on: {device}")

    seed           = config["training"].get("seed", 42)
    val_ratio      = config["training"].get("val_ratio", 0.1)
    grad_clip_norm = config["training"].get("grad_clip_norm", 1.0)
    aug_config     = config.get("augmentation", {})

    dataset = build_feature_dataset(config, training=True)
    logger.info(f"Full dataset size: {len(dataset)}")

    sample_features, sample_labels = dataset[0]
    feature_dim = sample_features.shape[-1]
    logger.info(f"feature_dim: {feature_dim}")

    model     = build_temporal_model(config, feature_dim).to(device)
    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    train_ds, val_ds = split_dataset(dataset, val_ratio, seed)
    train_loader, val_loader = build_dataloaders(config, train_ds, val_ds)

    best_val_loss = float("inf")

    for epoch in range(config["training"]["epochs"]):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion,
            device, logger, epoch, grad_clip_norm, aug_config,
        )
        val_loss = validate_one_epoch(model, val_loader, criterion, device)
        scheduler.step()

        logger.info(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if config["training"].get("save_every_epoch", True):
            ckpt_path = os.path.join(config["paths"]["checkpoint_dir"], f"epoch_{epoch}.pt")
            save_checkpoint(ckpt_path, model, optimizer, epoch, {
                "train_loss": train_loss, "val_loss": val_loss,
            })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(config["paths"]["checkpoint_dir"], "best.pt")
            save_checkpoint(best_path, model, optimizer, epoch, {
                "train_loss": train_loss, "val_loss": val_loss,
            })
            logger.info(f"Saved new best checkpoint at epoch {epoch}")
