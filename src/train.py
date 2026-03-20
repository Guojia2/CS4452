import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, Tuple

from src.dataset import THUMOSFeatureDataset
from src.models.temporal_model import build_temporal_model
from src.utils import get_logger, save_checkpoint


def build_feature_dataset(config: Dict) -> THUMOSFeatureDataset:
    return THUMOSFeatureDataset(
        feature_dir=config["paths"]["feature_dir"],
        ann_path=os.path.join(
            config["paths"]["data_root"],
            "annotations",
            "thumos_14_anno.json",
        ),
        window_size=config["data"].get("window_size", 128),
        stride=config["data"].get("stride", 64),
        subset=config["data"].get("subset", "training"),
    )


def split_dataset(dataset, val_ratio: float, seed: int):
    val_size = max(1, int(len(dataset) * val_ratio))
    train_size = len(dataset) - val_size
    if train_size <= 0:
        raise ValueError("Dataset is too small for the requested validation split.")
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=generator)


def build_dataloaders(config: Dict, train_ds, val_ds) -> Tuple[DataLoader, DataLoader]:
    batch_size  = config["training"]["batch_size"]
    num_workers = config["data"].get("num_workers", 4)
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
    return CosineAnnealingLR(optimizer, T_max=config["training"]["epochs"])


def train_one_epoch(model, loader, optimizer, criterion, device, logger, epoch, grad_clip_norm):
    model.train()
    running_loss = 0.0
    for step, (features, labels) in enumerate(loader):
        features = features.to(device, non_blocking=True)
        labels   = labels.to(device, non_blocking=True)
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

    dataset = build_feature_dataset(config)
    logger.info(f"Full dataset size: {len(dataset)}")

    sample_features, sample_labels = dataset[0]
    feature_dim = sample_features.shape[-1]
    logger.info(f"feature_dim: {feature_dim}")

    model     = build_temporal_model(config, feature_dim).to(device)
    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer)
    criterion = nn.BCEWithLogitsLoss()

    train_ds, val_ds = split_dataset(dataset, val_ratio, seed)
    train_loader, val_loader = build_dataloaders(config, train_ds, val_ds)

    best_val_loss = float("inf")

    for epoch in range(config["training"]["epochs"]):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion,
            device, logger, epoch, grad_clip_norm,
        )
        val_loss = validate_one_epoch(model, val_loader, criterion, device)
        scheduler.step()

        logger.info(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

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