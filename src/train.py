"""
train.py — Training loop for THUMOS-14 action recognition.

Entry points
------------
- ``train(cfg)``  : Called by Modal (or locally) to run a full training run.
- ``main()``      : CLI entry-point for local dry-runs without Modal.

Configuration
-------------
All hyper-parameters come from the YAML config dict (``configs/base_config.yaml``).
S3 bucket migration: update ``cfg["storage"]["bucket"]`` — nothing else changes.
"""

from __future__ import annotations

import argparse
import math
import os
import pathlib
import time
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.classifier import ActionRecognitionModel
from src.dataset import THUMOSVideoDataset, NUM_CLASSES, build_transforms
from src.utils import (
    AverageMeter,
    S3Store,
    get_logger,
    load_checkpoint,
    save_checkpoint,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Optimiser & LR scheduler
# ---------------------------------------------------------------------------

def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    train_cfg = cfg.get("training", {})
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: dict,
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    train_cfg = cfg.get("training", {})
    total_epochs   = int(train_cfg.get("epochs", 30))
    warmup_epochs  = int(train_cfg.get("warmup_epochs", 5))
    scheduler_name = train_cfg.get("lr_scheduler", "cosine")

    total_steps  = total_epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / max(1, warmup_steps)
        if scheduler_name == "cosine":
            progress = float(current_step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return 1.0  # constant after warmup

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Accuracy helper
# ---------------------------------------------------------------------------

def topk_accuracy(
    output: torch.Tensor,
    target: torch.Tensor,
    topk: tuple = (1, 5),
):
    """Compute top-k accuracy for the given predictions and targets."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        results = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            results.append(correct_k.mul_(100.0 / batch_size).item())
        return results


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    grad_clip: float = 1.0,
    log_interval: int = 20,
) -> float:
    model.train()

    loss_meter = AverageMeter("loss")
    top1_meter = AverageMeter("top1")
    t0 = time.time()

    for step, (clips, labels) in enumerate(loader):
        clips  = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(clips)
        loss   = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        top1, _ = topk_accuracy(logits.detach(), labels, topk=(1, 5))
        loss_meter.update(loss.item(), clips.size(0))
        top1_meter.update(top1, clips.size(0))

        if (step + 1) % log_interval == 0:
            elapsed = time.time() - t0
            lr = scheduler.get_last_lr()[0]
            logger.info(
                f"Epoch [{epoch}] step [{step+1}/{len(loader)}] "
                f"loss={loss_meter.avg:.4f} top1={top1_meter.avg:.2f}% "
                f"lr={lr:.2e} elapsed={elapsed:.1f}s"
            )

    return loss_meter.avg


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()

    loss_meter = AverageMeter("val_loss")
    top1_meter = AverageMeter("val_top1")

    for clips, labels in loader:
        clips  = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(clips)
        loss   = criterion(logits, labels)

        top1, _ = topk_accuracy(logits, labels, topk=(1, 5))
        loss_meter.update(loss.item(), clips.size(0))
        top1_meter.update(top1, clips.size(0))

    return loss_meter.avg, top1_meter.avg


# ---------------------------------------------------------------------------
# Main training function (called by Modal)
# ---------------------------------------------------------------------------

def train(cfg: dict) -> float:
    """Full training run.

    Parameters
    ----------
    cfg : Merged config dict (base_config.yaml + any CLI overrides).

    Returns
    -------
    Best validation top-1 accuracy achieved during training.
    """
    # ------------------------------------------------------------------ setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model_cfg    = cfg.get("model",    {})
    data_cfg     = cfg.get("data",     {})
    train_cfg    = cfg.get("training", {})
    paths_cfg    = cfg.get("paths",    {})
    storage_cfg  = cfg.get("storage",  {})

    num_classes    = int(model_cfg.get("num_classes",   NUM_CLASSES))
    backbone_name  = model_cfg.get("backbone",          "videomae_base")
    pooling        = model_cfg.get("pooling",            "mean")
    pretrained     = bool(model_cfg.get("pretrained",   True))
    dropout        = float(model_cfg.get("dropout",     0.5))

    clip_len       = int(data_cfg.get("clip_len",       16))
    frame_interval = int(data_cfg.get("frame_interval", 4))
    batch_size     = int(data_cfg.get("batch_size",     8))
    num_workers    = int(data_cfg.get("num_workers",    4))
    img_size       = int(data_cfg.get("img_size",       224))

    epochs         = int(train_cfg.get("epochs",        30))
    grad_clip      = float(train_cfg.get("grad_clip",   1.0))
    label_smoothing= float(train_cfg.get("label_smoothing", 0.1))

    checkpoint_dir = pathlib.Path(paths_cfg.get("checkpoint_dir", "/vol/checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    manifest_prefix = storage_cfg.get("manifest_prefix", "manifests")
    train_manifest  = f"{manifest_prefix}/train.jsonl"
    val_manifest    = f"{manifest_prefix}/val.jsonl"

    # ------------------------------------------------------------------ S3
    s3_store = S3Store.from_config(cfg)

    # ------------------------------------------------------------------ data
    train_dataset = THUMOSVideoDataset(
        manifest_path  = train_manifest,
        s3_store       = s3_store,
        split          = "train",
        clip_len       = clip_len,
        frame_interval = frame_interval,
        num_clips      = 1,
        img_size       = img_size,
    )

    val_dataset = THUMOSVideoDataset(
        manifest_path  = val_manifest,
        s3_store       = s3_store,
        split          = "val",
        clip_len       = clip_len,
        frame_interval = frame_interval,
        num_clips      = 1,   # single-clip for fast val; multi-clip used in eval.py
        img_size       = img_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
        pin_memory  = True,
        drop_last   = True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size  = batch_size * 2,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = True,
    )

    logger.info(
        f"Dataset: {len(train_dataset)} train samples, "
        f"{len(val_dataset)} val samples"
    )

    # ------------------------------------------------------------------ model
    model = ActionRecognitionModel(
        backbone_name = backbone_name,
        num_classes   = num_classes,
        pooling       = pooling,
        pretrained    = pretrained,
        dropout       = dropout,
    ).to(device)

    logger.info(
        f"Model: {backbone_name} | pooling={pooling} | "
        f"num_classes={num_classes} | pretrained={pretrained}"
    )

    # ------------------------------------------------------------------ optim
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=len(train_loader))
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # Resume from checkpoint if available
    resume_path = str(checkpoint_dir / "last.pt")
    start_epoch = 0
    best_top1   = 0.0
    if pathlib.Path(resume_path).exists():
        ckpt        = load_checkpoint(resume_path, model, optimizer, device)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_top1   = ckpt.get("best_metric", 0.0)
        logger.info(f"Resumed from epoch {start_epoch}, best_top1={best_top1:.2f}%")

    # ------------------------------------------------------------------ loop
    for epoch in range(start_epoch, epochs):
        t_epoch = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            criterion, device, epoch, grad_clip=grad_clip,
        )
        val_loss, val_top1 = validate(model, val_loader, criterion, device)

        is_best = val_top1 > best_top1
        best_top1 = max(best_top1, val_top1)

        logger.info(
            f"Epoch [{epoch}/{epochs-1}] "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_top1={val_top1:.2f}% "
            f"best={best_top1:.2f}% "
            f"time={time.time()-t_epoch:.1f}s"
        )

        ckpt_state = {
            "epoch":              epoch,
            "model_state_dict":   model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_top1":           val_top1,
            "best_metric":        best_top1,
            "cfg":                cfg,
        }

        last_path = str(checkpoint_dir / "last.pt")
        best_path = str(checkpoint_dir / "best.pt")
        save_checkpoint(ckpt_state, last_path, is_best=is_best, best_path=best_path)

        if is_best:
            logger.info(f"  ★ New best: {best_top1:.2f}%  (saved to {best_path})")

    logger.info(f"Training complete. Best val top-1: {best_top1:.2f}%")
    return best_top1


# ---------------------------------------------------------------------------
# CLI entry-point (local dry-run / no Modal)
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train action recognition model locally")
    parser.add_argument(
        "--config", default="configs/base_config.yaml",
        help="Path to YAML config file"
    )
    parser.add_argument(
        "--override", nargs="*", default=[],
        metavar="KEY=VALUE",
        help="Override config keys, e.g. training.epochs=5 model.backbone=x3d_m"
    )
    return parser.parse_args()


def _apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply ``key=value`` override strings into a nested config dict."""
    import re
    for override in overrides:
        key, _, val = override.partition("=")
        keys = key.split(".")
        d = cfg
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        # Try to cast to int/float/bool
        if val.lower() in ("true", "false"):
            val = val.lower() == "true"
        else:
            try:
                val = int(val)
            except ValueError:
                try:
                    val = float(val)
                except ValueError:
                    pass
        d[keys[-1]] = val
    return cfg


def main() -> None:
    import yaml

    args = parse_args()
    with open(args.config, "r") as fh:
        cfg = yaml.safe_load(fh)

    cfg = _apply_overrides(cfg, args.override)
    train(cfg)


if __name__ == "__main__":
    main()
