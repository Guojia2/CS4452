import csv
import json
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.evaluate import compute_binary_metrics
from src.utils import get_logger, save_checkpoint


def _write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _write_history_csv(path: str, history: list[dict]) -> None:
    if not history:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = list(history[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def _make_split_indices(dataset_size: int, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(dataset_size, generator=generator).tolist()
    val_size = max(1, int(val_fraction * dataset_size))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    if not train_indices:
        train_indices = val_indices
    return train_indices, val_indices


def _evaluate_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float,
) -> dict:
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for feats, labels in data_loader:
            feats, labels = feats.to(device), labels.to(device)
            logits = model(feats)
            total_loss += criterion(logits, labels).item()
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())

    metrics = compute_binary_metrics(
        torch.cat(all_logits, dim=0),
        torch.cat(all_labels, dim=0),
        threshold=threshold,
        prefix="val_",
    )
    metrics["val_loss"] = total_loss / max(1, len(data_loader))
    return metrics


def train(config: dict):
    logger = get_logger("train", log_dir=config["paths"]["log_dir"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on: {device}")

    use_features = config.get("use_features", True)
    artifact_dir = config["paths"].get("artifact_dir", config["paths"]["log_dir"])
    batch_size = config["data"].get(
        "batch_size",
        config.get("training", {}).get("batch_size", 8),
    )
    split_seed = config.get("training", {}).get("seed", 42)
    val_fraction = config.get("training", {}).get("val_fraction", 0.1)
    threshold = config.get("reporting", {}).get("threshold", 0.5)
    primary_metric = config.get("reporting", {}).get("primary_metric", "val_f1")

    os.makedirs(config["paths"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(artifact_dir, exist_ok=True)

    # --- Dataset ---
    if use_features:
        from src.dataset import THUMOSFeatureDataset
        dataset = THUMOSFeatureDataset(
            feature_dir=config["paths"]["feature_dir"],
            ann_path=os.path.join(
                config["paths"]["data_root"], "annotations", "thumos14.json"
            ),
            window_size=config["data"].get("window_size", 128),
            stride=config["data"].get("stride", 64),
        )
        # Infer feature dim from the first sample
        sample_feat, _ = dataset[0]
        feature_dim = sample_feat.shape[-1]
        logger.info(f"Feature dataset size: {len(dataset)}, feature_dim: {feature_dim}")

        from src.model import TemporalDetectionHead
        model = TemporalDetectionHead(
            feature_dim=feature_dim,
            num_classes=config["model"]["num_classes"],
        ).to(device)
    else:
        from src.dataset import THUMOSVideoDataset
        from src.model import ActionRecognitionModel
        dataset = THUMOSVideoDataset(
            video_dir=os.path.join(config["paths"]["data_root"], "videos"),
            ann_path=os.path.join(
                config["paths"]["data_root"], "annotations", "thumos14.json"
            ),
            clip_len_sec=config["data"].get("clip_len_sec", 2.0),
            stride_sec=config["data"].get("stride_sec", 1.0),
            num_frames=config["data"].get("num_frames", 16),
        )
        model = ActionRecognitionModel(
            backbone_name=config["model"]["backbone"],
            num_classes=config["model"]["num_classes"],
            pretrained=config["model"]["pretrained"],
        ).to(device)

    # --- Split ---
    train_indices, val_indices = _make_split_indices(len(dataset), val_fraction, split_seed)
    train_ds = Subset(dataset, train_indices)
    val_ds = Subset(dataset, val_indices)

    _write_json(
        os.path.join(artifact_dir, "split_indices.json"),
        {
            "seed": split_seed,
            "val_fraction": val_fraction,
            "train_indices": train_indices,
            "val_indices": val_indices,
        },
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
    )

    # --- Optimizer ---
    optimizer = AdamW(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config["training"]["epochs"])
    criterion = nn.BCEWithLogitsLoss()   # Multi-label for temporal detection

    best_metric = float("-inf")
    history = []

    for epoch in range(config["training"]["epochs"]):
        # --- Train ---
        model.train()
        running_loss = 0.0
        for step, (feats, labels) in enumerate(train_loader):
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(feats)          # (B, W, num_classes)
            loss   = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item()
            if step % 20 == 0:
                logger.info(f"Epoch {epoch} | Step {step}/{len(train_loader)} | Loss: {loss.item():.4f}")

        scheduler.step()
        avg_loss = running_loss / len(train_loader)

        # --- Validate ---
        val_metrics = _evaluate_epoch(model, val_loader, criterion, device, threshold)
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": float(avg_loss),
            **val_metrics,
        }
        history.append(epoch_metrics)
        logger.info(
            "Epoch %s | Train Loss: %.4f | Val Loss: %.4f | Val Acc: %.4f | Val F1: %.4f",
            epoch,
            avg_loss,
            val_metrics["val_loss"],
            val_metrics["val_binary_accuracy"],
            val_metrics["val_f1"],
        )

        _write_json(
            os.path.join(artifact_dir, "training_history.json"),
            {"history": history},
        )
        _write_history_csv(
            os.path.join(artifact_dir, "training_history.csv"),
            history,
        )

        # Save checkpoint every epoch; keep best separately
        ckpt_path = os.path.join(config["paths"]["checkpoint_dir"], f"epoch_{epoch}.pt")
        save_checkpoint(ckpt_path, model, optimizer, epoch, epoch_metrics)

        current_metric = epoch_metrics.get(primary_metric, epoch_metrics["val_f1"])
        if current_metric > best_metric:
            best_metric = current_metric
            save_checkpoint(
                os.path.join(config["paths"]["checkpoint_dir"], "best.pt"),
                model, optimizer, epoch, epoch_metrics,
            )

    summary = {
        "best_metric_name": primary_metric,
        "best_metric_value": best_metric,
        "epochs": config["training"]["epochs"],
        "history_path": os.path.join(artifact_dir, "training_history.json"),
        "split_path": os.path.join(artifact_dir, "split_indices.json"),
        "checkpoint_path": os.path.join(config["paths"]["checkpoint_dir"], "best.pt"),
        "final_epoch": history[-1] if history else {},
    }
    _write_json(os.path.join(artifact_dir, "training_summary.json"), summary)
    return summary