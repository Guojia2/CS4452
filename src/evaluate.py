import json
import os
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.dataset import THUMOSFeatureDataset
from src.model import TemporalDetectionHead
from src.utils import load_checkpoint


def compute_binary_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5,
    prefix: str = "",
) -> Dict[str, float]:
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= threshold).float()
    labels = labels.float()

    true_positive = ((predictions == 1) & (labels == 1)).sum().item()
    false_positive = ((predictions == 1) & (labels == 0)).sum().item()
    false_negative = ((predictions == 0) & (labels == 1)).sum().item()

    precision = true_positive / max(1.0, true_positive + false_positive)
    recall = true_positive / max(1.0, true_positive + false_negative)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    return {
        f"{prefix}binary_accuracy": float((predictions == labels).float().mean().item()),
        f"{prefix}precision": float(precision),
        f"{prefix}recall": float(recall),
        f"{prefix}f1": float(f1),
        f"{prefix}positive_rate": float(predictions.mean().item()),
    }


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> Dict[str, float]:
    criterion = nn.BCEWithLogitsLoss()
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for features, labels in data_loader:
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features)
            total_loss += criterion(logits, labels).item()
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())

    metrics = compute_binary_metrics(
        torch.cat(all_logits, dim=0),
        torch.cat(all_labels, dim=0),
        threshold=threshold,
    )
    metrics["val_loss"] = total_loss / max(1, len(data_loader))
    return metrics


def evaluate_checkpoint(
    config: dict,
    checkpoint_path: str,
    split_indices_path: str,
    output_path: str | None = None,
) -> Dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifact_dir = config["paths"].get("artifact_dir", config["paths"]["log_dir"])
    threshold = config.get("reporting", {}).get("threshold", 0.5)

    dataset = THUMOSFeatureDataset(
        feature_dir=config["paths"]["feature_dir"],
        ann_path=os.path.join(config["paths"]["data_root"], "annotations", "thumos14.json"),
        window_size=config["data"].get("window_size", 128),
        stride=config["data"].get("stride", 64),
    )

    with open(split_indices_path, "r", encoding="utf-8") as f:
        split_indices = json.load(f)
    val_dataset = Subset(dataset, split_indices["val_indices"])

    batch_size = config["data"].get(
        "batch_size",
        config.get("training", {}).get("batch_size", 8),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
    )

    sample_features, _ = dataset[0]
    model = TemporalDetectionHead(
        feature_dim=sample_features.shape[-1],
        num_classes=config["model"]["num_classes"],
    ).to(device)
    load_checkpoint(checkpoint_path, model)

    summary = evaluate_model(model, val_loader, device, threshold=threshold)
    summary.update(
        {
            "checkpoint_path": checkpoint_path,
            "split_indices_path": split_indices_path,
            "num_validation_windows": len(val_dataset),
            "artifact_dir": artifact_dir,
        }
    )

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    return summary