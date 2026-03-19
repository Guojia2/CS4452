import os
import logging
import torch
import numpy as np
from typing import Dict, Any


def get_logger(name: str, log_dir: str = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Always log to stdout (Modal streams this back to your terminal)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, Any],
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint(path: str, model: torch.nn.Module, optimizer=None):
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"], checkpoint.get("metrics", {})


def iou_1d(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """
    Compute temporal IoU between predicted and ground truth segments.
    pred, gt: arrays of shape (N, 2) and (M, 2), columns are [start, end] in seconds.
    Returns IoU matrix of shape (N, M).
    """
    inter_start = np.maximum(pred[:, 0:1], gt[:, 0])   # (N, M)
    inter_end   = np.minimum(pred[:, 1:2], gt[:, 1])   # (N, M)
    inter       = np.clip(inter_end - inter_start, 0, None)

    pred_dur = pred[:, 1] - pred[:, 0]                 # (N,)
    gt_dur   = gt[:, 1]   - gt[:, 0]                   # (M,)
    union    = pred_dur[:, None] + gt_dur[None, :] - inter

    return inter / (union + 1e-8)