"""
eval.py — Evaluation for THUMOS-14 action recognition.

Primary metric: mean Average Precision (mAP) over all classes.
Also reports top-1 and top-5 accuracy.

Entry points
------------
- ``evaluate(cfg, checkpoint_path)`` : Full evaluation run.
- ``compute_map(scores, labels, num_classes)`` : Standalone mAP helper.
- ``main()`` : CLI entry-point for local evaluation.

Multi-clip inference
--------------------
For each video, ``num_test_clips`` uniformly-spaced clips are sampled,
softmax scores are averaged across clips, and the per-class scores are used
for both mAP and accuracy computation.
"""

from __future__ import annotations

import argparse
import pathlib
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.classifier import ActionRecognitionModel
from src.dataset import (
    THUMOSVideoDataset,
    NUM_CLASSES,
    THUMOS14_CLASSES,
    multi_clip_collate,
)
from src.utils import S3Store, get_logger, load_checkpoint

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# mAP helpers
# ---------------------------------------------------------------------------

def compute_map(
    scores: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    ignore_empty: bool = True,
) -> tuple[float, np.ndarray]:
    """Compute per-class Average Precision and mean AP (mAP).

    Parameters
    ----------
    scores       : Shape ``(N, num_classes)`` — predicted class probabilities.
    labels       : Shape ``(N,)``             — ground-truth integer class indices.
    num_classes  : Number of classes.
    ignore_empty : If True, skip classes with no positive samples in ``labels``
                   (avoids undefined AP from sklearn).

    Returns
    -------
    mAP          : float — macro-averaged AP over non-empty classes.
    per_class_ap : np.ndarray of shape ``(num_classes,)`` — per-class AP values
                   (``np.nan`` for skipped classes).
    """
    from sklearn.metrics import average_precision_score

    # One-hot encode labels
    one_hot = np.zeros((len(labels), num_classes), dtype=np.float32)
    for i, lbl in enumerate(labels):
        one_hot[i, lbl] = 1.0

    per_class_ap = np.full(num_classes, np.nan)
    valid_aps = []

    for c in range(num_classes):
        pos_count = one_hot[:, c].sum()
        if ignore_empty and pos_count == 0:
            continue
        ap = average_precision_score(one_hot[:, c], scores[:, c])
        per_class_ap[c] = ap
        valid_aps.append(ap)

    mAP = float(np.mean(valid_aps)) if valid_aps else 0.0
    return mAP, per_class_ap


def _topk_accuracy_np(scores: np.ndarray, labels: np.ndarray, k: int) -> float:
    """Top-k accuracy from numpy score matrix and integer labels."""
    topk_preds = np.argsort(scores, axis=1)[:, -k:]  # (N, k) largest
    correct = np.any(topk_preds == labels[:, None], axis=1)
    return float(correct.mean() * 100.0)


# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    multi_clip: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Run forward pass over the entire dataset and collect softmax scores.

    Parameters
    ----------
    model      : Loaded model in eval mode.
    loader     : DataLoader returning ``(clips, labels)`` or
                 ``(list[clip_tensor], labels)`` in multi-clip mode.
    device     : Inference device.
    multi_clip : True → each batch item is a list of clips; average softmax.

    Returns
    -------
    all_scores : ``(N, num_classes)`` numpy array of class probabilities.
    all_labels : ``(N,)`` numpy array of ground-truth labels.
    """
    model.eval()
    all_scores = []
    all_labels = []

    for batch_clips, batch_labels in loader:
        if multi_clip:
            # batch_clips : List[ (B, C, T, H, W) ] — one tensor per clip position
            clip_logits = []
            for clips in batch_clips:
                clips = clips.to(device, non_blocking=True)
                logits = model(clips)  # (B, num_classes)
                clip_logits.append(F.softmax(logits, dim=1))
            # Average softmax over all clip positions
            scores = torch.stack(clip_logits, dim=0).mean(dim=0)  # (B, num_classes)
        else:
            batch_clips = batch_clips.to(device, non_blocking=True)
            scores = F.softmax(model(batch_clips), dim=1)

        all_scores.append(scores.cpu().numpy())
        all_labels.append(batch_labels.numpy())

    all_scores = np.concatenate(all_scores, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return all_scores, all_labels


# ---------------------------------------------------------------------------
# Pretty-print AP table
# ---------------------------------------------------------------------------

def _print_ap_table(
    per_class_ap: np.ndarray,
    class_index: dict,
    num_classes: int,
) -> None:
    """Print a nicely formatted per-class AP table to stdout."""
    header = f"{'Idx':>4}  {'Class':<30}  {'AP':>7}"
    print("\n" + header)
    print("-" * len(header))
    for c in range(num_classes):
        class_name = class_index.get(c + 1, f"class_{c}")
        ap_str = f"{per_class_ap[c]:.4f}" if not np.isnan(per_class_ap[c]) else "  N/A "
        print(f"{c+1:>4}  {class_name:<30}  {ap_str:>7}")
    print()


# ---------------------------------------------------------------------------
# Main evaluate function (called by Modal or locally)
# ---------------------------------------------------------------------------

def evaluate(
    cfg: dict,
    checkpoint_path: str,
    split: str = "test",
    num_test_clips: int = 5,
    print_table: bool = True,
) -> dict:
    """Load a checkpoint and evaluate on the requested split.

    Parameters
    ----------
    cfg             : Merged config dict.
    checkpoint_path : Local path to ``.pt`` checkpoint file.
    split           : Dataset split to evaluate (``"val"`` or ``"test"``).
    num_test_clips  : Number of uniformly-spaced clips per video for
                      multi-clip inference.
    print_table     : Print per-class AP table to stdout if True.

    Returns
    -------
    results : Dict with keys ``mAP``, ``top1``, ``top5``, ``per_class_ap``.
    """
    # ------------------------------------------------------------------ setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Evaluating on {device} | split={split} | num_test_clips={num_test_clips}")

    model_cfg   = cfg.get("model",   {})
    data_cfg    = cfg.get("data",    {})
    paths_cfg   = cfg.get("paths",   {})
    storage_cfg = cfg.get("storage", {})

    num_classes    = int(model_cfg.get("num_classes",   NUM_CLASSES))
    backbone_name  = model_cfg.get("backbone",          "videomae_base")
    pooling        = model_cfg.get("pooling",            "mean")
    dropout        = float(model_cfg.get("dropout",     0.0))  # no dropout at eval

    clip_len       = int(data_cfg.get("clip_len",       16))
    frame_interval = int(data_cfg.get("frame_interval", 4))
    batch_size     = int(data_cfg.get("batch_size",     4))
    num_workers    = int(data_cfg.get("num_workers",    4))
    img_size       = int(data_cfg.get("img_size",       224))

    manifest_prefix = storage_cfg.get("manifest_prefix", "manifests")
    manifest_key    = f"{manifest_prefix}/{split}.jsonl"

    # ------------------------------------------------------------------ S3
    s3_store = S3Store.from_config(cfg)

    # ------------------------------------------------------------------ data
    multi_clip = num_test_clips > 1
    dataset = THUMOSVideoDataset(
        manifest_path  = manifest_key,
        s3_store       = s3_store,
        split          = split,
        clip_len       = clip_len,
        frame_interval = frame_interval,
        num_clips      = num_test_clips,
        img_size       = img_size,
    )

    collate_fn = multi_clip_collate if multi_clip else None
    loader = DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = True,
        collate_fn  = collate_fn,
    )

    logger.info(f"Eval dataset: {len(dataset)} videos")

    # ------------------------------------------------------------------ model
    model = ActionRecognitionModel(
        backbone_name = backbone_name,
        num_classes   = num_classes,
        pooling       = pooling,
        pretrained    = False,   # weights come from checkpoint
        dropout       = dropout,
    ).to(device)

    load_checkpoint(checkpoint_path, model, optimizer=None, device=device)
    logger.info(f"Loaded checkpoint: {checkpoint_path}")

    # ------------------------------------------------------------------ infer
    all_scores, all_labels = run_inference(model, loader, device, multi_clip=multi_clip)

    # ------------------------------------------------------------------ mAP
    mAP, per_class_ap = compute_map(all_scores, all_labels, num_classes)
    top1 = _topk_accuracy_np(all_scores, all_labels, k=1)
    top5 = _topk_accuracy_np(all_scores, all_labels, k=5)

    logger.info(
        f"Results  mAP={mAP:.4f}  top-1={top1:.2f}%  top-5={top5:.2f}%"
    )

    if print_table:
        _print_ap_table(per_class_ap, THUMOS14_CLASSES, num_classes)

    results = {
        "mAP":          mAP,
        "top1":         top1,
        "top5":         top5,
        "per_class_ap": per_class_ap,
    }
    return results


# ---------------------------------------------------------------------------
# CLI entry-point (local evaluation / no Modal)
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate action recognition model locally")
    parser.add_argument(
        "--config", default="configs/base_config.yaml",
        help="Path to YAML config file"
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to model checkpoint (.pt)"
    )
    parser.add_argument(
        "--split", default="test", choices=["val", "test"],
        help="Dataset split to evaluate"
    )
    parser.add_argument(
        "--num-clips", type=int, default=5,
        help="Number of clips per video for multi-clip inference"
    )
    parser.add_argument(
        "--override", nargs="*", default=[],
        metavar="KEY=VALUE",
        help="Override config keys"
    )
    return parser.parse_args()


def main() -> None:
    import yaml

    args = parse_args()
    with open(args.config, "r") as fh:
        cfg = yaml.safe_load(fh)

    # Reuse _apply_overrides from train.py
    from src.train import _apply_overrides
    cfg = _apply_overrides(cfg, args.override)

    evaluate(cfg, args.checkpoint, split=args.split, num_test_clips=args.num_clips)


if __name__ == "__main__":
    main()
