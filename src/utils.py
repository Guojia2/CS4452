"""
utils.py — Shared utilities for CS4452 THUMOS-14 pipeline.

Contents
--------
- get_logger             : Configured stdlib logger
- S3Store                : Bucket-agnostic S3 I/O helper (change bucket via config/env)
- save_checkpoint        : Persist model + optimiser state to a local path
- load_checkpoint        : Restore model + optimiser from a checkpoint file
- iou_1d                 : Temporal intersection-over-union for two intervals
- load_class_index       : Parse data/Class Index.txt → {int: str}
- AverageMeter           : Running mean tracker for loss/accuracy logging
"""

from __future__ import annotations

import io
import logging
import os
import pathlib
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

def get_logger(name: str = "thumos14", level: int = logging.INFO) -> logging.Logger:
    """Return a module-level logger with a consistent format.

    Parameters
    ----------
    name  : Logger name — use ``__name__`` in calling modules.
    level : Logging level (default INFO).
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


# ---------------------------------------------------------------------------
# S3 Store — single place to swap the bucket
# ---------------------------------------------------------------------------

class S3Store:
    """Bucket-agnostic S3 I/O wrapper.

    All S3 operations in the project go through this class.  To migrate to a
    different bucket, update the ``bucket`` constructor argument (or the
    ``storage.bucket`` config key from which it is sourced) — **no other code
    needs to change**.

    Parameters
    ----------
    bucket : str
        S3 bucket name.
    region : str
        AWS region, e.g. ``"us-east-1"``.
    prefix : str
        Optional key prefix prepended to every operation (useful for
        namespace isolation inside a shared bucket).
    """

    def __init__(
        self,
        bucket: str,
        region: str = "us-east-1",
        prefix: str = "",
    ) -> None:
        import boto3  # lazy import so utils is importable without boto3 locally

        self.bucket = bucket
        self.region = region
        self.prefix = prefix.rstrip("/")

        self._client = boto3.client("s3", region_name=region)

    # ------------------------------------------------------------------
    # Internal key helper
    # ------------------------------------------------------------------

    def _full_key(self, key: str) -> str:
        if self.prefix:
            return f"{self.prefix}/{key.lstrip('/')}"
        return key.lstrip("/")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def upload_file(self, local_path: str, s3_key: str) -> None:
        """Upload a local file to S3."""
        full_key = self._full_key(s3_key)
        self._client.upload_file(local_path, self.bucket, full_key)

    def download_file(self, s3_key: str, local_path: str) -> None:
        """Download an S3 object to a local file."""
        full_key = self._full_key(s3_key)
        pathlib.Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        self._client.download_file(self.bucket, full_key, local_path)

    def read_bytes(self, s3_key: str) -> bytes:
        """Read an S3 object and return its raw bytes (streams into memory)."""
        full_key = self._full_key(s3_key)
        response = self._client.get_object(Bucket=self.bucket, Key=full_key)
        return response["Body"].read()

    def write_bytes(self, s3_key: str, data: bytes) -> None:
        """Write raw bytes to an S3 object."""
        full_key = self._full_key(s3_key)
        self._client.put_object(Bucket=self.bucket, Key=full_key, Body=data)

    def write_text(self, s3_key: str, text: str, encoding: str = "utf-8") -> None:
        """Write a text string to an S3 object."""
        self.write_bytes(s3_key, text.encode(encoding))

    def read_text(self, s3_key: str, encoding: str = "utf-8") -> str:
        """Read an S3 object and decode it as text."""
        return self.read_bytes(s3_key).decode(encoding)

    def list_prefix(self, prefix: str) -> List[str]:
        """Return all object keys that start with *prefix* (relative to self.prefix)."""
        full_prefix = self._full_key(prefix)
        paginator = self._client.get_paginator("list_objects_v2")
        keys: List[str] = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=full_prefix):
            for obj in page.get("Contents", []):
                k = obj["Key"]
                # Strip the store-level prefix so callers get relative keys
                if self.prefix:
                    k = k[len(self.prefix) + 1 :]
                keys.append(k)
        return keys

    def exists(self, s3_key: str) -> bool:
        """Return True if the S3 object exists."""
        import botocore.exceptions

        full_key = self._full_key(s3_key)
        try:
            self._client.head_object(Bucket=self.bucket, Key=full_key)
            return True
        except botocore.exceptions.ClientError:
            return False

    @classmethod
    def from_config(cls, cfg: dict) -> "S3Store":
        """Construct from the ``storage`` section of base_config.yaml.

        Expected keys under ``cfg["storage"]``:
            bucket, region, prefix (optional)
        """
        storage = cfg["storage"]
        return cls(
            bucket=storage["bucket"],
            region=storage.get("region", "us-east-1"),
            prefix=storage.get("prefix", ""),
        )


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    state: dict,
    path: str,
    is_best: bool = False,
    best_path: Optional[str] = None,
) -> None:
    """Save a training checkpoint.

    Parameters
    ----------
    state     : Dict containing ``model_state_dict``, ``optimizer_state_dict``,
                ``epoch``, ``best_metric``, and any extra keys.
    path      : Absolute path to write the checkpoint (e.g., inside Modal Volume).
    is_best   : If True, also copy to *best_path*.
    best_path : Destination for the best checkpoint copy.
    """
    import shutil

    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    if is_best and best_path:
        pathlib.Path(best_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, best_path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> dict:
    """Load a checkpoint and restore model (and optionally optimiser) state.

    Parameters
    ----------
    path      : Checkpoint file path.
    model     : The model to restore state into.
    optimizer : Optional optimiser to restore state into.
    device    : Map location; defaults to CPU if None.

    Returns
    -------
    The full checkpoint dict (contains ``epoch``, ``best_metric``, etc.).
    """
    map_location = device or torch.device("cpu")
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt


# ---------------------------------------------------------------------------
# Temporal IoU
# ---------------------------------------------------------------------------

def iou_1d(
    pred: Tuple[float, float],
    gt: Tuple[float, float],
) -> float:
    """Temporal intersection-over-union for two 1-D intervals.

    Parameters
    ----------
    pred, gt : ``(start, end)`` tuples in the same time unit.

    Returns
    -------
    float in [0, 1].
    """
    inter_start = max(pred[0], gt[0])
    inter_end   = min(pred[1], gt[1])
    inter       = max(0.0, inter_end - inter_start)

    union = (pred[1] - pred[0]) + (gt[1] - gt[0]) - inter
    if union <= 0:
        return 0.0
    return inter / union


# ---------------------------------------------------------------------------
# Class index parser
# ---------------------------------------------------------------------------

def load_class_index(path: str) -> Dict[int, str]:
    """Parse ``data/Class Index.txt`` into ``{1-based-index: class_name}``.

    File format (one entry per line)::

        1 ApplyEyeMakeup
        2 ApplyLipstick
        ...

    Parameters
    ----------
    path : Absolute or relative path to the class index text file.

    Returns
    -------
    Dict mapping 1-based integer index to class name string.
    """
    index: Dict[int, str] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                index[int(parts[0])] = parts[1]
    return index


# ---------------------------------------------------------------------------
# AverageMeter — running mean for loss / accuracy logging
# ---------------------------------------------------------------------------

class AverageMeter:
    """Computes and stores a running mean of a scalar value."""

    def __init__(self, name: str = "") -> None:
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.val   = 0.0
        self.avg   = 0.0
        self.sum   = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count

    def __repr__(self) -> str:
        return f"AverageMeter({self.name}: val={self.val:.4f}, avg={self.avg:.4f})"
