"""
modal_pipeline/app.py — Modal App for THUMOS-14 training and evaluation.

Resources
---------
- GPU:    A10G for training, T4 for evaluation
- Volume: "thumos14-vol" mounted at /vol — stores checkpoints and logs
- Secret: "aws-secret" — provides AWS credentials for S3 access
- Image:  debian slim + all project dependencies pre-installed

Bucket migration
----------------
Change ``storage.bucket`` in ``configs/base_config.yaml`` (or pass an
override like ``storage.bucket=my-new-bucket``).  No other code changes needed.

Usage (from repo root)
----------------------
# Run training
modal run modal_pipeline/app.py

# Run evaluation on a saved checkpoint
modal run modal_pipeline/app.py::run_eval --checkpoint-path /vol/checkpoints/best.pt

# Override config values
modal run modal_pipeline/app.py -- --override model.backbone=x3d_m training.epochs=5
"""

from __future__ import annotations

import pathlib
import sys

import modal

# ---------------------------------------------------------------------------
# App & shared resources
# ---------------------------------------------------------------------------

APP_NAME = "thumos14-action-recognition"

app = modal.App(APP_NAME)

# Persistent volume — stores checkpoints, logs, and any cached data
volume = modal.Volume.from_name("thumos14-vol", create_if_missing=True)
VOLUME_MOUNT = "/vol"

# AWS credentials for S3 access (set up once via `modal secret create aws-secret`)
aws_secret = modal.Secret.from_name("aws-secret")

# ---------------------------------------------------------------------------
# Container image
# ---------------------------------------------------------------------------

# Pin transformers for reproducibility; rest can float
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.2.2",
        "torchvision==0.17.2",
        "transformers>=4.40.0",
        "timm>=0.9.0",
        "einops",
        "pytorchvideo",
        "pyyaml",
        "av",           # PyAV — video decoding
        "boto3",        # S3 access
        "scikit-learn", # mAP computation
        "tqdm",
        "Pillow",
        "numpy",
    )
    # Copy project source into the image so Modal can import it
    .add_local_dir("src",     remote_path="/root/src")
    .add_local_dir("configs", remote_path="/root/configs")
    .add_local_dir("data",    remote_path="/root/data")
)

# ---------------------------------------------------------------------------
# Config loading helper (runs inside Modal container)
# ---------------------------------------------------------------------------

def _load_cfg(config_path: str = "/root/configs/base_config.yaml", overrides: list[str] | None = None) -> dict:
    import yaml
    with open(config_path, "r") as fh:
        cfg = yaml.safe_load(fh)

    # Wire Modal Volume paths into config if not already set
    cfg.setdefault("paths", {})
    cfg["paths"].setdefault("checkpoint_dir", f"{VOLUME_MOUNT}/checkpoints")
    cfg["paths"].setdefault("log_dir",        f"{VOLUME_MOUNT}/logs")

    if overrides:
        # Import here to avoid circular dependency at module load time
        sys.path.insert(0, "/root")
        from src.train import _apply_overrides
        cfg = _apply_overrides(cfg, overrides)

    return cfg


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

@app.function(
    image      = image,
    gpu        = "A10G",
    volumes    = {VOLUME_MOUNT: volume},
    secrets    = [aws_secret],
    timeout    = 86400,   # 24 h max
    memory     = 32768,   # 32 GB RAM
)
def run_training(
    config_path: str = "/root/configs/base_config.yaml",
    overrides: list[str] | None = None,
) -> float:
    """Train the model on Modal A10G GPU.

    Parameters
    ----------
    config_path : Path to YAML config inside the container.
    overrides   : List of ``"key=value"`` strings to override config keys.

    Returns
    -------
    Best validation top-1 accuracy.
    """
    import sys
    sys.path.insert(0, "/root")

    from src.train import train

    cfg = _load_cfg(config_path, overrides)
    best_top1 = train(cfg)
    volume.commit()   # flush Volume writes
    return best_top1


# ---------------------------------------------------------------------------
# Evaluation function
# ---------------------------------------------------------------------------

@app.function(
    image   = image,
    gpu     = "T4",
    volumes = {VOLUME_MOUNT: volume},
    secrets = [aws_secret],
    timeout = 7200,   # 2 h
)
def run_eval(
    checkpoint_path: str = f"{VOLUME_MOUNT}/checkpoints/best.pt",
    split: str = "test",
    num_test_clips: int = 5,
    config_path: str = "/root/configs/base_config.yaml",
    overrides: list[str] | None = None,
) -> dict:
    """Evaluate a saved checkpoint on Modal T4 GPU.

    Parameters
    ----------
    checkpoint_path : Path inside the Modal Volume (e.g. ``/vol/checkpoints/best.pt``).
    split           : Dataset split — ``"val"`` or ``"test"``.
    num_test_clips  : Number of clips per video for multi-clip inference.
    config_path     : Path to YAML config inside the container.
    overrides       : ``key=value`` config overrides.

    Returns
    -------
    Dict with ``mAP``, ``top1``, ``top5``, ``per_class_ap``.
    """
    import sys
    sys.path.insert(0, "/root")

    from src.eval import evaluate

    cfg = _load_cfg(config_path, overrides)
    results = evaluate(cfg, checkpoint_path, split=split, num_test_clips=num_test_clips)
    return results


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    mode: str = "train",
    checkpoint_path: str = f"{VOLUME_MOUNT}/checkpoints/best.pt",
    split: str = "test",
    num_clips: int = 5,
    override: list[str] = [],
):
    """Dispatch training or evaluation to Modal.

    Parameters
    ----------
    mode            : ``"train"`` or ``"eval"``.
    checkpoint_path : Checkpoint path for eval mode.
    split           : Eval split (``"val"`` or ``"test"``).
    num_clips       : Multi-clip count for eval.
    override        : Config key=value overrides (repeatable).

    Examples
    --------
    modal run modal_pipeline/app.py
    modal run modal_pipeline/app.py -- --mode eval --checkpoint-path /vol/checkpoints/best.pt
    modal run modal_pipeline/app.py -- --override model.backbone=x3d_m training.epochs=5
    """
    overrides = list(override) if override else []

    if mode == "train":
        print(f"Launching training on Modal A10G  (overrides: {overrides})")
        result = run_training.remote(overrides=overrides or None)
        print(f"Training complete. Best val top-1: {result:.2f}%")

    elif mode == "eval":
        print(f"Launching evaluation on Modal T4  checkpoint={checkpoint_path}")
        results = run_eval.remote(
            checkpoint_path = checkpoint_path,
            split           = split,
            num_test_clips  = num_clips,
            overrides       = overrides or None,
        )
        print(f"\nmAP  = {results['mAP']:.4f}")
        print(f"Top-1= {results['top1']:.2f}%")
        print(f"Top-5= {results['top5']:.2f}%")

    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose 'train' or 'eval'.")
