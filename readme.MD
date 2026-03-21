# THUMOS14 Temporal Action Detection

A two-stage pipeline for temporal action detection on the THUMOS14 benchmark, trained and evaluated on Modal cloud compute.

## Overview

This project detects and localizes actions in untrimmed videos across 20 action classes. Rather than training end-to-end, we use a two-stage approach:

1. **Feature Extraction** — A pre-trained video backbone (X3D-M or VideoMAE) encodes each video into a sequence of clip-level feature vectors. This is done once and the features are stored persistently.
2. **Temporal Head Training** — A lightweight transformer or CNN model is trained on the stored features to predict per-clip action labels. Since we only train the head, each training run takes ~20 minutes on an A100.

This approach is standard in the temporal action detection literature and makes iteration fast and cheap.

---

## Project Structure

```
thumos-action-recognition/
│
├── configs/
│   └── base_config.yaml          # All hyperparameters and paths — edit this to change runs
│
├── modal_pipeline/
│   ├── app.py                    # Shared Modal app/volume/image definitions (legacy, mostly unused)
│   ├── data_prep.py              # Downloads raw videos from Google Drive into Modal Volume
│   ├── extract_features_remote.py # Runs feature extraction on Modal GPU
│   ├── train_remote.py           # Runs temporal head training on Modal GPU
│   └── evaluate_remote.py        # Runs mAP evaluation on Modal GPU
│
├── src/
│   ├── dataset.py                # Dataset classes for raw video and pre-extracted features
│   ├── extract_features.py       # Feature extraction logic (called by extract_features_remote.py)
│   ├── train.py                  # Training loop (called by train_remote.py)
│   ├── evaluate.py               # mAP computation: compute_ap, compute_map
│   ├── utils.py                  # Logging, checkpointing, IoU utilities
│   └── models/
│       ├── __init__.py
│       ├── backbone.py           # Video backbones: X3D-M, VideoMAE-Base, VideoMAE-Large
│       └── temporal_model.py     # Temporal heads: TransformerModel, TemporalCNNModel
│
├── requirements.txt
└── README.md
```

---

## Dataset

We use **THUMOS14**, which contains untrimmed videos across 20 sports action classes. The data is stored in a persistent Modal Volume (`thumos-vol`) with the following layout:

```
thumos-vol/
├── raw/
│   ├── videos/
│   │   ├── val/         # 200 videos — used for training the temporal head
│   │   └── test/        # 213 videos — used for evaluation
│   └── annotations/
│       └── thumos_14_anno.json   # Action segment annotations from OpenTAD
│
├── features/
│   └── clip_level/      # Pre-extracted X3D-M features, one .pt file per video
│
├── checkpoints/         # Saved model weights, one per epoch + best.pt
└── logs/                # Training logs
```

**Important naming note:** In `thumos_14_anno.json`, the `training` subset corresponds to the `val/` videos on disk, and the `validation` subset corresponds to the `test/` videos on disk. This is standard THUMOS14 convention and is confusing — don't rename anything.

---

## Setup

**1. Install dependencies and authenticate with Modal:**
```bash
pip install -r requirements.txt
modal setup
```

**2. Set up rclone credentials for Google Drive access (one-time):**
```bash
rclone config   # follow prompts, name the remote "gdrive"
modal secret create rclone-config \
    RCLONE_CONFIG_CONTENT="$(cat ~/.config/rclone/rclone.conf)"
```

---

## Running the Pipeline

### Step 1 — Upload data to Modal Volume (one-time)
```bash
modal run modal_pipeline/data_prep.py
```

### Step 2 — Extract features (one-time per backbone)
```bash
modal run modal_pipeline/extract_features_remote.py
```
This takes ~35 minutes on an A100 for 200 training videos.

### Step 3 — Train the temporal head
```bash
modal run modal_pipeline/train_remote.py
```
Takes ~20 minutes on an A100 for 30 epochs.

### Step 4 — Evaluate
```bash
modal run modal_pipeline/evaluate_remote.py
```
Reports mAP at IoU thresholds 0.3, 0.4, 0.5, 0.6, 0.7.

### Step 5 — Download a checkpoint
```bash
modal volume get thumos-vol /checkpoints/best.pt ./local_checkpoints/best.pt
```

---

## Configuration

All hyperparameters live in `configs/base_config.yaml`. Key settings:

```yaml
model:
  backbone: "x3d_m"        # choices: x3d_m | videomae_base | videomae_large
  model: "transformer"      # choices: transformer | cnn
  num_classes: 20

data:
  clip_len_sec: 2.0         # length of each sliding window clip
  stride_sec: 1.0           # stride between clips
  window_size: 128          # number of clips per training sample

training:
  epochs: 30
  lr: 0.0001
  batch_size: 32
```

---

## Adding a New Model

To add a new backbone, add a branch to `build_backbone()` in `src/models/backbone.py`:

```python
elif backbone_name == "your_model":
    model = ...             # your model definition
    feature_dim = ...       # output feature dimension
    return model, feature_dim
```

To add a new temporal head, add a branch to `build_temporal_model()` in `src/models/temporal_model.py` and set `model: "your_model"` in `base_config.yaml`.

**Important:** If you change the backbone, you need to re-run feature extraction since the feature dimensions will be different. Store features in a new subdirectory (e.g. `/vol/features/videomae_base/`) to avoid overwriting X3D-M features, and update `feature_dir` in `base_config.yaml` accordingly.

---

## Evaluation Protocol

We follow the standard THUMOS14 evaluation protocol: mAP at IoU thresholds {0.3, 0.4, 0.5, 0.6, 0.7}. Per-clip logits from the temporal head are converted to temporal segments by thresholding sigmoid outputs and finding contiguous positive runs. mAP is computed per class and averaged.

---

## Modal Tips

- All training jobs are **ephemeral** — the app disappears after the job finishes, but data written to the Volume persists.
- Use `modal volume ls thumos-vol /path` to inspect Volume contents.
- Use `--detach` flag for long jobs if you don't want to keep your terminal open: `modal run --detach modal_pipeline/train_remote.py`
- GPU can be changed in each `_remote.py` file by editing the `GPU = "A100"` line. `"T4"` is cheapest for debugging, `"A100"` is fastest for full runs.
- Always call `volume.commit()` at the end of any function that writes to the Volume.