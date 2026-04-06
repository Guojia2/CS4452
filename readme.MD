# THUMOS14 Temporal Action Detection: I3D and VideoMAE v2 Backbones

This repository contains the code for a controlled 2×2 study comparing two feature backbones (I3D, VideoMAE v2) and two temporal detection heads (ActionFormer-style conv-transformer, CNN baseline) on the THUMOS14 benchmark. All four configurations share the same localization framework, training objective, and evaluation protocol — only the detection head is trained (features are pre-extracted and frozen).

---

## Model Architectures

### 1. ActionFormer (`backbone_type: convTransformer`)
A conv-transformer architecture. The input stage embeds clip features with 1D convolutional layers, followed by transformer blocks that use **Local Multi-Head Self-Attention (LMHSA)** with a window size of 19 (±9 neighbors). A multi-scale temporal feature pyramid (6 levels) is produced through downsampling transformer stages. Classification and regression heads (3 conv layers each) are applied densely at each temporal location.

- **Loss**: Sigmoid Focal Loss (α=0.25, γ=2.0) + Distance-IoU (DIoU) regression loss
- **Attention**: Local windowed self-attention, not global — computationally efficient over long sequences

### 2. CNN Baseline (`backbone_type: conv`)
A purely convolutional model sourced from the ActionFormer codebase. The encoder uses ResNet-style residual 1D temporal convolution blocks (`ConvBlock`) with no attention mechanism and no positional encoding. A 4-level feature pyramid is built through progressive temporal downsampling. Classification and regression heads (2 conv layers each) perform dense prediction.

- **Same loss functions** as ActionFormer
- No self-attention, no positional encoding

---

## Feature Backbones

Both backbones are used in **pre-extraction mode only** — features are extracted once, saved to disk, and loaded during training. No backbone weights are updated during head training.

### I3D (Inflated 3D ConvNet)
- **Model**: Two-stream Inflated 3D ConvNet pre-trained on Kinetics-400
- **Feature dimension**: 2048
- **Clip length**: 16 frames
- **Feature stride**: 4 (ActionFormer config), 16 (CNN Baseline config)
- **Features location (Modal volume)**: `/vol/features/pretrained/i3d_features`
- **Note**: I3D features were obtained from a pre-existing source; no I3D extraction script is included in this repo.

### VideoMAE v2 (`OpenGVLab/VideoMAEv2-Base`)
- **Model**: ViT-Base masked autoencoder, loaded via HuggingFace (`AutoModel`)
- **Feature dimension**: 768
- **Clip length**: 16 frames
- **Feature stride**: 8 (both ActionFormer and CNN Baseline configs)
- **Features location (Modal volume)**: `/vol/features/videomae2-new`
- **Extraction script**: `modal_pipeline/modal_extract.py`

---

## 2×2 Experimental Results

The table below shows the four configurations and their **actual results** from the paper.

| | **I3D Features (2048-dim, stride varies)** | **VideoMAE v2 Features (768-dim, stride 8)** |
|:---|:---|:---|
| **CNN Baseline** (4-level pyramid, stride 16 for I3D / 8 for VMv2) | **BL+I3D**: 1.66% avg mAP | **BL+VMv2**: 4.04% avg mAP |
| **ActionFormer** (6-level pyramid, stride 4 for I3D / 8 for VMv2) | **AF+I3D**: 66.76% avg mAP | **AF+VMv2**: 4.57% avg mAP |

**Key finding**: ActionFormer collapses with VideoMAE v2 features (66.76% → 4.57%) because it is tightly coupled to I3D feature statistics. No hyperparameter re-tuning was performed for VideoMAE v2. The CNN baseline, having no feature-specific tuning, correctly shows VideoMAE v2 is a stronger backbone in isolation (4.04% vs. 1.66%).

---

## Config Files

| Config file | Backbone | Head | Input dim | Feat stride | Pyramid levels | Output folder |
|---|---|---|---|---|---|---|
| `configs/thumos_i3d.yaml` | I3D | ActionFormer | 2048 | 4 | 6 | `/vol/ckpt/ActionFormer/` |
| `configs/thumos_i3d_baseline.yaml` | I3D | CNN Baseline | 2048 | 16 | 4 | `/vol/ckpt/Baseline-I3D/` |
| `configs/thumos_videomae2_actionformer.yaml` | VideoMAE v2 | ActionFormer | 768 | 8 | 6 | `/vol/ckpt/ActionFormer-Videomae2/` |
| `configs/thumos_videomae2_baseline.yaml` | VideoMAE v2 | CNN Baseline | 768 | 8 | 4 | `/vol/ckpt/Baseline-Videomae2/` |

All configs share: `lr=1e-4`, `weight_decay=0.05`, `batch_size=2`, `epochs=30` (+5 warmup), `AdamW`, cosine annealing schedule, EMA decay=0.999.

---

## Replication Guide

### Prerequisites
- [Modal](https://modal.com/) account and CLI installed (`pip install modal`, `modal setup`)
- A Modal volume named `Thumos14` containing:
  - `/vol/annotations/thumos14_af.json` — THUMOS14 annotation file
  - `/vol/features/pretrained/i3d_features/` — pre-extracted I3D `.npy` files (one per video)
  - `/vol/npy/validation_npy/` and `/vol/npy/test_npy/` — raw video frames as `.npy` arrays (needed only for VideoMAE v2 extraction)

---

### Step 1: Extract VideoMAE v2 Features (skip if using I3D only)

Features are extracted for both the `validation` (train) and `test` splits, parallelized across 4 shards per split.

```bash
modal run modal_pipeline/modal_extract.py --clip-len 16 --clip-stride 8
```

This saves features to `/vol/features/videomae2-new/<video_name>.npy` (shape: `[num_clips, 768]`).

---

### Step 2: Train

Each configuration is trained independently. Pass the config file path. Checkpoints are saved to the `output_folder` specified in the config, inside a timestamped subfolder named `<config_name>_<timestamp>/`.

Checkpoints are saved every 5 epochs (default `--ckpt-freq 5`). The final checkpoint after all 35 epochs (30 training + 5 warmup) is `epoch_035.pth.tar`.

```bash
# ActionFormer + I3D
modal run modal_pipeline/modal_train.py --config-path configs/thumos_i3d.yaml

# CNN Baseline + I3D
modal run modal_pipeline/modal_train.py --config-path configs/thumos_i3d_baseline.yaml

# ActionFormer + VideoMAE v2
modal run modal_pipeline/modal_train.py --config-path configs/thumos_videomae2_actionformer.yaml

# CNN Baseline + VideoMAE v2
modal run modal_pipeline/modal_train.py --config-path configs/thumos_videomae2_baseline.yaml
```

Optional flags:
- `--output <name>` — use a fixed name for the checkpoint subfolder instead of a timestamp
- `--ckpt-freq <n>` — save a checkpoint every n epochs (default: 5)
- `--resume /vol/ckpt/.../epoch_010.pth.tar` — resume from a checkpoint

---

### Step 3: Evaluate

The eval script accepts either a **path to a specific `.pth.tar` file** or a **path to a checkpoint directory** (it will automatically use the last checkpoint in the directory).

```bash
# Evaluate a specific checkpoint file
modal run modal_pipeline/modal_eval.py \
  --config-path configs/thumos_i3d.yaml \
  --ckpt-path /vol/ckpt/ActionFormer/thumos_i3d_<timestamp>/epoch_035.pth.tar

# Evaluate using a checkpoint directory (picks the last checkpoint automatically)
modal run modal_pipeline/modal_eval.py \
  --config-path configs/thumos_i3d.yaml \
  --ckpt-path /vol/ckpt/ActionFormer/thumos_i3d_<timestamp>
```

Replace `<timestamp>` with the actual folder name created during training (format: `YYYY-MM-DD HH:MM:SS`).

Config and checkpoint must match — use the same config file that was used for training.

| Config | Checkpoint base path |
|---|---|
| `thumos_i3d.yaml` | `/vol/ckpt/ActionFormer/` |
| `thumos_i3d_baseline.yaml` | `/vol/ckpt/Baseline-I3D/` |
| `thumos_videomae2_actionformer.yaml` | `/vol/ckpt/ActionFormer-Videomae2/` |
| `thumos_videomae2_baseline.yaml` | `/vol/ckpt/Baseline-Videomae2/` |

---

## Dataset

- **THUMOS14** — 413 untrimmed sports videos, 20 action classes
- **Training split**: 200 `validation` videos (3003 annotated segments)
- **Test split**: 213 videos on disk, 211 with ground-truth annotations used for mAP evaluation
- **Evaluation**: mAP at tIoU ∈ {0.3, 0.4, 0.5, 0.6, 0.7}, averaged

---

## Post-Processing (inference)

All configs use the following defaults (defined in `libs/core/config.py`; not present in YAML files):

| Parameter | Value |
|---|---|
| NMS method | Soft-NMS (linear decay) |
| IoU threshold | 0.1 |
| Soft-NMS σ | 0.5 |
| Pre-NMS score threshold | 0.001 |
| Pre-NMS top-k | 2000 |
| Post-NMS max segments | 200 |
| Min score (post-NMS) | 0.001 |
| Min duration threshold | 0.05 |
| Multiclass NMS | True (per-class) |
| Segment voting | Disabled (incompatible with multiclass NMS) |
