# ActionFormer: Localizing Actions with Transformers

This repository contains the implementation of **ActionFormer**, a powerful transformer-based model for temporal action detection (TAD), along with a convolutional baseline and support for multiple backbones (I3D and VideoMAE v2).

## Model Architectures

### 1. ActionFormer (LocPointTransformer)
ActionFormer is a purely transformer-based architecture designed for single-stage temporal action localization. It represents a shift from traditional anchor-based or convolutional detectors by leveraging the power of transformers to model long-range temporal dependencies.

- **Encoder**: Utilizes a stack of Transformer blocks with **Local Multi-Head Self-Attention (LMHSA)**. This allows the model to handle long video sequences by restricting attention to a local window, ensuring computational efficiency.
- **Feature Pyramid Network (FPN)**: Employs a multi-scale feature pyramid to detect actions of varying durations (from very short to very long).
- **Detection Heads**: 1D convolutional heads that perform point-based classification and regression (predicting distances to the start and end of action segments).
- **Training Objective**: Optimized using a combination of sigmoid focal loss for classification and Distance-IoU (DIoU) loss for regression.

### 2. Baseline (Convolutional TAD)
A simplified, convolution-only variant of ActionFormer. It serves as a strong baseline to highlight the benefits of the transformer encoder.

- **Architecture**: Replaces the Transformer blocks in the encoder with **1D Convolutional layers**.
- **Pyramid**: Uses an Identity or simplified FPN.
- **Heads**: Same point-based detection heads as ActionFormer.
- **Efficiency**: Characterized by faster training and lower memory footprint, suitable for faster prototyping.

---

## Feature Extraction (Backbones)

We compare two distinct backbones for extracting the underlying video features used by the detection models.

### I3D (Inflated 3D ConvNet)
- **Architecture**: Based on Inception-v1 inflated into 3D space.
- **Feature Dimension**: 2048-dimensional (RGB stream).
- **Pre-training**: Typically pre-trained on Kinetics-400.
- **Strengths**: Excels at capturing local motion and appearance features in a dense, short-range temporal window.

### VideoMAE V2 (Video Masked Autoencoder)
- **Architecture**: Vision Transformer (ViT-Base).
- **Feature Dimension**: 768-dimensional.
- **Pre-training**: Large-scale self-supervised pre-training using Masked Autoencoding.
- **Strengths**: Provides superior global temporal context and modeling capabilities due to the ViT architecture and massive pre-training scope.

---

## 2x2 Comparison: Feature Extraction vs. TAD Models

The following grid summarizes the interaction between the feature backbones and the temporal action detection models.

| | **I3D Features (2048D)** | **VideoMAE V2 Features (768D)** |
|:---:|:---|:---|
| **Baseline (CNN)** | **Standard Baseline**: Uses local 3D-CNN features with a 1D-CNN detector. Fastest and least computationally demanding. | **Modern Baseline**: Leverages superior ViT features but uses a simpler CNN-based detection pipeline. |
| **ActionFormer (Transformer)** | **Classic ActionFormer**: Multi-scale transformer detection on top of 3D-CNN features. A robust and high-performing setup. | **SOTA Pipeline**: The most powerful combination. Transformer-based detection on top of large-scale ViT features for maximum accuracy. |

---

## Usage

### Feature Extraction
Features can be extracted using the Modal pipeline:
```bash
modal run modal_pipeline/modal_extract.py --clip-len 16 --clip-stride 8
```

### Training
Train the model using the provided configurations:
```bash
# ActionFormer with VideoMAE v2
modal run modal_pipeline/modal_train.py --config-path configs/thumos_videomae2_actionformer.yaml

# Baseline with I3D
modal run modal_pipeline/modal_train.py --config-path configs/thumos_i3d_baseline.yaml
```

### Evaluation
Evaluate a trained checkpoint:
```bash
modal run modal_pipeline/modal_eval.py --config-path configs/thumos_videomae2_actionformer.yaml --resume /vol/ckpt/ActionFormer/checkpoint.pth.tar
```
