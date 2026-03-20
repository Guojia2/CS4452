import modal
import os

app = modal.App("thumos-action-recognition")
volume = modal.Volume.from_name("thumos-vol", create_if_missing=True)
VOLUME_MOUNT_PATH = "/vol"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.txt")
    .apt_install("ffmpeg")
    .add_local_dir("src", remote_path="/root/src")
    .add_local_dir("configs", remote_path="/root/configs")
)

GPU = "A100"


@app.function(
    image=image,
    gpu=GPU,
    volumes={VOLUME_MOUNT_PATH: volume},
    timeout=60 * 60 * 2,
)
def run_evaluation(
    checkpoint: str = "best.pt",
    config_path: str = "configs/base_config.yaml",
    iou_thresholds: list = [0.3, 0.4, 0.5, 0.6, 0.7],
):
    import sys
    import os
    import yaml
    import torch
    import numpy as np
    sys.path.insert(0, "/root")
    volume.reload()

    from src.models.temporal_model import build_temporal_model
    from src.utils import get_logger, load_checkpoint, iou_1d
    from src.dataset import THUMOS14_CLASSES, CLASS_TO_IDX, load_thumos_annotations

    logger = get_logger("evaluation")
    device = torch.device("cuda")

    with open(f"/root/{config_path}") as f:
        config = yaml.safe_load(f)

    config["paths"]["data_root"]      = os.path.join(VOLUME_MOUNT_PATH, "raw")
    config["paths"]["feature_dir"]    = os.path.join(VOLUME_MOUNT_PATH, "features", "test")
    config["paths"]["checkpoint_dir"] = os.path.join(VOLUME_MOUNT_PATH, "checkpoints")

    # --- Load model ---
    config["model"]["max_seq_len"] = 256
    feature_dim = 2048
    model = build_temporal_model(config, feature_dim).to(device)
    ckpt_path = os.path.join(VOLUME_MOUNT_PATH, "checkpoints", checkpoint)
    load_checkpoint(ckpt_path, model)
    model.eval()
    logger.info(f"Loaded checkpoint: {ckpt_path}")

    # --- Load test annotations (for ground truth segments) ---
    ann_path = os.path.join(VOLUME_MOUNT_PATH, "raw", "annotations", "thumos_14_anno.json")
    annotations = load_thumos_annotations(ann_path, subset="test")
    logger.info(f"Test videos with annotations: {len(annotations)}")

    # --- Run inference on test features ---
    feat_dir = config["paths"]["feature_dir"]
    feat_files = sorted([f for f in os.listdir(feat_dir) if f.endswith(".pt")])
    logger.info(f"Test feature files found: {len(feat_files)}")

    all_predictions  = {c: [] for c in THUMOS14_CLASSES}
    all_ground_truths = {c: [] for c in THUMOS14_CLASSES}

    window_size = config["data"].get("window_size", 128)
    stride_sec  = config["data"].get("stride_sec", 1.0)

    with torch.no_grad():
        for feat_file in feat_files:
            video_name = feat_file.replace(".pt", "")
            features = torch.load(os.path.join(feat_dir, feat_file))  # (N_clips, D)
            n_clips  = features.shape[0]

            # Run model over sliding windows
            all_logits = torch.zeros(n_clips, len(THUMOS14_CLASSES))
            counts     = torch.zeros(n_clips, 1)

            for start in range(0, max(1, n_clips - window_size + 1), window_size // 2):
                end = min(start + window_size, n_clips)
                window = features[start:end].unsqueeze(0).to(device)  # (1, W, D)

                # Pad if window is shorter than window_size
                if window.shape[1] < window_size:
                    pad = torch.zeros(
                        1, window_size - window.shape[1], window.shape[2]
                    ).to(device)
                    window = torch.cat([window, pad], dim=1)

                logits = model(window).squeeze(0)[:end - start]  # (W, C)
                all_logits[start:end] += logits.cpu()
                counts[start:end]     += 1

            # Average overlapping windows
            all_logits = all_logits / counts.clamp(min=1)
            scores = torch.sigmoid(all_logits).numpy()  # (N_clips, C)

            # Convert per-clip scores to segments using threshold
            threshold = 0.5
            for cls_idx, cls_name in enumerate(THUMOS14_CLASSES):
                cls_scores = scores[:, cls_idx]
                # Find contiguous runs above threshold
                above = cls_scores >= threshold
                in_seg = False
                seg_start = 0
                for t in range(n_clips):
                    if above[t] and not in_seg:
                        seg_start = t
                        in_seg = True
                    elif not above[t] and in_seg:
                        seg_end = t
                        score = float(cls_scores[seg_start:seg_end].mean())
                        start_sec = seg_start * stride_sec
                        end_sec   = seg_end   * stride_sec
                        all_predictions[cls_name].append(
                            (video_name, score, start_sec, end_sec)
                        )
                        in_seg = False
                if in_seg:
                    score = float(cls_scores[seg_start:].mean())
                    all_predictions[cls_name].append(
                        (video_name, score, seg_start * stride_sec, n_clips * stride_sec)
                    )

            # Accumulate ground truths from annotations
            if video_name in annotations:
                for ann in annotations[video_name].get("annotations", []):
                    cls_name = ann["label"]
                    if cls_name in CLASS_TO_IDX:
                        seg_start, seg_end = ann["segment"]
                        all_ground_truths[cls_name].append(
                            (video_name, seg_start, seg_end)
                        )

    # --- Compute mAP ---
    from src.evaluate import compute_map
    results = compute_map(all_predictions, all_ground_truths, iou_thresholds)

    logger.info("=" * 50)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 50)
    for k, v in results.items():
        logger.info(f"{k}: {v:.4f}")
    logger.info("=" * 50)

    return results


@app.local_entrypoint()
def main():
    results = run_evaluation.remote(checkpoint="epoch_29.pt") # change epoch_29.pt to best.pt once you fix the naming bug
    print("\nFinal mAP Results:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")