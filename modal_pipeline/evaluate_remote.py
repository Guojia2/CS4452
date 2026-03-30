"""
Evaluation on Modal — computes mAP@[0.3,0.4,0.5,0.6,0.7] on the test set.

Run:
  modal run modal_pipeline/evaluate_remote.py
  modal run modal_pipeline/evaluate_remote.py::main --checkpoint epoch_50.pt
"""

import modal
import os

from modal_pipeline.app import app, image, dataset_volume, work_volume, DATASET_PATH, WORK_PATH, GPU


@app.function(
    image=image,
    gpu=GPU,
    volumes={
        DATASET_PATH: dataset_volume,
        WORK_PATH:    work_volume,
    },
    timeout=60 * 60 * 2,
)
def run_evaluation(
    checkpoint: str = "best.pt",
    iou_thresholds: list = [0.3, 0.4, 0.5, 0.6, 0.7],
):
    import sys
    import yaml
    import torch
    sys.path.insert(0, "/root")
    work_volume.reload()

    from src.models.temporal_model import build_temporal_model
    from src.utils import get_logger, load_checkpoint
    from src.dataset import THUMOS14_CLASSES, CLASS_TO_IDX, load_thumos_annotations
    from src.evaluate import compute_map, nms_1d

    logger = get_logger("evaluation")
    device = torch.device("cuda")

    with open("/root/configs/base_config.yaml") as f:
        config = yaml.safe_load(f)

    config["paths"]["data_root"]      = f"{DATASET_PATH}/raw"
    config["paths"]["ann_path"]       = f"{DATASET_PATH}/raw/annotations/thumos_14_anno.json"
    config["paths"]["feature_dir"]    = f"{WORK_PATH}/features/clip_level"
    config["paths"]["checkpoint_dir"] = f"{WORK_PATH}/checkpoints"

    # Load model
    feature_dim = 2048
    model = build_temporal_model(config, feature_dim).to(device)
    ckpt_path = os.path.join(f"{WORK_PATH}/checkpoints", checkpoint)
    load_checkpoint(ckpt_path, model)
    model.eval()
    logger.info(f"Loaded checkpoint: {ckpt_path}")

    ann_path = config["paths"]["ann_path"]
    annotations = load_thumos_annotations(ann_path, subset="validation")
    logger.info(f"Annotated test videos: {len(annotations)}")

    feat_dir = config["paths"]["feature_dir"]
    feat_files = sorted(
        f for f in os.listdir(feat_dir)
        if f.endswith(".pt") and f.replace(".pt", "") in annotations
    )
    logger.info(f"Test feature files: {len(feat_files)}")

    all_predictions   = {c: [] for c in THUMOS14_CLASSES}
    all_ground_truths = {c: [] for c in THUMOS14_CLASSES}

    # Load ALL test set ground truths upfront (not just videos with features)
    for video_name, meta in annotations.items():
        for ann in meta.get("annotations", []):
            cls_name = ann["label"]
            if cls_name in CLASS_TO_IDX:
                s, e = ann["segment"]
                all_ground_truths[cls_name].append((video_name, s, e))

    total_gt = sum(len(v) for v in all_ground_truths.values())
    logger.info(f"Total GT segments across all {len(annotations)} test videos: {total_gt}")
    logger.info(f"Feature files available for inference: {len(feat_files)}")

    window_size = config["data"].get("window_size", 128)
    stride_sec  = config["data"].get("stride_sec", 1.0)

    with torch.no_grad():
        for feat_file in feat_files:
            video_name = feat_file.replace(".pt", "")
            features   = torch.load(os.path.join(feat_dir, feat_file), weights_only=True)
            n_clips    = features.shape[0]

            all_logits = torch.zeros(n_clips, len(THUMOS14_CLASSES))
            counts     = torch.zeros(n_clips, 1)

            for start in range(0, max(1, n_clips - window_size + 1), window_size // 2):
                end    = min(start + window_size, n_clips)
                window = features[start:end].unsqueeze(0).to(device)
                if window.shape[1] < window_size:
                    pad    = torch.zeros(1, window_size - window.shape[1], window.shape[2]).to(device)
                    window = torch.cat([window, pad], dim=1)
                logits = model(window).squeeze(0)[: end - start]
                all_logits[start:end] += logits.cpu()
                counts[start:end]     += 1

            all_logits /= counts.clamp(min=1)
            scores = torch.sigmoid(all_logits).numpy()

            threshold = 0.2
            for cls_idx, cls_name in enumerate(THUMOS14_CLASSES):
                cls_scores = scores[:, cls_idx]
                above = cls_scores >= threshold
                in_seg, seg_start = False, 0
                for t in range(n_clips):
                    if above[t] and not in_seg:
                        seg_start, in_seg = t, True
                    elif not above[t] and in_seg:
                        score = float(cls_scores[seg_start:t].mean())
                        all_predictions[cls_name].append(
                            (video_name, score, seg_start * stride_sec, t * stride_sec)
                        )
                        in_seg = False
                if in_seg:
                    score = float(cls_scores[seg_start:].mean())
                    all_predictions[cls_name].append(
                        (video_name, score, seg_start * stride_sec, n_clips * stride_sec)
                    )

    # Apply NMS per class
    for cls_name in THUMOS14_CLASSES:
        all_predictions[cls_name] = nms_1d(all_predictions[cls_name], iou_threshold=0.3)

    results = compute_map(all_predictions, all_ground_truths, iou_thresholds)

    logger.info("=" * 50)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 50)
    for k, v in results.items():
        logger.info(f"{k}: {v:.4f}")
    logger.info("=" * 50)

    # Persist results to the work volume as JSON
    import json
    results_dir = f"{WORK_PATH}/results"
    os.makedirs(results_dir, exist_ok=True)
    run_name = checkpoint.replace("/best.pt", "").replace(".pt", "").replace("/", "_")
    out_path = os.path.join(results_dir, f"{run_name}.json")
    with open(out_path, "w") as f:
        json.dump({"checkpoint": checkpoint, **results}, f, indent=2)
    work_volume.commit()
    logger.info(f"Results saved to {out_path}")

    return results


@app.local_entrypoint()
def eval_main():
    results = run_evaluation.remote(checkpoint="best.pt")
    print("\nFinal mAP Results:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")
