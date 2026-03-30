"""
Automated augmentation search — trains one run per augmentation, evaluates each,
then prints a comparison table.

Each experiment enables exactly one augmentation (all others off via base_config defaults).
Runs are sequential: train → evaluate → next experiment.

Run:
  modal run modal_pipeline/search_augmentations.py

Resume after interruption (skips training if checkpoint already exists):
  modal run modal_pipeline/search_augmentations.py --resume
"""

import modal
import os

from modal_pipeline.app import app, image, dataset_volume, work_volume, DATASET_PATH, WORK_PATH, GPU

# ── Experiment definitions ────────────────────────────────────────────────────

EXPERIMENTS = [
    ("baseline",          {}),
    ("window_jitter",     {"window_jitter": True}),
    ("balanced_sampling", {"class_balanced_sampling": True}),
    ("gaussian_noise",    {"gaussian_noise": {"enabled": True}}),
    ("feature_dropout",   {"feature_dropout": {"enabled": True}}),
    ("time_mask",         {"time_mask": {"enabled": True}}),
    ("feature_mask",      {"feature_mask": {"enabled": True}}),
    ("speed_perturb",     {"speed_perturb": {"enabled": True}}),
    ("mixup",             {"mixup": {"enabled": True}}),
    ("cutpaste",          {"cutpaste": {"enabled": True}}),
]

IOU_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]


# ── Remote functions defined here so Modal hydrates them in this app bundle ──

@app.function(
    image=image,
    gpu=GPU,
    volumes={DATASET_PATH: dataset_volume, WORK_PATH: work_volume},
    timeout=60 * 60 * 24,
    retries=1,
)
def run_training(
    aug_overrides: dict = {},
    run_name: str = "baseline",
    save_every_epoch: bool = True,
):
    import sys
    import yaml
    sys.path.insert(0, "/root")
    work_volume.reload()

    from src.train import train

    with open("/root/configs/base_config.yaml") as f:
        config = yaml.safe_load(f)

    config["paths"]["data_root"]      = f"{DATASET_PATH}/raw"
    config["paths"]["ann_path"]       = f"{DATASET_PATH}/raw/annotations/thumos_14_anno.json"
    config["paths"]["feature_dir"]    = f"{WORK_PATH}/features/clip_level"
    config["paths"]["checkpoint_dir"] = f"{WORK_PATH}/checkpoints/{run_name}"
    config["paths"]["log_dir"]        = f"{WORK_PATH}/logs/{run_name}"

    os.makedirs(config["paths"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(config["paths"]["log_dir"], exist_ok=True)

    aug_cfg = config.get("augmentation", {})
    for k, v in aug_overrides.items():
        if isinstance(v, dict) and isinstance(aug_cfg.get(k), dict):
            aug_cfg[k].update(v)
        else:
            aug_cfg[k] = v
    config["augmentation"] = aug_cfg
    config["training"]["save_every_epoch"] = save_every_epoch

    train(config)

    # Close all logger file handlers so the volume can be reloaded by the next run
    import logging
    for name in list(logging.Logger.manager.loggerDict):
        lgr = logging.getLogger(name)
        for handler in lgr.handlers[:]:
            handler.close()
            lgr.removeHandler(handler)

    work_volume.commit()


@app.function(
    image=image,
    gpu=GPU,
    volumes={DATASET_PATH: dataset_volume, WORK_PATH: work_volume},
    timeout=60 * 60 * 2,
)
def run_evaluation(
    checkpoint: str = "best.pt",
    iou_thresholds: list = [0.3, 0.4, 0.5, 0.6, 0.7],
):
    import sys
    import yaml
    import torch
    import json
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

    for k, v in results.items():
        logger.info(f"{k}: {v:.4f}")

    results_dir = f"{WORK_PATH}/results"
    os.makedirs(results_dir, exist_ok=True)
    run_name = checkpoint.replace("/best.pt", "").replace(".pt", "").replace("/", "_")
    out_path = os.path.join(results_dir, f"{run_name}.json")
    with open(out_path, "w") as f:
        json.dump({"checkpoint": checkpoint, **results}, f, indent=2)
    work_volume.commit()

    return results


# ── Local helpers ─────────────────────────────────────────────────────────────

def _checkpoint_exists(ckpt_relative: str) -> bool:
    """Check whether a checkpoint path exists in the alex-work volume."""
    import subprocess
    parent  = os.path.dirname(ckpt_relative)
    fname   = os.path.basename(ckpt_relative)
    ls_path = f"checkpoints/{parent}" if parent else "checkpoints"
    result  = subprocess.run(
        ["modal", "volume", "ls", "alex-work", ls_path],
        capture_output=True, text=True,
    )
    return fname in result.stdout


@app.local_entrypoint()
def main(resume: bool = False):
    results = {}

    for run_name, aug_overrides in EXPERIMENTS:
        print(f"\n{'='*60}")
        print(f"  Experiment: {run_name}")
        print(f"{'='*60}")

        ckpt = f"{run_name}/best.pt"

        if resume and _checkpoint_exists(ckpt):
            print(f"  Checkpoint found at {ckpt} — skipping training.")
        else:
            print(f"  Training...")
            run_training.remote(
                aug_overrides=aug_overrides,
                run_name=run_name,
                save_every_epoch=False,
            )
            print(f"  Training complete.")

        print(f"  Evaluating {ckpt} ...")
        mAP = run_evaluation.remote(checkpoint=ckpt)
        results[run_name] = mAP

        avg = mAP.get("mAP_avg", 0.0)
        print(f"  mAP_avg = {avg:.4f}")

    # ── Summary table ─────────────────────────────────────────────────────────
    baseline_avg = results.get("baseline", {}).get("mAP_avg", 0.0)

    print(f"\n\n{'='*72}")
    print("  AUGMENTATION SEARCH RESULTS")
    print(f"{'='*72}")

    header = f"{'Experiment':<22}"
    for t in IOU_THRESHOLDS:
        header += f"  mAP@{t}"
    header += "   mAP_avg   vs baseline"
    print(header)
    print("-" * 72)

    for run_name, r in results.items():
        row = f"{run_name:<22}"
        for t in IOU_THRESHOLDS:
            row += f"  {r.get(f'mAP@{t}', 0.0):6.4f}"
        avg   = r.get("mAP_avg", 0.0)
        delta = avg - baseline_avg
        sign  = "+" if delta >= 0 else ""
        row  += f"   {avg:.4f}   {sign}{delta:.4f}"
        print(row)

    print(f"\nBaseline mAP_avg: {baseline_avg:.4f}")
    winners = [
        name for name, r in results.items()
        if name != "baseline" and r.get("mAP_avg", 0.0) > baseline_avg
    ]
    if winners:
        print(f"Augmentations that beat baseline: {', '.join(winners)}")
    else:
        print("No augmentation beat baseline in this search.")
