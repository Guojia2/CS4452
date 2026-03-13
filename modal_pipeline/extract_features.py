import modal
import os
import torch
from modal_pipeline.app import (
    app,
    image,
    volume,
    ARTIFACT_DIR,
    FEATURE_DIR,
    RAW_DATA_DIR,
    VOLUME_MOUNT_PATH,
)

GPU = modal.gpu.A10G()   # Feature extraction benefits from a strong GPU


@app.function(
    image=image,
    gpu=GPU,
    volumes={VOLUME_MOUNT_PATH: volume},
    timeout=60 * 60 * 24,   # 24 hours — 200GB of video takes a while
    mounts=[
        modal.Mount.from_local_dir("src",     remote_path="/root/src"),
        modal.Mount.from_local_dir("configs", remote_path="/root/configs"),
    ],
    # Retry once on preemption — useful for long jobs
    retries=1,
)
def extract_features(
    config_path: str = "configs/base_config.yaml",
    backbone_name: str = "",
    clip_len_sec:  float = 2.0,
    stride_sec:    float = 1.0,
    num_frames:    int   = 0,
    batch_size:    int   = 16,
):
    import sys
    sys.path.insert(0, "/root")

    import yaml
    from torch.utils.data import DataLoader
    from src.dataset import THUMOSVideoDataset
    from src.model import build_backbone
    from src.utils import get_logger

    logger = get_logger("feature_extraction")
    device = torch.device("cuda")

    with open(f"/root/{config_path}") as f:
        config = yaml.safe_load(f)

    if not backbone_name:
        backbone_name = config["model"].get("backbone", "x3d_m")
    if num_frames <= 0:
        num_frames = config["data"].get("num_frames", config["data"].get("clip_len", 16))
    batch_size = config["data"].get("batch_size", batch_size)
    pretrained = config["model"].get("pretrained", True)
    paths = config.setdefault("paths", {})
    paths["data_root"] = RAW_DATA_DIR
    paths["feature_dir"] = FEATURE_DIR
    paths["artifact_dir"] = ARTIFACT_DIR
    os.makedirs(paths["feature_dir"], exist_ok=True)
    os.makedirs(paths["artifact_dir"], exist_ok=True)

    # --- Build backbone (inference only, no head) ---
    logger.info(f"Loading backbone: {backbone_name}")
    backbone, feature_dim = build_backbone(backbone_name, pretrained=pretrained)
    backbone = backbone.to(device).eval()

    # --- Dataset ---
    video_dir = os.path.join(paths["data_root"], "videos")
    ann_path = os.path.join(paths["data_root"], "annotations", "thumos14.json")
    feat_dir = paths["feature_dir"]

    dataset = THUMOSVideoDataset(
        video_dir=video_dir,
        ann_path=ann_path,
        clip_len_sec=clip_len_sec,
        stride_sec=stride_sec,
        num_frames=num_frames,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    logger.info(f"Extracting features for {len(dataset)} clips...")

    # Accumulate features per video in memory, then flush to disk
    video_features: dict = {}

    with torch.no_grad():
        for i, (clips, video_names, starts, ends) in enumerate(loader):
            clips = clips.to(device)                       # (B, C, T, H, W)
            feats = backbone(clips).cpu()                  # (B, feature_dim)

            for feat, vname in zip(feats, video_names):
                if vname not in video_features:
                    video_features[vname] = []
                video_features[vname].append(feat)

            # Periodically flush completed videos to the Volume
            # to avoid building up too much in RAM
            if i % 100 == 0:
                logger.info(f"  Step {i}/{len(loader)}")
                _flush_features(video_features, feat_dir, complete_only=True)

    # Final flush for any remaining videos
    _flush_features(video_features, feat_dir, complete_only=False)

    # Commit writes to the Modal Volume
    volume.commit()
    logger.info(f"Done. Features saved to {feat_dir}")
    logger.info(f"Feature dim: {feature_dim}")
    return {
        "backbone": backbone_name,
        "feature_dim": int(feature_dim),
        "feature_dir": feat_dir,
        "num_clips": len(dataset),
    }


def _flush_features(video_features: dict, feat_dir: str, complete_only: bool):
    """
    Save accumulated per-video features to .pt files and free memory.
    If complete_only=True, only saves videos whose loader has "moved on"
    (heuristic: skip the last video in the dict as it may still be accumulating).
    """
    keys = list(video_features.keys())
    if complete_only and len(keys) > 1:
        keys = keys[:-1]   # hold back the most recent video

    for vname in keys:
        feats = torch.stack(video_features.pop(vname), dim=0)  # (N_clips, D)
        save_path = os.path.join(feat_dir, f"{vname}.pt")
        torch.save(feats, save_path)


@app.local_entrypoint()
def main(
    config_path: str = "configs/base_config.yaml",
    backbone_name: str = "",
):
    extract_features.remote(
        config_path=config_path,
        backbone_name=backbone_name,
        clip_len_sec=2.0,
        stride_sec=1.0,
        num_frames=0,
        batch_size=16,
    )