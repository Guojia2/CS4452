import os
from typing import Dict, List

import torch
import yaml
from torch.utils.data import DataLoader

from src.dataset import THUMOSVideoDataset
from src.models.backbone import build_backbone
from src.utils import get_logger


def load_config(config_path: str) -> Dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_feature_extraction(
    config_path: str,
    subset: str = "training",
) -> None:
    config = load_config(config_path)

    backbone_name = config["model"]["backbone"]
    clip_len_sec = config["data"]["clip_len_sec"]
    stride_sec = config["data"]["stride_sec"]
    num_frames = config["data"]["num_frames"]
    batch_size = config["data"]["batch_size"]

    data_root = config["paths"]["data_root"]
    feat_dir = config["paths"]["feature_dir"]

    video_dir = os.path.join(data_root, "videos")
    ann_path = os.path.join(
        data_root,
        "annotations",
        "annotations",
        "thumos_14_anno.json",
    )

    os.makedirs(feat_dir, exist_ok=True)

    logger = get_logger("feature_extraction", log_dir=config["paths"].get("log_dir"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Loading backbone: {backbone_name}")
    backbone, feature_dim = build_backbone(backbone_name, pretrained=True)
    backbone = backbone.to(device).eval()

    dataset = THUMOSVideoDataset(
        video_dir=video_dir,
        ann_path=ann_path,
        subset=subset,
        clip_len_sec=clip_len_sec,
        stride_sec=stride_sec,
        num_frames=num_frames,
    )

    logger.info(f"data_root: {data_root}")
    logger.info(f"video_dir: {video_dir}")
    logger.info(f"ann_path: {ann_path}")
    logger.info(f"feature_dir: {feat_dir}")
    logger.info(f"ann_path exists: {os.path.exists(ann_path)}")
    logger.info(f"video_dir exists: {os.path.exists(video_dir)}")
    logger.info(
        f"Files in video_dir: {len(os.listdir(video_dir)) if os.path.exists(video_dir) else 'DIR NOT FOUND'}"
    )
    logger.info(f"Total clips: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    logger.info(f"Extracting features for {len(dataset)} clips...")

    video_features: Dict[str, List[torch.Tensor]] = {}

    with torch.no_grad():
        for i, (clips, video_names, starts, ends) in enumerate(loader):
            clips = clips.to(device, non_blocking=True)   # (B, C, T, H, W)
            feats = backbone(clips).cpu()                 # (B, D)

            for feat, vname in zip(feats, video_names):
                if vname not in video_features:
                    video_features[vname] = []
                video_features[vname].append(feat)

            if i % 100 == 0:
                logger.info(f"Step {i}/{len(loader)}")
                flush_features(video_features, feat_dir, complete_only=True)

    flush_features(video_features, feat_dir, complete_only=False)

    logger.info(f"Done. Features saved to {feat_dir}")
    logger.info(f"Feature dim: {feature_dim}")


def flush_features(
    video_features: Dict[str, List[torch.Tensor]],
    feat_dir: str,
    complete_only: bool,
) -> None:
    keys = list(video_features.keys())

    if complete_only and len(keys) > 1:
        keys = keys[:-1]

    for vname in keys:
        feats = torch.stack(video_features.pop(vname), dim=0)  # (N_clips, D)
        save_path = os.path.join(feat_dir, f"{vname}.pt")
        torch.save(feats, save_path)