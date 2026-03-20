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
    timeout=60 * 60 * 24,
    retries=1,
)
def extract_features(
    config_path: str = "configs/base_config.yaml",
    subset: str = "training",
):
    import sys
    import os
    import yaml
    sys.path.insert(0, "/root")
    volume.reload()

    from src.extract_features import run_feature_extraction

    with open(f"/root/{config_path}") as f:
        config = yaml.safe_load(f)

    # Remap paths to actual Volume layout
    config["paths"]["data_root"]      = os.path.join(VOLUME_MOUNT_PATH, "raw")
    config["paths"]["feature_dir"]    = os.path.join(VOLUME_MOUNT_PATH, "features", "clip_level")
    config["paths"]["checkpoint_dir"] = os.path.join(VOLUME_MOUNT_PATH, "checkpoints")
    config["paths"]["log_dir"]        = os.path.join(VOLUME_MOUNT_PATH, "logs")

    # Write the remapped config to a temp file so run_feature_extraction can read it
    import tempfile
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as tmp:
        yaml.dump(config, tmp)
        tmp_path = tmp.name

    run_feature_extraction(config_path=tmp_path, subset=subset)
    volume.commit()


@app.function(
    image=image,
    gpu=GPU,
    volumes={VOLUME_MOUNT_PATH: volume},
    timeout=60 * 60 * 24,
    retries=1,
)
def extract_test_features(
    config_path: str = "configs/base_config.yaml",
):
    import sys
    import os
    import yaml
    import torch
    import av
    sys.path.insert(0, "/root")
    volume.reload()

    from src.models.backbone import build_backbone
    from src.dataset import decode_video_frames
    from src.utils import get_logger
    from src.dataset import build_transforms

    with open(f"/root/{config_path}") as f:
        config = yaml.safe_load(f)

    backbone_name = config["model"]["backbone"]
    clip_len_sec  = config["data"]["clip_len_sec"]
    stride_sec    = config["data"]["stride_sec"]
    num_frames    = config["data"]["num_frames"]
    batch_size    = config["data"]["batch_size"]

    logger = get_logger("feature_extraction_test")
    device = torch.device("cuda")

    logger.info(f"Loading backbone: {backbone_name}")
    backbone, feature_dim = build_backbone(backbone_name, pretrained=True)
    backbone = backbone.to(device).eval()

    video_dir = os.path.join(VOLUME_MOUNT_PATH, "raw", "videos", "test")
    feat_dir  = os.path.join(VOLUME_MOUNT_PATH, "features", "test")
    os.makedirs(feat_dir, exist_ok=True)

    video_files = sorted([f for f in os.listdir(video_dir) if f.endswith(".mp4")])
    logger.info(f"Found {len(video_files)} test videos")

    with torch.no_grad():
        for video_file in video_files:
            video_path = os.path.join(video_dir, video_file)
            video_name = video_file.replace(".mp4", "")

            container = av.open(video_path)
            stream = container.streams.video[0]
            duration = float(stream.duration * stream.time_base)
            container.close()

            clips = []
            start = 0.0
            while start + clip_len_sec <= duration:
                clips.append((start, start + clip_len_sec))
                start += stride_sec

            if not clips:
                logger.info(f"Skipping {video_name} — too short")
                continue

        transform = build_transforms(img_size=config["data"].get("img_size", 224))

        all_feats = []
        for i in range(0, len(clips), batch_size):
            batch_clips = clips[i:i + batch_size]
            batch_tensors = torch.stack([
                torch.stack([
                    transform(f) for f in decode_video_frames(video_path, s, e, num_frames)
                ]).permute(1, 0, 2, 3)
                for s, e in batch_clips
            ]).to(device)
            feats = backbone(batch_tensors).cpu()
            all_feats.append(feats)

            all_feats = torch.cat(all_feats, dim=0)
            save_path = os.path.join(feat_dir, f"{video_name}.pt")
            torch.save(all_feats, save_path)
            logger.info(f"Saved {video_name}: {all_feats.shape}")

    volume.commit()
    logger.info(f"Done. Test features saved to {feat_dir}")


@app.local_entrypoint()
def main():
    extract_test_features.remote()