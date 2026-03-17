import modal
import yaml
import os


# --- Inline definitions (no import from modal_pipeline.app) ---
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
def run_training(
    config_path: str = "configs/base_config.yaml",
    use_features: bool = True,   # Set False to train end-to-end from raw video
):
    import sys
    sys.path.insert(0, "/root")

    with open(f"/root/{config_path}") as f:
        config = yaml.safe_load(f)

    # Remap paths to the Volume
    config["paths"]["data_root"]      = os.path.join(VOLUME_MOUNT_PATH, "raw")
    config["paths"]["feature_dir"]    = os.path.join(VOLUME_MOUNT_PATH, "features", "clip_level")
    config["paths"]["checkpoint_dir"] = os.path.join(VOLUME_MOUNT_PATH, "checkpoints")
    config["paths"]["log_dir"]        = os.path.join(VOLUME_MOUNT_PATH, "logs")
    config["use_features"]            = use_features

    from src.train import train
    train(config)
    volume.commit()


@app.local_entrypoint()
def main():
    run_training.remote(use_features=True)