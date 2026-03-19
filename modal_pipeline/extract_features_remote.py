import modal

from src.extract_features import run_feature_extraction


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
def extract_features(config_path: str = "configs/base_config.yaml"):
    import sys

    sys.path.insert(0, "/root")
    volume.reload()

    with open(f"/root/{config_path}") as f:
        config = yaml.safe_load(f)

    # Remap paths to the Volume
    config["paths"]["data_root"]      = os.path.join(VOLUME_MOUNT_PATH, "raw")
    config["paths"]["feature_dir"]    = os.path.join(VOLUME_MOUNT_PATH, "features", "clip_level")
    config["paths"]["checkpoint_dir"] = os.path.join(VOLUME_MOUNT_PATH, "checkpoints")
    config["paths"]["log_dir"]        = os.path.join(VOLUME_MOUNT_PATH, "logs")

    run_feature_extraction(config_path=config_path, subset="training")

    volume.commit()


@app.local_entrypoint()
def main():
    extract_features.remote()