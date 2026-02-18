import modal
from modal_pipeline.app import app, image, volume, VOLUME_MOUNT_PATH
import yaml
import os

GPU = modal.gpu.A10G()


@app.function(
    image=image,
    gpu=GPU,
    volumes={VOLUME_MOUNT_PATH: volume},
    timeout=60 * 60 * 12,
    mounts=[
        modal.Mount.from_local_dir("src",     remote_path="/root/src"),
        modal.Mount.from_local_dir("configs", remote_path="/root/configs"),
    ],
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