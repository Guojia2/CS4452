import os
import modal
import yaml

from modal_pipeline.app import (
    app,
    image,
    volume,
    ARTIFACT_DIR,
    CHECKPOINT_DIR,
    FEATURE_DIR,
    LOG_DIR,
    RAW_DATA_DIR,
    VOLUME_MOUNT_PATH,
)

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
    config.setdefault("paths", {})
    config["paths"]["data_root"] = RAW_DATA_DIR
    config["paths"]["feature_dir"] = FEATURE_DIR
    config["paths"]["checkpoint_dir"] = CHECKPOINT_DIR
    config["paths"]["log_dir"] = LOG_DIR
    config["paths"]["artifact_dir"] = ARTIFACT_DIR
    config["use_features"]            = use_features

    for path in (
        config["paths"]["feature_dir"],
        config["paths"]["checkpoint_dir"],
        config["paths"]["log_dir"],
        config["paths"]["artifact_dir"],
    ):
        os.makedirs(path, exist_ok=True)

    from src.train import train
    summary = train(config)
    volume.commit()
    return summary


@app.local_entrypoint()
def main(config_path: str = "configs/base_config.yaml", use_features: bool = True):
    run_training.remote(config_path=config_path, use_features=use_features)