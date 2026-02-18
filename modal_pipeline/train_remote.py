import modal
from modal_pipeline.app import app, image, volume, VOLUME_MOUNT_PATH
import yaml

# Adjust GPU to your budget:
#   modal.gpu.T4()   — cheapest, fine for prototyping
#   modal.gpu.A10G() — good mid-tier
#   modal.gpu.A100() — best, most expensive
GPU = modal.gpu.A10G()


@app.function(
    image=image,
    gpu=GPU,
    volumes={VOLUME_MOUNT_PATH: volume},
    timeout=60 * 60 * 12,  # 12 hours max
    # Mount your local src/ and configs/ into the container
    mounts=[
        modal.Mount.from_local_dir("src", remote_path="/root/src"),
        modal.Mount.from_local_dir("configs", remote_path="/root/configs"),
    ],
)
def run_training(config_path: str = "configs/base_config.yaml"):
    import sys
    sys.path.insert(0, "/root")  # So `import src.xxx` works

    # Remap config paths to the Volume mount
    with open(f"/root/{config_path}") as f:
        config = yaml.safe_load(f)

    config["paths"]["data_root"] = f"{VOLUME_MOUNT_PATH}/thumos"
    config["paths"]["checkpoint_dir"] = f"{VOLUME_MOUNT_PATH}/checkpoints"
    config["paths"]["log_dir"] = f"{VOLUME_MOUNT_PATH}/logs"

    from src.train import train
    train(config)

    # Commit so checkpoint writes are persisted to the Volume
    volume.commit()


@app.local_entrypoint()
def main():
    run_training.remote()