"""
Training on Modal.

Reads features from the work volume (alex-dev), writes checkpoints there too.
All augmentations are controlled by configs/base_config.yaml — set the flags
there before running.

Run baseline (all augmentations off):
  modal run modal_pipeline/train_remote.py

Run with specific augmentation (edit base_config.yaml first):
  modal run modal_pipeline/train_remote.py
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
    timeout=60 * 60 * 24,
    retries=1,
)
def run_training(
    aug_overrides: dict = {},
    run_name: str = "baseline",
    save_every_epoch: bool = True,
):
    """
    aug_overrides: dict of augmentation config keys to override (one level deep).
      e.g. {"gaussian_noise": {"enabled": True}} or {"window_jitter": True}
    run_name: used to namespace checkpoints and logs under their own subdir.
    save_every_epoch: set False during augmentation search to avoid filling the volume.
    """
    import sys
    import yaml
    import os
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

    # Apply augmentation overrides (one level deep merge)
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


@app.local_entrypoint()
def train_main():
    print("Submitting training job...")
    run_training.remote(run_name="baseline", save_every_epoch=True)
