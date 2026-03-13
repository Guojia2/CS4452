import json
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
from modal_pipeline.extract_features import extract_features
from modal_pipeline.train_remote import run_training

GPU = modal.gpu.A10G()


def _load_modal_config(config_path: str) -> dict:
    with open(f"/root/{config_path}", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config.setdefault("paths", {})
    config["paths"]["data_root"] = RAW_DATA_DIR
    config["paths"]["feature_dir"] = FEATURE_DIR
    config["paths"]["checkpoint_dir"] = CHECKPOINT_DIR
    config["paths"]["log_dir"] = LOG_DIR
    config["paths"]["artifact_dir"] = ARTIFACT_DIR
    return config


@app.function(
    image=image,
    gpu=GPU,
    volumes={VOLUME_MOUNT_PATH: volume},
    timeout=60 * 60 * 4,
    mounts=[
        modal.Mount.from_local_dir("src", remote_path="/root/src"),
        modal.Mount.from_local_dir("configs", remote_path="/root/configs"),
    ],
)
def run_evaluation(
    config_path: str = "configs/base_config.yaml",
    checkpoint_path: str = f"{CHECKPOINT_DIR}/best.pt",
    split_indices_path: str = f"{ARTIFACT_DIR}/split_indices.json",
):
    import sys

    sys.path.insert(0, "/root")
    from src.evaluate import evaluate_checkpoint

    config = _load_modal_config(config_path)
    evaluation_path = os.path.join(config["paths"]["artifact_dir"], "evaluation_summary.json")
    summary = evaluate_checkpoint(
        config=config,
        checkpoint_path=checkpoint_path,
        split_indices_path=split_indices_path,
        output_path=evaluation_path,
    )
    summary["evaluation_path"] = evaluation_path
    volume.commit()
    return summary


@app.function(
    image=image,
    volumes={VOLUME_MOUNT_PATH: volume},
    timeout=60 * 30,
    mounts=[
        modal.Mount.from_local_dir("src", remote_path="/root/src"),
        modal.Mount.from_local_dir("configs", remote_path="/root/configs"),
    ],
)  
def collect_run_outputs(
    history_path: str = f"{ARTIFACT_DIR}/training_history.json",
    evaluation_path: str = f"{ARTIFACT_DIR}/evaluation_summary.json",
    training_summary_path: str = f"{ARTIFACT_DIR}/training_summary.json",
    split_path: str = f"{ARTIFACT_DIR}/split_indices.json",
):
    import sys

    sys.path.insert(0, "/root")
    from src.reporting import load_json_artifact, summarize_history

    history_payload = load_json_artifact(history_path)
    evaluation_payload = load_json_artifact(evaluation_path)
    training_summary = load_json_artifact(training_summary_path)
    split_payload = load_json_artifact(split_path)
    primary_metric = training_summary.get("best_metric_name", "val_f1")

    manifest = {
        "training_summary": training_summary,
        "history": history_payload.get("history", []),
        "history_summary": summarize_history(
            history_payload.get("history", []),
            primary_metric=primary_metric,
        ),
        "evaluation_summary": evaluation_payload,
        "split_indices": split_payload,
        "paths": {
            "history_path": history_path,
            "evaluation_path": evaluation_path,
            "training_summary_path": training_summary_path,
            "split_path": split_path,
        },
        "history_path": history_path,
        "evaluation_path": evaluation_path,
    }

    manifest_path = os.path.join(ARTIFACT_DIR, "run_manifest.json")
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    volume.commit()
    return manifest


def run_baseline_workflow(
    config_path: str = "configs/base_config.yaml",
    backbone_name: str = "",
    skip_feature_extraction: bool = False,
) -> dict:
    if not skip_feature_extraction:
        extract_summary = extract_features.remote(
            config_path=config_path,
            backbone_name=backbone_name,
        )
    else:
        extract_summary = None

    training_summary = run_training.remote(config_path=config_path, use_features=True)
    evaluation_summary = run_evaluation.remote(
        config_path=config_path,
        checkpoint_path=training_summary["checkpoint_path"],
        split_indices_path=training_summary["split_path"],
    )
    report_summary = collect_run_outputs.remote(
        history_path=training_summary["history_path"],
        evaluation_path=evaluation_summary["evaluation_path"],
    )
    report_summary["feature_extraction"] = extract_summary
    report_summary["training_summary"] = training_summary
    report_summary["evaluation_summary"] = evaluation_summary
    return report_summary


@app.local_entrypoint()
def main(
    config_path: str = "configs/base_config.yaml",
    backbone_name: str = "",
    skip_feature_extraction: bool = False,
):
    report_summary = run_baseline_workflow(
        config_path=config_path,
        backbone_name=backbone_name,
        skip_feature_extraction=skip_feature_extraction,
    )
    print(json.dumps(report_summary, indent=2))
