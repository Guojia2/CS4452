import json


def load_json_artifact(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_history(history_path: str) -> list[dict]:
    payload = load_json_artifact(history_path)
    return payload.get("history", [])


def summarize_history(history: list[dict], primary_metric: str = "val_f1") -> dict:
    if not history:
        return {}

    best_epoch = max(history, key=lambda row: row.get(primary_metric, float("-inf")))
    latest_epoch = history[-1]
    return {
        "primary_metric": primary_metric,
        "best_epoch": best_epoch,
        "latest_epoch": latest_epoch,
        "num_epochs": len(history),
    }
