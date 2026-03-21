import numpy as np
from typing import List, Dict, Tuple
from src.utils import iou_1d


def compute_ap(
    predictions: np.ndarray,    # (N, 3): [score, pred_start, pred_end]
    ground_truths: np.ndarray,  # (M, 2): [gt_start, gt_end]
    iou_threshold: float,
) -> float:
    """
    Compute Average Precision for a single class at a given IoU threshold.
    Predictions must be pre-sorted by score (descending) before calling this.
    """
    if len(ground_truths) == 0:
        return 0.0

    scores    = predictions[:, 0]
    pred_segs = predictions[:, 1:]         # (N, 2)

    sort_idx  = np.argsort(-scores)
    pred_segs = pred_segs[sort_idx]

    iou_mat   = iou_1d(pred_segs, ground_truths)  # (N, M)
    matched_gt = np.zeros(len(ground_truths), dtype=bool)

    tp = np.zeros(len(pred_segs))
    fp = np.zeros(len(pred_segs))

    for i, iou_row in enumerate(iou_mat):
        best_gt = np.argmax(iou_row)
        if iou_row[best_gt] >= iou_threshold and not matched_gt[best_gt]:
            tp[i] = 1
            matched_gt[best_gt] = True
        else:
            fp[i] = 1

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recall    = cum_tp / len(ground_truths)
    precision = cum_tp / (cum_tp + cum_fp + 1e-8)

    # Area under PR curve using trapezoidal rule
    recall    = np.concatenate([[0.0], recall,    [recall[-1]]])
    precision = np.concatenate([[1.0], precision, [0.0]])
    ap = np.trapezoid(precision, recall)
    return float(ap)


def compute_map(
    all_predictions: Dict[str, List],   # {class_name: [(video, score, start, end), ...]}
    all_ground_truths: Dict[str, List], # {class_name: [(video, start, end), ...]}
    iou_thresholds: List[float] = [0.3, 0.4, 0.5, 0.6, 0.7],
) -> Dict[str, float]:
    """
    Compute mean Average Precision (mAP) at multiple IoU thresholds.
    This is the standard THUMOS-14 evaluation protocol.

    Returns a dict like:
    {
        "mAP@0.3": 0.52,
        "mAP@0.5": 0.41,
        "mAP_avg": 0.44,   # average across all thresholds
    }
    """
    results = {}
    all_aps = []

    for thresh in iou_thresholds:
        aps = []
        for cls_name in all_predictions:
            preds = np.array([
                [score, start, end]
                for (_, score, start, end) in all_predictions.get(cls_name, [])
            ]) if all_predictions.get(cls_name) else np.zeros((0, 3))

            gts = np.array([
                [start, end]
                for (_, start, end) in all_ground_truths.get(cls_name, [])
            ]) if all_ground_truths.get(cls_name) else np.zeros((0, 2))

            ap = compute_ap(preds, gts, thresh)
            aps.append(ap)

        map_thresh = float(np.mean(aps))
        results[f"mAP@{thresh}"] = map_thresh
        all_aps.append(map_thresh)

    results["mAP_avg"] = float(np.mean(all_aps))
    return results


def evaluate_model(model, data_loader, device, iou_thresholds=None):
    """
    Run model over a DataLoader and compute mAP.
    Expects model to output (scores, segments) per clip window.
    Adapt this to your detection head's output format.
    """
    import torch
    from src.dataset import THUMOS14_CLASSES  # avoid circular if needed
    if iou_thresholds is None:
        iou_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

    model.eval()
    all_predictions  = {c: [] for c in THUMOS14_CLASSES}
    all_ground_truths = {c: [] for c in THUMOS14_CLASSES}

    with torch.no_grad():
        for features, labels, meta in data_loader:
            features = features.to(device)
            # model output expected: list of dicts with keys
            # "scores" (C,), "segments" [(start, end), ...], "video_name"
            outputs = model(features)

            for out, lbl, m in zip(outputs, labels, meta):
                video_name = m["video_name"]
                for cls_idx, cls_name in enumerate(THUMOS14_CLASSES):
                    # Accumulate predictions
                    for score, (start, end) in zip(
                        out["scores"][cls_idx], out["segments"]
                    ):
                        all_predictions[cls_name].append(
                            (video_name, float(score), float(start), float(end))
                        )
                    # Accumulate ground truths from label tensor
                    gt_clips = (lbl[:, cls_idx] > 0).nonzero(as_tuple=True)[0]
                    if len(gt_clips) > 0:
                        all_ground_truths[cls_name].append(
                            (video_name, float(gt_clips[0]), float(gt_clips[-1]))
                        )

    return compute_map(all_predictions, all_ground_truths, iou_thresholds)