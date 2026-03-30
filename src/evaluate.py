import numpy as np
from typing import List, Dict, Tuple


def nms_1d(
    predictions: List[Tuple[str, float, float, float]],
    iou_threshold: float = 0.3,
) -> List[Tuple[str, float, float, float]]:
    """
    Non-Maximum Suppression for temporal segments.
    Applied per-video, per-class. Input/output: [(video, score, start, end), ...]
    """
    if len(predictions) <= 1:
        return predictions

    # Group by video
    by_video: Dict[str, List[Tuple[float, float, float, int]]] = {}
    for i, (vid, score, start, end) in enumerate(predictions):
        by_video.setdefault(vid, []).append((score, start, end, i))

    keep_indices = []
    for vid, segs in by_video.items():
        segs.sort(key=lambda x: -x[0])  # sort by score descending
        suppressed = set()
        for i, (s1, st1, en1, orig_i) in enumerate(segs):
            if i in suppressed:
                continue
            keep_indices.append(orig_i)
            for j in range(i + 1, len(segs)):
                if j in suppressed:
                    continue
                s2, st2, en2, _ = segs[j]
                inter = max(0.0, min(en1, en2) - max(st1, st2))
                union = (en1 - st1) + (en2 - st2) - inter
                if inter / (union + 1e-8) >= iou_threshold:
                    suppressed.add(j)

    keep_indices.sort()
    return [predictions[i] for i in keep_indices]


def compute_ap(
    predictions: List[Tuple[str, float, float, float]],   # [(video, score, start, end), ...]
    ground_truths: List[Tuple[str, float, float]],         # [(video, start, end), ...]
    iou_threshold: float,
) -> float:
    """
    Compute Average Precision for a single class at a given IoU threshold.
    Standard protocol: predictions ranked globally by score, matched only
    against ground truths from the same video.
    """
    if len(ground_truths) == 0:
        return 0.0
    if len(predictions) == 0:
        return 0.0

    n_gt = len(ground_truths)

    # Group ground truths by video
    gt_by_video: Dict[str, List[Tuple[float, float]]] = {}
    for vid, start, end in ground_truths:
        gt_by_video.setdefault(vid, []).append((start, end))

    # Track which GTs have been matched, per video
    matched = {vid: np.zeros(len(segs), dtype=bool) for vid, segs in gt_by_video.items()}

    # Sort predictions by score descending
    predictions = sorted(predictions, key=lambda x: -x[1])

    tp = np.zeros(len(predictions))
    fp = np.zeros(len(predictions))

    for i, (vid, score, pred_start, pred_end) in enumerate(predictions):
        if vid not in gt_by_video:
            fp[i] = 1
            continue

        # Compute IoU only against GTs from the same video
        best_iou = 0.0
        best_idx = -1
        for j, (gt_start, gt_end) in enumerate(gt_by_video[vid]):
            inter_start = max(pred_start, gt_start)
            inter_end   = min(pred_end, gt_end)
            inter = max(0.0, inter_end - inter_start)
            union = (pred_end - pred_start) + (gt_end - gt_start) - inter
            iou   = inter / (union + 1e-8)
            if iou > best_iou:
                best_iou = iou
                best_idx = j

        if best_iou >= iou_threshold and not matched[vid][best_idx]:
            tp[i] = 1
            matched[vid][best_idx] = True
        else:
            fp[i] = 1

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recall    = cum_tp / n_gt
    precision = cum_tp / (cum_tp + cum_fp)

    # Monotone-decreasing interpolation (standard VOC protocol)
    for j in range(len(precision) - 2, -1, -1):
        precision[j] = max(precision[j], precision[j + 1])

    # AP = sum of rectangular areas at each recall change
    mrec = np.concatenate([[0.0], recall])
    mpre = np.concatenate([[1.0], precision])
    changes = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[changes + 1] - mrec[changes]) * mpre[changes + 1])

    return float(ap)


def compute_map(
    all_predictions: Dict[str, List],   # {class_name: [(video, score, start, end), ...]}
    all_ground_truths: Dict[str, List], # {class_name: [(video, start, end), ...]}
    iou_thresholds: List[float] = [0.3, 0.4, 0.5, 0.6, 0.7],
) -> Dict[str, float]:
    """
    Compute mean Average Precision (mAP) at multiple IoU thresholds.
    Standard THUMOS-14 evaluation protocol.
    """
    results = {}
    all_aps = []

    for thresh in iou_thresholds:
        aps = []
        for cls_name in all_predictions:
            preds = all_predictions.get(cls_name, [])
            gts   = all_ground_truths.get(cls_name, [])
            ap = compute_ap(preds, gts, thresh)
            aps.append(ap)

        map_thresh = float(np.mean(aps)) if aps else 0.0
        results[f"mAP@{thresh}"] = map_thresh
        all_aps.append(map_thresh)

    results["mAP_avg"] = float(np.mean(all_aps)) if all_aps else 0.0
    return results
