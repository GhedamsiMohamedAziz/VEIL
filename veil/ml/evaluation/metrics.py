"""Aggregate metrics over detection records.

No composite "AI score" here on purpose: every number is a plain count or
mean over samples, and the sample count travels with it so a reader can
judge the uncertainty.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

Box = Sequence[float]


def iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)
    if inter <= 0:
        return 0.0
    area_a = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    area_b = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% CI for a detection rate. A rate over 8 samples is not a rate over
    800, and the report says so."""
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    return (max(centre - margin, 0.0), min(centre + margin, 1.0))


def summarize(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    records = list(records)
    n = len(records)
    if n == 0:
        return {"samples": 0, "detection_rate": None, "mean_confidence": None}
    detected = sum(1 for r in records if r["detected"])
    scores = [r["max_score"] for r in records]
    positive = [s for s in scores if s > 0]
    lo, hi = _wilson_interval(detected, n)
    return {
        "samples": n,
        "detections": detected,
        "detection_rate": detected / n,
        "detection_rate_ci95": [lo, hi],
        "mean_confidence": sum(scores) / n,
        "mean_confidence_when_detected": (sum(positive) / len(positive)) if positive else 0.0,
        "max_confidence": max(scores),
        "total_detections": sum(r["detection_count"] for r in records),
    }


def _keeps_geometry(transform: dict[str, Any]) -> bool:
    """True when the transformation left every pixel where the annotation says
    it is. Brightness, blur and noise do; rotation, scale, shift, tilt and
    cloth deformation move the subject away from its labelled box."""
    return (transform.get("scale", 1.0) == 1.0 and not any(
        transform.get(k) for k in ("rotation_deg", "translate_x", "translate_y",
                                   "perspective", "deformation")))


def ground_truth_metrics(
    records: Iterable[dict[str, Any]],
    annotations: dict[str, list[dict[str, Any]]],
    target_label: str,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Precision / recall / mean IoU against labelled boxes.

    Only records whose image has annotations contribute; if the dataset is
    unlabelled the caller gets `{"available": False}` rather than a fake 0.0.
    Geometrically transformed records are skipped: their boxes live in the
    warped frame, and matching them to unwarped annotations measures the warp.
    `annotations` must already be in the records' pixel frame (see
    `runner._annotations_at`).
    """
    tp = fp = fn = 0
    ious: list[float] = []
    used = 0
    for record in records:
        if not _keeps_geometry(record.get("transform") or {}):
            continue
        gt_boxes = [
            a["box"] for a in annotations.get(record.get("image_id") or "", [])
            if a.get("label") == target_label
        ]
        if not gt_boxes:
            continue
        used += 1
        remaining = list(gt_boxes)
        for box in record["boxes"]:
            if not remaining:
                fp += 1
                continue
            scored = [(iou(box, gt), i) for i, gt in enumerate(remaining)]
            best_v, best_i = max(scored)
            if best_v >= iou_threshold:
                tp += 1
                ious.append(best_v)
                remaining.pop(best_i)
            else:
                fp += 1
        fn += len(remaining)
    if used == 0:
        return {"available": False, "reason": "no ground-truth annotations for this label"}
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "available": True,
        "annotated_samples": used,
        "true_positives": tp, "false_positives": fp, "false_negatives": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "mean_iou": (sum(ious) / len(ious)) if ious else 0.0,
        "iou_threshold": iou_threshold,
    }
