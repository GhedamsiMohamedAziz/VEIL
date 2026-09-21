"""Metric arithmetic, checked against hand-computed values."""

from __future__ import annotations

from veil.ml.evaluation import metrics, robustness
from veil.ml.evaluation.comparison import compare


def record(detected: bool, score: float, boxes=None, image_id="img", **transform):
    return {
        "image_index": 0, "image_id": image_id, "transform_index": 0,
        "transform": {"rotation_deg": 0.0, "scale": 1.0, "brightness": 1.0,
                      "contrast": 1.0, "perspective": 0.0, "blur_sigma": 0.0,
                      "noise_std": 0.0, "deformation": 0.0, **transform},
        "detected": detected, "max_score": score,
        "detection_count": len(boxes or []), "boxes": boxes or [],
    }


def test_iou_matches_hand_calculation():
    # 10x10 and 10x10 overlapping in a 5x5 corner: 25 / (100+100-25)
    assert metrics.iou([0, 0, 10, 10], [5, 5, 15, 15]) == 25 / 175
    assert metrics.iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert metrics.iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0


def test_summarize_counts_and_averages():
    result = metrics.summarize([record(True, 0.8), record(False, 0.0), record(True, 0.6)])
    assert result["samples"] == 3
    assert result["detections"] == 2
    assert result["detection_rate"] == 2 / 3
    assert abs(result["mean_confidence"] - 1.4 / 3) < 1e-9
    assert abs(result["mean_confidence_when_detected"] - 0.7) < 1e-9
    lo, hi = result["detection_rate_ci95"]
    assert lo < 2 / 3 < hi


def test_summarize_of_nothing_is_none_not_zero():
    assert metrics.summarize([])["detection_rate"] is None


def test_ground_truth_precision_and_recall():
    annotations = {"img": [{"label": "person", "box": [0, 0, 10, 10]},
                           {"label": "person", "box": [50, 50, 60, 60]}]}
    records = [record(True, 0.9, boxes=[[0, 0, 10, 10], [90, 90, 99, 99]])]
    result = metrics.ground_truth_metrics(records, annotations, "person")
    assert result["true_positives"] == 1   # exact match on the first box
    assert result["false_positives"] == 1  # the far-away box
    assert result["false_negatives"] == 1  # the unmatched annotation
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["mean_iou"] == 1.0


def test_ground_truth_reports_unavailable_rather_than_zero():
    assert metrics.ground_truth_metrics([record(True, 0.9)], {}, "person") == {
        "available": False, "reason": "no ground-truth annotations for this label"
    }


def test_veil_score_axes_are_measured_independently():
    records = [
        record(True, 0.9, rotation_deg=30.0),    # angle active, detected
        record(False, 0.0, rotation_deg=-30.0),  # angle active, missed
        record(False, 0.0, brightness=0.6),      # lighting active, missed
    ]
    score = robustness.veil_score(records)
    assert score["angle_robustness"] == 0.5
    assert score["lighting_robustness"] == 1.0
    assert score["scale_robustness"] is None          # never exercised
    assert score["physical_robustness"] is None       # no physical samples
    assert score["sample_counts"]["axis_angle"] == 2


def test_comparison_delta_direction():
    baseline = [record(True, 0.9), record(True, 0.8)]
    candidate = [record(True, 0.7), record(False, 0.0)]
    result = compare(baseline, candidate)
    assert result["delta"]["detection_rate"] == -0.5
    assert result["delta"]["mean_confidence"] < 0


def test_ground_truth_ignores_geometrically_transformed_records():
    annotations = {"img": [{"label": "person", "box": [0, 0, 10, 10]}]}
    rotated = {**record(True, 0.9, boxes=[[40, 40, 50, 50]]), "transform": {"rotation_deg": 30.0}}
    dimmed = {**record(True, 0.9, boxes=[[0, 0, 10, 10]]), "transform": {"brightness": 0.6, "scale": 1.0}}
    result = metrics.ground_truth_metrics([rotated, dimmed], annotations, "person")
    assert result["annotated_samples"] == 1 and result["false_positives"] == 0
    assert result["recall"] == 1.0


def test_annotations_are_scaled_into_the_run_frame():
    from veil.ml.runner import _annotations_at

    scaled = _annotations_at({"a": [{"label": "person", "box": [120, 60, 360, 480]}]},
                             {"a": (480, 480)}, 320)
    assert scaled["a"][0]["box"] == [80.0, 40.0, 240.0, 320.0]


def test_ground_truth_says_so_when_the_sweep_has_no_unwarped_point():
    annotations = {"img": [{"label": "person", "box": [0, 0, 10, 10]}]}
    rotated = {**record(True, 0.9, boxes=[[0, 0, 10, 10]]), "transform": {"rotation_deg": -30.0}}
    result = metrics.ground_truth_metrics([rotated], annotations, "person")
    assert result["available"] is False and "labelled frame" in result["reason"]
