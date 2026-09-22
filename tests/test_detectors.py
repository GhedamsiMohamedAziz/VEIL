"""The detector contract - checked on the built-in detector, which needs no
network access. Torchvision detectors are covered by test_detectors_torch.py."""

from __future__ import annotations

import pytest
import torch

from veil.ml.detectors import registry
from veil.ml.detectors.base import Detector
from veil.ml.detectors.blob import LABEL, ColorBlobDetector


def red(size: int = 64) -> torch.Tensor:
    image = torch.zeros(1, 3, size, size)
    image[:, 0] = 0.9
    return image


def test_blob_detector_finds_a_red_square_and_ignores_grey():
    detector = ColorBlobDetector().load()
    assert detector.predict(red())[0][0].label == LABEL
    assert detector.predict(torch.full((1, 3, 64, 64), 0.5))[0] == []


def test_blob_detector_box_covers_the_blob():
    image = torch.zeros(1, 3, 64, 64)
    image[:, 0, 10:30, 20:40] = 0.9
    (detection,) = ColorBlobDetector().predict(image)[0]
    assert detection.box == (20.0, 10.0, 40.0, 30.0)


def test_detector_is_deterministic():
    detector = ColorBlobDetector()
    assert detector.predict(red())[0][0].score == detector.predict(red())[0][0].score


def test_non_differentiable_detector_refuses_gradients_explicitly():
    with pytest.raises(NotImplementedError):
        ColorBlobDetector().score(red(), LABEL)


def test_registry_lists_and_resolves():
    ids = {info.id for info in registry.available()}
    assert {"colorblob-v1", "fasterrcnn-mobilenet-320"} <= ids
    assert registry.exists("colorblob-v1")
    assert not registry.exists("some-model-a-user-uploaded")
    assert isinstance(registry.get("colorblob-v1"), Detector)
    with pytest.raises(KeyError):
        registry.get("some-model-a-user-uploaded")


def test_evaluate_summarizes_a_batch():
    batch = torch.cat([red(), torch.full((1, 3, 64, 64), 0.5)])
    summary = ColorBlobDetector().evaluate(batch, LABEL)
    # Compared to literals: a dict rebuilt from `summary`'s own fields equals itself.
    assert summary["images"] == 2 and summary["detected"] == 1
    assert summary["max_scores"][0] > 0.5 and summary["max_scores"][1] == 0.0
    assert [len(d) for d in summary["detections"]] == [1, 0]
