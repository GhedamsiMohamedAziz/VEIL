"""The torchvision path: real weights, real gradients.

Skipped automatically when the weights are not in the torch hub cache and
there is no network, so CI without egress still runs green (and says so).
"""

from __future__ import annotations

import pytest
import torch

from veil.ml.detectors import registry
from veil.ml.patterns.optimizer import OptimizationConfig, optimize
from veil.ml.simulation.renderer import Placement
from veil.ml.simulation.transforms import TransformSpec

pytestmark = pytest.mark.slow

MODEL = "fasterrcnn-mobilenet-320"


@pytest.fixture(scope="module")
def detector():
    try:
        return registry.get(MODEL)
    except Exception as exc:  # pragma: no cover - offline CI
        pytest.skip(f"torchvision weights unavailable: {exc}")


def silhouette() -> torch.Tensor:
    """A crude dark human-shaped figure on a flat background. Faster R-CNN
    reports it as a person at ~0.4 confidence, which is enough signal to
    test the machinery (it is not a substitute for real photographs)."""
    image = torch.full((1, 3, 320, 320), 0.55)
    image[:, :, 60:120, 140:180] = 0.25
    image[:, :, 120:260, 110:210] = 0.25
    image[:, :, 260:310, 120:150] = 0.25
    image[:, :, 260:310, 170:200] = 0.25
    return image


def test_detects_a_person_and_reports_coco_labels(detector):
    labels = {d.label for d in detector.predict(silhouette(), threshold=0.3)[0]}
    assert "person" in labels
    assert "person" in detector.metadata().labels


def test_score_is_differentiable_with_respect_to_the_image(detector):
    image = silhouette().requires_grad_(True)
    score = detector.score(image, "person")
    assert score.shape == (1,)
    score.sum().backward()
    assert image.grad is not None and float(image.grad.abs().sum()) > 0


def test_score_stays_connected_when_nothing_is_detected(detector):
    """An empty result must still produce a usable gradient, or optimization
    stops the moment it starts working."""
    blank = torch.full((1, 3, 320, 320), 0.5, requires_grad=True)
    score = detector.score(blank, "person")
    assert detector.predict(blank.detach(), threshold=0.05) == [[]]  # truly nothing detected
    assert 0.0 < float(score.detach()) < 0.05  # below anything selection lets through
    score.sum().backward()
    assert float(blank.grad.abs().sum()) > 0  # `is not None` is true of an all-zero gradient


def test_unknown_label_is_rejected(detector):
    with pytest.raises(KeyError):
        detector.score(silhouette(), "unicorn")


def test_gradient_strategy_is_selected_and_produces_a_printable_pattern(detector):
    spec = TransformSpec(rotation_deg=[-10.0, 10.0], scale=[1.0], translate=[0.0],
                         perspective=[0.0], brightness=[1.0], contrast=[1.0],
                         blur_sigma=[0.0], noise_std=[0.0], deformation=[0.0])
    result = optimize(
        detector, silhouette(), Placement(0.5, 0.58, 0.14, 0.2), spec,
        OptimizationConfig(target_label="person", pattern_size=32, iterations=2,
                           batch_transforms=1),
    )
    assert result.strategy == "gradient"
    assert len(result.history) == 2
    assert result.pattern.shape == (3, 32, 32)

    from veil.ml.patterns.constraints import PRINTABLE_PALETTE, non_printability

    assert float(non_printability(result.pattern)) < 1e-5  # quantized on export


def test_pre_selection_scores_are_not_shared_between_threads(detector):
    """The registry hands one detector to the run worker and to request threads.
    A forward elsewhere must not replace what `score` is about to read."""
    import threading

    blank = torch.full((2, 3, 320, 320), 0.5)
    expected = detector.score(blank, "person").detach()
    detector._forward(blank)  # this thread's state now describes a 2-image batch
    other = threading.Thread(target=lambda: detector.predict(torch.rand(1, 3, 320, 320)))
    other.start()
    other.join()
    here = torch.stack([detector._raw_confidence(i, 1) for i in range(2)]).detach()
    assert torch.allclose(here, expected)
