"""The set of detectors VEIL is allowed to test against.

Registration is code, not user input: an experiment stores a detector *id*,
and this module is the only thing that turns an id into a loaded model. That
is what keeps "no arbitrary model loading" true (docs/security.md).
"""

from __future__ import annotations

from collections.abc import Callable

from veil.ml.detectors.base import Detector, DetectorInfo
from veil.ml.detectors.blob import ColorBlobDetector
from veil.ml.detectors.torchvision_detector import _MODELS, TorchvisionDetector

_FACTORIES: dict[str, Callable[[], Detector]] = {
    "colorblob-v1": ColorBlobDetector,
    **{mid: (lambda m=mid: TorchvisionDetector(m)) for mid in _MODELS},
}

_CACHE: dict[str, Detector] = {}


def available() -> list[DetectorInfo]:
    return [_FACTORIES[k]().metadata() for k in sorted(_FACTORIES)]


def info(detector_id: str) -> DetectorInfo:
    """A detector's metadata without loading its weights - for validation and
    reports, which only need labels and version."""
    if detector_id not in _FACTORIES:
        raise KeyError(f"unknown detector: {detector_id}")
    return _FACTORIES[detector_id]().metadata()


def exists(detector_id: str) -> bool:
    return detector_id in _FACTORIES


def get(detector_id: str) -> Detector:
    """Return a loaded detector. Cached because weights are expensive."""
    if detector_id not in _FACTORIES:
        raise KeyError(f"unknown detector: {detector_id}")
    if detector_id not in _CACHE:
        _CACHE[detector_id] = _FACTORIES[detector_id]().load()
    return _CACHE[detector_id]
