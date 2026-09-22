"""A small, deterministic, dependency-free detector.

This is not a stand-in for a real model: it is a genuine (if primitive)
colour-blob detector used as the default in CI and on machines that cannot
download weights. It makes every layer above it testable without a 100 MB
download, and it exercises the non-differentiable branch of the optimizer.
"""

from __future__ import annotations

import torch

from veil.ml.detectors.base import Detection, Detector, DetectorInfo

LABEL = "blob"


class ColorBlobDetector(Detector):
    """Detects a saturated region of a target hue: one bounding box around every
    matching pixel (no connected components - two separate blobs give one box).

    A region counts as a detection when enough pixels are both bright and
    dominated by the target channel. Score is the fraction of such pixels in
    the region's bounding box, which makes it monotonic and comparable
    across images.
    """

    differentiable = False

    def __init__(self, channel: int = 0, min_pixels: int = 64, dominance: float = 0.25):
        self.channel = channel
        self.min_pixels = min_pixels
        self.dominance = dominance

    def load(self) -> "ColorBlobDetector":
        return self

    def _mask(self, image: torch.Tensor) -> torch.Tensor:
        target = image[self.channel]
        others = (image.sum(0) - target) / 2.0
        return (target - others) > self.dominance

    @torch.no_grad()
    def predict(self, images: torch.Tensor, threshold: float = 0.5) -> list[list[Detection]]:
        out: list[list[Detection]] = []
        for image in images:
            mask = self._mask(image)
            count = int(mask.sum())
            if count < self.min_pixels:
                out.append([])
                continue
            ys, xs = torch.nonzero(mask, as_tuple=True)
            x1, x2 = float(xs.min()), float(xs.max()) + 1
            y1, y2 = float(ys.min()), float(ys.max()) + 1
            area = max((x2 - x1) * (y2 - y1), 1.0)
            score = min(count / area, 1.0)
            out.append([Detection(LABEL, score, (x1, y1, x2, y2))] if score >= threshold else [])
        return out

    def metadata(self) -> DetectorInfo:
        return DetectorInfo(
            id="colorblob-v1",
            name="Colour blob detector",
            version="1.0.0",
            labels=[LABEL],
            differentiable=False,
            license="MIT (VEIL)",
            source="veil.ml.detectors.blob",
            notes=(
                "Deterministic non-ML detector. Used for CI and for offline "
                "smoke tests; results are not representative of neural detectors."
            ),
        )
