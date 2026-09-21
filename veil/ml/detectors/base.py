"""Detector abstraction.

Everything downstream of this file speaks one dialect:

    images  : torch.Tensor, float32, shape [B, 3, H, W], values in [0, 1]
    output  : list[list[Detection]], one list per image

`score(...)` is the differentiable hook used by pattern optimization. A
detector that cannot provide gradients sets `differentiable = False` and the
optimizer switches to a gradient-free search (see ml/patterns/optimizer.py).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(frozen=True)
class Detection:
    label: str
    score: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels

    def as_dict(self) -> dict[str, Any]:
        return {"label": self.label, "score": self.score, "box": list(self.box)}


@dataclass
class DetectorInfo:
    id: str
    name: str
    version: str
    task: str = "object-detection"
    labels: list[str] = field(default_factory=list)
    differentiable: bool = False
    license: str = ""
    source: str = ""
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        # Label lists can be long (COCO has 91); expose a count plus a sample.
        d["label_count"] = len(self.labels)
        d["labels"] = self.labels[:20]
        return d


class Detector(ABC):
    """A computer-vision model VEIL is authorized to test against."""

    differentiable: bool = False

    @abstractmethod
    def load(self) -> "Detector":
        """Fetch weights / build the graph. Idempotent."""

    @abstractmethod
    def predict(self, images: torch.Tensor, threshold: float = 0.5) -> list[list[Detection]]:
        """Run inference. Must not require gradients."""

    @abstractmethod
    def metadata(self) -> DetectorInfo:
        """Everything needed to reproduce a result with this detector."""

    def score(self, images: torch.Tensor, label: str) -> torch.Tensor:
        """Differentiable per-image confidence for `label`, shape [B].

        Returns 0.0 for images with no detection of that label, kept
        connected to the input graph so autograd does not break.
        """
        raise NotImplementedError(
            f"{type(self).__name__} is not differentiable; use a gradient-free optimizer"
        )

    def evaluate(self, images: torch.Tensor, label: str, threshold: float = 0.5) -> dict[str, Any]:
        """Convenience summary for one batch. Real metrics live in
        ml/evaluation/metrics.py - this is the per-batch primitive."""
        batches = self.predict(images, threshold=threshold)
        hits = [[d for d in dets if d.label == label] for dets in batches]
        return {
            "images": len(batches),
            "detected": sum(1 for h in hits if h),
            "max_scores": [max((d.score for d in h), default=0.0) for h in hits],
            "detections": [[d.as_dict() for d in h] for h in hits],
        }
