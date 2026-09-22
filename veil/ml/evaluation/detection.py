"""Run a detector over a transformation sweep and record raw samples.

Everything downstream aggregates these records. They are kept flat and
JSON-serializable so a report can be recomputed from stored results without
re-running the model.
"""

from __future__ import annotations

from typing import Any

import torch

from veil.ml.detectors.base import Detector
from veil.ml.simulation import transforms as T
from veil.ml.simulation.pipeline import render_scene
from veil.ml.simulation.renderer import Placement


def sweep(
    detector: Detector,
    images: torch.Tensor,
    spec: T.TransformSpec,
    target_label: str,
    pattern: torch.Tensor | None = None,
    placement: Placement | list[Placement] | None = None,
    seed: int = 0,
    threshold: float = 0.5,
    image_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """One record per (image, transformation point)."""
    records: list[dict[str, Any]] = []
    for point, params in enumerate(spec.sample(seed=seed)):
        scene = render_scene(images, pattern, placement, params, seed=seed)
        batches = detector.predict(scene, threshold=threshold)
        for i, dets in enumerate(batches):
            # Highest confidence first: ground-truth matching is greedy, and a weak
            # box must not claim the annotation a strong one would have matched.
            hits = sorted((d for d in dets if d.label == target_label),
                          key=lambda d: d.score, reverse=True)
            records.append(
                {
                    "image_index": i,
                    "image_id": (image_ids or [])[i] if image_ids and i < len(image_ids) else None,
                    "transform_index": point,
                    "transform": params.as_dict(),
                    "detected": bool(hits),
                    "max_score": max((d.score for d in hits), default=0.0),
                    "detection_count": len(hits),
                    "boxes": [list(d.box) for d in hits],
                }
            )
    return records
