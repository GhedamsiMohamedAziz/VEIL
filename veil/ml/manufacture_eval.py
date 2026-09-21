"""Does the pattern still work once a machine has made it?

Optimization produces float pixels. A knitting machine produces stitches in
yarn colours; a printer produces dots in ink. Committing to either loses
information, and the honest question is how much *effect* is lost with it.

This re-measures the pattern under a production spec against the same
detector, dataset and transformation grid as the original run, so the
retained effect is a like-for-like comparison and not a new experiment.
"""

from __future__ import annotations

from typing import Any

import torch

from veil.ml.detectors.base import Detector
from veil.ml.evaluation.detection import sweep
from veil.ml.evaluation.metrics import summarize
from veil.ml.patterns.manufacture import ProductionSpec, colour_usage, to_artwork, to_production
from veil.ml.simulation.renderer import Placement
from veil.ml.simulation.transforms import TransformSpec


def evaluate_production(
    detector: Detector,
    images: torch.Tensor,
    image_ids: list[str],
    pattern: torch.Tensor,
    placement: Placement | list[Placement],
    spec: TransformSpec,
    production: ProductionSpec,
    *,
    target_label: str,
    threshold: float,
    seed: int,
    control_rate: float | None = None,
    digital_rate: float | None = None,
) -> dict[str, Any]:
    """Sweep the manufacturable pattern and report what survived."""
    artwork = to_artwork(pattern, production)
    # Back to the optimizer's working resolution so the renderer composites it
    # the same way it did during the run - the production grid is far larger.
    resized = torch.nn.functional.interpolate(
        artwork.unsqueeze(0), size=pattern.shape[-2:], mode="area"
    ).squeeze(0)

    records = sweep(detector, images, spec, target_label, pattern=resized,
                    placement=placement, seed=seed, threshold=threshold,
                    image_ids=image_ids)
    stats = summarize(records)
    produced_rate = stats["detection_rate"]

    retained = None
    if control_rate is not None and digital_rate is not None and produced_rate is not None:
        digital_effect = control_rate - digital_rate
        produced_effect = control_rate - produced_rate
        retained = (produced_effect / digital_effect) if digital_effect > 0 else None

    grid_h, grid_w = production.grid
    return {
        "production": production.as_dict(),
        "grid": {"rows": grid_h, "columns": grid_w,
                 "units": "stitches" if production.method != "print" else "pixels"},
        "metrics": stats,
        "detection_rate": produced_rate,
        "digital_detection_rate": digital_rate,
        "control_detection_rate": control_rate,
        "effect_retained": retained,
        "colour_usage": colour_usage(to_production(pattern, production), production),
        "note": (
            "effect_retained = (control - produced) / (control - digital). "
            "1.0 means manufacturing cost nothing; below ~0.8 the pattern "
            "should be re-optimized under the production constraint rather "
            "than quantized after the fact."
        ),
    }
