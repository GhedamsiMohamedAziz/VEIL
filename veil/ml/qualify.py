"""Is this dataset usable before you spend compute on it?

The failure this prevents: optimizing a pattern for an hour against images
the detector barely sees the target in, then discovering post-hoc that there
was no headroom (`MIN_BASELINE_RATE_FOR_VERDICT` in comparison.py catches it
only after the fact).

Qualification is the same baseline sweep an experiment starts with, run
per-image and reported per-image, so a weak dataset can be fixed by dropping
the photos that do not work rather than abandoned.
"""

from __future__ import annotations

from typing import Any

import torch

from veil.ml.detectors.base import Detector
from veil.ml.evaluation.detection import sweep
from veil.ml.evaluation.metrics import summarize
from veil.ml.simulation.transforms import TransformSpec

# An image the detector sees the target in less than half the time across the
# transformation grid cannot support a measurable drop. This is stricter than
# the post-hoc verdict floor (0.2), because at qualification time the cost of
# rejecting a borderline image is one photo, not a wasted run.
MIN_IMAGE_RATE = 0.5
MIN_DATASET_RATE = 0.6


def qualify(
    detector: Detector,
    images: torch.Tensor,
    image_ids: list[str],
    target_label: str,
    *,
    threshold: float = 0.5,
    spec: TransformSpec | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Baseline-sweep a dataset and report whether it can carry an experiment."""
    spec = spec or TransformSpec(
        rotation_deg=[-15.0, 0.0, 15.0], scale=[1.0], translate=[0.0],
        perspective=[0.0], brightness=[0.8, 1.0, 1.2], contrast=[1.0],
        blur_sigma=[0.0], noise_std=[0.0], deformation=[0.0],
        mode="grid", max_samples=9,
    )
    records = sweep(detector, images, spec, target_label, seed=seed,
                    threshold=threshold, image_ids=image_ids)

    per_image = []
    for index, image_id in enumerate(image_ids):
        subset = [r for r in records if r["image_index"] == index]
        stats = summarize(subset)
        per_image.append({
            "artifact_id": image_id,
            "detection_rate": stats["detection_rate"],
            "mean_confidence": stats["mean_confidence"],
            "samples": stats["samples"],
            "usable": (stats["detection_rate"] or 0.0) >= MIN_IMAGE_RATE,
        })

    overall = summarize(records)
    usable = [image for image in per_image if image["usable"]]
    rate = overall["detection_rate"] or 0.0
    dataset_usable = rate >= MIN_DATASET_RATE and len(usable) >= 1

    unusable_count = len(per_image) - len(usable)
    if dataset_usable:
        verdict = (
            f"usable: the detector sees {target_label!r} in {rate * 100:.0f}% of "
            f"baseline samples, leaving room to measure a drop"
        )
    elif usable and unusable_count:
        verdict = (
            f"partially usable: {len(usable)} of {len(per_image)} images clear "
            f"the {MIN_IMAGE_RATE * 100:.0f}% bar. Drop the other "
            f"{unusable_count} and re-qualify"
        )
    elif usable:
        # Every image clears the per-image bar, yet the aggregate is still
        # short: the images are uniformly marginal rather than mixed, and
        # there is nothing to drop.
        verdict = (
            f"marginal: every image clears the {MIN_IMAGE_RATE * 100:.0f}% bar "
            f"but the overall rate is {rate * 100:.0f}%, below the "
            f"{MIN_DATASET_RATE * 100:.0f}% needed for a measurable drop. Use "
            f"imagery the detector sees more reliably, or accept wide intervals"
        )
    else:
        verdict = (
            f"not usable: the detector sees {target_label!r} in only "
            f"{rate * 100:.0f}% of baseline samples. There is nothing for a "
            f"pattern to suppress, so any measured drop would be noise"
        )

    return {
        "detector_id": detector.metadata().id,
        "detector_version": detector.metadata().version,
        "target_label": target_label,
        "threshold": threshold,
        "transformations": spec.as_dict(),
        "overall": overall,
        "per_image": per_image,
        "usable_image_ids": [image["artifact_id"] for image in usable],
        "unusable_image_ids": [i["artifact_id"] for i in per_image if not i["usable"]],
        "dataset_usable": dataset_usable,
        "thresholds": {"min_image_rate": MIN_IMAGE_RATE, "min_dataset_rate": MIN_DATASET_RATE},
        "verdict": verdict,
    }
