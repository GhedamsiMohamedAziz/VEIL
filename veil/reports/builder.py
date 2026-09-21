"""Assemble the machine-readable experiment report.

Rule: the report contains only numbers that were measured in the run it
describes, plus the configuration needed to reproduce them. It always
carries a limitations section - a VEIL result without its test conditions
is not a VEIL result.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from veil.db import scoped
from veil.ml.detectors import registry
from veil.models import Dataset, Evaluation, Experiment, ExperimentRun, Pattern, PhysicalTest

NOT_HELD_OUT = (
    "The pattern was evaluated on the same image(s) it was optimized on: the "
    "dataset was too small to hold any out. Every rate in this report is a "
    "training score and says nothing about a new photograph."
)

LIMITATIONS = [
    "This report describes experimental results obtained under the specified "
    "test conditions. Results may not generalize to other models, cameras, "
    "environments, or deployment systems.",
    "Digital results are measured on a simulated transformation distribution. "
    "Simulation is an approximation of physical capture, not a substitute for it.",
    "Detection rates are estimated from a finite sample; 95% confidence "
    "intervals are reported alongside each rate. The intervals treat every "
    "image-transformation pair as independent and describe the rate on the "
    "evaluated images only; with few images they say little about new ones.",
    "A pattern optimized against one detector is not expected to transfer to "
    "detectors it was not measured against.",
    "The baseline-to-candidate drop includes the effect of the patch covering "
    "part of the subject. Only the control-to-candidate difference is "
    "attributable to the pattern itself; where no control arm was run, no such "
    "attribution can be made.",
    "Random patterns vary in how much they suppress detection. The candidate "
    "is compared against every control draw with an exact paired McNemar "
    "test, and the worst p-value is reported; a difference that does not "
    "reach significance against all of them is not a finding.",
    "Statistical significance is not effect size. A difference can be real "
    "and still too small to matter, and both numbers are reported for that "
    "reason.",
    "Transfer results, where present, describe only the detectors listed. "
    "They say nothing about models that were not measured, and a pattern "
    "that transfers to one model is not thereby general.",
]


def _jsonable(value: Any) -> Any:
    """The report is stored in a JSON column and served as JSON, so it holds
    no Python objects - timestamps become ISO-8601 strings once, here."""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def build(session: Session, experiment: Experiment, run: ExperimentRun | None = None) -> dict[str, Any]:
    org = experiment.organization_id
    if run is None:
        run = session.scalar(
            scoped(ExperimentRun, org)
            .where(ExperimentRun.experiment_id == experiment.id, ExperimentRun.status == "completed")
            .order_by(ExperimentRun.created_at.desc())
        )
    query = scoped(Evaluation, org).where(Evaluation.experiment_id == experiment.id)
    if run is not None:
        query = query.where(Evaluation.run_id == run.id)
    evaluation = session.scalar(query.order_by(Evaluation.created_at.desc()))
    if evaluation is None:
        raise ValueError("experiment has no completed evaluation to report on")

    pattern = session.get(Pattern, evaluation.pattern_id) if evaluation.pattern_id else None
    dataset = session.get(Dataset, experiment.dataset_id)
    physical = list(session.scalars(
        scoped(PhysicalTest, org).where(PhysicalTest.experiment_id == experiment.id)
    ))
    detector_info = registry.info(experiment.detector_id).as_dict()

    comparison = evaluation.metrics.get("comparison", {})
    # Evaluations stored before the split existed were not held out either.
    split = evaluation.metrics.get("split") or {"held_out": False}
    return _jsonable({
        "title": f"VEIL Experiment Report #{experiment.number:03d}",
        "experiment": {
            "id": experiment.id, "number": experiment.number, "name": experiment.name,
            "description": experiment.description, "status": experiment.status,
            "created_at": experiment.created_at, "completed_at": experiment.completed_at,
        },
        "reproducibility": {
            "run_id": run.id if run else None,
            "seed": run.seed if run else None,
            "code_version": run.code_version if run else None,
            "detector_version": run.detector_version if run else None,
            "dataset_version": run.dataset_version if run else None,
            "configuration": run.configuration if run else experiment.configuration,
        },
        "detector": detector_info,
        "dataset": {
            "id": dataset.id, "name": dataset.name, "version": dataset.version,
            "source": dataset.source, "license": dataset.license,
            "item_count": len(dataset.artifact_ids),
            "has_ground_truth": bool(dataset.annotations),
        } if dataset else None,
        "pattern": {
            "id": pattern.id, "version": pattern.version,
            "generation_parameters": pattern.generation_parameters,
            "optimization": pattern.metrics,
        } if pattern else None,
        "transformations": evaluation.transformations,
        "results": {
            "sample_count": evaluation.sample_count,
            "baseline": evaluation.baseline_metrics,
            "control": evaluation.control_metrics or None,
            "candidate": evaluation.candidate_metrics,
            "delta": comparison.get("delta", {}),
            "delta_vs_control": comparison.get("delta_vs_control", {}),
            "attribution": comparison.get("attribution", {}),
            "per_axis": comparison.get("per_axis", {}),
            "by_rotation": comparison.get("by_rotation", {}),
            "by_brightness": comparison.get("by_brightness", {}),
            "veil_score": evaluation.metrics.get("veil_score", {}),
            "transfer": evaluation.metrics.get("transfer", {}),
            "ground_truth": evaluation.metrics.get("ground_truth", {}),
            "split": split,
        },
        "physical_tests": [
            {
                "id": t.id, "arm": t.arm or "unspecified", "camera": t.camera, "resolution": t.resolution, "fps": t.fps,
                "distance_m": t.distance_m, "angle_deg": t.angle_deg, "lighting": t.lighting,
                "environment": t.environment, "frame_count": t.frame_count, "result": t.result,
            }
            for t in physical
        ],
        "limitations": LIMITATIONS if split.get("held_out") else [NOT_HELD_OUT, *LIMITATIONS],
    })
