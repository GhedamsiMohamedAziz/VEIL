"""Per-axis robustness breakdown and the VEIL score.

The VEIL score is not a black box. Each component is:

    robustness(axis) = 1 - detection_rate(samples where that axis is active)

"active" means the axis is away from its neutral value (no rotation, no
tilt, neutral lighting, no blur/noise). A component is `None` - never 0 -
when no sample exercised that axis, because "not measured" and "the model
always saw through it" are different facts.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

from veil.ml.evaluation.metrics import summarize

# axis name -> predicate on the transform parameters of one sample
AXES: dict[str, Callable[[dict[str, float]], bool]] = {
    "angle": lambda t: abs(t.get("rotation_deg", 0.0)) > 1e-6 or abs(t.get("perspective", 0.0)) > 1e-6,
    "lighting": lambda t: abs(t.get("brightness", 1.0) - 1.0) > 1e-6 or abs(t.get("contrast", 1.0) - 1.0) > 1e-6,
    "scale": lambda t: abs(t.get("scale", 1.0) - 1.0) > 1e-6,
    "camera_noise": lambda t: t.get("blur_sigma", 0.0) > 1e-6 or t.get("noise_std", 0.0) > 1e-6,
    "deformation": lambda t: t.get("deformation", 0.0) > 1e-6,
}


def by_axis(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    records = list(records)
    out: dict[str, Any] = {}
    for axis, active in AXES.items():
        subset = [r for r in records if active(r["transform"])]
        out[axis] = summarize(subset) if subset else {"samples": 0, "detection_rate": None}
    return out


def by_value(records: Iterable[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    """Detection rate bucketed by one transform parameter's exact value -
    this is what the 'detection rate by angle' chart plots."""
    records = list(records)
    values = sorted({r["transform"].get(field, 0.0) for r in records})
    rows = []
    for value in values:
        subset = [r for r in records if r["transform"].get(field, 0.0) == value]
        rows.append({"value": value, **summarize(subset)})
    return rows


def veil_score(
    digital_records: Iterable[dict[str, Any]],
    physical_records: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Transparent experimental score. Values are fractions in [0,1]."""
    digital_records = list(digital_records)
    axes = by_axis(digital_records)
    overall = summarize(digital_records)

    def robustness(block: dict[str, Any]) -> float | None:
        rate = block.get("detection_rate")
        return None if rate is None else 1.0 - rate

    physical = summarize(physical_records) if physical_records else {"samples": 0, "detection_rate": None}
    return {
        "digital_robustness": robustness(overall),
        "angle_robustness": robustness(axes["angle"]),
        "lighting_robustness": robustness(axes["lighting"]),
        "scale_robustness": robustness(axes["scale"]),
        "camera_noise_robustness": robustness(axes["camera_noise"]),
        "physical_robustness": robustness(physical),
        "definition": "robustness = 1 - detection_rate over the samples exercising that axis",
        "sample_counts": {
            "digital": overall.get("samples", 0),
            **{f"axis_{k}": v.get("samples", 0) for k, v in axes.items()},
            "physical": physical.get("samples", 0),
        },
    }
