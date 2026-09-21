"""Baseline vs control vs candidate.

Three arms, because two are not enough to make a claim:

* **baseline**  - the scene as it is, no pattern.
* **control**   - *several* unoptimized patterns of the same size, palette
  and placement. Everything the candidate has except the optimization. Several
  because one random pattern is a sample of one: without its spread, "the
  candidate beat the control" can be noise between two arbitrary draws.
* **candidate** - the optimized pattern.

`baseline -> candidate` mixes two effects: the patch physically covers part
of the subject, and the pattern was optimized against the detector. Only
`control -> candidate` isolates the second. A platform that reports the first
and calls it robustness is measuring occlusion and mislabelling it.
"""

from __future__ import annotations

from typing import Any

from veil.ml.evaluation.metrics import summarize
from veil.ml.evaluation.robustness import by_axis, by_value
from veil.ml.evaluation.significance import against_every_draw

# Below this many samples, an attribution verdict is noise. The Wilson
# interval on each arm is reported regardless; this only gates the verdict.
MIN_SAMPLES_FOR_VERDICT = 20

# If the detector barely saw the target *without* any pattern, there was
# nothing to suppress and every downstream number is measuring noise around
# zero. This is the usual shape of a meaningless transfer result: a model
# that never detected the subject in the first place cannot be shown to stop
# detecting it.
MIN_BASELINE_RATE_FOR_VERDICT = 0.2


def _delta(a: dict[str, Any], b: dict[str, Any], key: str) -> float | None:
    """b - a, or None when either side was not measured."""
    x, y = a.get(key), b.get(key)
    return None if x is None or y is None else y - x


def control_spread(draws: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize the per-draw control detection rates.

    `best` is the *lowest* rate: the luckiest unoptimized pattern. A candidate
    that does not beat it has not been shown to do anything a random pattern
    could not.
    """
    rates = [d["detection_rate"] for d in draws if d.get("detection_rate") is not None]
    if not rates:
        return {"draws": 0}
    mean = sum(rates) / len(rates)
    variance = sum((r - mean) ** 2 for r in rates) / len(rates)
    return {
        "draws": len(rates),
        "rates": rates,
        "mean": mean,
        "best": min(rates),       # lowest detection = strongest random pattern
        "worst": max(rates),
        "std": variance**0.5,
        "range": max(rates) - min(rates),
    }


def attribution(
    baseline: dict[str, Any],
    control: dict[str, Any] | None,
    candidate: dict[str, Any],
    spread: dict[str, Any] | None = None,
    significance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Split the observed drop into occlusion and optimization.

        total       = baseline_rate - candidate_rate
        occlusion   = baseline_rate - control_rate      (pooled over draws)
        attributable= control_rate  - candidate_rate    <- the headline claim

    With several control draws we also report the stricter test:

        vs_best_control = best_control_rate - candidate_rate

    where `best` is the lowest-detection unoptimized draw. A candidate that
    beats the mean but not the best has only beaten average luck.

    `significance` carries the paired McNemar result against every control
    draw. Effect size and significance answer different questions and both
    gate the verdict: a large drop that could be chance is not a finding, and
    a significant but tiny drop is not a product.
    """
    if control is None or control.get("detection_rate") is None:
        return {
            "available": False,
            "reason": "no control arm was run; the observed drop cannot be "
            "separated from occlusion",
        }
    base_rate = baseline.get("detection_rate")
    control_rate = control.get("detection_rate")
    candidate_rate = candidate.get("detection_rate")
    if base_rate is None or candidate_rate is None:
        return {"available": False, "reason": "an arm produced no samples"}

    total = base_rate - candidate_rate
    occlusion = base_rate - control_rate
    attributable = control_rate - candidate_rate
    samples = min(baseline.get("samples", 0), control.get("samples", 0), candidate.get("samples", 0))

    spread = spread or {}
    draws = spread.get("draws", 0)
    best_control = spread.get("best")
    vs_best = None if best_control is None else best_control - candidate_rate

    if base_rate < MIN_BASELINE_RATE_FOR_VERDICT:
        # Checked before the sample count: no number of samples rescues an
        # experiment whose detector never saw the target to begin with.
        verdict = (
            f"no headroom: the detector saw the target in only "
            f"{base_rate * 100:.1f}% of baseline samples, so there was almost "
            f"nothing for a pattern to suppress"
        )
    elif samples < MIN_SAMPLES_FOR_VERDICT:
        verdict = "inconclusive: too few samples to attribute the effect"
    elif draws > 1 and vs_best is not None and vs_best <= 0:
        verdict = (
            f"within control variance: the candidate did not beat the best of "
            f"{draws} unoptimized patterns"
        )
    elif significance and significance.get("available") and not significance["significant_against_all"]:
        # Paired McNemar against every control draw, worst p-value taken.
        verdict = (
            f"not statistically significant: the difference from {significance['draws']} "
            f"unoptimized pattern(s) could be chance "
            f"(worst p = {significance['worst_p_value']:.3f}, alpha = {significance['alpha']})"
        )
    elif attributable <= 0:
        verdict = (
            "no measurable optimization effect: the optimized pattern did not "
            "beat an unoptimized one of the same size"
        )
    elif total <= 0:
        verdict = "the pattern did not reduce detection relative to the baseline"
    elif attributable / total < 0.25:
        verdict = "mostly occlusion: under a quarter of the drop is attributable to the pattern"
    elif attributable / total < 0.5:
        verdict = "minority effect: occlusion accounts for most of the measured drop"
    else:
        verdict = "optimization accounts for the majority of the measured drop"

    return {
        "available": True,
        "total_drop": total,
        "occlusion_drop": occlusion,
        "attributable_drop": attributable,
        "attributable_fraction": (attributable / total) if total > 0 else None,
        "attributable_vs_best_control": vs_best,
        "control_spread": spread or None,
        "significance": significance or None,
        "samples_per_arm": samples,
        "baseline_rate": base_rate,
        "has_headroom": base_rate >= MIN_BASELINE_RATE_FOR_VERDICT,
        "verdict": verdict,
        "definition": (
            "attributable_drop = control_detection_rate - candidate_detection_rate, "
            "where the control is one or more unoptimized patterns of identical "
            "size, palette and placement. attributable_vs_best_control uses the "
            "lowest-detection control draw and is the stricter test."
        ),
    }


def compare(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    control: list[dict[str, Any]] | None = None,
    control_draws: list[list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """`control` is the pooled records; `control_draws` the per-draw records."""
    base, cand = summarize(baseline), summarize(candidate)
    ctrl = summarize(control) if control else None
    spread = control_spread([summarize(d) for d in control_draws]) if control_draws else None
    significance = against_every_draw(control_draws, candidate) if control_draws else None

    axes_b, axes_c = by_axis(baseline), by_axis(candidate)
    axes_ctrl = by_axis(control) if control else {}

    result: dict[str, Any] = {
        "baseline": base,
        "control": ctrl,
        "candidate": cand,
        "delta": {
            "detection_rate": _delta(base, cand, "detection_rate"),
            "mean_confidence": _delta(base, cand, "mean_confidence"),
            "max_confidence": _delta(base, cand, "max_confidence"),
        },
        "delta_vs_control": {
            "detection_rate": _delta(ctrl, cand, "detection_rate") if ctrl else None,
            "mean_confidence": _delta(ctrl, cand, "mean_confidence") if ctrl else None,
        },
        "attribution": attribution(base, ctrl, cand, spread, significance),
        "control_spread": spread,
        "significance": significance,
        "per_axis": {
            axis: {
                "baseline_detection_rate": axes_b[axis].get("detection_rate"),
                "control_detection_rate": axes_ctrl.get(axis, {}).get("detection_rate"),
                "candidate_detection_rate": axes_c[axis].get("detection_rate"),
                "samples": axes_c[axis].get("samples", 0),
            }
            for axis in axes_b
        },
        "by_rotation": {
            "baseline": by_value(baseline, "rotation_deg"),
            "candidate": by_value(candidate, "rotation_deg"),
        },
        "by_brightness": {
            "baseline": by_value(baseline, "brightness"),
            "candidate": by_value(candidate, "brightness"),
        },
    }
    if control:
        result["by_rotation"]["control"] = by_value(control, "rotation_deg")
        result["by_brightness"]["control"] = by_value(control, "brightness")
    return result
