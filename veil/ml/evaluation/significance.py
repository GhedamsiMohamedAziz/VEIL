"""Is the difference between two arms real, or could it be chance?

The control and candidate arms are measured on the **same** images at the
**same** transformation points. They are paired, not independent, so the
right test is McNemar's on the discordant pairs - the cases where one arm
detected the target and the other did not. Concordant pairs carry no
information about a difference and are correctly ignored.

The exact (binomial) form is used rather than the chi-square approximation,
because VEIL runs routinely produce a handful of discordant pairs, which is
exactly where the approximation misbehaves.

No SciPy: `math.comb` is all an exact binomial tail needs.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

SIGNIFICANCE_LEVEL = 0.05


def _key(record: dict[str, Any]) -> tuple[Any, Any]:
    return (record.get("image_index"), record.get("transform_index"))


def discordance(
    reference: Iterable[dict[str, Any]], candidate: Iterable[dict[str, Any]]
) -> dict[str, int]:
    """Count paired outcomes between two arms.

    `only_reference` is the count where the reference detected and the
    candidate did not - the direction that favours the candidate.
    """
    ref = {_key(r): bool(r["detected"]) for r in reference}
    cand = {_key(r): bool(r["detected"]) for r in candidate}
    shared = ref.keys() & cand.keys()
    only_reference = sum(1 for k in shared if ref[k] and not cand[k])
    only_candidate = sum(1 for k in shared if cand[k] and not ref[k])
    return {
        "pairs": len(shared),
        "only_reference": only_reference,
        "only_candidate": only_candidate,
        "discordant": only_reference + only_candidate,
    }


def mcnemar_exact(only_reference: int, only_candidate: int) -> float:
    """Two-sided exact McNemar p-value.

    Under the null the discordant pairs split 50/50, so the count in either
    direction is Binomial(n, 0.5). With no discordant pairs there is no
    evidence of a difference and the p-value is 1.0 - not 0.
    """
    n = only_reference + only_candidate
    if n == 0:
        return 1.0
    smaller = min(only_reference, only_candidate)
    tail = sum(math.comb(n, i) for i in range(smaller + 1)) / (2**n)
    return min(2.0 * tail, 1.0)


def paired_test(
    reference: list[dict[str, Any]], candidate: list[dict[str, Any]]
) -> dict[str, Any]:
    """McNemar's test for one control draw against the candidate."""
    counts = discordance(reference, candidate)
    p_value = mcnemar_exact(counts["only_reference"], counts["only_candidate"])
    return {
        **counts,
        "p_value": p_value,
        "significant": p_value < SIGNIFICANCE_LEVEL,
        "favours_candidate": counts["only_reference"] > counts["only_candidate"],
    }


def against_every_draw(
    draws: list[list[dict[str, Any]]], candidate: list[dict[str, Any]]
) -> dict[str, Any]:
    """Test the candidate against each control draw and take the worst result.

    A claim that the optimized pattern works must hold against *every*
    unoptimized draw, not the most flattering one. Taking the largest
    p-value is an intersection-union test: conservative by construction, and
    it removes the temptation to report the best of N comparisons.
    """
    if not draws:
        return {"available": False, "reason": "no control draws to test against"}
    tests = [paired_test(draw, candidate) for draw in draws]
    worst = max(tests, key=lambda t: t["p_value"])
    return {
        "available": True,
        "test": "exact McNemar, paired on (image, transformation point)",
        "draws": len(tests),
        "per_draw": tests,
        "worst_p_value": worst["p_value"],
        "significant_against_all": all(
            t["significant"] and t["favours_candidate"] for t in tests
        ),
        "alpha": SIGNIFICANCE_LEVEL,
        "note": (
            "The worst of the per-draw p-values is reported: the claim must "
            "hold against every unoptimized control, not the weakest one."
        ),
    }


def holm(p_values: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, in the order given. Testing a pattern
    against m transfer detectors gives m chances of a spurious "significant";
    the adjusted value is the one to hold against alpha."""
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted, running = [0.0] * len(p_values), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted
