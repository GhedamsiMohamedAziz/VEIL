"""The paired test, checked against hand-computable binomial values."""

from __future__ import annotations

import pytest

from veil.ml.evaluation.significance import (
    SIGNIFICANCE_LEVEL,
    against_every_draw,
    discordance,
    mcnemar_exact,
    paired_test,
)


def records(detected: list[bool], image: int = 0) -> list[dict]:
    return [
        {"image_index": image, "transform_index": i, "detected": hit,
         "max_score": 0.9 if hit else 0.0, "detection_count": int(hit),
         "boxes": [], "transform": {}}
        for i, hit in enumerate(detected)
    ]


def test_mcnemar_matches_the_binomial_tail():
    # Under H0 the discordant pairs are Binomial(n, 0.5); two-sided p is
    # 2 * P(X <= min(b, c)). These are exact, not approximations.
    assert mcnemar_exact(0, 10) == 2 * (1 / 2**10)
    assert mcnemar_exact(1, 9) == 2 * ((1 + 10) / 2**10)
    assert mcnemar_exact(5, 5) == 1.0
    assert mcnemar_exact(3, 0) == 2 * (1 / 2**3)


def test_no_discordant_pairs_means_no_evidence_not_certainty():
    """Identical arms must give p = 1.0. A naive implementation returns 0
    here, which would report 'highly significant' for no difference at all."""
    assert mcnemar_exact(0, 0) == 1.0
    identical = records([True, False, True])
    assert paired_test(identical, identical)["p_value"] == 1.0


def test_discordance_pairs_on_image_and_transform_not_position():
    reference = records([True, True, True])
    candidate = list(reversed(records([True, False, False])))  # order shuffled
    counts = discordance(reference, candidate)
    assert counts["pairs"] == 3
    assert counts["only_reference"] == 2
    assert counts["only_candidate"] == 0


def test_unmatched_records_are_ignored_not_guessed():
    reference = records([True] * 3, image=0)
    candidate = records([False] * 3, image=1)  # a different image entirely
    counts = discordance(reference, candidate)
    assert counts["pairs"] == 0
    assert paired_test(reference, candidate)["p_value"] == 1.0


def test_direction_is_reported_so_a_worse_candidate_is_not_a_win():
    reference = records([False] * 12)
    candidate = records([True] * 12)  # candidate detected MORE often
    result = paired_test(reference, candidate)
    assert result["p_value"] < SIGNIFICANCE_LEVEL  # a real difference...
    assert result["favours_candidate"] is False    # ...in the wrong direction


def test_worst_draw_decides_not_the_most_flattering():
    candidate = records([False] * 12)
    strong = records([True] * 12)                    # candidate clearly better
    weak = records([True] * 6 + [False] * 6)         # weaker evidence
    result = against_every_draw([strong, weak], candidate)
    assert result["draws"] == 2
    assert result["worst_p_value"] == max(t["p_value"] for t in result["per_draw"])
    assert result["worst_p_value"] > min(t["p_value"] for t in result["per_draw"])


def test_significance_requires_every_draw_to_agree():
    candidate = records([False] * 12)
    convincing = records([True] * 12)
    tied = records([False] * 12)  # no discordance at all -> p = 1.0
    assert against_every_draw([convincing], candidate)["significant_against_all"] is True
    assert against_every_draw([convincing, tied], candidate)["significant_against_all"] is False


def test_no_draws_is_reported_as_unavailable():
    assert against_every_draw([], records([True]))["available"] is False


def test_holm_adjusts_across_the_family_and_keeps_the_order_given():
    from veil.ml.evaluation.significance import holm

    assert holm([]) == []
    assert holm([0.03]) == [0.03]  # one test: nothing to correct
    adjusted = holm([0.04, 0.01, 0.03])
    assert adjusted == pytest.approx([0.06, 0.03, 0.06])  # 3*0.01, max(.03, 2*0.03), max(.06, 1*0.04)
    assert holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])
