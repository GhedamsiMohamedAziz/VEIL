"""Effect attribution: separating the pattern from the patch.

The failure this guards against is the one the platform is most likely to
commit: reporting an occlusion effect as a robustness result.
"""

from __future__ import annotations

from veil.ml.evaluation.comparison import MIN_SAMPLES_FOR_VERDICT, attribution, compare
from veil.ml.evaluation.metrics import summarize


def arm(detected: int, missed: int):
    def one(hit: bool):
        return {
            "image_index": 0, "image_id": "img", "transform_index": 0,
            "transform": {"rotation_deg": 0.0, "brightness": 1.0},
            "detected": hit, "max_score": 0.9 if hit else 0.0,
            "detection_count": int(hit), "boxes": [],
        }
    return [one(True)] * detected + [one(False)] * missed


N = MIN_SAMPLES_FOR_VERDICT


def test_drop_is_split_into_occlusion_and_optimization():
    # baseline 100% -> control 30% -> candidate 0%.
    result = attribution(summarize(arm(N, 0)), summarize(arm(int(N * 0.3), N - int(N * 0.3))), summarize(arm(0, N)))
    assert result["available"] is True
    assert result["total_drop"] == 1.0
    assert abs(result["occlusion_drop"] - 0.7) < 1e-9
    assert abs(result["attributable_drop"] - 0.3) < 1e-9
    assert abs(result["attributable_fraction"] - 0.3) < 1e-9


def test_a_useless_optimizer_is_reported_as_useless():
    """Control and candidate identical: the patch did everything."""
    result = attribution(summarize(arm(N, 0)), summarize(arm(0, N)), summarize(arm(0, N)))
    assert result["attributable_drop"] == 0.0
    assert "no measurable optimization effect" in result["verdict"]


def test_a_worse_than_control_candidate_is_not_dressed_up():
    result = attribution(summarize(arm(N, 0)), summarize(arm(0, N)), summarize(arm(N // 2, N // 2)))
    assert result["attributable_drop"] < 0
    assert "no measurable optimization effect" in result["verdict"]


def test_verdict_wording_matches_the_arithmetic():
    cases = {0.1: "mostly occlusion", 0.4: "minority effect", 0.8: "majority"}
    for fraction, expected in cases.items():
        detected_in_control = round(N * fraction)
        result = attribution(
            summarize(arm(N, 0)),
            summarize(arm(detected_in_control, N - detected_in_control)),
            summarize(arm(0, N)),
        )
        assert abs(result["attributable_fraction"] - fraction) < 0.05
        assert expected in result["verdict"], (fraction, result["verdict"])


def test_small_samples_produce_no_verdict():
    result = attribution(summarize(arm(4, 0)), summarize(arm(2, 2)), summarize(arm(0, 4)))
    assert "inconclusive" in result["verdict"]
    assert result["samples_per_arm"] == 4


def test_missing_control_is_stated_not_assumed():
    result = attribution(summarize(arm(N, 0)), None, summarize(arm(0, N)))
    assert result["available"] is False
    assert "cannot be separated from occlusion" in result["reason"]

    # ...and compare() must not silently invent one.
    without = compare(arm(N, 0), arm(0, N))
    assert without["control"] is None
    assert without["attribution"]["available"] is False
    assert without["delta_vs_control"]["detection_rate"] is None


def test_compare_carries_all_three_arms_per_axis():
    result = compare(arm(N, 0), arm(0, N), arm(N // 2, N - N // 2))
    assert result["control"]["detection_rate"] == 0.5
    assert result["delta_vs_control"]["detection_rate"] == -0.5
    angle = result["per_axis"]["angle"]
    # No rotation in these records, so the angle axis was never exercised.
    assert angle["candidate_detection_rate"] is None
    assert "control" in result["by_rotation"]


def test_control_spread_summarizes_the_draws():
    from veil.ml.evaluation.comparison import control_spread

    draws = [summarize(arm(2, 8)), summarize(arm(5, 5)), summarize(arm(8, 2))]
    spread = control_spread(draws)
    assert spread["draws"] == 3
    assert spread["rates"] == [0.2, 0.5, 0.8]
    assert abs(spread["mean"] - 0.5) < 1e-9
    assert spread["best"] == 0.2   # lowest detection = strongest random pattern
    assert spread["worst"] == 0.8
    assert abs(spread["range"] - 0.6) < 1e-9


def test_beating_the_mean_control_but_not_the_best_is_not_a_result():
    """The candidate (30%) beats the mean control (50%) but loses to the
    luckiest random pattern (20%). That is not evidence of anything."""
    base, cand = summarize(arm(N, 0)), summarize(arm(int(N * 0.3), N - int(N * 0.3)))
    pooled = summarize(arm(int(N * 0.5), N - int(N * 0.5)))
    spread = {"draws": 3, "rates": [0.2, 0.5, 0.8], "mean": 0.5, "best": 0.2,
              "worst": 0.8, "std": 0.245, "range": 0.6}
    result = attribution(base, pooled, cand, spread)
    assert result["attributable_drop"] > 0            # beats the mean
    assert result["attributable_vs_best_control"] < 0  # loses to the best draw
    assert "within control variance" in result["verdict"]


def test_an_effect_that_could_be_chance_is_not_a_finding():
    """The candidate beats the best draw, but the paired test says the
    difference is not distinguishable from chance."""
    base, cand = summarize(arm(N, 0)), summarize(arm(int(N * 0.3), N - int(N * 0.3)))
    pooled = summarize(arm(int(N * 0.5), N - int(N * 0.5)))
    spread = {"draws": 3, "rates": [0.35, 0.5, 0.65], "mean": 0.5, "best": 0.35,
              "worst": 0.65, "std": 0.12, "range": 0.3}
    significance = {"available": True, "draws": 3, "worst_p_value": 0.219,
                    "significant_against_all": False, "alpha": 0.05}
    result = attribution(base, pooled, cand, spread, significance)
    assert result["attributable_vs_best_control"] > 0      # it did beat the best draw
    assert "not statistically significant" in result["verdict"]
    assert "worst p = 0.219" in result["verdict"]


def test_significance_alone_does_not_excuse_a_tiny_effect():
    """A significant but occlusion-dominated result still reads as occlusion:
    the two checks gate different things and both must be reported."""
    base, cand = summarize(arm(N, 0)), summarize(arm(1, N - 1))
    pooled = summarize(arm(3, N - 3))
    spread = {"draws": 2, "rates": [0.1, 0.2], "mean": 0.15, "best": 0.1,
              "worst": 0.2, "std": 0.05, "range": 0.1}
    significance = {"available": True, "draws": 2, "worst_p_value": 0.001,
                    "significant_against_all": True, "alpha": 0.05}
    result = attribution(base, pooled, cand, spread, significance)
    assert "mostly occlusion" in result["verdict"]
    assert result["significance"]["significant_against_all"] is True


def test_a_real_effect_survives_the_variance_checks():
    base, cand = summarize(arm(N, 0)), summarize(arm(0, N))
    pooled = summarize(arm(int(N * 0.6), N - int(N * 0.6)))
    spread = {"draws": 3, "rates": [0.55, 0.6, 0.65], "mean": 0.6, "best": 0.55,
              "worst": 0.65, "std": 0.04, "range": 0.1}
    significance = {"available": True, "draws": 3, "worst_p_value": 0.002,
                    "significant_against_all": True, "alpha": 0.05}
    result = attribution(base, pooled, cand, spread, significance)
    assert result["attributable_vs_best_control"] > 0
    assert result["attributable_drop"] > spread["range"]
    assert "majority" in result["verdict"]


def test_a_single_draw_skips_the_variance_verdicts():
    """With one draw there is no spread to reason about; the verdict must
    fall back to the occlusion split rather than invent a variance claim."""
    base, cand = summarize(arm(N, 0)), summarize(arm(0, N))
    pooled = summarize(arm(int(N * 0.6), N - int(N * 0.6)))
    spread = {"draws": 1, "rates": [0.6], "mean": 0.6, "best": 0.6,
              "worst": 0.6, "std": 0.0, "range": 0.0}
    result = attribution(base, pooled, cand, spread)
    assert "variance" not in result["verdict"]
    assert "majority" in result["verdict"]


def test_a_detector_that_never_saw_the_target_yields_no_verdict():
    """The usual shape of a meaningless transfer result: the model barely
    detected the subject without any pattern, so there is nothing to suppress
    and the arithmetic below is noise around zero."""
    base = summarize(arm(1, N - 1))          # 5% baseline
    control = summarize(arm(1, N - 1))
    cand = summarize(arm(0, N))
    result = attribution(base, control, cand)
    assert result["has_headroom"] is False
    assert "no headroom" in result["verdict"]
    assert "5.0% of baseline samples" in result["verdict"]


def test_no_headroom_outranks_the_sample_count():
    """More samples cannot rescue an experiment with no baseline detections,
    so that verdict must win over 'inconclusive'."""
    base, control, cand = summarize(arm(0, 4)), summarize(arm(0, 4)), summarize(arm(0, 4))
    result = attribution(base, control, cand)
    assert "no headroom" in result["verdict"]
    assert "inconclusive" not in result["verdict"]


def test_a_healthy_baseline_has_headroom():
    result = attribution(summarize(arm(N, 0)), summarize(arm(0, N)), summarize(arm(0, N)))
    assert result["has_headroom"] is True
    assert result["baseline_rate"] == 1.0
    assert "no headroom" not in result["verdict"]
