"""The PDF must contain the numbers, not merely be a valid PDF.

Asserting `startswith(b"%PDF-")` passes on an empty document. These tests
read back the text reportlab actually drew, by disabling stream compression
for the duration of the test so the content stream is greppable - no extra
PDF-parsing dependency.
"""

from __future__ import annotations

import re

import pytest
from reportlab import rl_config

from veil.reports.render import to_pdf


@pytest.fixture
def drawn_text(monkeypatch):
    """Return a function that renders a report and returns its drawn strings."""
    monkeypatch.setattr(rl_config, "pageCompression", 0)

    def render(report: dict) -> str:
        content = to_pdf(report).decode("latin-1")
        drawn = re.findall(r"Tm \((.*?)\) Tj", content)
        # PDF escapes parentheses and backslashes inside string literals.
        unescape = lambda s: s.replace(r"\(", "(").replace(r"\)", ")").replace(r"\\", "\\")
        return "\n".join(unescape(d) for d in drawn)

    return render


def report(**overrides) -> dict:
    base = {
        "title": "VEIL Experiment Report #007",
        "experiment": {"name": "Three-arm sweep", "status": "completed", "description": ""},
        "reproducibility": {"run_id": "abc", "seed": 42, "code_version": "deadbee",
                            "detector_version": "torchvision-0.29.0", "dataset_version": 1,
                            "configuration": {}},
        "detector": {"id": "fasterrcnn-mobilenet-320", "name": "frcnn", "version": "0.29.0",
                     "license": "BSD-3-Clause", "differentiable": True},
        "dataset": {"name": "lab", "license": "CC0"},
        "transformations": {"rotation_deg": [-15.0, 0.0, 15.0]},
        "results": {
            "sample_count": 36,
            "baseline": {"detection_rate": 0.5, "mean_confidence": 0.2527},
            "control": {"detection_rate": 0.111, "mean_confidence": 0.046},
            "candidate": {"detection_rate": 0.083, "mean_confidence": 0.0305},
            "delta": {"detection_rate": -0.417},
            "delta_vs_control": {"detection_rate": -0.028},
            "attribution": {
                "available": True, "total_drop": 0.417, "occlusion_drop": 0.389,
                "attributable_drop": 0.028, "samples_per_arm": 36,
                "verdict": "mostly occlusion: under a quarter of the drop is attributable to the pattern",
            },
            "veil_score": {"digital_robustness": 0.917, "physical_robustness": None,
                           "definition": "robustness = 1 - detection_rate"},
            "ground_truth": {"candidate": {"available": False}},
        },
        "physical_tests": [],
        "limitations": ["Results may not generalize to other models, cameras, environments."],
    }
    base.update(overrides)
    return base


def test_pdf_contains_all_three_arms(drawn_text):
    text = drawn_text(report())
    assert "baseline detection rate: 50.0%" in text
    assert "control detection rate (unoptimized pattern): 11.1%" in text
    assert "candidate detection rate: 8.3%" in text


def test_pdf_states_the_attribution_split_and_verdict(drawn_text):
    text = drawn_text(report())
    assert "Attribution" in text
    assert "of which occlusion (baseline to control): 38.9%" in text
    assert "attributable to the pattern (control to candidate): 2.8%" in text
    assert "mostly occlusion" in text
    # The caveat must come before the headline score, not after it.
    assert text.index("Attribution") < text.index("VEIL score")


def test_pdf_says_not_measured_rather_than_zero(drawn_text):
    text = drawn_text(report())
    assert "physical robustness: not measured" in text
    assert "physical robustness: 0.0%" not in text


def test_pdf_without_a_control_arm_says_so(drawn_text):
    payload = report()
    payload["results"]["control"] = None
    payload["results"]["attribution"] = {"available": False, "reason": "no control arm was run"}
    text = drawn_text(payload)
    assert "control detection rate" not in text
    assert "no control arm was run" in text
    assert "cannot be separated" in text


def test_limitations_are_wrapped_not_truncated(drawn_text):
    """A clipped limitation is a lie, so long text must wrap and survive."""
    sentence = ("This report describes experimental results obtained under the specified "
                "test conditions and may not generalize to other deployment systems.")
    text = drawn_text(report(limitations=[sentence]))
    rebuilt = text.replace("\n", "")
    assert sentence in rebuilt, rebuilt[-200:]


def test_pdf_reports_the_control_spread_and_stricter_test(drawn_text):
    payload = report()
    payload["results"]["control"]["spread"] = {
        "draws": 3, "rates": [0.08, 0.111, 0.2], "mean": 0.13,
        "best": 0.08, "worst": 0.2, "std": 0.05, "range": 0.12,
    }
    payload["results"]["attribution"]["attributable_vs_best_control"] = -0.003
    payload["results"]["attribution"]["control_spread"] = payload["results"]["control"]["spread"]
    payload["results"]["attribution"]["verdict"] = (
        "within control variance: the candidate did not beat the best of 3 unoptimized patterns"
    )
    text = drawn_text(payload)
    assert "mean of 3 unoptimized patterns" in text
    assert "control draws (best to worst): 8.0%, 11.1%, 20.0%" in text
    assert "versus the best control draw (stricter test): -0.3%" in text
    assert "spread across control draws: 12.0%" in text
    assert "within control variance" in text


def test_single_draw_does_not_claim_a_spread(drawn_text):
    payload = report()
    payload["results"]["control"]["spread"] = {"draws": 1, "rates": [0.111]}
    text = drawn_text(payload)
    assert "control detection rate (unoptimized pattern)" in text
    assert "control draws (best to worst)" not in text
    assert "spread across control draws" not in text


def test_pdf_reports_the_paired_test(drawn_text):
    payload = report()
    payload["results"]["attribution"]["significance"] = {
        "available": True, "draws": 3, "worst_p_value": 0.2188,
        "significant_against_all": False, "alpha": 0.05,
        "test": "exact McNemar, paired on (image, transformation point)",
    }
    text = drawn_text(payload)
    assert "paired test: exact McNemar" in text
    assert "worst p-value across control draws: 0.2188" in text
    assert "significant against every control draw: no" in text


def test_pdf_omits_the_paired_test_when_it_was_not_run(drawn_text):
    text = drawn_text(report())
    assert "worst p-value" not in text
    assert "paired test" not in text


def test_pdf_reports_transfer_results(drawn_text):
    payload = report()
    payload["results"]["transfer"] = {
        "retinanet-resnet50": {
            "available": True, "detector_version": "torchvision-0.29.0",
            "baseline": {"detection_rate": 0.72}, "control": {"detection_rate": 0.61},
            "candidate": {"detection_rate": 0.58},
            "attribution": {"verdict": "not statistically significant: could be chance"},
        },
        "colorblob-v1": {"available": False, "reason": "detector cannot produce the label 'person'"},
    }
    text = drawn_text(payload)
    assert "Transfer to detectors not optimized against" in text
    assert "retinanet-resnet50" in text
    assert "baseline detection rate: 72.0%" in text
    assert "candidate detection rate: 58.0%" in text
    assert "verdict: not statistically significant" in text
    # A detector that could not be measured says so, rather than being dropped.
    assert "colorblob-v1: not measured" in text


def test_pdf_omits_the_transfer_section_when_none_was_run(drawn_text):
    assert "Transfer to detectors" not in drawn_text(report())


def test_physical_tests_are_printed_with_their_arm(drawn_text):
    test = {"arm": "baseline", "camera": "C920", "resolution": "1280x720", "distance_m": 3.0,
            "angle_deg": 0.0, "lighting": "office", "environment": "lab", "result": {"detected": True}}
    assert "[baseline] C920" in drawn_text(report(physical_tests=[test]))


def test_transfer_verdicts_are_printed_with_their_family_correction(drawn_text):
    block = {"available": True, "baseline": {}, "control": {}, "candidate": {},
             "attribution": {"verdict": "significant"},
             "family": {"detectors_tested": 3, "holm_p_value": 0.12,
                        "significant_after_correction": False}}
    text = drawn_text(report(results={**report()["results"], "transfer": {"retinanet": block}}))
    text = text.replace("\n", "")  # long lines are hard-wrapped across drawn strings
    assert "3 transfer detectors" in text and "NOT significant once corrected" in text
