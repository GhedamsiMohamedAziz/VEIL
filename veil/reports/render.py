"""Render a report dict to PDF.

piggy: reportlab's low-level canvas with a tiny line-writer, not a template
engine. The report is a linear document; flowing text through a layout
framework would buy nothing.
"""

from __future__ import annotations

import io
from typing import Any

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

_LEFT = 20 * mm
_TOP = 277 * mm
_BOTTOM = 20 * mm


def _pct(value: Any) -> str:
    if value is None:
        return "not measured"
    if isinstance(value, (int, float)):
        return f"{value * 100:.1f}%"
    return str(value)


class _Writer:
    def __init__(self, pdf: canvas.Canvas):
        self.pdf = pdf
        self.y = _TOP

    def _space(self, height: float) -> None:
        if self.y - height < _BOTTOM:
            self.pdf.showPage()
            self.y = _TOP

    def line(self, text: str, size: int = 9, font: str = "Helvetica", gap: float = 4.6 * mm) -> None:
        self._space(gap)
        self.pdf.setFont(font, size)
        # Hard-wrap rather than clip: a truncated limitation is a lie.
        width = int(175 / (size * 0.5) * 1.6)
        chunks = [text[i : i + width] for i in range(0, len(text), width)] or [""]
        for chunk in chunks:
            self._space(gap)
            self.pdf.setFont(font, size)  # showPage() in _space resets the font
            self.pdf.drawString(_LEFT, self.y, chunk)
            self.y -= gap

    def heading(self, text: str) -> None:
        self.y -= 3 * mm
        self.line(text, size=12, font="Helvetica-Bold")

    def kv(self, key: str, value: Any) -> None:
        self.line(f"{key}: {value}")


def to_pdf(report: dict[str, Any]) -> bytes:
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    w = _Writer(pdf)

    w.line(report["title"], size=18, font="Helvetica-Bold", gap=9 * mm)
    experiment = report["experiment"]
    w.line(f"{experiment['name']} - status {experiment['status']}", size=10)
    if experiment.get("description"):
        w.line(experiment["description"])

    w.heading("Reproducibility")
    for key, value in report["reproducibility"].items():
        if key != "configuration":
            w.kv(key, value)

    w.heading("Detector")
    detector = report["detector"]
    for key in ("id", "name", "version", "license", "differentiable"):
        w.kv(key, detector.get(key))

    if report.get("dataset"):
        w.heading("Dataset")
        for key, value in report["dataset"].items():
            w.kv(key, value)

    w.heading("Transformations")
    for key, value in (report.get("transformations") or {}).items():
        w.kv(key, value)

    results = report["results"]
    w.heading("Results")
    w.kv("samples per condition set", results["sample_count"])
    baseline, candidate = results["baseline"], results["candidate"]
    split = results.get("split") or {}
    w.kv("images evaluated", f"{candidate.get('images') or 'not recorded'}"
         + (" (held out from optimization)" if split.get("held_out") else " (NOT held out)"))
    control = results.get("control")
    spread = (control or {}).get("spread") or {}
    w.kv("baseline detection rate", _pct(baseline.get("detection_rate")))
    if control:
        label = (
            f"control detection rate (mean of {spread['draws']} unoptimized patterns)"
            if spread.get("draws", 0) > 1
            else "control detection rate (unoptimized pattern)"
        )
        w.kv(label, _pct(control.get("detection_rate")))
        if spread.get("draws", 0) > 1:
            w.kv("control draws (best to worst)",
                 ", ".join(_pct(r) for r in sorted(spread.get("rates", []))))
    w.kv("candidate detection rate", _pct(candidate.get("detection_rate")))
    w.kv("delta vs baseline", _pct(results["delta"].get("detection_rate")))
    if control:
        w.kv("delta vs control", _pct(results.get("delta_vs_control", {}).get("detection_rate")))
    w.kv("baseline mean confidence", round(baseline.get("mean_confidence") or 0.0, 4))
    w.kv("candidate mean confidence", round(candidate.get("mean_confidence") or 0.0, 4))

    w.heading("Attribution")
    attribution = results.get("attribution") or {}
    if attribution.get("available"):
        w.kv("total drop (baseline to candidate)", _pct(attribution["total_drop"]))
        w.kv("of which occlusion (baseline to control)", _pct(attribution["occlusion_drop"]))
        w.kv("attributable to the pattern (control to candidate)", _pct(attribution["attributable_drop"]))
        if attribution.get("attributable_vs_best_control") is not None:
            w.kv("versus the best control draw (stricter test)",
                 _pct(attribution["attributable_vs_best_control"]))
        control_spread = attribution.get("control_spread") or {}
        if control_spread.get("draws", 0) > 1:
            w.kv("spread across control draws", _pct(control_spread["range"]))
        w.kv("samples per arm", attribution["samples_per_arm"])
        significance = attribution.get("significance") or {}
        if significance.get("available"):
            w.kv("paired test", significance["test"])
            w.kv("worst p-value across control draws", f"{significance['worst_p_value']:.4f}")
            w.kv("significant against every control draw",
                 "yes" if significance["significant_against_all"] else "no")
        w.line(f"Verdict: {attribution['verdict']}", size=9, font="Helvetica-Bold")
    else:
        w.line(f"Not available - {attribution.get('reason', 'no control arm')}.", size=9)
        w.line("Without a control arm the measured drop cannot be separated from "
               "the patch covering part of the subject.", size=8)

    w.heading("VEIL score (experimental)")
    score = results.get("veil_score", {})
    for key in ("digital_robustness", "angle_robustness", "lighting_robustness",
                "scale_robustness", "camera_noise_robustness", "physical_robustness"):
        w.kv(key.replace("_", " "), _pct(score.get(key)))
    w.line(score.get("definition", ""), size=8)

    ground_truth = results.get("ground_truth", {}).get("candidate", {})
    if ground_truth.get("available"):
        w.heading("Ground-truth metrics")
        for key in ("precision", "recall", "f1", "mean_iou", "annotated_samples"):
            w.kv(key, round(ground_truth[key], 4) if isinstance(ground_truth[key], float) else ground_truth[key])

    transfer = results.get("transfer") or {}
    if transfer:
        w.heading("Transfer to detectors not optimized against")
        for detector_id, block in transfer.items():
            if not block.get("available"):
                w.line(f"{detector_id}: not measured - {block.get('reason', 'unavailable')}")
                continue
            control = block.get("control") or {}
            w.line(detector_id, size=9, font="Helvetica-Bold")
            w.kv("  baseline detection rate", _pct(block["baseline"].get("detection_rate")))
            if control:
                w.kv("  control detection rate", _pct(control.get("detection_rate")))
            w.kv("  candidate detection rate", _pct(block["candidate"].get("detection_rate")))
            verdict = block.get("attribution", {}).get("verdict") \
                or block.get("attribution", {}).get("reason", "no attribution")
            w.line(f"  verdict: {verdict}", size=8)

    w.heading("Physical tests")
    if report["physical_tests"]:
        for test in report["physical_tests"]:
            # The arm first: a rate means nothing until you know what was worn.
            w.line(
                f"[{test.get('arm', 'unspecified')}] {test['camera']} @ {test['resolution']}, {test['distance_m']}m, "
                f"{test['angle_deg']} deg, {test['lighting']}, {test['environment']}: "
                f"{test['result']}"
            )
    else:
        w.line("No physical tests recorded for this experiment.")

    w.heading("Limitations")
    for item in report["limitations"]:
        w.line(f"- {item}", size=8, gap=3.8 * mm)

    pdf.showPage()
    pdf.save()
    return buffer.getvalue()
