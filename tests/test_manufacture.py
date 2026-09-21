"""Turning a pattern into something a machine can make."""

from __future__ import annotations

import pytest
import torch

from veil.ml.patterns.manufacture import (
    KNIT_YARNS,
    ProductionSpec,
    colour_usage,
    to_artwork,
    to_production,
)


def test_knit_grid_follows_gauge_and_size():
    spec = ProductionSpec(method="knit", stitches_per_cm=5.0, width_cm=30, height_cm=40)
    assert spec.grid == (200, 150)  # 40cm*5, 30cm*5
    fine = ProductionSpec(method="knit", stitches_per_cm=10.0, width_cm=30, height_cm=40)
    assert fine.grid == (400, 300)


def test_print_grid_follows_dpi():
    spec = ProductionSpec(method="print", dpi=300, width_cm=2.54, height_cm=2.54)
    assert spec.grid == (300, 300)  # one inch square


def test_knit_output_uses_only_loaded_yarns():
    spec = ProductionSpec(method="knit", stitches_per_cm=4, width_cm=10, height_cm=10, max_yarns=4)
    produced = to_production(torch.rand(3, 64, 64), spec)
    allowed = {tuple(round(c, 4) for c in colour) for colour in KNIT_YARNS[:4]}
    used = {tuple(round(float(v), 4) for v in produced[:, y, x])
            for y in range(produced.shape[1]) for x in range(produced.shape[2])}
    assert used <= allowed
    assert len(used) > 1  # a single-colour garment would be a plain shirt


def test_max_yarns_is_respected_because_each_costs_a_feeder():
    spec = ProductionSpec(method="knit", stitches_per_cm=4, width_cm=10, height_cm=10, max_yarns=2)
    usage = colour_usage(to_production(torch.rand(3, 64, 64), spec), spec)
    assert len(usage) <= 2


def test_colour_usage_reports_every_yarn_actually_used():
    spec = ProductionSpec(method="knit", stitches_per_cm=4, width_cm=10, height_cm=10)
    produced = to_production(torch.rand(3, 64, 64), spec)
    usage = colour_usage(produced, spec)
    assert abs(sum(u["fraction"] for u in usage) - 1.0) < 1e-6
    assert usage == sorted(usage, key=lambda u: -u["fraction"])
    assert all(u["hex"].startswith("#") and len(u["hex"]) == 7 for u in usage)
    # A yarn used for a fraction of a percent still needs a cone on the
    # machine, so it must appear rather than being rounded away.
    assert all(u["stitches"] >= 1 for u in usage)


def test_downsample_then_snap_never_invents_a_colour():
    """Snapping first and downsampling after would blend yarns into shades
    the machine cannot produce. This pins the order."""
    spec = ProductionSpec(method="knit", stitches_per_cm=2, width_cm=10, height_cm=10)
    # A hard checkerboard of two yarns: averaging would produce a mid tone.
    pattern = torch.zeros(3, 64, 64)
    pattern[:, ::2, ::2] = torch.tensor(KNIT_YARNS[1]).view(3, 1, 1)
    produced = to_production(pattern, spec)
    allowed = torch.tensor(spec.palette)
    flat = produced.reshape(3, -1).t()
    distance = torch.cdist(flat.unsqueeze(0), allowed.unsqueeze(0),
                           compute_mode="donot_use_mm_for_euclid_dist").squeeze(0).min(dim=1).values
    assert float(distance.max()) == 0.0  # exact: snapped values are the palette's own floats


def test_artwork_upscale_renders_stitches_as_blocks():
    spec = ProductionSpec(method="knit", stitches_per_cm=2, width_cm=5, height_cm=5)
    artwork = to_artwork(torch.rand(3, 32, 32), spec, upscale=4)
    rows, cols = spec.grid
    assert artwork.shape == (3, rows * 4, cols * 4)
    # Each stitch is a flat block, so a 4x4 cell has one colour.
    cell = artwork[:, 0:4, 0:4]
    assert torch.allclose(cell, cell[:, :1, :1].expand_as(cell))


@pytest.mark.parametrize("method", ["knit", "weave", "print"])
def test_every_method_produces_a_valid_image(method):
    spec = ProductionSpec(method=method, width_cm=5, height_cm=5, dpi=72)
    produced = to_production(torch.rand(3, 48, 48), spec)
    assert produced.shape[0] == 3
    assert float(produced.min()) >= 0.0 and float(produced.max()) <= 1.0


API = "/api/v1"


def test_manufacture_endpoint_reports_retained_effect(client, auth):
    """End to end: run an experiment, then ask what survives knitting."""
    import io

    import numpy as np
    from PIL import Image

    from veil import jobs

    array = np.full((128, 128, 3), 120, dtype=np.uint8)
    array[32:96, 32:96] = (230, 20, 20)
    buffer = io.BytesIO()
    Image.fromarray(array, mode="RGB").save(buffer, format="PNG")

    project_id = client.post(f"{API}/projects", json={"name": "M"}, headers=auth).json()["id"]
    artifact = client.post(f"{API}/projects/{project_id}/artifacts",
                           files={"file": ("s.png", buffer.getvalue(), "image/png")},
                           headers=auth).json()
    dataset = client.post(f"{API}/datasets", json={
        "project_id": project_id, "name": "d", "source": "synthetic, generated in tests",
        "artifact_ids": [artifact["id"]]}, headers=auth).json()
    experiment = client.post(f"{API}/experiments", json={
        "project_id": project_id, "name": "e", "detector_id": "colorblob-v1",
        "dataset_id": dataset["id"],
        "configuration": {
            "target_label": "blob", "threshold": 0.3, "image_size": 128, "control_draws": 1,
            "transforms": {"rotation_deg": [0.0], "scale": [1.0], "translate": [0.0],
                           "perspective": [0.0], "brightness": [1.0], "contrast": [1.0],
                           "blur_sigma": [0.0], "noise_std": [0.0], "deformation": [0.0],
                           "mode": "grid", "max_samples": 2},
            "optimization": {"pattern_size": 32, "iterations": 2, "batch_transforms": 1},
        }}, headers=auth).json()

    response = client.post(f"{API}/experiments/{experiment['id']}/run",
                           json={"seed": 4}, headers=auth)
    jobs.wait(response.json()["id"], timeout=300)

    pattern = client.get(f"{API}/experiments/{experiment['id']}/patterns", headers=auth).json()[0]
    result = client.post(f"{API}/patterns/{pattern['id']}/manufacture",
                         json={"method": "knit", "stitches_per_cm": 4,
                               "width_cm": 20, "height_cm": 25, "max_yarns": 4},
                         headers=auth)
    assert result.status_code == 200
    body = result.json()
    assert body["grid"] == {"rows": 100, "columns": 80, "units": "stitches"}
    assert 0 <= len(body["colour_usage"]) <= 4
    assert body["detection_rate"] is not None
    # The ratio is only as good as the two rates it is built from: they must be
    # the run's own, and the ratio must follow from them (None when the digital
    # pattern had no effect to retain).
    evaluation = client.get(f"{API}/experiments/{experiment['id']}/results", headers=auth).json()[0]
    digital = evaluation["candidate_metrics"]["detection_rate"]
    control = evaluation["control_metrics"]["detection_rate"]
    assert body["digital_detection_rate"] == digital and body["control_detection_rate"] == control
    if control - digital > 0:
        expected = (control - body["detection_rate"]) / (control - digital)
        assert body["effect_retained"] == pytest.approx(expected)
    else:
        assert body["effect_retained"] is None

    # The mill gets a real file, not a promise of one.
    artwork = client.get(f"{API}/artifacts/{body['artwork_artifact_id']}/download", headers=auth)
    assert artwork.status_code == 200
    assert artwork.content.startswith(b"\x89PNG")
    assert "attachment" in artwork.headers["content-disposition"]


def test_manufacture_is_scoped_to_the_owning_organization(client, auth, api_key):
    from veil.db import create_organization, session_scope

    with session_scope() as session:
        _, _, other = create_organization(session, "Rival", "r@example.com")
    response = client.post(f"{API}/patterns/deadbeef/manufacture",
                           json={"method": "knit"}, headers={"X-API-Key": other})
    assert response.status_code == 404


def test_effect_retained_is_the_share_of_the_digital_effect_that_survives(monkeypatch):
    from veil.ml import manufacture_eval
    from veil.ml.patterns import generator
    from veil.ml.simulation.renderer import Placement
    from veil.ml.simulation.transforms import TransformSpec

    def half_detected(*args, **kwargs):  # produced rate 0.5
        return [{"detected": hit, "max_score": float(hit), "detection_count": int(hit), "boxes": [],
                 "transform": {}, "image_id": "a", "image_index": 0, "transform_index": i}
                for i, hit in enumerate([True, False])]

    monkeypatch.setattr(manufacture_eval, "sweep", half_detected)

    def retained(control_rate, digital_rate):
        return manufacture_eval.evaluate_production(
            None, torch.zeros(1, 3, 32, 32), ["a"], generator.initialize(16, "gray", seed=0),
            Placement(0.5, 0.5, 0.3, 0.3), TransformSpec(), ProductionSpec(),
            target_label="x", threshold=0.5, seed=1,
            control_rate=control_rate, digital_rate=digital_rate)["effect_retained"]

    assert retained(0.8, 0.2) == pytest.approx(0.5)   # (0.8-0.5)/(0.8-0.2)
    assert retained(0.8, 0.5) == pytest.approx(1.0)   # manufacturing cost nothing
    assert retained(0.5, 0.5) is None                 # no digital effect: nothing to retain
    assert retained(None, 0.2) is None                # no control arm was run
