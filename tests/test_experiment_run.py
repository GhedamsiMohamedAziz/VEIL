"""End-to-end: upload -> experiment -> run -> results -> report.

Uses the built-in colour-blob detector so the whole chain runs offline and
in seconds. The torchvision path is exercised by test_detectors_torch.py.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

API = "/api/v1"


@pytest.fixture
def scene_png() -> bytes:
    """A red patch on grey - the detector sees exactly one blob, and there is
    room around it for a pattern to be pasted."""
    array = np.full((128, 128, 3), 120, dtype=np.uint8)
    array[32:96, 32:96] = (230, 20, 20)
    buffer = io.BytesIO()
    Image.fromarray(array, mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def experiment(client, auth, scene_png):
    project_id = client.post(f"{API}/projects", json={"name": "E2E"}, headers=auth).json()["id"]
    artifact = client.post(f"{API}/projects/{project_id}/artifacts",
                           files={"file": ("scene.png", scene_png, "image/png")},
                           headers=auth).json()
    dataset = client.post(f"{API}/datasets", json={
        "project_id": project_id, "name": "lab scenes", "source": "synthetic, generated in-repo",
        "license": "CC0", "artifact_ids": [artifact["id"]],
        "annotations": {artifact["id"]: [{"label": "blob", "box": [32, 32, 96, 96]}]},
    }, headers=auth).json()
    return client.post(f"{API}/experiments", json={
        "project_id": project_id, "name": "Blob robustness #1",
        "description": "controlled lab sweep", "detector_id": "colorblob-v1",
        "dataset_id": dataset["id"],
        "configuration": {
            "target_label": "blob", "threshold": 0.3, "image_size": 128,
            "transforms": {"rotation_deg": [-20.0, 0.0, 20.0], "scale": [1.0],
                           "translate": [0.0], "perspective": [0.0],
                           "brightness": [0.7, 1.0], "contrast": [1.0],
                           "blur_sigma": [0.0], "noise_std": [0.0],
                           "deformation": [0.0], "mode": "grid", "max_samples": 6},
            "optimization": {"pattern_size": 32, "iterations": 3, "batch_transforms": 1},
            "control_draws": 2,
        },
    }, headers=auth).json()


def run_to_completion(client, auth, experiment_id, seed=7):
    from veil import jobs

    response = client.post(f"{API}/experiments/{experiment_id}/run",
                           json={"seed": seed}, headers=auth)
    assert response.status_code == 202
    run_id = response.json()["id"]
    jobs.wait(run_id, timeout=300)
    return client.get(f"{API}/runs/{run_id}", headers=auth).json()


def test_full_experiment_produces_real_measurements(client, auth, experiment):
    run = run_to_completion(client, auth, experiment["id"])
    assert run["status"] == "completed", run["error"]
    assert run["seed"] == 7
    assert run["detector_version"] and run["code_version"]
    assert any("baseline sweep" in line for line in run["log"])

    results = client.get(f"{API}/experiments/{experiment['id']}/results", headers=auth).json()
    assert len(results) == 1
    evaluation = results[0]

    # 1 image x 6 transformation points, measured twice (baseline + candidate).
    assert evaluation["sample_count"] == 6
    assert evaluation["baseline_metrics"]["samples"] == 6
    assert 0.0 <= evaluation["baseline_metrics"]["detection_rate"] <= 1.0
    assert 0.0 <= evaluation["candidate_metrics"]["detection_rate"] <= 1.0

    score = evaluation["metrics"]["veil_score"]
    assert score["angle_robustness"] is not None      # rotations were swept
    assert score["lighting_robustness"] is not None   # brightness was swept
    assert score["camera_noise_robustness"] is None   # blur/noise were not
    assert score["physical_robustness"] is None       # no physical tests yet
    assert score["sample_counts"]["digital"] == 6

    ground_truth = evaluation["metrics"]["ground_truth"]["baseline"]
    assert ground_truth["available"] is True
    assert ground_truth["annotated_samples"] == 6


def test_control_arm_runs_and_is_attributed(client, auth, experiment):
    """Three arms are measured, and the drop is split between them."""
    run = run_to_completion(client, auth, experiment["id"])
    assert run["status"] == "completed", run["error"]
    assert sum("control draw" in line for line in run["log"]) == 2
    assert any("attribution:" in line for line in run["log"])

    evaluation = client.get(f"{API}/experiments/{experiment['id']}/results", headers=auth).json()[0]
    # Two control draws x 6 transformation points, pooled.
    assert evaluation["control_metrics"]["samples"] == 12
    spread = evaluation["control_metrics"]["spread"]
    assert spread["draws"] == 2
    assert len(spread["rates"]) == 2
    assert spread["best"] <= spread["mean"] <= spread["worst"]

    attribution = evaluation["metrics"]["comparison"]["attribution"]
    assert attribution["available"] is True
    # 6 samples is below the verdict threshold - it must say so, not guess.
    assert "inconclusive" in attribution["verdict"]
    assert attribution["samples_per_arm"] == 6
    assert (
        abs(attribution["occlusion_drop"] + attribution["attributable_drop"]
            - attribution["total_drop"]) < 1e-9
    )


def test_control_can_be_disabled_and_the_report_says_so(client, auth, experiment):
    """Turning the control off must degrade the claim, not hide the gap."""
    from veil import jobs

    config = {**experiment["configuration"], "control_draws": 0}
    response = client.post(f"{API}/experiments/{experiment['id']}/run",
                           json={"seed": 3, "configuration": config}, headers=auth)
    assert response.status_code == 202
    jobs.wait(response.json()["id"], timeout=300)

    evaluation = client.get(f"{API}/experiments/{experiment['id']}/results", headers=auth).json()[0]
    assert evaluation["control_metrics"] == {}
    attribution = evaluation["metrics"]["comparison"]["attribution"]
    assert attribution["available"] is False
    assert "occlusion" in attribution["reason"]

    report = client.post(f"{API}/experiments/{experiment['id']}/report", headers=auth).json()
    assert report["payload"]["results"]["control"] is None
    assert report["payload"]["results"]["attribution"]["available"] is False
    assert any("attributable to the pattern itself" in line
               for line in report["payload"]["limitations"])


def test_run_is_reproducible_for_the_same_seed(client, auth, experiment):
    first = run_to_completion(client, auth, experiment["id"], seed=11)
    second = run_to_completion(client, auth, experiment["id"], seed=11)
    assert first["status"] == second["status"] == "completed"

    results = client.get(f"{API}/experiments/{experiment['id']}/results", headers=auth).json()
    assert len(results) == 2
    assert results[0]["baseline_metrics"] == results[1]["baseline_metrics"]
    assert results[0]["candidate_metrics"] == results[1]["candidate_metrics"]

    patterns = client.get(f"{API}/experiments/{experiment['id']}/patterns", headers=auth).json()
    assert len(patterns) == 2
    a = client.get(f"{API}/patterns/{patterns[0]['id']}/image", headers=auth)
    b = client.get(f"{API}/patterns/{patterns[1]['id']}/image", headers=auth)
    assert a.status_code == 200 and a.headers["content-type"] == "image/png"
    assert a.content == b.content  # identical seed -> identical artifact


def test_concurrent_runs_are_refused(client, auth, experiment):
    from veil import jobs

    first = client.post(f"{API}/experiments/{experiment['id']}/run", json={"seed": 1}, headers=auth)
    second = client.post(f"{API}/experiments/{experiment['id']}/run", json={"seed": 2}, headers=auth)
    assert first.status_code == 202
    assert second.status_code == 409
    jobs.wait(first.json()["id"], timeout=300)


def test_report_carries_conditions_and_limitations(client, auth, experiment):
    run_to_completion(client, auth, experiment["id"])
    report = client.post(f"{API}/experiments/{experiment['id']}/report", headers=auth).json()
    payload = report["payload"]

    assert payload["reproducibility"]["seed"] == 7
    assert payload["reproducibility"]["detector_version"]
    assert payload["detector"]["id"] == "colorblob-v1"
    assert payload["dataset"]["license"] == "CC0"
    assert payload["transformations"]["rotation_deg"] == [-20.0, 0.0, 20.0]
    assert payload["results"]["baseline"]["samples"] == 6
    assert payload["results"]["control"]["samples"] == 12  # 2 draws x 6 points
    assert payload["results"]["control"]["spread"]["draws"] == 2
    assert payload["results"]["attribution"]["available"] is True
    assert len(payload["limitations"]) >= 3
    assert any("not generalize" in line for line in payload["limitations"])

    pdf = client.get(f"{API}/reports/{report["id"]}/pdf", headers=auth)
    assert pdf.status_code == 200
    assert pdf.content.startswith(b"%PDF-")
    # A valid header proves nothing about the contents; tests/test_report_pdf.py
    # asserts on the text actually drawn. Here just check it is not an empty doc.
    assert len(pdf.content) > 2000


def test_physical_test_feeds_the_physical_robustness_score(client, auth, experiment):
    run_to_completion(client, auth, experiment["id"])
    pattern = client.get(f"{API}/experiments/{experiment['id']}/patterns", headers=auth).json()[0]

    created = client.post(f"{API}/physical-tests", json={
        "experiment_id": experiment["id"], "pattern_id": pattern["id"], "arm": "candidate",
        "camera": "Logitech C920", "resolution": "1280x720", "fps": 30.0,
        "distance_m": 3.0, "angle_deg": 15.0, "lighting": "office fluorescent",
        "environment": "lab", "frame_count": 60,
        "result": {"detected": False, "max_score": 0.0, "detection_count": 0},
    }, headers=auth)
    assert created.status_code == 201

    report = client.post(f"{API}/experiments/{experiment['id']}/report", headers=auth).json()
    tests = report["payload"]["physical_tests"]
    assert len(tests) == 1
    assert tests[0]["distance_m"] == 3.0 and tests[0]["lighting"] == "office fluorescent"

    # A second run now has physical samples to score against.
    run_to_completion(client, auth, experiment["id"])


def test_transfer_detector_is_measured_with_the_same_discipline(client, auth, scene_png):
    """A pattern optimized against one detector, measured against another."""
    from veil import jobs

    project_id = client.post(f"{API}/projects", json={"name": "Transfer"}, headers=auth).json()["id"]
    artifact = client.post(f"{API}/projects/{project_id}/artifacts",
                           files={"file": ("s.png", scene_png, "image/png")},
                           headers=auth).json()
    dataset = client.post(f"{API}/datasets", json={
        "project_id": project_id, "name": "d", "source": "synthetic, generated in tests",
        "artifact_ids": [artifact["id"]]}, headers=auth).json()
    experiment = client.post(f"{API}/experiments", json={
        "project_id": project_id, "name": "Transfer sweep", "detector_id": "colorblob-v1",
        "dataset_id": dataset["id"],
        "configuration": {
            "target_label": "blob", "threshold": 0.3, "image_size": 128,
            "control_draws": 1,
            # colorblob is the only detector producing 'blob', so the transfer
            # list names it explicitly and the runner must skip it as "that is
            # the primary result, not a transfer".
            "transfer_detectors": ["colorblob-v1"],
            "transforms": {"rotation_deg": [0.0], "scale": [1.0], "translate": [0.0],
                           "perspective": [0.0], "brightness": [1.0], "contrast": [1.0],
                           "blur_sigma": [0.0], "noise_std": [0.0], "deformation": [0.0],
                           "mode": "grid", "max_samples": 2},
            "optimization": {"pattern_size": 32, "iterations": 2, "batch_transforms": 1},
        }}, headers=auth).json()

    response = client.post(f"{API}/experiments/{experiment['id']}/run",
                           json={"seed": 5}, headers=auth)
    jobs.wait(response.json()["id"], timeout=300)
    run = client.get(f"{API}/runs/{response.json()['id']}", headers=auth).json()
    assert run["status"] == "completed", run["error"]

    evaluation = client.get(f"{API}/experiments/{experiment['id']}/results", headers=auth).json()[0]
    # The primary detector is not double-counted as its own transfer result.
    assert evaluation["metrics"]["transfer"] == {}


def test_transfer_detector_must_be_able_to_see_the_target(client, auth, scene_png):
    project_id = client.post(f"{API}/projects", json={"name": "T"}, headers=auth).json()["id"]
    artifact = client.post(f"{API}/projects/{project_id}/artifacts",
                           files={"file": ("s.png", scene_png, "image/png")},
                           headers=auth).json()
    dataset = client.post(f"{API}/datasets", json={
        "project_id": project_id, "name": "d", "source": "synthetic, generated in tests",
        "artifact_ids": [artifact["id"]]}, headers=auth).json()
    body = {"project_id": project_id, "name": "e", "detector_id": "colorblob-v1",
            "dataset_id": dataset["id"],
            "configuration": {"target_label": "blob",
                              "transfer_detectors": ["fasterrcnn-mobilenet-320"]}}
    response = client.post(f"{API}/experiments", json=body, headers=auth)
    assert response.status_code == 400
    assert "cannot detect" in response.json()["detail"]

    body["configuration"]["transfer_detectors"] = ["no-such-model"]
    response = client.post(f"{API}/experiments", json=body, headers=auth)
    assert response.status_code == 400
    assert "unknown transfer detector" in response.json()["detail"]


def test_garment_provenance_links_back_to_the_experiment(client, auth, experiment):
    run_to_completion(client, auth, experiment["id"])
    pattern = client.get(f"{API}/experiments/{experiment['id']}/patterns", headers=auth).json()[0]
    garment = client.post(f"{API}/garments", json={
        "pattern_id": pattern["id"], "sku": "VEIL-TS-0001", "batch_id": "B-2026-01",
        "product_type": "tshirt", "material": "100% polyester, 180gsm",
        "print_method": "dye sublimation",
        "test_conditions": {"distance_m": 3.0, "lighting": "office fluorescent"},
    }, headers=auth)
    assert garment.status_code == 201
    assert client.post(f"{API}/garments", json={
        "pattern_id": pattern["id"], "sku": "VEIL-TS-0001", "batch_id": "B",
        "product_type": "tshirt"}, headers=auth).status_code == 409

    provenance = client.get(f"{API}/garments/VEIL-TS-0001/provenance", headers=auth).json()
    assert provenance["experiment"]["id"] == experiment["id"]
    assert provenance["pattern"]["id"] == pattern["id"]
    assert "do not generalize" in provenance["limitations"]


def test_physical_test_response_carries_every_recorded_condition(client, auth, experiment):
    """A condition that is stored but not returned cannot be checked by
    whoever reads the result, so the response must carry all of them."""
    run_to_completion(client, auth, experiment["id"])
    pattern = client.get(f"{API}/experiments/{experiment['id']}/patterns", headers=auth).json()[0]
    created = client.post(f"{API}/physical-tests", json={
        "experiment_id": experiment["id"], "pattern_id": pattern["id"], "arm": "candidate",
        "camera": "Logitech C920", "resolution": "1920x1080", "fps": 30.0,
        "distance_m": 3.0, "angle_deg": 15.0, "lighting": "office fluorescent",
        "environment": "lab", "frame_count": 60, "notes": "first print",
        "result": {"detected": True, "detection_rate": 1.0, "max_score": 0.99},
    }, headers=auth).json()

    for field, expected in (
        ("camera", "Logitech C920"), ("resolution", "1920x1080"), ("fps", 30.0),
        ("distance_m", 3.0), ("angle_deg", 15.0), ("lighting", "office fluorescent"),
        ("environment", "lab"), ("frame_count", 60), ("notes", "first print"),
        ("pattern_id", pattern["id"]),
    ):
        assert created[field] == expected, field

    listed = client.get(f"{API}/physical-tests?experiment_id={experiment['id']}",
                        headers=auth).json()[0]
    assert listed["fps"] == 30.0 and listed["notes"] == "first print"


def test_physical_arm_is_required_and_constrained(client, auth, experiment):
    """A physical record without an arm is uninterpretable, so it is refused
    rather than stored and silently scored."""
    body = {"experiment_id": experiment["id"], "camera": "c", "distance_m": 3.0,
            "angle_deg": 0.0, "lighting": "l", "environment": "e", "frame_count": 10,
            "result": {"detected": True, "detection_rate": 1.0}}
    assert client.post(f"{API}/physical-tests", json=body, headers=auth).status_code == 422
    assert client.post(f"{API}/physical-tests", json={**body, "arm": "whatever"},
                       headers=auth).status_code == 422
    ok = client.post(f"{API}/physical-tests", json={**body, "arm": "baseline"}, headers=auth)
    assert ok.status_code == 201
    assert ok.json()["arm"] == "baseline"


def test_only_candidate_arm_records_score_physical_robustness(client, auth, experiment):
    """A baseline measurement must not be counted as the pattern's result."""
    run_to_completion(client, auth, experiment["id"])
    pattern = client.get(f"{API}/experiments/{experiment['id']}/patterns", headers=auth).json()[0]
    common = {"experiment_id": experiment["id"], "pattern_id": pattern["id"],
              "camera": "c", "distance_m": 3.0, "angle_deg": 0.0, "lighting": "l",
              "environment": "e", "frame_count": 10}

    # Two baseline measurements where the detector always saw the subject.
    for _ in range(2):
        client.post(f"{API}/physical-tests", json={
            **common, "arm": "baseline",
            "result": {"detected": True, "detection_rate": 1.0, "max_score": 0.99},
        }, headers=auth)

    run_to_completion(client, auth, experiment["id"], seed=9)
    evaluation = client.get(f"{API}/experiments/{experiment['id']}/results", headers=auth).json()[0]
    # Baselines are counted and reported, never scored.
    assert evaluation["metrics"]["physical"]["by_arm"]["baseline"]["samples"] == 2
    assert evaluation["metrics"]["physical"]["excluded"]["baseline"] == 2
    assert evaluation["metrics"]["veil_score"]["physical_robustness"] is None

    # A candidate measurement does score.
    client.post(f"{API}/physical-tests", json={
        **common, "arm": "candidate",
        "result": {"detected": False, "detection_rate": 0.0, "max_score": 0.0},
    }, headers=auth)
    run_to_completion(client, auth, experiment["id"], seed=10)
    evaluation = client.get(f"{API}/experiments/{experiment['id']}/results", headers=auth).json()[0]
    assert evaluation["metrics"]["veil_score"]["physical_robustness"] == 1.0
    assert evaluation["metrics"]["physical"]["by_arm"]["candidate"]["samples"] == 1


def test_pattern_is_evaluated_on_images_the_optimizer_never_saw(client, auth, experiment):
    """Four scenes with the blob in four different places: the run must split
    them, score only the held-out half, and place the print on each blob."""
    from veil.ml.runner import split_indices

    assert split_indices(1, 7) == ([0], [0], False)
    fit, held, disjoint = split_indices(5, 7)
    assert disjoint and not set(fit) & set(held) and sorted(fit + held) == list(range(5))
    assert split_indices(5, 7) == (fit, held, True)

    project_id = experiment["project_id"]
    ids, centres = [], {}
    for index, (top, left) in enumerate([(8, 8), (8, 56), (56, 8), (56, 56)]):
        array = np.full((128, 128, 3), 120, dtype=np.uint8)
        array[top:top + 64, left:left + 64] = (230, 20, 20)
        buffer = io.BytesIO()
        Image.fromarray(array, mode="RGB").save(buffer, format="PNG")
        artifact = client.post(f"{API}/projects/{project_id}/artifacts",
                               files={"file": (f"s{index}.png", buffer.getvalue(), "image/png")},
                               headers=auth).json()
        ids.append(artifact["id"])
        centres[artifact["id"]] = ((left + 32) / 128, (top + 32) / 128)
    dataset = client.post(f"{API}/datasets", json={
        "project_id": project_id, "name": "four scenes", "source": "synthetic, generated in-repo",
        "license": "CC0", "artifact_ids": ids}, headers=auth).json()
    multi = client.post(f"{API}/experiments", json={
        "project_id": project_id, "name": "held out", "detector_id": "colorblob-v1",
        "dataset_id": dataset["id"], "configuration": experiment["configuration"],
    }, headers=auth).json()

    run = run_to_completion(client, auth, multi["id"])
    assert run["status"] == "completed", run.get("error")
    results = client.get(f"{API}/experiments/{multi['id']}/results", headers=auth).json()[0]
    split = results["metrics"]["split"]
    assert split["held_out"] is True
    assert len(split["optimize"]) == 2 and len(split["evaluate"]) == 2
    assert not set(split["optimize"]) & set(split["evaluate"])

    records = results["metrics"]["records"]
    for arm in ("baseline", "candidate"):
        assert {r["image_id"] for r in records[arm]} == set(split["evaluate"])

    pattern = client.get(f"{API}/patterns/{results['pattern_id']}", headers=auth).json()
    placements = pattern["generation_parameters"]["placements"]
    assert set(placements) == set(split["evaluate"])
    for image_id, placement in placements.items():
        cx, cy = centres[image_id]
        assert abs(placement["cx"] - cx) < 0.05 and abs(placement["cy"] - cy) < 0.05

    report = client.post(f"{API}/experiments/{multi['id']}/report", headers=auth).json()
    assert not any("training score" in line for line in report["payload"]["limitations"])
    # ...while the single-image experiment must say so, first.
    run_to_completion(client, auth, experiment["id"])
    single = client.post(f"{API}/experiments/{experiment['id']}/report", headers=auth).json()
    assert "training score" in single["payload"]["limitations"][0]
