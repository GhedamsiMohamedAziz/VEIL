"""Dataset qualification: the cheap check before the expensive run."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

API = "/api/v1"


def png(red: bool, size: int = 96) -> bytes:
    """A red square the blob detector fires on, or a grey one it ignores."""
    array = np.full((size, size, 3), 120, dtype=np.uint8)
    if red:
        array[20:76, 20:76] = (230, 20, 20)
    buffer = io.BytesIO()
    Image.fromarray(array, mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def project(client, auth):
    return client.post(f"{API}/projects", json={"name": "Q"}, headers=auth).json()["id"]


def upload(client, auth, project_id, data, name):
    return client.post(f"{API}/projects/{project_id}/artifacts",
                       files={"file": (name, data, "image/png")}, headers=auth).json()["id"]


def make_dataset(client, auth, project_id, artifact_ids, **extra):
    body = {"project_id": project_id, "name": "d", "source": "generated in tests",
            "artifact_ids": artifact_ids, **extra}
    return client.post(f"{API}/datasets", json=body, headers=auth)


def test_a_dataset_the_detector_sees_is_usable(client, auth, project):
    good = upload(client, auth, project, png(True), "a.png")
    dataset = make_dataset(client, auth, project, [good]).json()
    result = client.post(f"{API}/datasets/{dataset['id']}/qualify",
                         json={"detector_id": "colorblob-v1", "target_label": "blob",
                               "threshold": 0.3, "image_size": 96},
                         headers=auth).json()
    assert result["dataset_usable"] is True
    assert "usable" in result["verdict"]
    assert result["per_image"][0]["usable"] is True
    assert result["usable_image_ids"] == [good]


def test_a_dataset_the_detector_cannot_see_is_rejected_before_the_run(client, auth, project):
    """This is the check that would have saved every synthetic-scene run."""
    blank = upload(client, auth, project, png(False), "b.png")
    dataset = make_dataset(client, auth, project, [blank]).json()
    result = client.post(f"{API}/datasets/{dataset['id']}/qualify",
                         json={"detector_id": "colorblob-v1", "target_label": "blob",
                               "threshold": 0.3, "image_size": 96},
                         headers=auth).json()
    assert result["dataset_usable"] is False
    assert "not usable" in result["verdict"]
    assert "nothing for a pattern to suppress" in result["verdict"]
    assert result["usable_image_ids"] == []


def test_a_mixed_dataset_names_the_images_to_drop(client, auth, project):
    good = upload(client, auth, project, png(True), "a.png")
    bad = upload(client, auth, project, png(False), "b.png")
    dataset = make_dataset(client, auth, project, [good, bad]).json()
    result = client.post(f"{API}/datasets/{dataset['id']}/qualify",
                         json={"detector_id": "colorblob-v1", "target_label": "blob",
                               "threshold": 0.3, "image_size": 96},
                         headers=auth).json()
    assert result["usable_image_ids"] == [good]
    assert result["unusable_image_ids"] == [bad]
    assert "re-qualify" in result["verdict"] or result["dataset_usable"] is False


def test_qualify_rejects_a_label_the_detector_cannot_produce(client, auth, project):
    good = upload(client, auth, project, png(True), "a.png")
    dataset = make_dataset(client, auth, project, [good]).json()
    response = client.post(f"{API}/datasets/{dataset['id']}/qualify",
                           json={"detector_id": "colorblob-v1", "target_label": "person"},
                           headers=auth)
    assert response.status_code == 400
    assert "cannot detect" in response.json()["detail"]


def test_dataset_provenance_is_required(client, auth, project):
    good = upload(client, auth, project, png(True), "a.png")
    response = client.post(f"{API}/datasets", json={
        "project_id": project, "name": "d", "artifact_ids": [good]}, headers=auth)
    assert response.status_code == 422  # source is required


def test_imagery_of_people_requires_a_recorded_basis(client, auth, project):
    good = upload(client, auth, project, png(True), "a.png")
    response = make_dataset(client, auth, project, [good], contains_people=True)
    assert response.status_code == 400
    assert "consent is required" in response.json()["detail"]

    ok = make_dataset(client, auth, project, [good], contains_people=True,
                      consent="Staff volunteers, written consent on file, ref C-2026-01")
    assert ok.status_code == 201
    assert ok.json()["contains_people"] is True
    assert "written consent" in ok.json()["consent"]


def test_uniformly_marginal_images_are_not_told_to_drop_a_nonexistent_rest(client, auth, project):
    """Every image clears the per-image bar but the aggregate is short. There
    is nothing to drop, so the advice must not say otherwise."""
    from veil.ml import qualify as qualify_module

    good = upload(client, auth, project, png(True), "a.png")
    dataset = make_dataset(client, auth, project, [good]).json()
    # Raise the dataset bar above what this image can reach, leaving the
    # per-image bar below it.
    original = qualify_module.MIN_DATASET_RATE
    qualify_module.MIN_DATASET_RATE = 1.5
    try:
        result = client.post(f"{API}/datasets/{dataset['id']}/qualify",
                             json={"detector_id": "colorblob-v1", "target_label": "blob",
                                   "threshold": 0.3, "image_size": 96},
                             headers=auth).json()
    finally:
        qualify_module.MIN_DATASET_RATE = original

    assert result["dataset_usable"] is False
    assert "marginal" in result["verdict"]
    assert "Drop the other" not in result["verdict"]


def test_torso_box_marks_the_figure_that_was_actually_drawn():
    """The annotation must follow the drawing's RNG stream: it once boxed a
    figure that was never drawn (IoU down to 0.0007 on some seeds)."""
    import random

    import numpy as np
    from PIL import Image, ImageDraw, ImageFilter

    from veil.ml import scenes
    from veil.ml.evaluation.metrics import iou

    for seed in range(12):
        figure = np.asarray(scenes.person(240, seed)).astype(int)
        empty = Image.new("RGB", (240, 240))
        scenes._background(ImageDraw.Draw(empty), 240, random.Random(seed))
        empty = np.asarray(empty.filter(ImageFilter.GaussianBlur(radius=0.6))).astype(int)
        ys, xs = np.nonzero(np.abs(figure - empty).sum(-1) > 30)
        drawn = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
        assert iou(list(scenes.torso_box(240, seed)), drawn) > 0.75, seed
