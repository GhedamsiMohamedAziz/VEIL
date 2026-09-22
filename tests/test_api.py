"""API contract, authentication and tenant isolation."""

from __future__ import annotations

import pytest

API = "/api/v1"


def test_health_needs_no_auth(client):
    assert client.get("/health").json()["status"] == "ok"


def test_unauthenticated_requests_are_rejected(client):
    assert client.get(f"{API}/projects").status_code == 401
    assert client.get(f"{API}/projects", headers={"X-API-Key": "nope"}).status_code == 401


def test_bearer_token_also_works(client, api_key):
    response = client.get(f"{API}/projects", headers={"Authorization": f"Bearer {api_key}"})
    assert response.status_code == 200


def test_project_lifecycle(client, auth):
    created = client.post(f"{API}/projects", json={"name": "Urban", "description": "d"}, headers=auth)
    assert created.status_code == 201
    project_id = created.json()["id"]
    assert client.get(f"{API}/projects/{project_id}", headers=auth).json()["name"] == "Urban"
    assert [p["id"] for p in client.get(f"{API}/projects", headers=auth).json()] == [project_id]


def test_upload_rejects_unidentifiable_content(client, auth):
    project_id = client.post(f"{API}/projects", json={"name": "p"}, headers=auth).json()["id"]
    response = client.post(
        f"{API}/projects/{project_id}/artifacts",
        files={"file": ("payload.png", b"#!/bin/sh\nrm -rf /\n", "image/png")},
        headers=auth,
    )
    # The client claimed image/png; the bytes say otherwise.
    assert response.status_code == 415


def test_upload_rejects_riff_that_is_not_webp_and_unknown_kind(client, auth, red_png):
    project_id = client.post(f"{API}/projects", json={"name": "p"}, headers=auth).json()["id"]
    url = f"{API}/projects/{project_id}/artifacts"
    wav = b"RIFF\x24\x00\x00\x00WAVEfmt "
    assert client.post(url, files={"file": ("a.webp", wav, "image/webp")},
                       headers=auth).status_code == 415
    # `kind` ends up in a Content-Disposition header: only the known values get in.
    injected = client.post(url, files={"file": ("a.png", red_png, "image/png")},
                           data={"kind": 'x"; filename="evil.html'}, headers=auth)
    assert injected.status_code == 422


def test_upload_is_content_addressed(client, auth, red_png):
    project_id = client.post(f"{API}/projects", json={"name": "p"}, headers=auth).json()["id"]
    first = client.post(f"{API}/projects/{project_id}/artifacts",
                        files={"file": ("a.png", red_png, "image/png")}, headers=auth).json()
    second = client.post(f"{API}/projects/{project_id}/artifacts",
                         files={"file": ("b.png", red_png, "image/png")}, headers=auth).json()
    assert first["sha256"] == second["sha256"]
    assert first["media_type"] == "image/png"


def test_detectors_are_read_only_and_listed(client, auth):
    ids = {d["id"] for d in client.get(f"{API}/detectors", headers=auth).json()}
    assert "colorblob-v1" in ids
    # There is deliberately no way to register a model over the API.
    assert client.post(f"{API}/detectors", json={}, headers=auth).status_code == 405


def test_experiment_rejects_unknown_detector_and_wrong_label(client, auth, red_png):
    project_id = client.post(f"{API}/projects", json={"name": "p"}, headers=auth).json()["id"]
    artifact = client.post(f"{API}/projects/{project_id}/artifacts",
                           files={"file": ("a.png", red_png, "image/png")}, headers=auth).json()
    dataset = client.post(f"{API}/datasets", json={
        "project_id": project_id, "name": "d", "source": "synthetic, generated in tests",
        "artifact_ids": [artifact["id"]]}, headers=auth).json()

    body = {"project_id": project_id, "name": "e", "detector_id": "not-a-model",
            "dataset_id": dataset["id"]}
    assert client.post(f"{API}/experiments", json=body, headers=auth).status_code == 400

    body = {**body, "detector_id": "colorblob-v1", "configuration": {"target_label": "person"}}
    response = client.post(f"{API}/experiments", json=body, headers=auth)
    assert response.status_code == 400
    assert "cannot detect" in response.json()["detail"]


@pytest.fixture
def other_org_key():
    from veil.db import create_organization, session_scope

    with session_scope() as session:
        _, _, key = create_organization(session, "Rival Corp", "rival@example.com")
    return key


def test_one_organization_cannot_read_anothers_data(client, auth, other_org_key, red_png):
    project_id = client.post(f"{API}/projects", json={"name": "secret"}, headers=auth).json()["id"]
    artifact = client.post(f"{API}/projects/{project_id}/artifacts",
                           files={"file": ("a.png", red_png, "image/png")}, headers=auth).json()
    intruder = {"X-API-Key": other_org_key}

    assert client.get(f"{API}/projects/{project_id}", headers=intruder).status_code == 404
    assert client.get(f"{API}/projects", headers=intruder).json() == []
    assert client.get(f"{API}/projects/{project_id}/artifacts", headers=intruder).status_code == 404
    # ...and cannot smuggle a foreign artifact into its own dataset.
    own_project = client.post(f"{API}/projects", json={"name": "mine"}, headers=intruder).json()["id"]
    response = client.post(f"{API}/datasets", json={
        "project_id": own_project, "name": "d", "source": "synthetic, generated in tests",
        "artifact_ids": [artifact["id"]]}, headers=intruder)
    assert response.status_code == 404


def test_report_before_any_run_is_a_conflict_not_a_fabrication(client, auth, red_png):
    project_id = client.post(f"{API}/projects", json={"name": "p"}, headers=auth).json()["id"]
    artifact = client.post(f"{API}/projects/{project_id}/artifacts",
                           files={"file": ("a.png", red_png, "image/png")}, headers=auth).json()
    dataset = client.post(f"{API}/datasets", json={
        "project_id": project_id, "name": "d", "source": "synthetic, generated in tests",
        "artifact_ids": [artifact["id"]]}, headers=auth).json()
    experiment = client.post(f"{API}/experiments", json={
        "project_id": project_id, "name": "e", "detector_id": "colorblob-v1",
        "dataset_id": dataset["id"], "configuration": {"target_label": "blob"}},
        headers=auth).json()
    response = client.post(f"{API}/experiments/{experiment['id']}/report", headers=auth)
    assert response.status_code == 409


def test_validation_and_catalogue_never_load_detector_weights(client, auth, monkeypatch):
    """Labels and version are static: creating an experiment must not build a
    ResNet per request, and the catalogue must name the weights a run records."""
    from veil.ml.detectors import registry
    from veil.ml.detectors.torchvision_detector import TorchvisionDetector

    def refuse(self):
        raise AssertionError("weights loaded on a metadata path")

    monkeypatch.setattr(TorchvisionDetector, "load", refuse)
    info = registry.info("fasterrcnn-mobilenet-320")
    assert "person" in info.labels and info.version.endswith("COCO_V1")
    listed = {d["id"]: d for d in client.get(f"{API}/detectors", headers=auth).json()}
    assert listed["fasterrcnn-mobilenet-320"]["version"] == info.version


def test_a_sku_taken_by_one_organization_is_free_for_another(client, auth, other_org_key):
    """Uniqueness is per tenant: a global constraint turned 'is this SKU taken?'
    into a way to enumerate another organization's garments."""
    from veil.db import session_scope
    from veil.models import Pattern, User

    body = {"sku": "SHARED-0001", "batch_id": "B", "product_type": "tshirt"}
    with session_scope() as session:
        pattern_ids = []
        for user in session.query(User).order_by(User.created_at):
            pattern = Pattern(organization_id=user.organization_id, experiment_id="e" * 32, version=1)
            session.add(pattern)
            session.flush()
            pattern_ids.append(pattern.id)
    mine = client.post(f"{API}/garments", json={**body, "pattern_id": pattern_ids[0]}, headers=auth)
    theirs = client.post(f"{API}/garments", json={**body, "pattern_id": pattern_ids[1]},
                         headers={"X-API-Key": other_org_key})
    assert (mine.status_code, theirs.status_code) == (201, 201)
    again = client.post(f"{API}/garments", json={**body, "pattern_id": pattern_ids[0]}, headers=auth)
    assert again.status_code == 409


def test_experiment_numbers_are_never_reused_and_active_runs_are_unique(client, auth, red_png):
    import pytest as _pytest
    from sqlalchemy.exc import IntegrityError

    from veil.db import session_scope
    from veil.models import Experiment, ExperimentRun

    project_id = client.post(f"{API}/projects", json={"name": "p"}, headers=auth).json()["id"]
    artifact = client.post(f"{API}/projects/{project_id}/artifacts",
                           files={"file": ("a.png", red_png, "image/png")}, headers=auth).json()
    dataset = client.post(f"{API}/datasets", json={
        "project_id": project_id, "name": "d", "source": "synthetic, generated in tests",
        "artifact_ids": [artifact["id"]]}, headers=auth).json()
    body = {"project_id": project_id, "name": "e", "detector_id": "colorblob-v1",
            "dataset_id": dataset["id"], "configuration": {"target_label": "blob"}}
    made = [client.post(f"{API}/experiments", json=body, headers=auth).json() for _ in range(3)]
    assert [e["number"] for e in made] == [1, 2, 3]
    with session_scope() as session:  # experiment #2 goes away; its number must not come back
        session.delete(session.get(Experiment, made[1]["id"]))
    assert client.post(f"{API}/experiments", json=body, headers=auth).json()["number"] == 4

    # The database itself refuses a second active run, whatever the route checked.
    with _pytest.raises(IntegrityError), session_scope() as session:
        organization_id = session.get(Experiment, made[0]["id"]).organization_id
        for _ in range(2):
            session.add(ExperimentRun(organization_id=organization_id,
                                      experiment_id=made[0]["id"], status="queued", seed=1))


def test_oversized_uploads_are_refused_before_and_after_the_body(client, auth, red_png, monkeypatch):
    from veil.config import get_settings

    project_id = client.post(f"{API}/projects", json={"name": "p"}, headers=auth).json()["id"]
    url = f"{API}/projects/{project_id}/artifacts"
    monkeypatch.setattr(get_settings(), "max_upload_bytes", len(red_png) - 1)
    # Within the framing slack, so it reaches the route: the stream ceiling stops it.
    assert client.post(url, files={"file": ("a.png", red_png, "image/png")},
                       headers=auth).status_code == 413
    # Declared far beyond the limit: refused on the header alone.
    huge = client.post(url, content=b"x", headers={**auth, "Content-Length": str(10**12),
                                                   "Content-Type": "multipart/form-data; boundary=b"})
    assert huge.status_code == 413


def test_chunked_bodies_are_accepted_and_cut_off_at_the_limit(client, auth, monkeypatch):
    """curl -T and streamed fetches declare no length. They must work, and an
    oversized one must get a 413 the browser can read (CORS headers on it)."""
    from veil.config import get_settings

    def chunks(total):
        yield b'{"name": "p'
        for _ in range(total // 1024):
            yield b"x" * 1024
        yield b'"}'

    headers = {**auth, "Content-Type": "application/json", "Origin": "http://localhost:3000"}
    small = client.post(f"{API}/projects", content=chunks(0), headers=headers)
    assert small.status_code == 201

    monkeypatch.setattr(get_settings(), "max_upload_bytes", 1024)
    big = client.post(f"{API}/projects", content=chunks(200 * 1024), headers=headers)
    assert big.status_code == 413
    assert big.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_artifact_paths_must_look_like_artifacts():
    from veil.storage import local_path

    digest = "ab" + "0" * 62
    assert local_path(f"file:///anywhere/org/ab/{digest}").name == digest
    for uri in ("file:///etc/passwd", f"file:///anywhere/org/cd/{digest}"):
        with pytest.raises(ValueError):
            local_path(uri)
