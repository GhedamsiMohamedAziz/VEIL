"""Projects, artifact upload, and datasets."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile, status

from veil.api.deps import SessionDep, UserDep, audit, fetch
from veil.config import get_settings
from veil.db import scoped
from veil.models import Artifact, Dataset, Project
from veil.schemas import (
    ArtifactOut,
    DatasetCreate,
    DatasetOut,
    ProjectCreate,
    ProjectOut,
    QualifyRequest,
)
from veil.ml.detectors import registry
from veil.ml.qualify import qualify
from veil.ml.runner import _load_images
from veil.storage import put_bytes, sniff_media_type

router = APIRouter(tags=["projects"])


@router.post("/projects", response_model=ProjectOut, status_code=201)
def create_project(body: ProjectCreate, session: SessionDep, user: UserDep) -> Project:
    project = Project(organization_id=user.organization_id, name=body.name, description=body.description)
    session.add(project)
    session.flush()
    audit(session, user, "project.create", project.id)
    return project


@router.get("/projects", response_model=list[ProjectOut])
def list_projects(session: SessionDep, user: UserDep, limit: int = Query(100, le=500)) -> list[Project]:
    return list(session.scalars(scoped(Project, user.organization_id).limit(limit)))


@router.get("/projects/{project_id}", response_model=ProjectOut)
def get_project(project_id: str, session: SessionDep, user: UserDep) -> Project:
    return fetch(session, Project, project_id, user)


@router.post("/projects/{project_id}/artifacts", response_model=ArtifactOut, status_code=201)
async def upload_artifact(
    project_id: str, session: SessionDep, user: UserDep, file: UploadFile = File(...),
    kind: Literal["input", "pattern", "report", "frame", "artwork"] = Form("input"),
) -> Artifact:
    """Upload an authorized test image.

    The media type comes from the file's magic bytes, not the client's
    header, and anything we do not recognize is rejected outright - VEIL
    never stores a blob it cannot identify.
    """
    project = fetch(session, Project, project_id, user)
    settings = get_settings()
    data = await file.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            f"file exceeds {settings.max_upload_bytes} bytes")
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "empty file")
    media_type = sniff_media_type(data[:16])
    if media_type is None or media_type not in settings.allowed_upload_types:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                            f"unsupported file content; allowed: {list(settings.allowed_upload_types)}")
    uri, digest, size = put_bytes(user.organization_id, data)
    artifact = Artifact(
        organization_id=user.organization_id, project_id=project.id, kind=kind,
        uri=uri, media_type=media_type, size_bytes=size, sha256=digest,
        meta={"original_filename": file.filename},
    )
    session.add(artifact)
    session.flush()
    audit(session, user, "artifact.upload", artifact.id, sha256=digest, size=size)
    return artifact


@router.get("/projects/{project_id}/artifacts", response_model=list[ArtifactOut])
def list_artifacts(project_id: str, session: SessionDep, user: UserDep) -> list[Artifact]:
    fetch(session, Project, project_id, user)
    return list(session.scalars(
        scoped(Artifact, user.organization_id).where(Artifact.project_id == project_id)
    ))


@router.post("/datasets", response_model=DatasetOut, status_code=201)
def create_dataset(body: DatasetCreate, session: SessionDep, user: UserDep) -> Dataset:
    fetch(session, Project, body.project_id, user)
    for artifact_id in body.artifact_ids:
        fetch(session, Artifact, artifact_id, user)  # ownership check per item
    if body.contains_people and not body.consent.strip():
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "consent is required when the dataset contains identifiable people; "
            "record the basis for holding this imagery",
        )
    dataset = Dataset(
        organization_id=user.organization_id, project_id=body.project_id, name=body.name,
        source=body.source, license=body.license, artifact_ids=body.artifact_ids,
        annotations=body.annotations, contains_people=body.contains_people,
        consent=body.consent,
    )
    session.add(dataset)
    session.flush()
    audit(session, user, "dataset.create", dataset.id, items=len(body.artifact_ids))
    return dataset


@router.get("/datasets", response_model=list[DatasetOut])
def list_datasets(session: SessionDep, user: UserDep, project_id: str | None = None) -> list[Dataset]:
    query = scoped(Dataset, user.organization_id)
    if project_id:
        query = query.where(Dataset.project_id == project_id)
    return list(session.scalars(query))


@router.get("/datasets/{dataset_id}", response_model=DatasetOut)
def get_dataset(dataset_id: str, session: SessionDep, user: UserDep) -> Dataset:
    return fetch(session, Dataset, dataset_id, user)


@router.post("/datasets/{dataset_id}/qualify")
def qualify_dataset(
    dataset_id: str, body: QualifyRequest, session: SessionDep, user: UserDep
) -> dict:
    """Baseline-sweep a dataset before running an experiment on it.

    Reports, per image, how often the detector sees the target with no
    pattern present. An image it rarely sees cannot support a measurable
    drop, so this is the cheap check that prevents an expensive meaningless
    run. Synchronous: it is one baseline sweep over a small grid.
    """
    dataset = fetch(session, Dataset, dataset_id, user)
    if not registry.exists(body.detector_id):
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            "unknown detector; see GET /detectors")
    detector = registry.get(body.detector_id)
    labels = detector.metadata().labels
    if labels and body.target_label not in labels:
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            f"{body.detector_id} cannot detect {body.target_label!r}")
    try:
        images, image_ids = _load_images(session, dataset, body.image_size)
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc

    result = qualify(detector, images, image_ids, body.target_label,
                     threshold=body.threshold)
    audit(session, user, "dataset.qualify", dataset.id,
          detector=body.detector_id, usable=result["dataset_usable"])
    return {"dataset_id": dataset.id, **result}
