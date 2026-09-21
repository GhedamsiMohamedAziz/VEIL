"""Patterns, their image artifacts, and the garments printed from them."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Response, status
from sqlalchemy.exc import IntegrityError

from veil.api.deps import SessionDep, UserDep, audit, fetch
from veil.db import scoped
from veil.ml.detectors import registry
from veil.ml.evaluation.metrics import summarize
from veil.ml.manufacture_eval import evaluate_production
from veil.ml.patterns.manufacture import ProductionSpec, to_artwork
from veil.ml.patterns.serialization import from_png, to_png
from veil.ml.runner import _load_images
from veil.ml.simulation.renderer import Placement
from veil.ml.simulation.transforms import TransformSpec
from veil.models import (
    Artifact,
    Dataset,
    Evaluation,
    Experiment,
    ExperimentRun,
    Garment,
    Pattern,
    utcnow,
)
from veil.schemas import GarmentCreate, GarmentOut, PatternOut, ProductionRequest
from veil.storage import put_bytes, read_bytes

router = APIRouter(tags=["patterns"])


@router.get("/patterns", response_model=list[PatternOut])
def list_patterns(session: SessionDep, user: UserDep) -> list[Pattern]:
    return list(session.scalars(
        scoped(Pattern, user.organization_id).order_by(Pattern.created_at.desc())
    ))


@router.get("/patterns/{pattern_id}", response_model=PatternOut)
def get_pattern(pattern_id: str, session: SessionDep, user: UserDep) -> Pattern:
    return fetch(session, Pattern, pattern_id, user)


@router.get("/patterns/{pattern_id}/image")
def get_pattern_image(pattern_id: str, session: SessionDep, user: UserDep) -> Response:
    pattern = fetch(session, Pattern, pattern_id, user)
    if not pattern.artifact_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "pattern has no image artifact")
    artifact = fetch(session, Artifact, pattern.artifact_id, user)
    return Response(read_bytes(artifact.uri), media_type=artifact.media_type)


@router.post("/garments", response_model=GarmentOut, status_code=201)
def create_garment(body: GarmentCreate, session: SessionDep, user: UserDep) -> Garment:
    """Register a physical item. The SKU is what a QR/NFC tag resolves to."""
    pattern = fetch(session, Pattern, body.pattern_id, user)
    if session.scalar(scoped(Garment, user.organization_id).where(Garment.sku == body.sku)):
        raise HTTPException(status.HTTP_409_CONFLICT, "sku already registered")
    garment = Garment(
        organization_id=user.organization_id, pattern_id=pattern.id,
        experiment_id=pattern.experiment_id, sku=body.sku, batch_id=body.batch_id,
        product_type=body.product_type, material=body.material,
        print_method=body.print_method, test_conditions=body.test_conditions,
        tested_at=utcnow(),
    )
    session.add(garment)
    try:
        session.flush()
    except IntegrityError as exc:  # lost a race, or a pre-migration global constraint
        session.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, "sku already registered") from exc
    audit(session, user, "garment.create", garment.id, sku=body.sku)
    return garment


@router.get("/garments", response_model=list[GarmentOut])
def list_garments(session: SessionDep, user: UserDep) -> list[Garment]:
    return list(session.scalars(scoped(Garment, user.organization_id)))


@router.get("/garments/{sku}/provenance")
def garment_provenance(sku: str, session: SessionDep, user: UserDep) -> dict:
    """What a QR code on a VEIL Wear item resolves to: which pattern, which
    experiment, under which measured conditions."""
    garment = session.scalar(scoped(Garment, user.organization_id).where(Garment.sku == sku))
    if garment is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown sku")
    pattern = fetch(session, Pattern, garment.pattern_id, user)
    experiment = fetch(session, Experiment, garment.experiment_id, user)
    return {
        "sku": garment.sku,
        "batch_id": garment.batch_id,
        "product_type": garment.product_type,
        "material": garment.material,
        "print_method": garment.print_method,
        "pattern": {"id": pattern.id, "version": pattern.version},
        "experiment": {"id": experiment.id, "number": experiment.number,
                       "detector_id": experiment.detector_id, "name": experiment.name},
        "tested_at": garment.tested_at,
        "test_conditions": garment.test_conditions,
        "limitations": (
            "Measurements describe the detector, dataset and physical conditions "
            "recorded in the linked experiment. They do not generalize to other "
            "models, cameras or environments."
        ),
    }


@router.post("/patterns/{pattern_id}/manufacture")
def manufacture_pattern(
    pattern_id: str, body: ProductionRequest, session: SessionDep, user: UserDep
) -> dict:
    """Quantize the pattern to a production constraint and re-measure it.

    A pattern that works as float pixels may not survive being knitted in six
    yarns at five stitches per centimetre. This runs the same sweep against
    the same detector and dataset so the retained effect is comparable, and
    stores the production artwork as an artifact the mill can be sent.
    """
    pattern_row = fetch(session, Pattern, pattern_id, user)
    if not pattern_row.artifact_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "pattern has no image artifact")
    experiment = fetch(session, Experiment, pattern_row.experiment_id, user)
    dataset = fetch(session, Dataset, experiment.dataset_id, user)
    source = fetch(session, Artifact, pattern_row.artifact_id, user)
    pattern = from_png(read_bytes(source.uri))

    config = dict(experiment.configuration or {})
    target_label = config.get("target_label", "person")
    threshold = float(config.get("threshold", 0.5))
    image_size = int(config.get("image_size", 320))
    transform_spec = TransformSpec.from_config(config.get("transforms"))
    generation = pattern_row.generation_parameters or {}
    single = Placement(**generation.get("placement", {}))

    evaluation = session.scalar(
        scoped(Evaluation, user.organization_id)
        .where(Evaluation.pattern_id == pattern_row.id)
        .order_by(Evaluation.created_at.desc())
    )
    control_rate = (evaluation.control_metrics or {}).get("detection_rate") if evaluation else None
    digital_rate = (evaluation.candidate_metrics or {}).get("detection_rate") if evaluation else None

    production = ProductionSpec(
        method=body.method, stitches_per_cm=body.stitches_per_cm, dpi=body.dpi,
        width_cm=body.width_cm, height_cm=body.height_cm, max_yarns=body.max_yarns,
    )
    try:
        images, image_ids = _load_images(session, dataset, image_size)
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    # Measure on the images the run evaluated on, each with its own placement:
    # `digital_rate` came from exactly those, and a production rate over the
    # optimizer's images would not be comparable to it.
    evaluated = (generation.get("split") or {}).get("evaluate")
    if evaluated:
        keep = [i for i, image_id in enumerate(image_ids) if image_id in set(evaluated)]
        if not keep:  # measuring on other images would give a rate comparable to nothing
            raise HTTPException(status.HTTP_409_CONFLICT,
                                "the images this pattern was evaluated on are no longer in the dataset")
        images, image_ids = images[keep], [image_ids[i] for i in keep]
    stored = generation.get("placements") or {}
    placement = [Placement(**stored[i]) if i in stored else single for i in image_ids]

    # The run's own seed: it drives the cloth deformation and the sensor noise,
    # and `digital_rate` below was measured under it. Another seed would put
    # noise, not manufacturing, into effect_retained.
    run = session.get(ExperimentRun, pattern_row.run_id) if pattern_row.run_id else None
    seed = run.seed if run is not None and run.organization_id == user.organization_id else 42

    detector = registry.get(experiment.detector_id)
    result = evaluate_production(
        detector, images, image_ids, pattern, placement, transform_spec, production,
        target_label=target_label, threshold=threshold, seed=seed,
        control_rate=control_rate, digital_rate=digital_rate,
    )

    # The artwork file itself, at production resolution.
    artwork_png = to_png(to_artwork(pattern, production))
    uri, digest, size = put_bytes(user.organization_id, artwork_png)
    artwork = Artifact(
        organization_id=user.organization_id, project_id=experiment.project_id,
        kind="artwork", uri=uri, media_type="image/png", size_bytes=size, sha256=digest,
        meta={"pattern_id": pattern_row.id, "production": production.as_dict(),
              "effect_retained": result["effect_retained"]},
    )
    session.add(artwork)
    session.flush()
    audit(session, user, "pattern.manufacture", pattern_row.id, method=body.method)
    return {"pattern_id": pattern_row.id, "artwork_artifact_id": artwork.id, **result}


@router.get("/artifacts/{artifact_id}/download")
def download_artifact(artifact_id: str, session: SessionDep, user: UserDep) -> Response:
    """Fetch a stored artifact - production artwork, an uploaded image."""
    artifact = fetch(session, Artifact, artifact_id, user)
    return Response(
        read_bytes(artifact.uri), media_type=artifact.media_type,
        headers={"Content-Disposition":
                 f'attachment; filename="veil-{artifact.kind}-{artifact.id[:8]}.png"'},
    )
