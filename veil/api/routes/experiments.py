"""Experiments, runs and results."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from veil import jobs
from veil.api.deps import SessionDep, UserDep, audit, fetch
from veil.db import scoped
from veil.ml.detectors import registry
from veil.models import Dataset, Evaluation, Experiment, ExperimentRun, Pattern, Project
from veil.schemas import (
    EvaluationOut,
    ExperimentCreate,
    ExperimentOut,
    PatternOut,
    RunCreate,
    RunOut,
)

router = APIRouter(tags=["experiments"])


@router.post("/experiments", response_model=ExperimentOut, status_code=201)
def create_experiment(body: ExperimentCreate, session: SessionDep, user: UserDep) -> Experiment:
    fetch(session, Project, body.project_id, user)
    dataset = fetch(session, Dataset, body.dataset_id, user)
    if dataset.project_id != body.project_id:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "dataset belongs to another project")
    if not registry.exists(body.detector_id):
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            f"unknown detector; see GET /detectors")
    detector_labels = registry.info(body.detector_id).labels
    target = body.configuration.target_label
    if detector_labels and target not in detector_labels:
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            f"detector {body.detector_id} cannot detect {target!r}")
    for transfer_id in body.configuration.transfer_detectors:
        if not registry.exists(transfer_id):
            raise HTTPException(status.HTTP_400_BAD_REQUEST,
                                f"unknown transfer detector {transfer_id!r}; see GET /detectors")
        transfer_labels = registry.info(transfer_id).labels
        if transfer_labels and target not in transfer_labels:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"transfer detector {transfer_id} cannot detect {target!r}; "
                "a transfer result against a detector that cannot produce the "
                "target label would be meaningless",
            )
    # max+1, not count+1: a count reuses the number of a deleted experiment, and
    # "#003" in a report must name one experiment forever. Two concurrent
    # creations can still pick the same number; the unique constraint catches
    # that and the loser simply takes the next one.
    for _attempt in range(3):
        next_number = 1 + (session.scalar(
            select(func.max(Experiment.number)).where(Experiment.project_id == body.project_id)
        ) or 0)
        experiment = Experiment(
            organization_id=user.organization_id, project_id=body.project_id, number=next_number,
            name=body.name, description=body.description, detector_id=body.detector_id,
            dataset_id=body.dataset_id, configuration=body.configuration.model_dump(), status="draft",
        )
        try:
            with session.begin_nested():
                session.add(experiment)
            break
        except IntegrityError:
            continue
    else:
        raise HTTPException(status.HTTP_409_CONFLICT, "experiment numbering is contended; retry")
    audit(session, user, "experiment.create", experiment.id)
    return experiment


@router.get("/experiments", response_model=list[ExperimentOut])
def list_experiments(session: SessionDep, user: UserDep, project_id: str | None = None) -> list[Experiment]:
    query = scoped(Experiment, user.organization_id).order_by(Experiment.created_at.desc())
    if project_id:
        query = query.where(Experiment.project_id == project_id)
    return list(session.scalars(query))


@router.get("/experiments/{experiment_id}", response_model=ExperimentOut)
def get_experiment(experiment_id: str, session: SessionDep, user: UserDep) -> Experiment:
    return fetch(session, Experiment, experiment_id, user)


@router.post("/experiments/{experiment_id}/run", response_model=RunOut, status_code=202)
def start_run(experiment_id: str, body: RunCreate, session: SessionDep, user: UserDep) -> ExperimentRun:
    """Queue a run. Returns immediately; poll the run for status."""
    experiment = fetch(session, Experiment, experiment_id, user)
    active = session.scalar(
        scoped(ExperimentRun, user.organization_id).where(
            ExperimentRun.experiment_id == experiment.id,
            ExperimentRun.status.in_(("queued", "running")),
        )
    )
    if active is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, f"run {active.id} is already {active.status}")
    config = body.configuration.model_dump() if body.configuration else experiment.configuration
    run = ExperimentRun(
        organization_id=user.organization_id, experiment_id=experiment.id,
        status="queued", seed=body.seed, configuration=config,
    )
    experiment.status = "queued"
    session.add(run)
    try:
        session.flush()
    except IntegrityError as exc:  # uq_active_run: a concurrent request queued first
        session.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, "a run is already active") from exc
    audit(session, user, "experiment.run", run.id, seed=body.seed)
    run_id = run.id
    session.commit()  # the worker reads this row from another thread
    jobs.submit(run_id)
    return run


@router.get("/experiments/{experiment_id}/runs", response_model=list[RunOut])
def list_runs(experiment_id: str, session: SessionDep, user: UserDep) -> list[ExperimentRun]:
    fetch(session, Experiment, experiment_id, user)
    return list(session.scalars(
        scoped(ExperimentRun, user.organization_id)
        .where(ExperimentRun.experiment_id == experiment_id)
        .order_by(ExperimentRun.created_at.desc())
    ))


@router.get("/runs/{run_id}", response_model=RunOut)
def get_run(run_id: str, session: SessionDep, user: UserDep) -> ExperimentRun:
    return fetch(session, ExperimentRun, run_id, user)


@router.get("/experiments/{experiment_id}/results", response_model=list[EvaluationOut])
def get_results(experiment_id: str, session: SessionDep, user: UserDep) -> list[Evaluation]:
    fetch(session, Experiment, experiment_id, user)
    return list(session.scalars(
        scoped(Evaluation, user.organization_id)
        .where(Evaluation.experiment_id == experiment_id)
        .order_by(Evaluation.created_at.desc())
    ))


@router.get("/experiments/{experiment_id}/patterns", response_model=list[PatternOut])
def list_experiment_patterns(experiment_id: str, session: SessionDep, user: UserDep) -> list[Pattern]:
    fetch(session, Experiment, experiment_id, user)
    return list(session.scalars(
        scoped(Pattern, user.organization_id).where(Pattern.experiment_id == experiment_id)
    ))
