"""Physical test records: measurements taken with a real camera."""

from __future__ import annotations

from fastapi import APIRouter

from veil import events
from veil.api.deps import SessionDep, UserDep, audit, fetch
from veil.db import scoped
from veil.models import Artifact, Experiment, Garment, Pattern, PhysicalTest
from veil.schemas import PhysicalTestCreate, PhysicalTestOut

router = APIRouter(tags=["physical-tests"])


@router.post("/physical-tests", response_model=PhysicalTestOut, status_code=201)
def create_physical_test(body: PhysicalTestCreate, session: SessionDep, user: UserDep) -> PhysicalTest:
    """Record one physical measurement.

    Conditions are stored with the result and are not optional in the report:
    a physical number without its distance, angle and lighting is unusable.
    """
    experiment = fetch(session, Experiment, body.experiment_id, user)
    for model, object_id in ((Pattern, body.pattern_id), (Garment, body.garment_id),
                             (Artifact, body.artifact_id)):
        if object_id:
            fetch(session, model, object_id, user)
    test = PhysicalTest(
        organization_id=user.organization_id, experiment_id=experiment.id,
        **body.model_dump(exclude={"experiment_id"}),
    )
    session.add(test)
    session.flush()
    audit(session, user, "physical_test.create", test.id)
    events.emit(events.PHYSICAL_TEST_COMPLETED, experiment_id=experiment.id,
                project_id=experiment.project_id, physical_test_id=test.id,
                detected=bool(test.result.get("detected")))
    return test


@router.get("/physical-tests", response_model=list[PhysicalTestOut])
def list_physical_tests(
    session: SessionDep, user: UserDep, experiment_id: str | None = None
) -> list[PhysicalTest]:
    query = scoped(PhysicalTest, user.organization_id).order_by(PhysicalTest.created_at.desc())
    if experiment_id:
        query = query.where(PhysicalTest.experiment_id == experiment_id)
    return list(session.scalars(query))
