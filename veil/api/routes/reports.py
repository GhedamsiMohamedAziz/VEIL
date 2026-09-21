"""Report generation (JSON + PDF)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Response, status

from veil import events
from veil.api.deps import SessionDep, UserDep, audit, fetch
from veil.db import scoped
from veil.models import Artifact, Experiment, ExperimentRun, Report
from veil.reports.builder import build
from veil.reports.render import to_pdf
from veil.schemas import ReportOut
from veil.storage import put_bytes, read_bytes

router = APIRouter(tags=["reports"])


@router.post("/experiments/{experiment_id}/report", response_model=ReportOut, status_code=201)
def generate_report(
    experiment_id: str, session: SessionDep, user: UserDep, run_id: str | None = None
) -> Report:
    experiment = fetch(session, Experiment, experiment_id, user)
    run = fetch(session, ExperimentRun, run_id, user) if run_id else None
    if run is not None and run.experiment_id != experiment.id:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "run belongs to another experiment")
    try:
        payload = build(session, experiment, run)
    except ValueError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc

    pdf = to_pdf(payload)
    uri, digest, size = put_bytes(user.organization_id, pdf)
    artifact = Artifact(
        organization_id=user.organization_id, project_id=experiment.project_id, kind="report",
        uri=uri, media_type="application/pdf", size_bytes=size, sha256=digest,
        meta={"experiment_id": experiment.id},
    )
    session.add(artifact)
    session.flush()
    report = Report(
        organization_id=user.organization_id, experiment_id=experiment.id,
        run_id=payload["reproducibility"]["run_id"], payload=payload,
        pdf_artifact_id=artifact.id,
    )
    session.add(report)
    session.flush()
    audit(session, user, "report.generate", report.id)
    events.emit(events.REPORT_GENERATED, experiment_id=experiment.id,
                project_id=experiment.project_id, report_id=report.id)
    return report


@router.get("/reports", response_model=list[ReportOut])
def list_reports(session: SessionDep, user: UserDep, experiment_id: str | None = None) -> list[Report]:
    query = scoped(Report, user.organization_id).order_by(Report.created_at.desc())
    if experiment_id:
        query = query.where(Report.experiment_id == experiment_id)
    return list(session.scalars(query))


@router.get("/reports/{report_id}", response_model=ReportOut)
def get_report(report_id: str, session: SessionDep, user: UserDep) -> Report:
    return fetch(session, Report, report_id, user)


@router.get("/reports/{report_id}/pdf")
def get_report_pdf(report_id: str, session: SessionDep, user: UserDep) -> Response:
    report = fetch(session, Report, report_id, user)
    if not report.pdf_artifact_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "report has no PDF")
    artifact = fetch(session, Artifact, report.pdf_artifact_id, user)
    return Response(
        read_bytes(artifact.uri), media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="veil-report-{report.id}.pdf"'},
    )
