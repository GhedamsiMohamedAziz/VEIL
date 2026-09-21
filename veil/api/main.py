"""VEIL Lab API."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from veil import __version__
from veil.api.routes import detectors, experiments, patterns, physical, projects, reports
from veil.config import get_settings
from veil.db import create_organization, init_db, session_scope
from veil.models import Experiment, ExperimentRun, Organization, utcnow

DESCRIPTION = """
A research and testing platform for measuring how computer-vision systems
perceive the physical world.

Every number this API returns is an experimental measurement made under
recorded conditions - a specific detector version, dataset, transformation
distribution and seed. Results are not guarantees and do not generalize to
models, cameras or environments that were not measured.

Authenticate with `X-API-Key: <key>` or `Authorization: Bearer <key>`.
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)
    init_db()
    if settings.bootstrap_api_key:
        with session_scope() as session:
            if session.query(Organization).count() == 0:
                create_organization(session, "VEIL (bootstrap)", "dev@veil.local",
                                    settings.bootstrap_api_key)
                logging.getLogger("veil").info("bootstrap organization created")
    # The worker is a thread of this process: at startup nothing can own a run
    # still marked active. Left alone it would block its experiment with a 409
    # forever, so it is closed as failed and the experiment can be run again.
    with session_scope() as session:
        for run in session.query(ExperimentRun).filter(ExperimentRun.status.in_(("queued", "running"))):
            run.status, run.finished_at = "failed", utcnow()
            run.error = "interrupted: the API process restarted before this run finished"
            experiment = session.get(Experiment, run.experiment_id)
            if experiment is not None and experiment.status in ("queued", "running"):
                experiment.status = "failed"
    yield


app = FastAPI(
    title="VEIL Lab API",
    version=__version__,
    description=DESCRIPTION,
    lifespan=lifespan,
)

# The dashboard is a separate origin; set VEIL_CORS_ORIGINS for anything but localhost.
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(get_settings().cors_origins),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def refuse_oversized_bodies(request: Request, call_next):
    """The per-file limit in the upload route only runs after the multipart
    parser has spooled the whole body to disk. Refuse on the declared length,
    before a byte is read; a body that declares none is refused too."""
    if request.method in ("POST", "PUT", "PATCH"):
        declared = request.headers.get("content-length")
        if declared is None:
            return JSONResponse({"detail": "Content-Length required"}, status_code=411)
        # Slack for multipart framing and the other form fields.
        if not declared.isdigit() or int(declared) > get_settings().max_upload_bytes + 64 * 1024:
            return JSONResponse({"detail": "request body too large"}, status_code=413)
    return await call_next(request)


for module in (projects, experiments, patterns, detectors, physical, reports):
    app.include_router(module.router, prefix="/api/v1")


@app.get("/health", tags=["meta"])
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}
