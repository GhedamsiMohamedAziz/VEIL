"""VEIL Lab API."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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

# The dashboard is a separate origin in development. Tighten this list before
# any deployment that is not localhost.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

for module in (projects, experiments, patterns, detectors, physical, reports):
    app.include_router(module.router, prefix="/api/v1")


@app.get("/health", tags=["meta"])
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}
