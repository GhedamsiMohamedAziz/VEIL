"""VEIL Lab API."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
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
        active = ("queued", "running") if settings.fail_orphaned_runs_on_start else ()
        for run in session.query(ExperimentRun).filter(ExperimentRun.status.in_(active)):
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

class BodyLimit:
    """Refuse oversized request bodies before they reach the multipart parser,
    which spools a whole body to disk before the upload route can check it.

    A declared Content-Length is refused outright; a chunked body (curl -T,
    streamed fetch, generator bodies) declares nothing, so its bytes are
    counted as they arrive and the request is cut off at the ceiling."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] not in ("POST", "PUT", "PATCH"):
            return await self.app(scope, receive, send)
        # Slack for multipart framing and the other form fields.
        limit = get_settings().max_upload_bytes + 64 * 1024
        refusal = JSONResponse({"detail": "request body too large"}, status_code=413)
        declared = dict(scope["headers"]).get(b"content-length")
        if declared is not None and (not declared.isdigit() or int(declared) > limit):
            return await refusal(scope, receive, send)

        seen, over = 0, False

        async def counted_receive():
            nonlocal seen, over
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > limit:
                    over = True
                    return {"type": "http.disconnect"}  # the app stops reading here
            return message

        async def send_unless_over(message):
            if not over:
                await send(message)

        try:
            await self.app(scope, counted_receive, send_unless_over)
        except Exception:
            if not over:
                raise
        if over:
            await refusal(scope, receive, send)


# Order matters: the last middleware added is the outermost. CORS must wrap
# BodyLimit, or the browser hides its 413 behind a generic network error.
app.add_middleware(BodyLimit)
# The dashboard is a separate origin; set VEIL_CORS_ORIGINS for anything but localhost.
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(get_settings().cors_origins),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


for module in (projects, experiments, patterns, detectors, physical, reports):
    app.include_router(module.router, prefix="/api/v1")


@app.get("/health", tags=["meta"])
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}
