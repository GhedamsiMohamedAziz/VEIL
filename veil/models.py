"""Domain model.

Tenancy rule (see docs/security.md): every row a user can reach carries
`organization_id`, and every query goes through `veil.db.scoped()`. There is
no "global" experiment or artifact.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _uuid() -> str:
    return uuid.uuid4().hex


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    type_annotation_map = {dict[str, Any]: JSON, list[str]: JSON}


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Organization(Base, TimestampMixin):
    __tablename__ = "organizations"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(200))


class User(Base, TimestampMixin):
    """A principal. `api_key_hash` is a SHA-256 of the key; the key itself is
    shown once at creation and never stored."""

    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), index=True)
    email: Mapped[str] = mapped_column(String(320), unique=True)
    api_key_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    role: Mapped[str] = mapped_column(String(32), default="member")

    organization: Mapped[Organization] = relationship()


class Project(Base, TimestampMixin):
    __tablename__ = "projects"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text, default="")


class Artifact(Base, TimestampMixin):
    """Any stored blob: an uploaded test image, a generated pattern PNG, a
    report PDF, a physical-test frame."""

    __tablename__ = "artifacts"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), index=True)
    project_id: Mapped[str | None] = mapped_column(ForeignKey("projects.id"), index=True)
    kind: Mapped[str] = mapped_column(String(32))  # input | pattern | report | frame
    uri: Mapped[str] = mapped_column(String(500))
    media_type: Mapped[str] = mapped_column(String(100))
    size_bytes: Mapped[int] = mapped_column(Integer)
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    meta: Mapped[dict[str, Any]] = mapped_column(default=dict)


class Dataset(Base, TimestampMixin):
    """An ordered, versioned set of authorized test artifacts."""

    __tablename__ = "datasets"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), index=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    version: Mapped[int] = mapped_column(Integer, default=1)
    # Provenance is mandatory: we only run on data we are allowed to use.
    source: Mapped[str] = mapped_column(Text, default="")
    license: Mapped[str] = mapped_column(String(200), default="")
    # Photographs of people are the realistic input for this platform, so the
    # basis for holding them is recorded with the data rather than assumed.
    consent: Mapped[str] = mapped_column(Text, default="")
    contains_people: Mapped[bool] = mapped_column(Boolean, default=False)
    artifact_ids: Mapped[list[str]] = mapped_column(default=list)
    # Optional ground truth: {artifact_id: [{"label": str, "box": [x1,y1,x2,y2]}]}
    annotations: Mapped[dict[str, Any]] = mapped_column(default=dict)


class Experiment(Base, TimestampMixin):
    __tablename__ = "experiments"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), index=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    number: Mapped[int] = mapped_column(Integer)  # per-project human number (#042)
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(20), default="draft", index=True)
    detector_id: Mapped[str] = mapped_column(String(100))
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id"))
    configuration: Mapped[dict[str, Any]] = mapped_column(default=dict)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (UniqueConstraint("project_id", "number", name="uq_experiment_number"),)


class ExperimentRun(Base, TimestampMixin):
    """One reproducible execution of an experiment. Pins every version that
    can change the numbers."""

    __tablename__ = "experiment_runs"
    # At most one active run per experiment, enforced where a check-then-insert
    # in the route cannot be. piggy: create_all adds it to new databases only;
    # existing ones get it with the first Alembic migration.
    __table_args__ = (
        Index("uq_active_run", "experiment_id", unique=True,
              sqlite_where=text("status IN ('queued', 'running')"),
              postgresql_where=text("status IN ('queued', 'running')")),
    )
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), index=True)
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.id"), index=True)
    status: Mapped[str] = mapped_column(String(20), default="queued", index=True)
    seed: Mapped[int] = mapped_column(Integer)
    code_version: Mapped[str] = mapped_column(String(64), default="")
    detector_version: Mapped[str] = mapped_column(String(100), default="")
    dataset_version: Mapped[int] = mapped_column(Integer, default=1)
    configuration: Mapped[dict[str, Any]] = mapped_column(default=dict)
    error: Mapped[str] = mapped_column(Text, default="")
    log: Mapped[list[str]] = mapped_column(default=list)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Pattern(Base, TimestampMixin):
    __tablename__ = "patterns"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), index=True)
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.id"), index=True)
    run_id: Mapped[str | None] = mapped_column(ForeignKey("experiment_runs.id"))
    version: Mapped[int] = mapped_column(Integer, default=1)
    artifact_id: Mapped[str | None] = mapped_column(ForeignKey("artifacts.id"))
    generation_parameters: Mapped[dict[str, Any]] = mapped_column(default=dict)
    metrics: Mapped[dict[str, Any]] = mapped_column(default=dict)


class Evaluation(Base, TimestampMixin):
    __tablename__ = "evaluations"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), index=True)
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.id"), index=True)
    run_id: Mapped[str | None] = mapped_column(ForeignKey("experiment_runs.id"))
    pattern_id: Mapped[str | None] = mapped_column(ForeignKey("patterns.id"))
    baseline_metrics: Mapped[dict[str, Any]] = mapped_column(default=dict)
    # An unoptimized pattern of identical size/palette/placement. Empty when
    # the run disabled the control arm - see docs/experiments.md.
    control_metrics: Mapped[dict[str, Any]] = mapped_column(default=dict)
    candidate_metrics: Mapped[dict[str, Any]] = mapped_column(default=dict)
    transformations: Mapped[dict[str, Any]] = mapped_column(default=dict)
    metrics: Mapped[dict[str, Any]] = mapped_column(default=dict)
    sample_count: Mapped[int] = mapped_column(Integer, default=0)


class PhysicalTest(Base, TimestampMixin):
    """A measurement taken with a real camera on a real garment/print."""

    __tablename__ = "physical_tests"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), index=True)
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.id"), index=True)
    pattern_id: Mapped[str | None] = mapped_column(ForeignKey("patterns.id"))
    garment_id: Mapped[str | None] = mapped_column(ForeignKey("garments.id"))
    # Which arm this measurement belongs to. A physical record whose arm is
    # unknown cannot be scored: "the detector saw a person" means nothing
    # until you say whether a pattern was on the subject. Rows predating this
    # column read back as "unspecified" and are excluded from the score.
    arm: Mapped[str] = mapped_column(String(20), default="unspecified", index=True)
    camera: Mapped[str] = mapped_column(String(200), default="")
    resolution: Mapped[str] = mapped_column(String(32), default="")
    fps: Mapped[float] = mapped_column(Float, default=0.0)
    distance_m: Mapped[float | None] = mapped_column(Float)
    angle_deg: Mapped[float | None] = mapped_column(Float)
    lighting: Mapped[str] = mapped_column(String(200), default="")
    environment: Mapped[str] = mapped_column(String(200), default="")
    frame_count: Mapped[int] = mapped_column(Integer, default=0)
    result: Mapped[dict[str, Any]] = mapped_column(default=dict)
    artifact_id: Mapped[str | None] = mapped_column(ForeignKey("artifacts.id"))
    notes: Mapped[str] = mapped_column(Text, default="")


class Garment(Base, TimestampMixin):
    """VEIL Wear physical item metadata (the QR/NFC payload target)."""

    __tablename__ = "garments"
    __table_args__ = (UniqueConstraint("organization_id", "sku", name="uq_garment_org_sku"),)
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), index=True)
    pattern_id: Mapped[str] = mapped_column(ForeignKey("patterns.id"), index=True)
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.id"))
    # Unique per organization, like every lookup of it. A global constraint
    # let one tenant learn which SKUs another had registered (201 vs error).
    # piggy: databases created before this keep the global constraint until
    # Alembic lands (README "Deliberately not built"); the route answers 409
    # either way, never 500.
    sku: Mapped[str] = mapped_column(String(64))
    batch_id: Mapped[str] = mapped_column(String(64))
    product_type: Mapped[str] = mapped_column(String(64))  # tshirt | hoodie | jacket
    material: Mapped[str] = mapped_column(String(200), default="")
    print_method: Mapped[str] = mapped_column(String(100), default="")
    tested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    test_conditions: Mapped[dict[str, Any]] = mapped_column(default=dict)


class Report(Base, TimestampMixin):
    __tablename__ = "reports"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), index=True)
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.id"), index=True)
    run_id: Mapped[str | None] = mapped_column(ForeignKey("experiment_runs.id"))
    payload: Mapped[dict[str, Any]] = mapped_column(default=dict)  # the JSON report
    pdf_artifact_id: Mapped[str | None] = mapped_column(ForeignKey("artifacts.id"))


class AuditEvent(Base, TimestampMixin):
    __tablename__ = "audit_events"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), index=True)
    user_id: Mapped[str | None] = mapped_column(ForeignKey("users.id"))
    action: Mapped[str] = mapped_column(String(64))
    target: Mapped[str] = mapped_column(String(200), default="")
    detail: Mapped[dict[str, Any]] = mapped_column(default=dict)
