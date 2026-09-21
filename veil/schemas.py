"""Public API contracts. These types are the API - internal ML structures
never leak through them."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# --- input -------------------------------------------------------------


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=5000)


class DatasetCreate(BaseModel):
    project_id: str
    name: str = Field(min_length=1, max_length=200)
    source: str = Field(
        min_length=3, max_length=1000,
        description="Where the data came from. Required: an experiment whose "
                    "inputs have no recorded provenance is not reproducible.",
    )
    license: str = Field(default="", max_length=200)
    contains_people: bool = Field(
        default=False, description="Whether the imagery depicts identifiable people."
    )
    consent: str = Field(
        default="", max_length=2000,
        description="The basis for holding this imagery (consent, licence, "
                    "own staff, synthetic). Required when contains_people is true.",
    )
    artifact_ids: list[str] = Field(default_factory=list)
    annotations: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)


class TransformConfig(BaseModel):
    rotation_deg: list[float] = [-30.0, 0.0, 30.0]
    scale: list[float] = [0.8, 1.0, 1.2]
    translate: list[float] = [0.0]
    perspective: list[float] = [0.0, 0.15]
    brightness: list[float] = [0.6, 1.0, 1.4]
    contrast: list[float] = [1.0]
    blur_sigma: list[float] = [0.0, 1.0]
    noise_std: list[float] = [0.0, 0.02]
    deformation: list[float] = [0.0, 0.05]
    mode: str = Field(default="grid", pattern="^(grid|random)$")
    samples: int = Field(default=16, ge=1, le=512)
    max_samples: int = Field(default=256, ge=1, le=4096)


class OptimizationConfigIn(BaseModel):
    pattern_size: int = Field(default=128, ge=16, le=512)
    init_method: str = Field(default="palette_noise", pattern="^(palette_noise|uniform_noise|gray)$")
    iterations: int = Field(default=30, ge=1, le=1000)
    batch_transforms: int = Field(default=4, ge=1, le=64)
    learning_rate: float = Field(default=0.05, gt=0, le=1.0)
    tv_weight: float = Field(default=0.05, ge=0, le=10)
    nps_weight: float = Field(default=0.05, ge=0, le=10)
    step_size: float = Field(default=0.08, gt=0, le=1.0)


class ExperimentConfig(BaseModel):
    target_label: str = Field(default="person", max_length=100)
    control_draws: int = Field(
        default=3,
        ge=0,
        le=10,
        description=(
            "How many unoptimized control patterns to sweep alongside the "
            "candidate. Each is identical in size, palette and placement and "
            "differs only in its random draw. 1 separates the pattern's effect "
            "from simple occlusion; 2+ also measures the spread across random "
            "patterns, which is what distinguishes a real effect from luck. "
            "0 disables the control arm - the run then reports that no "
            "attribution can be made."
        ),
    )
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    image_size: int = Field(default=320, ge=64, le=1024)
    transfer_detectors: list[str] = Field(
        default_factory=list,
        max_length=5,
        description=(
            "Detectors the finished pattern is also measured against, with the "
            "same three-arm discipline and the same placement. This is how the "
            "claim 'a pattern optimized against one model does not transfer' "
            "becomes a measurement instead of an assumption."
        ),
    )
    transforms: TransformConfig = Field(default_factory=TransformConfig)
    optimization: OptimizationConfigIn = Field(default_factory=OptimizationConfigIn)


class ExperimentCreate(BaseModel):
    project_id: str
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=5000)
    detector_id: str
    dataset_id: str
    configuration: ExperimentConfig = Field(default_factory=ExperimentConfig)


class ProductionRequest(BaseModel):
    method: str = Field(default="knit", pattern="^(knit|weave|print)$")
    stitches_per_cm: float = Field(default=5.0, gt=0, le=40)
    dpi: int = Field(default=300, ge=72, le=1200)
    width_cm: float = Field(default=30.0, gt=0, le=200)
    height_cm: float = Field(default=40.0, gt=0, le=200)
    max_yarns: int = Field(default=6, ge=2, le=12)


class QualifyRequest(BaseModel):
    detector_id: str
    target_label: str = Field(default="person", max_length=100)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    image_size: int = Field(default=320, ge=64, le=1024)


class RunCreate(BaseModel):
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    configuration: ExperimentConfig | None = Field(
        default=None, description="Overrides the experiment's stored configuration for this run"
    )


class PhysicalTestCreate(BaseModel):
    experiment_id: str
    arm: str = Field(
        pattern="^(baseline|control|candidate)$",
        description=(
            "What was on the subject when this was measured. 'baseline' - "
            "nothing; 'control' - an unoptimized printed pattern; 'candidate' "
            "- the optimized one. Required, because a detection rate without "
            "it cannot be interpreted, and only 'candidate' records contribute "
            "to physical robustness."
        ),
    )
    pattern_id: str | None = None
    garment_id: str | None = None
    camera: str = Field(default="", max_length=200)
    resolution: str = Field(default="", max_length=32)
    fps: float = Field(default=0.0, ge=0)
    distance_m: float | None = Field(default=None, ge=0, le=1000)
    angle_deg: float | None = Field(default=None, ge=-180, le=180)
    lighting: str = Field(default="", max_length=200)
    environment: str = Field(default="", max_length=200)
    frame_count: int = Field(default=0, ge=0)
    result: dict[str, Any] = Field(default_factory=dict)
    artifact_id: str | None = None
    notes: str = Field(default="", max_length=5000)


class GarmentCreate(BaseModel):
    pattern_id: str
    sku: str = Field(min_length=1, max_length=64)
    batch_id: str = Field(min_length=1, max_length=64)
    product_type: str = Field(pattern="^(tshirt|hoodie|jacket|accessory)$")
    material: str = Field(default="", max_length=200)
    print_method: str = Field(default="", max_length=100)
    test_conditions: dict[str, Any] = Field(default_factory=dict)


class OrganizationCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    email: str = Field(min_length=3, max_length=320, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


# --- output ------------------------------------------------------------


class ProjectOut(ORMModel):
    id: str
    name: str
    description: str
    created_at: datetime


class ArtifactOut(ORMModel):
    id: str
    kind: str
    media_type: str
    size_bytes: int
    sha256: str
    created_at: datetime


class DatasetOut(ORMModel):
    id: str
    project_id: str
    name: str
    version: int
    source: str
    license: str
    contains_people: bool
    consent: str
    artifact_ids: list[str]
    created_at: datetime


class ExperimentOut(ORMModel):
    id: str
    project_id: str
    number: int
    name: str
    description: str
    status: str
    detector_id: str
    dataset_id: str
    configuration: dict[str, Any]
    created_at: datetime
    completed_at: datetime | None


class RunOut(ORMModel):
    id: str
    experiment_id: str
    status: str
    seed: int
    code_version: str
    detector_version: str
    dataset_version: int
    error: str
    log: list[str]
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class PatternOut(ORMModel):
    id: str
    experiment_id: str
    run_id: str | None
    version: int
    artifact_id: str | None
    generation_parameters: dict[str, Any]
    metrics: dict[str, Any]
    created_at: datetime


class EvaluationOut(ORMModel):
    id: str
    experiment_id: str
    run_id: str | None
    pattern_id: str | None
    baseline_metrics: dict[str, Any]
    control_metrics: dict[str, Any]
    candidate_metrics: dict[str, Any]
    transformations: dict[str, Any]
    sample_count: int
    metrics: dict[str, Any]
    created_at: datetime


class PhysicalTestOut(ORMModel):
    id: str
    experiment_id: str
    arm: str
    pattern_id: str | None
    garment_id: str | None
    camera: str
    resolution: str
    fps: float
    distance_m: float | None
    angle_deg: float | None
    lighting: str
    environment: str
    frame_count: int
    result: dict[str, Any]
    notes: str
    created_at: datetime


class GarmentOut(ORMModel):
    id: str
    pattern_id: str
    experiment_id: str
    sku: str
    batch_id: str
    product_type: str
    material: str
    print_method: str
    tested_at: datetime | None
    test_conditions: dict[str, Any]


class ReportOut(ORMModel):
    id: str
    experiment_id: str
    run_id: str | None
    payload: dict[str, Any]
    pdf_artifact_id: str | None
    created_at: datetime


class DetectorOut(BaseModel):
    id: str
    name: str
    version: str
    task: str
    labels: list[str]
    label_count: int
    differentiable: bool
    license: str
    source: str
    notes: str


class OrganizationOut(BaseModel):
    organization_id: str
    user_id: str
    api_key: str = Field(description="Shown once. Store it now; it is not recoverable.")
