"""The experiment engine: one reproducible run, start to finish.

A run is deterministic given (experiment configuration, seed, detector
version, dataset version, code version). Those five things are written to
`experiment_runs` before any work happens, so a result can always be traced
back to what produced it.

Stages: baseline sweep -> placement -> pattern optimization -> candidate
sweep -> comparison. Each stage emits a structured event.
"""

from __future__ import annotations

import random
import subprocess
import traceback
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

import torch

from veil import events
from veil.db import scoped, session_scope
from veil.ml.detectors import registry
from veil.ml.evaluation import comparison, metrics, robustness
from veil.ml.evaluation.detection import sweep
from veil.ml.patterns import constraints, generator
from veil.ml.patterns.optimizer import OptimizationConfig, optimize
from veil.ml.patterns.serialization import image_to_tensor, to_png
from veil.ml.simulation.renderer import Placement
from veil.ml.simulation.transforms import TransformSpec
from veil.models import (
    Artifact,
    Dataset,
    Evaluation,
    Experiment,
    ExperimentRun,
    Pattern,
    PhysicalTest,
    utcnow,
)
from veil.storage import put_bytes, read_bytes


def code_version() -> str:
    """Git commit if we are in a checkout, else the package version."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:  # pragma: no cover - git absent
        pass
    from veil import __version__

    return f"v{__version__}"


def _load_images(
    session, dataset: Dataset, size: int, original_sizes: dict[str, tuple[int, int]] | None = None,
) -> tuple[torch.Tensor, list[str]]:
    """Decode the dataset's artifacts into one padded batch.

    Images are resized to a common square so a batch can be formed; the
    resize factor is part of the recorded configuration. Pass a dict as
    `original_sizes` to receive each image's (height, width) before the resize.
    """
    tensors, ids = [], []
    for artifact_id in dataset.artifact_ids:
        artifact = session.get(Artifact, artifact_id)
        if artifact is None or artifact.organization_id != dataset.organization_id:
            continue
        if not artifact.media_type.startswith("image/"):
            continue  # piggy: video sweeps land in v0.2 (see docs/roadmap.md)
        tensor = image_to_tensor(read_bytes(artifact.uri), size=None)
        if original_sizes is not None:
            original_sizes[artifact.id] = (int(tensor.shape[-2]), int(tensor.shape[-1]))
        tensor = torch.nn.functional.interpolate(
            tensor, size=(size, size), mode="bilinear", align_corners=False
        )
        tensors.append(tensor)
        ids.append(artifact.id)
    if not tensors:
        raise ValueError("dataset contains no usable images")
    return torch.cat(tensors, dim=0), ids


def _annotations_at(
    annotations: dict[str, list[dict[str, Any]]],
    original_sizes: dict[str, tuple[int, int]], size: int,
) -> dict[str, list[dict[str, Any]]]:
    """Annotation boxes are drawn on the original photo; detections come from
    the `size`-square copy. Bring the boxes into that frame before matching."""
    scaled = {}
    for image_id, items in annotations.items():
        if image_id not in original_sizes:
            continue
        height, width = original_sizes[image_id]
        sx, sy = size / width, size / height
        scaled[image_id] = [
            {**a, "box": [a["box"][0] * sx, a["box"][1] * sy, a["box"][2] * sx, a["box"][3] * sy]}
            for a in items
        ]
    return scaled


def _placements_for(detector, images: torch.Tensor, target_label: str, threshold: float) -> list[Placement]:
    """Put the print where the detector currently sees the target, so the
    experiment tests the interesting region rather than a corner of the sky.

    One placement per image: subjects stand in different places, and image 0's
    torso coordinates are background in every other photo."""
    size = images.shape[-2:]
    placements = []
    for dets in detector.predict(images, threshold=threshold):
        hit = next((d for d in dets if d.label == target_label), None)
        placements.append(
            Placement.from_detection(hit, (int(size[0]), int(size[1]))) if hit
            else Placement(cx=0.5, cy=0.5, width=0.3, height=0.3)
        )
    return placements


def split_indices(count: int, seed: int) -> tuple[list[int], list[int], bool]:
    """Disjoint (optimize, evaluate) image indices, and whether they are disjoint.

    A pattern scored on the images it was optimized on reports a training
    score: controls are never optimized, so the candidate's advantage would be
    guaranteed rather than measured. One image cannot be split; the run still
    executes, and `held_out=False` travels with every number it produces."""
    if count < 2:
        return list(range(count)), list(range(count)), False
    order = list(range(count))
    random.Random(seed).shuffle(order)
    half = count // 2  # the evaluate side gets the odd image: it carries the claims
    return sorted(order[:half]), sorted(order[half:]), True


def evaluate_pattern(
    detector,
    images: torch.Tensor,
    spec: TransformSpec,
    target_label: str,
    pattern: torch.Tensor,
    placement: Placement | list[Placement],
    *,
    seed: int,
    threshold: float,
    image_ids: list[str],
    control_draws: int,
    pattern_size: int,
    init_method: str,
    baseline: list[dict[str, Any]] | None = None,
    note: Callable[[str], None] = lambda _message: None,
) -> dict[str, list]:
    """Measure one pattern against one detector, with the full three-arm
    discipline: baseline, N unoptimized controls, then the candidate.

    Transfer evaluation calls this with a different detector and the same
    pattern and placement. Sharing the function is the point - a transfer
    number produced by a weaker method than the primary one would be
    comparable to nothing.

    `baseline` can be supplied when the caller already swept it (the primary
    path needs it to choose the placement).
    """
    if baseline is None:
        baseline = sweep(detector, images, spec, target_label, seed=seed,
                         threshold=threshold, image_ids=image_ids)
        note(f"baseline sweep: detection rate "
             f"{metrics.summarize(baseline)['detection_rate']:.3f}")

    control_draws_records: list[list[dict[str, Any]]] = []
    for draw in range(control_draws):
        control_pattern = constraints.quantize_to_palette(
            generator.initialize(pattern_size, init_method, seed=seed + 100_000 + draw)
        )
        records = sweep(detector, images, spec, target_label, pattern=control_pattern,
                        placement=placement, seed=seed, threshold=threshold,
                        image_ids=image_ids)
        control_draws_records.append(records)
        note(f"control draw {draw}: detection rate "
             f"{metrics.summarize(records)['detection_rate']:.3f}")

    candidate = sweep(detector, images, spec, target_label, pattern=pattern,
                      placement=placement, seed=seed, threshold=threshold,
                      image_ids=image_ids)
    note(f"candidate sweep: detection rate "
         f"{metrics.summarize(candidate)['detection_rate']:.3f}")

    return {
        "baseline": baseline,
        "control_draws": control_draws_records,
        "control": [r for draw_records in control_draws_records for r in draw_records],
        "candidate": candidate,
    }


def execute_run(run_id: str) -> dict[str, Any]:
    """Execute a queued run. Safe to call from a worker thread."""
    with session_scope() as session:
        run = session.get(ExperimentRun, run_id)
        if run is None:
            raise KeyError(f"no such run: {run_id}")
        experiment = session.get(Experiment, run.experiment_id)
        assert experiment is not None
        dataset = session.get(Dataset, experiment.dataset_id)
        if dataset is None or dataset.organization_id != run.organization_id:
            raise ValueError("dataset not found in this organization")

        config = dict(run.configuration or {})
        target_label = config.get("target_label", "person")
        threshold = float(config.get("threshold", 0.5))
        image_size = int(config.get("image_size", 320))
        spec = TransformSpec.from_config(config.get("transforms"))
        opt_cfg = OptimizationConfig.from_config({**(config.get("optimization") or {}), "target_label": target_label, "seed": run.seed})
        # `control` (bool) is the pre-0.2 spelling; keep reading it so older
        # stored experiment configurations still run.
        if "control_draws" in config:
            control_draws = max(int(config["control_draws"]), 0)
        else:
            control_draws = 3 if config.get("control", True) else 0

        detector = registry.get(experiment.detector_id)
        run.detector_version = detector.metadata().version
        run.dataset_version = dataset.version
        run.code_version = code_version()
        run.status = "running"
        run.started_at = utcnow()
        experiment.status = "running"
        # Commit, not flush: a flushed-only status is invisible to the client
        # polling GET /runs/{id}, and on SQLite it holds the write lock for the
        # whole run. The session keeps its objects (expire_on_commit=False).
        session.commit()

        log: list[str] = []

        def note(message: str) -> None:
            log.append(f"{datetime.now(timezone.utc).isoformat()} {message}")
            run.log = list(log)
            session.commit()  # progress is only progress if another session can read it

        base = {"run_id": run.id, "experiment_id": experiment.id, "project_id": experiment.project_id}
        try:
            events.emit(events.EXPERIMENT_STARTED, **base, seed=run.seed, detector=experiment.detector_id)
            note(f"loading dataset {dataset.name} v{dataset.version}")
            original_sizes: dict[str, tuple[int, int]] = {}
            images, image_ids = _load_images(session, dataset, image_size, original_sizes)
            note(f"{images.shape[0]} image(s) at {image_size}px")

            placements = _placements_for(detector, images, target_label, threshold)
            fit_idx, eval_idx, held_out = split_indices(images.shape[0], run.seed)
            split = {
                "held_out": held_out,
                "optimize": [image_ids[i] for i in fit_idx],
                "evaluate": [image_ids[i] for i in eval_idx],
            }
            run.configuration = {**config, "split": split}
            note(f"split: optimize on {len(fit_idx)}, evaluate on {len(eval_idx)} image(s)"
                 + ("" if held_out else " - NOT held out: a single image cannot be split"))
            fit_images, fit_placements = images[fit_idx], [placements[i] for i in fit_idx]
            # From here on `images` is the evaluation side only: every reported
            # number comes from images the optimizer never saw.
            images, image_ids = images[eval_idx], split["evaluate"]
            placement = [placements[i] for i in eval_idx]

            events.emit(events.EVALUATION_STARTED, **base, stage="baseline")
            baseline = sweep(detector, images, spec, target_label, seed=run.seed,
                             threshold=threshold, image_ids=image_ids)
            note(f"baseline sweep: {len(baseline)} samples, "
                 f"detection rate {metrics.summarize(baseline)['detection_rate']:.3f}")

            events.emit(events.PATTERN_GENERATION_STARTED, **base, strategy=(
                "gradient" if detector.differentiable else "evolution"))

            def on_iteration(step: int, record: dict[str, float]) -> None:
                if step % 5 == 0 or step == opt_cfg.iterations - 1:
                    events.emit(events.PATTERN_ITERATION_COMPLETED, **base, **record)
                    note(f"iteration {step}: loss {record['loss']:.4f}")

            result = optimize(detector, fit_images, fit_placements, spec, opt_cfg, on_iteration=on_iteration)

            png = to_png(result.pattern)
            uri, digest, size_bytes = put_bytes(run.organization_id, png)
            artifact = Artifact(
                organization_id=run.organization_id, project_id=experiment.project_id,
                kind="pattern", uri=uri, media_type="image/png", size_bytes=size_bytes,
                sha256=digest, meta={"experiment_id": experiment.id, "run_id": run.id},
            )
            session.add(artifact)
            session.flush()

            version = 1 + session.query(Pattern).filter_by(experiment_id=experiment.id).count()
            pattern_row = Pattern(
                organization_id=run.organization_id, experiment_id=experiment.id, run_id=run.id,
                version=version, artifact_id=artifact.id,
                generation_parameters={
                    **opt_cfg.as_dict(),
                    # `placement` (first evaluated image) is the pre-split spelling,
                    # kept for stored readers; `placements` is the real record.
                    "placement": placement[0].as_dict(),
                    "placements": {i: p.as_dict() for i, p in zip(image_ids, placement)},
                    "split": split,
                    "transform_spec": spec.as_dict(),
                },
                metrics=result.summary(),
            )
            session.add(pattern_row)
            session.flush()

            events.emit(events.EVALUATION_STARTED, **base, stage="control")
            arms = evaluate_pattern(
                detector, images, spec, target_label, result.pattern, placement,
                seed=run.seed, threshold=threshold, image_ids=image_ids,
                control_draws=control_draws, pattern_size=opt_cfg.pattern_size,
                init_method=opt_cfg.init_method, baseline=baseline, note=note,
            )
            control_draws_records = arms["control_draws"]
            control = arms["control"]
            candidate = arms["candidate"]

            # Transfer: does the pattern do anything to a model it was never
            # optimized against? Measured with the same three-arm discipline
            # and the same placement - the print does not move between
            # detectors, only the model looking at it changes.
            transfer: dict[str, Any] = {}
            for transfer_id in config.get("transfer_detectors") or []:
                if transfer_id == experiment.detector_id:
                    continue  # that is the primary result, not a transfer
                try:
                    other = registry.get(transfer_id)
                except KeyError:
                    note(f"transfer: unknown detector {transfer_id}, skipped")
                    continue
                other_labels = other.metadata().labels
                if other_labels and target_label not in other_labels:
                    note(f"transfer: {transfer_id} cannot detect {target_label!r}, skipped")
                    transfer[transfer_id] = {
                        "available": False,
                        "reason": f"detector cannot produce the label {target_label!r}",
                    }
                    continue
                events.emit(events.EVALUATION_STARTED, **base, stage="transfer",
                            detector=transfer_id)
                note(f"transfer to {transfer_id}:")
                other_arms = evaluate_pattern(
                    other, images, spec, target_label, result.pattern, placement,
                    seed=run.seed, threshold=threshold, image_ids=image_ids,
                    control_draws=control_draws, pattern_size=opt_cfg.pattern_size,
                    init_method=opt_cfg.init_method, note=lambda m: note(f"  {m}"),
                )
                other_comparison = comparison.compare(
                    other_arms["baseline"], other_arms["candidate"],
                    other_arms["control"] or None, other_arms["control_draws"] or None,
                )
                transfer[transfer_id] = {
                    "available": True,
                    "detector_version": other.metadata().version,
                    "baseline": metrics.summarize(other_arms["baseline"]),
                    "control": metrics.summarize(other_arms["control"]) if other_arms["control"] else None,
                    "candidate": metrics.summarize(other_arms["candidate"]),
                    "attribution": other_comparison["attribution"],
                    "veil_score": robustness.veil_score(other_arms["candidate"]),
                }
                verdict = transfer[transfer_id]["attribution"].get("verdict") \
                    or transfer[transfer_id]["attribution"].get("reason")
                note(f"  -> {verdict}")
                events.emit(events.EVALUATION_COMPLETED, **base, stage="transfer",
                            detector=transfer_id,
                            candidate_rate=transfer[transfer_id]["candidate"]["detection_rate"])

            # Physical records only mean something once you know what was on
            # the subject. Only 'candidate' records score; 'unspecified' ones
            # (predating the arm column, or recorded without it) are counted
            # and reported, never scored.
            def as_record(test: PhysicalTest) -> dict[str, Any]:
                return {"detected": bool(test.result.get("detected")),
                        "max_score": float(test.result.get("max_score", 0.0)),
                        "detection_count": int(test.result.get("detection_count", 0)),
                        "boxes": [], "transform": {}, "image_id": None}

            by_arm: dict[str, list[dict[str, Any]]] = {}
            for test in session.scalars(
                scoped(PhysicalTest, run.organization_id).where(
                    PhysicalTest.experiment_id == experiment.id
                )
            ):
                by_arm.setdefault(test.arm or "unspecified", []).append(as_record(test))
            physical = by_arm.get("candidate", [])
            physical_summary = {
                "by_arm": {arm: metrics.summarize(records) for arm, records in by_arm.items()},
                "scored_arm": "candidate",
                "excluded": {
                    arm: len(records) for arm, records in by_arm.items()
                    if arm not in ("candidate",)
                },
                "note": (
                    "Only 'candidate' records contribute to physical robustness. "
                    "A measurement whose arm is unspecified cannot be interpreted: "
                    "a detection rate says nothing until you state what was on "
                    "the subject."
                ),
            }
            if by_arm:
                note("physical tests: " + ", ".join(
                    f"{arm}={len(records)}" for arm, records in sorted(by_arm.items())))

            annotations = _annotations_at(dataset.annotations or {}, original_sizes, image_size)
            evaluation = Evaluation(
                organization_id=run.organization_id, experiment_id=experiment.id, run_id=run.id,
                pattern_id=pattern_row.id,
                baseline_metrics=metrics.summarize(baseline),
                control_metrics=(
                    {
                        **metrics.summarize(control),
                        "spread": comparison.control_spread(
                            [metrics.summarize(d) for d in control_draws_records]
                        ),
                    }
                    if control
                    else {}
                ),
                candidate_metrics=metrics.summarize(candidate),
                transformations=spec.as_dict(),
                sample_count=len(candidate),
                metrics={
                    "comparison": comparison.compare(
                        baseline, candidate, control or None, control_draws_records or None
                    ),
                    "transfer": transfer,
                    "split": split,
                    "physical": physical_summary,
                    "veil_score": robustness.veil_score(candidate, physical or None),
                    "baseline_veil_score": robustness.veil_score(baseline),
                    "ground_truth": {
                        "baseline": metrics.ground_truth_metrics(baseline, annotations, target_label),
                        "candidate": metrics.ground_truth_metrics(candidate, annotations, target_label),
                    },
                    "records": {"baseline": baseline, "control": control,
                                "control_draws": control_draws_records, "candidate": candidate},
                },
            )
            session.add(evaluation)

            run.status = "completed"
            run.finished_at = utcnow()
            experiment.status = "completed"
            experiment.completed_at = run.finished_at
            attribution = evaluation.metrics["comparison"]["attribution"]
            note(f"attribution: {attribution.get('verdict', attribution.get('reason'))}")
            note("run completed")
            events.emit(events.EVALUATION_COMPLETED, **base,
                        baseline_rate=evaluation.baseline_metrics["detection_rate"],
                        control_rate=evaluation.control_metrics.get("detection_rate"),
                        candidate_rate=evaluation.candidate_metrics["detection_rate"],
                        attributable_drop=attribution.get("attributable_drop"))
            events.emit(events.EXPERIMENT_COMPLETED, **base)
            return {"run_id": run.id, "status": "completed", "evaluation_id": evaluation.id,
                    "pattern_id": pattern_row.id}
        except Exception as exc:  # noqa: BLE001 - the failure is the result
            run.status = "failed"
            run.error = f"{type(exc).__name__}: {exc}"
            run.finished_at = utcnow()
            experiment.status = "failed"
            note(f"FAILED: {run.error}")
            events.emit(events.EXPERIMENT_FAILED, **base, error=run.error,
                        traceback=traceback.format_exc(limit=5))
            return {"run_id": run.id, "status": "failed", "error": run.error}
