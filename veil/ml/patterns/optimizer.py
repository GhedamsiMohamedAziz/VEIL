"""Expectation-over-transformation pattern optimization.

Objective (docs/ml-pipeline.md):

    P* = argmin_P  E_theta[ s(f(T(P, theta))) ] + lambda_tv*TV(P) + lambda_nps*NPS(P)

where s is the detector's confidence for the target label, T the simulation
pipeline and theta a transformation drawn from the experiment's spec.

Two search strategies, chosen by the detector's own `differentiable` flag:

* gradient  - backprop through the renderer, transforms and detector head.
* evolution - a (1+1) evolution strategy with smoothed mutations, for
  detectors that expose no gradients (or run behind an API).

Both report the same history, so an experiment's report does not depend on
which one ran.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from veil.ml.detectors.base import Detector
from veil.ml.patterns import constraints, generator
from veil.ml.simulation import transforms as T
from veil.ml.simulation.pipeline import render_scene
from veil.ml.simulation.renderer import Placement


@dataclass
class OptimizationConfig:
    target_label: str = "person"
    pattern_size: int = 128
    init_method: str = "palette_noise"
    iterations: int = 30
    batch_transforms: int = 4  # transformation draws averaged per step
    learning_rate: float = 0.05
    tv_weight: float = 0.05
    nps_weight: float = 0.05
    step_size: float = 0.08  # evolution-strategy mutation scale
    seed: int = 42

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> "OptimizationConfig":
        if not config:
            return cls()
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in config.items() if k in known})

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class OptimizationResult:
    pattern: torch.Tensor
    history: list[dict[str, float]] = field(default_factory=list)
    strategy: str = ""
    initial_loss: float = 0.0
    final_loss: float = 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "iterations": len(self.history),
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "improvement": self.initial_loss - self.final_loss,
            "history": self.history,
        }


_REPORT_STEP = 1_000_003  # seeds the reporting draw; far outside any iteration count


def _penalties(pattern: torch.Tensor, cfg: OptimizationConfig) -> torch.Tensor:
    return (
        cfg.tv_weight * constraints.total_variation(pattern)
        + cfg.nps_weight * constraints.non_printability(pattern)
    )


def _expected_score(
    detector: Detector,
    images: torch.Tensor,
    pattern: torch.Tensor,
    placement: Placement | list[Placement],
    spec: T.TransformSpec,
    cfg: OptimizationConfig,
    step: int,
    differentiable: bool,
) -> torch.Tensor:
    """E_theta[ detector confidence ] over `cfg.batch_transforms` draws."""
    # A fresh sub-seed per step: the expectation is estimated over *different*
    # transformations each iteration, which is what prevents the pattern from
    # overfitting one pose. The seed stays a pure function of (seed, step).
    draw_spec = T.TransformSpec(**{**spec.as_dict(), "mode": "random", "samples": cfg.batch_transforms})
    params_list = draw_spec.sample(seed=cfg.seed + step)
    scores = []
    for params in params_list:
        scene = render_scene(images, pattern, placement, params, seed=cfg.seed + step)
        if differentiable:
            scores.append(detector.score(scene, cfg.target_label).mean())
        else:
            summary = detector.evaluate(scene, cfg.target_label)
            scores.append(torch.tensor(float(sum(summary["max_scores"]) / max(len(summary["max_scores"]), 1))))
    return torch.stack(scores).mean()


def optimize(
    detector: Detector,
    images: torch.Tensor,
    placement: Placement | list[Placement],
    spec: T.TransformSpec,
    cfg: OptimizationConfig,
    on_iteration: Callable[[int, dict[str, float]], None] | None = None,
) -> OptimizationResult:
    # On the images' device: a CPU pattern under a GPU detector round-trips the
    # whole scene batch every step.
    pattern = generator.initialize(cfg.pattern_size, cfg.init_method, cfg.seed).to(images.device)
    differentiable = bool(getattr(detector, "differentiable", False))
    strategy = "gradient" if differentiable else "evolution"
    history: list[dict[str, float]] = []

    def loss_of(p: torch.Tensor, step: int) -> torch.Tensor:
        return _expected_score(detector, images, p, placement, spec, cfg, step, differentiable) + _penalties(p, cfg)

    start = pattern.clone()
    if differentiable:
        param = pattern.clone().requires_grad_(True)
        opt = torch.optim.Adam([param], lr=cfg.learning_rate)
        for step in range(cfg.iterations):
            opt.zero_grad()
            loss = loss_of(param, step)
            loss.backward()
            opt.step()
            with torch.no_grad():
                param.clamp_(0, 1)
            value = float(loss.detach())
            record = {"iteration": step, "loss": value}
            history.append(record)
            if on_iteration:
                on_iteration(step, record)
        # The final iterate, not the step with the lowest loss: each step's loss
        # is measured on its own transformation draw, so the minimum picks the
        # easiest draw rather than the best pattern.
        pattern = param.detach()
    else:
        gen = torch.Generator().manual_seed(cfg.seed)
        with torch.no_grad():
            current = pattern
            for step in range(cfg.iterations):
                # Smooth mutation: pixel-level noise would not survive print.
                coarse = torch.randn(3, 16, 16, generator=gen)
                mutation = torch.nn.functional.interpolate(
                    coarse.unsqueeze(0), size=current.shape[-2:], mode="bilinear", align_corners=False
                ).squeeze(0).to(current.device)
                candidate = (current + cfg.step_size * mutation).clamp(0, 1)
                # Both scored on this step's transformation draw. Comparing the
                # candidate's draw to an older one accepts whatever met the
                # easiest transformations, not whatever is better.
                best_loss = float(loss_of(current, step))
                value = float(loss_of(candidate, step))
                if value < best_loss:
                    best_loss, current = value, candidate
                record = {"iteration": step, "loss": value, "best_loss": best_loss}
                history.append(record)
                if on_iteration:
                    on_iteration(step, record)
            pattern = current

    # What gets reported is measured once, after the fact: start and result,
    # both snapped to the palette (the snap can erase the effect, and only the
    # snapped pattern is ever exported), on one transformation draw that no
    # iteration used. The minimum of N noisy per-step losses would show an
    # "improvement" for an optimizer that did nothing.
    result = constraints.quantize_to_palette(pattern.detach())
    with torch.no_grad():
        initial = float(loss_of(constraints.quantize_to_palette(start), _REPORT_STEP))
        final = float(loss_of(result, _REPORT_STEP))
    return OptimizationResult(
        pattern=result, history=history, strategy=strategy,
        initial_loss=initial, final_loss=final,
    )
