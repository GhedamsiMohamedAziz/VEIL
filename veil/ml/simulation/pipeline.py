"""Compose a scene: image -> pattern paste -> camera transformations -> detector."""

from __future__ import annotations

from collections.abc import Iterator

import torch

from veil.ml.simulation import transforms as T
from veil.ml.simulation.renderer import Placement, paste


def transformed_views(
    images: torch.Tensor, spec: T.TransformSpec, seed: int = 0, count: int | None = None
) -> Iterator[tuple[T.TransformParams, torch.Tensor]]:
    """Yield (params, transformed batch) for every point in the spec."""
    for params in spec.sample(seed=seed, count=count):
        yield params, T.apply(images, params, seed=seed)


def render_scene(
    images: torch.Tensor,
    pattern: torch.Tensor | None,
    placement: Placement | None,
    params: T.TransformParams,
    seed: int = 0,
) -> torch.Tensor:
    """One full forward pass of the physical-simulation chain.

    The pattern is pasted *before* the camera transformations, because in the
    real world the camera sees a garment that is already deformed and lit -
    it does not transform the print separately from the scene.
    """
    scene = images
    if pattern is not None and placement is not None:
        scene = paste(scene, pattern, placement, deformation=params.deformation, seed=seed)
    return T.apply(scene, params, seed=seed)
