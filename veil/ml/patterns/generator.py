"""Candidate pattern initialization."""

from __future__ import annotations

import torch

from veil.ml.patterns.constraints import PRINTABLE_PALETTE


def initialize(
    size: int = 128, method: str = "palette_noise", seed: int = 0, device: str | torch.device = "cpu"
) -> torch.Tensor:
    """Return a [3,size,size] pattern in [0,1].

    Initialization matters more than it looks: starting from blocky palette
    colours instead of white noise gives the optimizer a printable starting
    point and converges to patterns that survive the blur of a real camera.
    """
    gen = torch.Generator().manual_seed(seed)
    if method == "uniform_noise":
        pattern = torch.rand(3, size, size, generator=gen)
    elif method == "gray":
        pattern = torch.full((3, size, size), 0.5)
    elif method == "palette_noise":
        colors = torch.tensor(PRINTABLE_PALETTE)
        blocks = max(size // 16, 2)
        idx = torch.randint(len(colors), (blocks, blocks), generator=gen)
        small = colors[idx].permute(2, 0, 1)  # [3,blocks,blocks]
        pattern = torch.nn.functional.interpolate(
            small.unsqueeze(0), size=(size, size), mode="nearest"
        ).squeeze(0)
    else:
        raise ValueError(f"unknown init method: {method}")
    return pattern.clamp(0, 1).to(device)
