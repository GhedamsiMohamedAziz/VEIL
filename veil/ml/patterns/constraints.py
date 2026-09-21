"""Physical-realizability constraints on a candidate pattern.

A pattern that only exists as float pixels is not a product. These penalties
push the optimizer towards something a textile printer can actually produce:
a limited ink palette, and no per-pixel high-frequency noise that a weave
would destroy.
"""

from __future__ import annotations

import torch

# A small, deliberately coarse ink set. Sublimation/screen printing on
# polyester reproduces saturated primaries far better than pastels.
PRINTABLE_PALETTE: list[tuple[float, float, float]] = [
    (0.05, 0.05, 0.05), (0.95, 0.95, 0.95), (0.80, 0.10, 0.10),
    (0.10, 0.45, 0.85), (0.15, 0.60, 0.25), (0.95, 0.80, 0.10),
    (0.55, 0.20, 0.65), (0.95, 0.50, 0.10), (0.45, 0.45, 0.45),
]


def total_variation(pattern: torch.Tensor) -> torch.Tensor:
    """Mean absolute neighbour difference. Lower = smoother = printable."""
    dx = (pattern[:, :, 1:] - pattern[:, :, :-1]).abs().mean()
    dy = (pattern[:, 1:, :] - pattern[:, :-1, :]).abs().mean()
    return dx + dy


def non_printability(pattern: torch.Tensor, palette: list[tuple[float, float, float]] | None = None) -> torch.Tensor:
    """Sharma et al.'s non-printability score: distance from every pixel to
    the nearest reproducible ink. Zero when the pattern uses palette colours."""
    colors = torch.tensor(palette or PRINTABLE_PALETTE, device=pattern.device, dtype=pattern.dtype)
    flat = pattern.reshape(3, -1).t()  # [N,3]
    dist = torch.cdist(flat.unsqueeze(0), colors.unsqueeze(0)).squeeze(0)  # [N,K]
    return dist.min(dim=1).values.mean()


def quantize_to_palette(pattern: torch.Tensor, palette: list[tuple[float, float, float]] | None = None) -> torch.Tensor:
    """Snap to the ink set. Applied once at export so the shipped artifact is
    exactly what gets printed (non-differentiable, hence export-only)."""
    colors = torch.tensor(palette or PRINTABLE_PALETTE, device=pattern.device, dtype=pattern.dtype)
    flat = pattern.reshape(3, -1).t()
    idx = torch.cdist(flat.unsqueeze(0), colors.unsqueeze(0)).squeeze(0).argmin(dim=1)
    return colors[idx].t().reshape(pattern.shape)
