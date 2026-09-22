"""Controlled, deterministic, differentiable image transformations.

Two properties matter and are both tested:

1. **Deterministic** - the same (spec, seed, index) always produces the same
   parameters and the same output tensor. Reproducibility is the product.
2. **Differentiable** - the same code path is used to optimize a pattern
   (expectation over transformations) and to evaluate it. If training and
   evaluation used different pipelines the numbers would be meaningless.

Geometry (rotation, scale, translation, perspective, cloth deformation) is
composed into a single sampling grid and applied with one `grid_sample`, so
the image is resampled once instead of five times.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

Range = tuple[float, float] | list[float]


@dataclass(frozen=True)
class TransformParams:
    """One concrete point in transformation space."""

    rotation_deg: float = 0.0
    scale: float = 1.0
    translate_x: float = 0.0  # fraction of width
    translate_y: float = 0.0
    perspective: float = 0.0  # 0 = none, 0.3 = strong tilt
    brightness: float = 1.0  # multiplicative
    contrast: float = 1.0
    blur_sigma: float = 0.0  # pixels
    noise_std: float = 0.0  # in [0,1] image units
    deformation: float = 0.0  # cloth wrinkle amplitude, fraction of size

    def as_dict(self) -> dict[str, float]:
        return asdict(self)

    def seed(self, base: int) -> int:
        digest = hashlib.sha256(f"{base}|{sorted(self.as_dict().items())}".encode()).digest()
        return int.from_bytes(digest[:4], "big")


@dataclass
class TransformSpec:
    """The transformation distribution an experiment is measured over.

    `mode="grid"` walks the cartesian product of the listed values - use it
    for evaluation, where "detection rate at 30 degrees" must be a real
    measurement rather than a random draw. `mode="random"` samples uniformly
    inside each [min, max] range - use it for optimization.
    """

    rotation_deg: Range = field(default_factory=lambda: [-30.0, 0.0, 30.0])
    scale: Range = field(default_factory=lambda: [0.8, 1.0, 1.2])
    translate: Range = field(default_factory=lambda: [0.0])
    perspective: Range = field(default_factory=lambda: [0.0, 0.15])
    brightness: Range = field(default_factory=lambda: [0.6, 1.0, 1.4])
    contrast: Range = field(default_factory=lambda: [1.0])
    blur_sigma: Range = field(default_factory=lambda: [0.0, 1.0])
    noise_std: Range = field(default_factory=lambda: [0.0, 0.02])
    deformation: Range = field(default_factory=lambda: [0.0, 0.05])
    mode: str = "grid"
    samples: int = 16  # used by mode="random"
    max_samples: int = 256  # hard ceiling on a grid sweep

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> "TransformSpec":
        if not config:
            return cls()
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in config.items() if k in known})

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def _grid(self) -> list[TransformParams]:
        axes = [
            list(self.rotation_deg), list(self.scale), list(self.translate),
            list(self.perspective), list(self.brightness), list(self.contrast),
            list(self.blur_sigma), list(self.noise_std), list(self.deformation),
        ]
        # A fixed-seed subsample, not the first N: product() varies the last
        # axis fastest, so truncating it drops whole values of the first axis
        # (the default grid lost rotation=+30 entirely). A stride would alias
        # with the axis lengths instead. Indices are drawn from the size and
        # decoded one by one: the product itself is never built, since axis
        # lengths come from user configuration.
        total = math.prod(len(axis) for axis in axes)
        picks = range(total) if total <= self.max_samples else sorted(
            random.Random(0).sample(range(total), self.max_samples))
        out = []
        for index in picks:
            combo = []
            for axis in reversed(axes):  # last axis fastest, like itertools.product
                index, position = divmod(index, len(axis))
                combo.append(axis[position])
            deform, noise, blur, con, bri, persp, tr, sc, rot = combo
            out.append(TransformParams(rot, sc, tr, tr, persp, bri, con, blur, noise, deform))
        return out

    def _random(self, seed: int) -> list[TransformParams]:
        gen = torch.Generator().manual_seed(seed)

        def draw(r: Range) -> float:
            vals = list(r)
            if len(vals) == 1:
                return float(vals[0])
            lo, hi = min(vals), max(vals)
            return float(lo + (hi - lo) * torch.rand(1, generator=gen).item())

        return [
            TransformParams(
                draw(self.rotation_deg), draw(self.scale), draw(self.translate),
                draw(self.translate), draw(self.perspective), draw(self.brightness),
                draw(self.contrast), draw(self.blur_sigma), draw(self.noise_std),
                draw(self.deformation),
            )
            for _ in range(self.samples)
        ]

    def sample(self, seed: int = 0, count: int | None = None) -> list[TransformParams]:
        params = self._grid() if self.mode == "grid" else self._random(seed)
        if count is not None and count < len(params):
            params = params[:count]
        return params


def _geometry_grid(params: TransformParams, shape: torch.Size, device, dtype) -> torch.Tensor:
    """Build one normalized sampling grid combining every geometric effect."""
    b, _, h, w = shape
    theta = math.radians(params.rotation_deg)
    s = max(params.scale, 1e-3)
    cos, sin = math.cos(theta) / s, math.sin(theta) / s
    # Inverse map: output coords -> input coords.
    mat = torch.tensor(
        [[cos, -sin, params.translate_x * 2.0], [sin, cos, params.translate_y * 2.0]],
        device=device, dtype=dtype,
    ).unsqueeze(0).expand(b, 2, 3)
    grid = F.affine_grid(mat, (b, 1, h, w), align_corners=False)

    if params.perspective:
        # Tilt about the horizontal axis: rows further "back" sample wider.
        y = grid[..., 1]
        factor = 1.0 + params.perspective * y
        grid = torch.stack([grid[..., 0] * factor, y], dim=-1)

    if params.deformation:
        # Smooth sinusoidal displacement standing in for cloth wrinkles.
        gen = torch.Generator(device="cpu").manual_seed(params.seed(1))
        freq = 2.0 + 6.0 * torch.rand(2, generator=gen).to(device=device, dtype=dtype)
        phase = 6.283 * torch.rand(2, generator=gen).to(device=device, dtype=dtype)
        gx, gy = grid[..., 0], grid[..., 1]
        dx = params.deformation * torch.sin(freq[0] * gy + phase[0])
        dy = params.deformation * torch.sin(freq[1] * gx + phase[1])
        grid = torch.stack([gx + dx, gy + dy], dim=-1)
    return grid


def _gaussian_blur(image: torch.Tensor, sigma: float) -> torch.Tensor:
    radius = max(int(3 * sigma), 1)
    x = torch.arange(-radius, radius + 1, device=image.device, dtype=image.dtype)
    kernel = torch.exp(-(x**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    c = image.shape[1]
    out = F.conv2d(
        F.pad(image, (radius, radius, 0, 0), mode="reflect"),
        kernel.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c,
    )
    return F.conv2d(
        F.pad(out, (0, 0, radius, radius), mode="reflect"),
        kernel.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c,
    )


def apply(image: torch.Tensor, params: TransformParams, seed: int = 0) -> torch.Tensor:
    """Apply one transformation point. Input and output are [B,3,H,W] in [0,1]."""
    if image.dim() != 4 or image.shape[1] != 3:
        raise ValueError(f"expected [B,3,H,W], got {tuple(image.shape)}")
    out = image
    needs_geometry = any(
        (params.rotation_deg, params.scale != 1.0, params.translate_x,
         params.translate_y, params.perspective, params.deformation)
    )
    if needs_geometry:
        grid = _geometry_grid(params, out.shape, out.device, out.dtype)
        out = F.grid_sample(out, grid, mode="bilinear", padding_mode="border", align_corners=False)
    if params.contrast != 1.0:
        mean = out.mean(dim=(1, 2, 3), keepdim=True)
        out = (out - mean) * params.contrast + mean
    if params.brightness != 1.0:
        out = out * params.brightness
    if params.blur_sigma > 0:
        out = _gaussian_blur(out, params.blur_sigma)
    if params.noise_std > 0:
        gen = torch.Generator(device="cpu").manual_seed(params.seed(seed))
        noise = torch.randn(out.shape, generator=gen).to(device=out.device, dtype=out.dtype)
        out = out + noise * params.noise_std
    # Same values as a hard clamp, but the gradient passes straight through. A
    # hard clamp gives saturated pixels a gradient of exactly zero, and at
    # brightness 1.4 most of a bright pattern saturates: those pixels would stay
    # frozen at whatever colour they started with.
    return out + (out.clamp(0.0, 1.0) - out).detach()
