"""Differentiable compositing of a pattern onto a scene.

The pattern is warped into a target region with `grid_sample`, so gradients
flow from the detector's score back to the pattern pixels. The same warp
carries the cloth-deformation displacement, which is what makes the
optimization "physical" rather than a pure pixel attack.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from veil.ml.detectors.base import Detection


@dataclass(frozen=True)
class Placement:
    """Where the print sits, in normalized image coordinates (0..1)."""

    cx: float = 0.5
    cy: float = 0.5
    width: float = 0.3
    height: float = 0.3
    rotation_deg: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_detection(cls, det: Detection, image_hw: tuple[int, int], coverage: float = 0.45) -> "Placement":
        """Put the print on the torso of a detected box.

        `coverage` is the fraction of the box's area the garment print takes
        up; the centre sits slightly above the box's middle, where a chest
        print would be.
        """
        h, w = image_hw
        x1, y1, x2, y2 = det.box
        bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
        return cls(
            cx=((x1 + x2) / 2) / w,
            cy=(y1 + bh * 0.42) / h,
            width=(bw * coverage) / w,
            height=(bh * coverage) / h,
        )


def paste(
    images: torch.Tensor,
    pattern: torch.Tensor,
    placement: "Placement | list[Placement]",
    deformation: float = 0.0,
    seed: int = 0,
) -> torch.Tensor:
    """Composite `pattern` ([3,ph,pw] in [0,1]) into every image of the batch.

    `placement` may be a list with one entry per image: different subjects
    stand in different places, and training on many subjects - the only cure
    for single-image overfitting - needs the print on each one's own torso.
    """
    if pattern.dim() != 3 or pattern.shape[0] != 3:
        raise ValueError(f"pattern must be [3,H,W], got {tuple(pattern.shape)}")
    if isinstance(placement, (list, tuple)):
        if len(placement) != images.shape[0]:
            raise ValueError(f"{len(placement)} placements for {images.shape[0]} images")
        return torch.cat([
            paste(images[i : i + 1], pattern, placement[i], deformation, seed)
            for i in range(images.shape[0])
        ], dim=0)
    b, _, h, w = images.shape
    device, dtype = images.device, images.dtype

    # Output-pixel grid in normalized [-1,1] coords.
    ys = torch.linspace(-1, 1, h, device=device, dtype=dtype).view(h, 1).expand(h, w)
    xs = torch.linspace(-1, 1, w, device=device, dtype=dtype).view(1, w).expand(h, w)

    # Map scene coords -> pattern coords (inverse of the placement box).
    cx, cy = placement.cx * 2 - 1, placement.cy * 2 - 1
    half_w, half_h = max(placement.width, 1e-3), max(placement.height, 1e-3)
    u = (xs - cx) / half_w
    v = (ys - cy) / half_h

    if placement.rotation_deg:
        t = torch.tensor(placement.rotation_deg * torch.pi / 180.0, device=device, dtype=dtype)
        u, v = u * torch.cos(t) + v * torch.sin(t), -u * torch.sin(t) + v * torch.cos(t)

    if deformation:
        gen = torch.Generator(device="cpu").manual_seed(seed)
        freq = (3.0 + 5.0 * torch.rand(2, generator=gen)).to(device=device, dtype=dtype)
        amp = deformation / half_h
        u = u + amp * torch.sin(freq[0] * v)
        v = v + amp * torch.sin(freq[1] * u)

    grid = torch.stack([u, v], dim=-1).unsqueeze(0).expand(b, h, w, 2)
    warped = F.grid_sample(
        pattern.unsqueeze(0).expand(b, -1, -1, -1), grid,
        mode="bilinear", padding_mode="zeros", align_corners=False,
    )
    inside = ((u.abs() <= 1.0) & (v.abs() <= 1.0)).to(dtype).view(1, 1, h, w)
    return images * (1 - inside) + warped * inside
