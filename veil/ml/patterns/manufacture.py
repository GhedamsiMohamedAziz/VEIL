"""Turning an optimized pattern into something a factory can make.

A pattern that only exists as float pixels is not a garment. Two production
routes, with genuinely different constraints:

* **print** (DTG / dye sublimation) - a fine ink palette, resolution limited
  by the printer, colour limited by the substrate.
* **knit / weave** (what a jacquard machine does) - each "pixel" is a stitch,
  the palette is the yarn cones loaded on the machine (typically 2-6), and
  the gauge fixes the stitch count per centimetre. This is a far coarser
  constraint than printing and it changes what the optimizer should be
  producing.

Quantizing to the production constraint *after* optimization loses effect.
`effect_retained` measures how much, so the loss is reported rather than
discovered on the first sample garment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from veil.ml.patterns.constraints import PRINTABLE_PALETTE

# A jacquard knit carries a handful of yarn colours, not a gamut. These are
# ordinary mill-stock shades; a real run uses the mill's card.
KNIT_YARNS: list[tuple[float, float, float]] = [
    (0.06, 0.06, 0.07),   # black
    (0.93, 0.92, 0.89),   # ecru
    (0.72, 0.13, 0.15),   # red
    (0.12, 0.28, 0.60),   # navy
    (0.20, 0.50, 0.28),   # green
    (0.90, 0.72, 0.16),   # ochre
]


@dataclass
class ProductionSpec:
    """How the pattern will physically be made."""

    method: str = "knit"                 # knit | weave | print
    # Knit/weave: stitches (or picks) per centimetre. A 12-gauge machine is
    # roughly 5 stitches/cm; finer gauges cost more and run slower.
    stitches_per_cm: float = 5.0
    # Print: dots per inch the file is exported at.
    dpi: int = 300
    # Finished artwork size on the garment, in centimetres.
    width_cm: float = 30.0
    height_cm: float = 40.0
    yarns: list[tuple[float, float, float]] = field(default_factory=lambda: list(KNIT_YARNS))
    max_yarns: int = 6

    def as_dict(self) -> dict[str, Any]:
        return {**self.__dict__, "yarns": [list(y) for y in self.yarns]}

    @property
    def grid(self) -> tuple[int, int]:
        """Pattern resolution in production units (stitches, or print pixels)."""
        if self.method == "print":
            return (round(self.height_cm / 2.54 * self.dpi),
                    round(self.width_cm / 2.54 * self.dpi))
        return (max(round(self.height_cm * self.stitches_per_cm), 1),
                max(round(self.width_cm * self.stitches_per_cm), 1))

    @property
    def palette(self) -> list[tuple[float, float, float]]:
        if self.method == "print":
            return list(PRINTABLE_PALETTE)
        return self.yarns[: self.max_yarns]


def _snap(pattern: torch.Tensor, palette: list[tuple[float, float, float]]) -> torch.Tensor:
    colors = torch.tensor(palette, device=pattern.device, dtype=pattern.dtype)
    flat = pattern.reshape(3, -1).t()
    index = torch.cdist(flat.unsqueeze(0), colors.unsqueeze(0)).squeeze(0).argmin(dim=1)
    return colors[index].t().reshape(pattern.shape)


def to_production(pattern: torch.Tensor, spec: ProductionSpec) -> torch.Tensor:
    """Resample to the production grid, then snap to the available colours.

    Order matters: downsampling first means each stitch averages the pixels
    it covers, and only then commits to a yarn. Snapping first and then
    downsampling would blend yarns into colours the machine cannot make.
    """
    height, width = spec.grid
    resampled = F.interpolate(pattern.unsqueeze(0), size=(height, width),
                              mode="area").squeeze(0)
    return _snap(resampled.clamp(0, 1), spec.palette)


def to_artwork(pattern: torch.Tensor, spec: ProductionSpec, upscale: int = 1) -> torch.Tensor:
    """The production pattern rendered back to image size for evaluation, and
    for the artwork file. `upscale` renders each stitch as a block."""
    production = to_production(pattern, spec)
    if upscale > 1:
        production = F.interpolate(production.unsqueeze(0), scale_factor=upscale,
                                   mode="nearest").squeeze(0)
    return production


def colour_usage(production: torch.Tensor, spec: ProductionSpec) -> list[dict[str, Any]]:
    """Which yarns the pattern actually uses, and how much of each.

    A pattern using one yarn for 0.3% of stitches still costs a cone and a
    feeder on the machine; the mill needs to know before quoting.
    """
    colors = torch.tensor(spec.palette, device=production.device, dtype=production.dtype)
    flat = production.reshape(3, -1).t()
    index = torch.cdist(flat.unsqueeze(0), colors.unsqueeze(0)).squeeze(0).argmin(dim=1)
    total = index.numel()
    usage = []
    for slot, colour in enumerate(spec.palette):
        count = int((index == slot).sum())
        if count:
            usage.append({
                "slot": slot,
                "rgb": [round(c, 4) for c in colour],
                "hex": "#%02x%02x%02x" % tuple(round(c * 255) for c in colour),
                "stitches": count,
                "fraction": count / total,
            })
    return sorted(usage, key=lambda u: -u["fraction"])
