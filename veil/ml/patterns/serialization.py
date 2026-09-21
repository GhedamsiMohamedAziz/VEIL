"""Pattern <-> PNG. Lossless, 8-bit, exactly what goes to the printer."""

from __future__ import annotations

import io

import numpy as np
import torch
from PIL import Image


def to_png(pattern: torch.Tensor) -> bytes:
    if pattern.dim() != 3 or pattern.shape[0] != 3:
        raise ValueError(f"expected [3,H,W], got {tuple(pattern.shape)}")
    array = (pattern.detach().clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array.transpose(1, 2, 0), mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def from_png(data: bytes) -> torch.Tensor:
    image = Image.open(io.BytesIO(data)).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array.transpose(2, 0, 1)).contiguous()


def image_to_tensor(data: bytes, size: int | None = 640) -> torch.Tensor:
    """Decode an uploaded image to a [1,3,H,W] batch in [0,1]."""
    image = Image.open(io.BytesIO(data)).convert("RGB")
    if size:
        image.thumbnail((size, size), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0).contiguous()
