"""Pattern initialization, constraints and serialization."""

from __future__ import annotations

import pytest
import torch

from veil.ml.patterns import constraints, generator, serialization


def test_initialization_is_seeded_and_in_range():
    a = generator.initialize(64, "palette_noise", seed=1)
    assert a.shape == (3, 64, 64)
    assert 0.0 <= float(a.min()) and float(a.max()) <= 1.0
    assert torch.equal(a, generator.initialize(64, "palette_noise", seed=1))
    assert not torch.equal(a, generator.initialize(64, "palette_noise", seed=2))


def test_unknown_initialization_is_rejected():
    with pytest.raises(ValueError):
        generator.initialize(32, "definitely-not-a-method")


def test_png_roundtrip_is_lossless_at_8_bit():
    pattern = generator.initialize(32, "palette_noise", seed=5)
    restored = serialization.from_png(serialization.to_png(pattern))
    assert restored.shape == pattern.shape
    assert float((restored - pattern).abs().max()) <= 1 / 255 + 1e-6


def test_total_variation_rewards_flat_patterns():
    flat = torch.full((3, 16, 16), 0.5)
    noisy = torch.rand(3, 16, 16, generator=torch.Generator().manual_seed(0))
    assert float(constraints.total_variation(flat)) == 0.0
    assert float(constraints.total_variation(noisy)) > 0.1


def test_palette_colours_have_zero_non_printability():
    colour = torch.tensor(constraints.PRINTABLE_PALETTE[2]).view(3, 1, 1).expand(3, 8, 8)
    assert float(constraints.non_printability(colour)) < 1e-6
    off_palette = torch.full((3, 8, 8), 0.5) + torch.tensor([0.2, -0.3, 0.1]).view(3, 1, 1)
    assert float(constraints.non_printability(off_palette)) > 0.0


def test_quantization_snaps_every_pixel_to_the_ink_set():
    quantized = constraints.quantize_to_palette(torch.rand(3, 16, 16))
    palette = {tuple(round(c, 4) for c in colour) for colour in constraints.PRINTABLE_PALETTE}
    pixels = {tuple(round(float(v), 4) for v in quantized[:, y, x])
              for y in range(16) for x in range(16)}
    assert pixels <= palette
