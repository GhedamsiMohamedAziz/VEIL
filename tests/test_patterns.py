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


def test_an_optimizer_that_cannot_improve_reports_no_improvement():
    """A detector blind to the pattern: its score follows the brightness draw
    only. Accepting on a lucky draw, or reporting the minimum of noisy step
    losses, would both claim progress here. Nothing may be claimed."""
    from veil.ml.patterns.optimizer import OptimizationConfig, optimize
    from veil.ml.simulation.renderer import Placement
    from veil.ml.simulation.transforms import TransformSpec

    class CornerBrightness:
        differentiable = False

        def evaluate(self, scene, label):
            return {"max_scores": [float(scene[..., :4, :4].mean())]}

    spec = TransformSpec(rotation_deg=[0.0], scale=[1.0], translate=[0.0], perspective=[0.0],
                         brightness=[0.5, 1.5], contrast=[1.0], blur_sigma=[0.0],
                         noise_std=[0.0], deformation=[0.0])
    cfg = OptimizationConfig(target_label="x", pattern_size=16, iterations=8,
                             batch_transforms=1, tv_weight=0.0, nps_weight=0.0, seed=5)
    result = optimize(CornerBrightness(), torch.full((1, 3, 64, 64), 0.5),
                      Placement(0.5, 0.5, 0.2, 0.2), spec, cfg)
    start = constraints.quantize_to_palette(generator.initialize(16, cfg.init_method, cfg.seed))
    assert torch.equal(result.pattern, start)
    assert result.summary()["improvement"] == 0.0
    assert len({round(h["loss"], 6) for h in result.history}) > 1  # the draws did vary


def test_gradient_strategy_exports_a_pattern_that_really_lowers_the_loss():
    """A differentiable stand-in whose confidence is the scene's mean
    brightness: the optimizer must darken the patch, export it quantized, and
    report an improvement."""
    from veil.ml.patterns.optimizer import OptimizationConfig, optimize
    from veil.ml.simulation.renderer import Placement
    from veil.ml.simulation.transforms import TransformSpec

    class Brightness:
        differentiable = True

        def score(self, scene, label):
            return scene.mean(dim=(1, 2, 3))

    spec = TransformSpec(rotation_deg=[-10.0, 10.0], scale=[1.0], translate=[0.0],
                         perspective=[0.0], brightness=[0.8, 1.2], contrast=[1.0],
                         blur_sigma=[0.0], noise_std=[0.0], deformation=[0.0])
    cfg = OptimizationConfig(target_label="x", pattern_size=16, iterations=25, batch_transforms=2,
                             learning_rate=0.1, tv_weight=0.0, nps_weight=0.0, seed=1)
    result = optimize(Brightness(), torch.full((2, 3, 64, 64), 0.5),
                      Placement(0.5, 0.5, 0.4, 0.4), spec, cfg)
    start = constraints.quantize_to_palette(generator.initialize(16, cfg.init_method, cfg.seed))
    assert result.strategy == "gradient" and len(result.history) == 25
    assert float(result.pattern.mean()) < float(start.mean())
    assert result.summary()["improvement"] > 0
    assert float(constraints.non_printability(result.pattern)) < 1e-5


def test_production_sweep_measures_only_colours_the_mill_can_make(monkeypatch):
    from veil.ml import manufacture_eval
    from veil.ml.patterns.manufacture import ProductionSpec, to_artwork
    from veil.ml.simulation.renderer import Placement
    from veil.ml.simulation.transforms import TransformSpec

    swept = {}

    def fake_sweep(detector, images, spec, label, pattern=None, **kwargs):
        swept["pattern"] = pattern
        return [{"detected": False, "max_score": 0.0, "detection_count": 0, "boxes": [],
                 "transform": {}, "image_id": "a", "image_index": 0, "transform_index": 0}]

    monkeypatch.setattr(manufacture_eval, "sweep", fake_sweep)
    pattern = generator.initialize(48, "uniform_noise", seed=2)
    production = ProductionSpec(method="knit", stitches_per_cm=4, width_cm=10, height_cm=12, max_yarns=4)
    manufacture_eval.evaluate_production(
        None, torch.zeros(1, 3, 64, 64), ["a"], pattern, Placement(0.5, 0.5, 0.3, 0.3),
        TransformSpec(), production, target_label="x", threshold=0.5, seed=1)

    def colours(t):
        return {tuple(round(float(v), 4) for v in px) for px in t.reshape(3, -1).t()}

    assert colours(swept["pattern"]) <= colours(to_artwork(pattern, production))
