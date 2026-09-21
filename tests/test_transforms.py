"""The reproducibility guarantee, tested."""

from __future__ import annotations

import pytest
import torch

from veil.ml.simulation import transforms as T


def test_grid_sampling_is_exhaustive_and_capped():
    spec = T.TransformSpec(rotation_deg=[-30, 0, 30], scale=[0.8, 1.0], translate=[0.0],
                           perspective=[0.0], brightness=[1.0], contrast=[1.0],
                           blur_sigma=[0.0], noise_std=[0.0], deformation=[0.0])
    params = spec.sample()
    assert len(params) == 6
    assert {p.rotation_deg for p in params} == {-30, 0, 30}

    capped = T.TransformSpec(max_samples=5).sample()
    assert len(capped) == 5


def test_same_seed_gives_identical_parameters_and_pixels():
    spec = T.TransformSpec(mode="random", samples=8)
    assert spec.sample(seed=7) == spec.sample(seed=7)
    assert spec.sample(seed=7) != spec.sample(seed=8)

    image = torch.rand(2, 3, 48, 48)
    params = T.TransformParams(rotation_deg=17.0, noise_std=0.05, deformation=0.04)
    assert torch.equal(T.apply(image, params, seed=3), T.apply(image, params, seed=3))
    assert not torch.equal(T.apply(image, params, seed=3), T.apply(image, params, seed=4))


def test_transforms_are_differentiable():
    image = torch.rand(1, 3, 48, 48, requires_grad=True)
    params = T.TransformParams(rotation_deg=10.0, scale=0.9, brightness=1.2,
                               blur_sigma=1.0, perspective=0.1, deformation=0.03)
    T.apply(image, params).sum().backward()
    assert image.grad is not None and float(image.grad.abs().sum()) > 0


def test_output_stays_a_valid_image():
    image = torch.rand(1, 3, 32, 32)
    out = T.apply(image, T.TransformParams(brightness=4.0, noise_std=0.5))
    assert out.shape == image.shape
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_identity_transform_is_a_no_op():
    image = torch.rand(1, 3, 32, 32)
    assert torch.equal(T.apply(image, T.TransformParams()), image)


def test_rejects_wrong_shape():
    with pytest.raises(ValueError):
        T.apply(torch.rand(3, 32, 32), T.TransformParams())
