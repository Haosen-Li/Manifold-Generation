import pytest
import torch
from diffusers import DDIMScheduler

from manifold_gen import SAMG, ManifoldDDIMScheduler, Z2Sampling, register_guidance
from manifold_gen.oracle import _joint_loss, spherical_update
from manifold_gen.samg import optimal_unconditional


def test_samg_spatial_guidance():
    torch.manual_seed(3)
    u, c = torch.randn(1, 4, 8, 8), torch.randn(1, 4, 8, 8)
    base, strong = u + 2 * (c - u), u + 7 * (c - u)
    delta = strong - base
    energy = torch.nn.functional.avg_pool2d(delta.square().mean(1, keepdim=True), 3, 1, 1)
    weights = 1 - (energy - energy.min()) / (energy.max() - energy.min() + 1e-8)
    expected = base + weights * delta
    torch.testing.assert_close(SAMG(base_scale=2, strong_scale=7, smooth_kernel=3)(u, c), expected)


def test_batch_independence_and_video_layouts():
    u, c = torch.randn(2, 4, 3, 8, 8), torch.randn(2, 4, 3, 8, 8)
    c[1] *= 100
    samg = SAMG(smooth_kernel=3)
    result = samg(u, c)
    torch.testing.assert_close(result[0], samg(u[:1], c[:1])[0])
    torch.testing.assert_close(result, samg(u.movedim(1, 2), c.movedim(1, 2), channel_dim=2).movedim(2, 1))


def test_float16_energy_and_constant_inputs():
    u = torch.zeros(1, 4, 4, 4, dtype=torch.float16)
    c = torch.full_like(u, 1000.0)
    result = SAMG(strong_scale=3)(u, c)
    assert torch.isfinite(result).all()
    torch.testing.assert_close(SAMG()(u, u), u)


def test_cfg_zero_projection_and_initialization():
    u, c = torch.randn(2, 4, 3, 3), torch.randn(2, 4, 3, 3)
    scaled = optimal_unconditional(u, c)
    assert ((c - scaled) * u).flatten(1).sum(1).abs().max() < 1e-5
    assert SAMG(variant="cfg_zero")(u, c, step_index=0).count_nonzero() == 0


def test_geodesic_radius_zero_and_gradient_direction():
    x, g = torch.randn(3, 4, 8, 8), torch.randn(3, 4, 8, 8)
    result = spherical_update(x, g, 0.02)
    torch.testing.assert_close(result.flatten(1).norm(dim=1), x.flatten(1).norm(dim=1))
    assert ((result - x) * g).flatten(1).sum(1).min() > 0
    torch.testing.assert_close(spherical_update(x, torch.zeros_like(x), 0.5), x)
    assert spherical_update(torch.zeros_like(x), g, 0.1).count_nonzero() == 0


def test_joint_chunked_denominator_and_gradients_match_dense():
    tensors = [torch.randn(1, 2, n, 4, requires_grad=True) for n in (3, 7, 5, 3, 7, 5)]
    cq, ck, ct, uq, uk, ut = tensors
    mask = torch.rand(1, 5)
    c = cq @ torch.cat([ck, ct], -2).transpose(-1, -2) / 2
    u = uq @ torch.cat([uk, ut], -2).transpose(-1, -2) / 2
    expected = ((u + 4.5 * (c - u)).softmax(-1)[..., 7:] * mask[:, None, None]).sum()
    result = _joint_loss(*tensors, mask, 4.5, 2)
    torch.testing.assert_close(result, expected)
    actual_grads = torch.autograd.grad(result, tensors, retain_graph=True)
    expected_grads = torch.autograd.grad(expected, tensors)
    for a, b in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("prediction_type", ["epsilon", "v_prediction", "sample"])
def test_ddim_matches_upstream_on_uniform_grid(prediction_type):
    a = DDIMScheduler(clip_sample=False, prediction_type=prediction_type)
    b = ManifoldDDIMScheduler.from_config(a.config)
    a.set_timesteps(10)
    b.set_timesteps(10)
    x, noise = torch.randn(1, 4, 8, 8), torch.randn(1, 4, 8, 8)
    for t in a.timesteps:
        expected = a.step(noise, t, x).prev_sample
        actual = b.step(noise, t, x).prev_sample
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_custom_grid_uses_next_alpha():
    s = ManifoldDDIMScheduler(clip_sample=False)
    s.set_timesteps(timesteps=[999, 810, 500, 42])
    x, e = torch.randn(1, 4, 4, 4), torch.randn(1, 4, 4, 4)
    a, b = s.alphas_cumprod[999], s.alphas_cumprod[810]
    expected = b.sqrt() * (x - (1 - a).sqrt() * e) / a.sqrt() + (1 - b).sqrt() * e
    torch.testing.assert_close(s.step(e, 999, x).prev_sample, expected)


def test_custom_guidance_and_invalid_settings():
    register_guidance("test_identity", lambda module, u, c, **kw: c)
    u, c = torch.randn(1, 4, 3, 3), torch.randn(1, 4, 3, 3)
    torch.testing.assert_close(SAMG(variant="test_identity")(u, c), c)
    with pytest.raises(ValueError):
        SAMG(smooth_kernel=2)
    with pytest.raises(ValueError):
        Z2Sampling(warmup_steps=0)
