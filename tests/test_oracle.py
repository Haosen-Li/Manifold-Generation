import pytest
import torch
from conftest import make_pipeline
from diffusers import EulerDiscreteScheduler

from manifold_gen import SAMG, OracleNoise, plug
from manifold_gen.oracle import token_importance


@pytest.mark.parametrize("family", ["sdxl", "sd3"])
@pytest.mark.parametrize("cfg", [False, True])
def test_oracle_optimizes_without_parameter_grads_or_hook_leaks(family, cfg, tokenizer):
    pipe, _ = make_pipeline(family, tokenizer)
    model = pipe.unet if family == "sdxl" else pipe.transformer
    before = sum(len(m._forward_hooks) for m in model.modules())
    params = list(model.parameters())
    flags = [p.requires_grad for p in params]
    x = torch.randn(2, 4, 8, 8)
    result = OracleNoise(steps=2, cfg_aware=cfg, query_chunk_size=5, key_chunk_size=5).optimize(
        pipe, ["a b", "c d"], x, guidance_scale=5, height=16, width=16, max_sequence_length=8
    )
    assert torch.isfinite(result).all()
    assert not torch.allclose(result, x)
    torch.testing.assert_close(result.flatten(1).norm(dim=1), x.flatten(1).norm(dim=1))
    assert [p.requires_grad for p in params] == flags
    assert all(p.grad is None for p in params)
    assert sum(len(m._forward_hooks) for m in model.modules()) == before


@pytest.mark.parametrize("family", ["sdxl", "sd3"])
def test_oracle_native_pipeline_batch(family, tokenizer):
    pipe, _ = make_pipeline(family, tokenizer)
    options = dict(
        prompt=["a b", "c d"],
        height=16,
        width=16,
        num_images_per_prompt=2,
        num_inference_steps=2,
        guidance_scale=5,
        output_type="latent",
    )
    if family == "sd3":
        options["max_sequence_length"] = 8
    wrapped = plug(pipe, oracle_noise=OracleNoise())
    result = wrapped(**options, generator=torch.Generator().manual_seed(5)).images
    assert result.shape[0] == 4
    assert torch.isfinite(result).all()
    assert wrapped.last_stats["oracle_denoiser_calls"] == 2


def test_turbo_conditional_mode_and_original_sigma_scaling(tokenizer):
    pipe, _ = make_pipeline("sdxl", tokenizer)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    x = torch.randn(1, 4, 8, 8)
    oracle = OracleNoise(cfg_aware=False)
    optimized = oracle.optimize(pipe, "a b", x, height=16, width=16, guidance_scale=0)
    kwargs = dict(
        prompt="a b", height=16, width=16, num_inference_steps=1, guidance_scale=0.0, output_type="latent"
    )
    expected = pipe(latents=optimized, **kwargs).images
    actual = plug(pipe, oracle_noise=oracle)(latents=x, **kwargs).images
    torch.testing.assert_close(actual, expected, atol=5e-6, rtol=2e-5)


def test_empty_prompt_and_inference_mode(tokenizer):
    pipe, _ = make_pipeline("sdxl", tokenizer)
    weights = token_importance(pipe, [""], device="cpu")
    assert weights.count_nonzero() == 0
    x = torch.randn(1, 4, 8, 8)
    result = OracleNoise().optimize(pipe, "", x, height=16, width=16)
    torch.testing.assert_close(result, x)
    with torch.inference_mode(), pytest.raises(RuntimeError, match="inference_mode"):
        OracleNoise().optimize(pipe, "a", x)


def test_oracle_restores_hooks_and_checkpointing_after_error(tokenizer):
    pipe, _ = make_pipeline("sdxl", tokenizer)
    pipe.unet.enable_gradient_checkpointing()

    def fail(*args):
        raise RuntimeError("intentional failure")

    handle = pipe.unet.mid_block.register_forward_hook(fail)
    before = sum(len(m._forward_hooks) for m in pipe.unet.modules())
    with pytest.raises(RuntimeError, match="intentional failure"):
        OracleNoise(use_token_weights=False).optimize(pipe, "a", torch.randn(1, 4, 8, 8))
    assert sum(len(m._forward_hooks) for m in pipe.unet.modules()) == before
    assert pipe.unet.is_gradient_checkpointing
    handle.remove()


def test_three_component_composition_runs(tokenizer):
    from manifold_gen import Z2Sampling

    pipe, _ = make_pipeline("sdxl", tokenizer)
    wrapped = plug(pipe, oracle_noise=OracleNoise(), samg=SAMG(), z2=Z2Sampling())
    result = wrapped(
        "a b", height=16, width=16, num_inference_steps=3, guidance_scale=5, output_type="latent"
    ).images
    assert torch.isfinite(result).all()
    assert wrapped.last_stats["denoiser_calls"] == 3
    assert wrapped.last_stats["oracle_denoiser_calls"] == 2
