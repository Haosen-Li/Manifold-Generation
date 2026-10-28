import pytest
import torch
from conftest import make_pipeline

from manifold_gen import SAMG, Z2Sampling, plug


def get_latents(output):
    return output.images if hasattr(output, "images") else output.frames


@pytest.mark.parametrize("family", ["sd", "sdxl", "sd3", "hunyuan", "cogvideox", "modelscope"])
@pytest.mark.parametrize("component", ["identity", "samg", "cfg_zero", "z2", "cfgpp", "samg_z2"])
def test_native_pipeline_adapter(family, component):
    if family == "sd3" and component == "cfgpp":
        pytest.skip("CFG++ x0/epsilon renoising is intentionally incompatible with flow.")
    pipe, options = make_pipeline(family)
    samg = None
    z2 = None
    if component in {"samg", "samg_z2"}:
        samg = SAMG(base_scale=1, strong_scale=5)
    elif component == "cfg_zero":
        samg = SAMG(variant="cfg_zero", base_scale=1, strong_scale=5)
    elif component == "cfgpp":
        samg = SAMG(variant="cfg++", base_scale=0.4, strong_scale=1.0)
    if component in {"z2", "samg_z2"}:
        z2 = Z2Sampling(zigzag_steps=3)
    model = pipe.unet if hasattr(pipe, "unet") else pipe.transformer
    original_forward = model.forward
    original_step = pipe.scheduler.step
    wrapped = plug(pipe, samg=samg, z2=z2)
    output = wrapped(**options, generator=torch.Generator().manual_seed(99))
    result = get_latents(output)
    assert torch.isfinite(result).all()
    assert result.shape[0] == 1
    assert model.forward == original_forward
    assert pipe.scheduler.step == original_step
    assert not hasattr(pipe, "_manifold_gen_active")
    assert wrapped.last_stats["scheduler_steps"] == 4
    assert wrapped.last_stats["denoiser_calls"] == (3 if component == "cfg_zero" else 4)
    if z2:
        assert wrapped.last_stats["z2_corrected_steps"] == 2
    if component == "identity":
        expected = get_latents(pipe(**options, generator=torch.Generator().manual_seed(99)))
        torch.testing.assert_close(result, expected)


def test_constant_samg_reduces_to_native_cfg():
    pipe, options = make_pipeline("sdxl")
    actual = plug(pipe, samg=SAMG(base_scale=5, strong_scale=5))(
        **options, generator=torch.Generator().manual_seed(9)
    ).images
    expected = pipe(**options, generator=torch.Generator().manual_seed(9)).images
    torch.testing.assert_close(actual, expected)


def test_repeated_calls_reset_surrogate_and_v_prediction():
    pipe, options = make_pipeline("sd", prediction_type="v_prediction")
    wrapped = plug(pipe, z2=Z2Sampling())
    a = wrapped(**options, generator=torch.Generator().manual_seed(9)).images
    b = wrapped(**options, generator=torch.Generator().manual_seed(9)).images
    torch.testing.assert_close(a, b)


def test_patches_restored_after_model_error():
    pipe, options = make_pipeline("sd")
    original = pipe.unet.forward
    original_step = pipe.scheduler.step

    def fail(*args, **kwargs):
        raise RuntimeError("test failure")

    # Keep the signature so the wrapper can bind pipeline arguments.
    import functools

    pipe.unet.forward = functools.wraps(original)(fail)
    failing_forward = pipe.unet.forward
    with pytest.raises(RuntimeError, match="test failure"):
        plug(pipe, samg=SAMG())(**options)
    assert pipe.unet.forward is failing_forward
    assert pipe.scheduler.step == original_step
    assert not hasattr(pipe, "_manifold_gen_active")


def test_invalid_combinations_are_rejected():
    pipe, options = make_pipeline("sd3")
    with pytest.raises(ValueError, match="flow solvers"):
        plug(pipe, samg=SAMG(variant="cfg++"))(**options)
    options["guidance_scale"] = 0.0
    with pytest.raises(ValueError, match="branches"):
        plug(pipe, samg=SAMG())(**options)
