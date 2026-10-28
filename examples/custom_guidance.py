"""Expose an additional prediction-space variant without editing the package."""

from manifold_gen import SAMG, register_guidance


def scheduled_samg(component, unconditional, conditional, *, channel_dim, step_index):
    # Example policy: plain base guidance for the first two steps, then SAMG.
    # The warmup policy is configurable independently of spatial guidance.
    if step_index < 2:
        return unconditional + component.base_scale * (conditional - unconditional)
    return component.mix(unconditional, conditional, channel_dim=channel_dim)


def make_component():
    register_guidance("scheduled_samg", scheduled_samg, replace=True)
    return SAMG(variant="scheduled_samg", base_scale=1.0, strong_scale=5.0)


if __name__ == "__main__":
    import torch

    from manifold_gen import load_pipeline

    pipe = load_pipeline("sdxl", samg=make_component())
    image = pipe(
        "A red cube",
        guidance_scale=5.0,
        num_inference_steps=30,
        generator=torch.Generator("cuda").manual_seed(42),
    ).images[0]
    image.save("custom_guidance.png")
