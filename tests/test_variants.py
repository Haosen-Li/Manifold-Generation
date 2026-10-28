import torch
from conftest import make_pipeline

from manifold_gen import SAMG, plug


def test_cfg_zero_flow_projection():
    pipe, options = make_pipeline("sd3")
    x = torch.randn(1, 4, 8, 8)
    sampler = pipe.scheduler
    sampler.set_timesteps(4)
    latent = x.clone()
    with torch.no_grad():
        for i, t in enumerate(sampler.timesteps):
            if i == 0:
                prediction = torch.zeros_like(latent)
            else:
                raw = pipe.transformer(
                    hidden_states=torch.cat([latent, latent]),
                    timestep=t.expand(2),
                    encoder_hidden_states=torch.cat(
                        [options["negative_prompt_embeds"], options["prompt_embeds"]]
                    ),
                    pooled_projections=torch.cat(
                        [options["negative_pooled_prompt_embeds"], options["pooled_prompt_embeds"]]
                    ),
                    return_dict=False,
                )[0]
                u, c = raw.chunk(2)
                coefficient = (u * c).flatten(1).sum(1) / (u.square().flatten(1).sum(1) + 1e-8)
                u = u * coefficient[:, None, None, None]
                base = u + (c - u)
                delta = (u + 5 * (c - u)) - base
                energy = delta.square().mean(1, keepdim=True)
                weights = 1 - (energy - energy.min()) / (energy.max() - energy.min() + 1e-8)
                prediction = base + weights * delta
            latent = sampler.step(prediction, t, latent).prev_sample
    actual = plug(pipe, samg=SAMG(variant="cfg_zero", base_scale=1, strong_scale=5))(
        **options, latents=x
    ).images
    torch.testing.assert_close(actual, latent, atol=3e-6, rtol=3e-5)
