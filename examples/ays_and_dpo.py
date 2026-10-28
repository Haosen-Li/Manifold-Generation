"""Custom AYS grid and optional compatible DPO checkpoint.

python examples/ays_and_dpo.py
python examples/ays_and_dpo.py --model-id /path/to/compatible-dpo-sdxl

model-id is a complete Diffusers SDXL checkpoint. For UNet-only DPO weights,
load that UNet into a base SDXL pipeline first, then use plug(base, z2=...).
"""

import argparse

import torch
from diffusers.schedulers import AysSchedules

from manifold_gen import Z2Sampling, load_pipeline


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    pipe = load_pipeline(
        "sdxl", model_id=args.model_id, device=args.device, z2=Z2Sampling(warmup_steps=1, zigzag_steps=8)
    )
    grid = AysSchedules["StableDiffusionXLTimesteps"]
    image = pipe(
        "A small yellow dog in a green field",
        timesteps=grid,
        num_inference_steps=len(grid),
        guidance_scale=5.0,
        generator=torch.Generator(args.device).manual_seed(42),
    ).images[0]
    image.save("ays_z2.png")


if __name__ == "__main__":
    main()
