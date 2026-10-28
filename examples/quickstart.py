"""Run any ONE component against the same SDXL architecture.

python examples/quickstart.py --component samg
python examples/quickstart.py --component oracle
python examples/quickstart.py --component z2
"""

import argparse
from pathlib import Path

import torch

from manifold_gen import SAMG, OracleNoise, Z2Sampling, load_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--component", choices=["samg", "oracle", "z2"], default="samg")
    parser.add_argument("--prompt", default="A red cube to the left of a blue sphere")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    components = {
        "samg": {"samg": SAMG(base_scale=1.0, strong_scale=5.0)},
        "oracle": {"oracle_noise": OracleNoise(theta=0.02, steps=1)},
        "z2": {"z2": Z2Sampling(warmup_steps=1, zigzag_steps=10)},
    }
    pipe = load_pipeline("sdxl", device=args.device, **components[args.component])
    output = pipe(
        args.prompt,
        num_inference_steps=30,
        guidance_scale=5.0,
        generator=torch.Generator(args.device).manual_seed(42),
    )
    Path("outputs").mkdir(exist_ok=True)
    output.images[0].save(f"outputs/{args.component}.png")
    print(pipe.last_stats)


if __name__ == "__main__":
    main()
