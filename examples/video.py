"""Video adapter example. No model download occurs until main() runs."""

import argparse
from pathlib import Path

import torch
from diffusers.utils import export_to_video

from manifold_gen import MODELS, SAMG, Z2Sampling, load_pipeline


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["cogvideox", "modelscope"], default="cogvideox")
    p.add_argument("--component", choices=["samg", "z2"], default="z2")
    p.add_argument("--device", default="cuda")
    p.add_argument("--prompt", default="A train crossing a stone bridge over a river")
    args = p.parse_args()
    spec = MODELS[args.model]
    components = (
        {"samg": SAMG(strong_scale=spec.guidance)}
        if args.component == "samg"
        else {"z2": Z2Sampling(warmup_steps=10, zigzag_steps=30)}
    )
    pipe = load_pipeline(args.model, device=args.device, **components)
    result = pipe(
        args.prompt,
        height=spec.height,
        width=spec.width,
        num_frames=spec.frames,
        num_inference_steps=50,
        guidance_scale=spec.guidance,
        output_type="np",
        generator=torch.Generator(args.device).manual_seed(42),
    )
    Path("outputs").mkdir(exist_ok=True)
    export_to_video(result.frames[0], f"outputs/{args.model}_{args.component}.mp4", fps=8)


if __name__ == "__main__":
    main()
