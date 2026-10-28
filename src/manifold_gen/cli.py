"""Reproducible image/video generation, including prompt files and baselines."""

import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from . import __version__
from .adapters import MODELS, load_pipeline
from .oracle import OracleNoise
from .samg import SAMG
from .z2 import Z2Sampling


def read_prompts(path, column="caption"):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        values = [r if isinstance(r, str) else r[column] for r in records]
    elif suffix == ".json":
        records = json.loads(path.read_text(encoding="utf-8"))
        values = [r if isinstance(r, str) else r[column] for r in records]
    elif suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as f:
            values = [r[column] for r in csv.DictReader(f)]
    elif suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError('Parquet input requires pip install -e ".[data]"') from exc
        values = pd.read_parquet(path, columns=[column])[column].tolist()
    else:
        values = [s.strip() for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]
    if not values or not all(isinstance(v, str) for v in values):
        raise ValueError(
            "Prompt files must contain at least one text prompt; nulls/numbers are not accepted."
        )
    return values


def parser():
    p = argparse.ArgumentParser(description="Manifold Gen: Oracle Noise, SAMG and Z²-Sampling")
    p.add_argument("--list-models", action="store_true")
    p.add_argument("--model", choices=list(MODELS), default="sdxl")
    p.add_argument("--model-id", help="Compatible Hugging Face repository or local checkpoint directory")
    prompts = p.add_mutually_exclusive_group()
    prompts.add_argument("--prompt")
    prompts.add_argument("--prompts", help="TXT, JSON, JSONL, CSV or Parquet file")
    p.add_argument("--prompt-column", default="caption")
    p.add_argument("--limit", type=int)
    p.add_argument("--negative-prompt", default="")
    p.add_argument("--components", default="samg", help="Comma-separated oracle,samg,z2; or baseline")
    p.add_argument("--samg-variant", choices=["cfg", "cfg++", "cfg_zero"], default="cfg")
    p.add_argument("--base-scale", type=float)
    p.add_argument("--strong-scale", type=float)
    p.add_argument("--lambda-min", type=float, default=0.0)
    p.add_argument("--lambda-max", type=float, default=1.0)
    p.add_argument("--smooth-kernel", type=int, default=1)
    p.add_argument("--temporal-kernel", type=int, default=1)
    p.add_argument("--zero-init-steps", type=int, default=1)
    p.add_argument("--theta", type=float, default=0.02)
    p.add_argument("--oracle-steps", type=int, default=1)
    p.add_argument("--oracle-cfg", choices=["auto", "on", "off"], default="auto")
    p.add_argument("--uniform-token-weights", action="store_true")
    p.add_argument("--query-chunk-size", type=int, default=256)
    p.add_argument("--key-chunk-size", type=int, default=256)
    p.add_argument("--zigzag-steps", type=int, default=10)
    p.add_argument("--warmup-steps", type=int, default=1)
    p.add_argument("--inversion-guidance", type=float, default=0.0)
    p.add_argument("--steps", type=int)
    p.add_argument("--guidance-scale", type=float)
    p.add_argument("--height", type=int)
    p.add_argument("--width", type=int)
    p.add_argument("--frames", type=int)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--timesteps", help="Custom DDIM grid, e.g. 999,845,730,...,13 (SD/SDXL only)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--offline", action="store_true")
    p.add_argument("--weight-variant", help="Optional checkpoint file variant, e.g. fp16")
    p.add_argument("--output", default="outputs")
    p.add_argument("--compare-baseline", action="store_true")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.list_models:
        for name, spec in MODELS.items():
            print(f"{name:18s} {spec.model_id}")
        return
    if not (args.prompt is not None or args.prompts):
        parser().error("Provide --prompt or --prompts.")
    components = set(args.components.split(","))
    if not components <= {"oracle", "samg", "z2", "baseline"} or (
        "baseline" in components and len(components) > 1
    ):
        parser().error("--components accepts oracle,samg,z2, or baseline alone.")
    spec = MODELS[args.model]
    guidance = args.guidance_scale if args.guidance_scale is not None else spec.guidance
    pp = args.samg_variant == "cfg++"
    samg = (
        SAMG(
            base_scale=args.base_scale if args.base_scale is not None else (0.4 if pp else 1.0),
            strong_scale=args.strong_scale if args.strong_scale is not None else (1.0 if pp else guidance),
            variant=args.samg_variant,
            lambda_min=args.lambda_min,
            lambda_max=args.lambda_max,
            smooth_kernel=args.smooth_kernel,
            temporal_kernel=args.temporal_kernel,
            zero_init_steps=args.zero_init_steps,
        )
        if "samg" in components
        else None
    )
    oracle = (
        OracleNoise(
            theta=args.theta,
            steps=args.oracle_steps,
            cfg_aware={"auto": None, "on": True, "off": False}[args.oracle_cfg],
            use_token_weights=not args.uniform_token_weights,
            query_chunk_size=args.query_chunk_size,
            key_chunk_size=args.key_chunk_size,
        )
        if "oracle" in components
        else None
    )
    z2 = (
        Z2Sampling(
            zigzag_steps=args.zigzag_steps,
            warmup_steps=args.warmup_steps,
            inversion_guidance=args.inversion_guidance,
        )
        if "z2" in components
        else None
    )
    prompts = read_prompts(args.prompts, args.prompt_column) if args.prompts else [args.prompt]
    if args.limit is not None:
        if args.limit <= 0:
            parser().error("--limit must be positive.")
        prompts = prompts[: args.limit]
    options = {"local_files_only": args.offline}
    if args.weight_variant:
        options["variant"] = args.weight_variant
    wrapped = load_pipeline(
        args.model,
        model_id=args.model_id,
        device=args.device,
        torch_dtype=getattr(torch, args.dtype) if args.dtype else None,
        samg=samg,
        oracle_noise=oracle,
        z2=z2,
        **options,
    )
    kwargs = dict(
        height=args.height or spec.height,
        width=args.width or spec.width,
        num_inference_steps=args.steps or spec.steps,
        guidance_scale=guidance,
        negative_prompt=args.negative_prompt,
    )
    if spec.frames is not None:
        kwargs.update(num_frames=args.frames or spec.frames, output_type="np")
    if args.timesteps:
        if spec.family not in {"sd", "sdxl"}:
            parser().error("Custom grids currently use the SD/SDXL ManifoldDDIMScheduler adapter.")
        from .schedulers import ManifoldDDIMScheduler

        wrapped.pipe.scheduler = ManifoldDDIMScheduler.from_config(
            wrapped.pipe.scheduler.config, clip_sample=False, thresholding=False
        )
        kwargs["timesteps"] = [int(t.strip()) for t in args.timesteps.split(",")]
        kwargs["num_inference_steps"] = len(kwargs["timesteps"])
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for i, prompt in enumerate(prompts):
        seed = args.seed + i
        for label, runner in ([("baseline", wrapped.pipe)] if args.compare_baseline else []) + [
            ("result", wrapped)
        ]:
            if str(args.device).startswith("cuda"):
                torch.cuda.synchronize()
            start = time.perf_counter()
            output = runner(prompt, generator=torch.Generator(device=args.device).manual_seed(seed), **kwargs)
            if str(args.device).startswith("cuda"):
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if spec.frames is None:
                name = f"{i:05d}_{label}.png"
                output.images[0].save(out / name)
            else:
                from diffusers.utils import export_to_video

                name = f"{i:05d}_{label}.mp4"
                export_to_video(output.frames[0], str(out / name), fps=args.fps)
            record = dict(
                index=i,
                prompt=prompt,
                seed=seed,
                label=label,
                file=name,
                seconds=elapsed,
                stats=dict(wrapped.last_stats) if label == "result" else None,
            )
            records.append(record)
            print(f"{name}  {elapsed:.3f}s")
        # Checkpoint completed generations so an interrupted batch keeps its metadata.
        metadata = dict(
            manifold_gen_version=__version__,
            model=args.model,
            model_id=args.model_id or spec.model_id,
            arguments=vars(args),
            samg=asdict(samg) if samg else None,
            oracle=asdict(oracle) if oracle else None,
            z2=asdict(z2) if z2 else None,
            records=records,
        )
        (out / "run.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
