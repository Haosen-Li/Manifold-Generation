# Manifold Gen

**Geometry-aware inference for image and video generation.**

Manifold Gen provides three composable components for Diffusers pipelines:

| Component | Function | Paper |
|---|---|---|
| **Oracle Noise** | Semantic optimization of the initial latent on a hypersphere | [arXiv](https://arxiv.org/abs/2604.23540) |
| **SAMG** | Spatially adaptive guidance with CFG, CFG++ and CFG-Zero* | [arXiv](https://arxiv.org/abs/2604.26503) |
| **Z²-Sampling** | Zero-extra-call trajectory correction using a temporal semantic surrogate | [arXiv](https://arxiv.org/abs/2604.23536) |

[中文](README.zh-CN.md) · [API reference](docs/api.md) · [Model compatibility](docs/compatibility.md)

## Installation

Requires Python 3.10+, PyTorch 2.2+ and Diffusers 0.35.1. Install the PyTorch build
appropriate for your device, then run from the repository root:

```bash
python -m pip install -e ".[text,video]"
```

Optional dependencies: `.[data]` for Parquet prompts; `.[dev]` for development.
Model weights are downloaded by Diffusers or loaded from a local directory.

## Quick start

```python
import torch
from manifold_gen import SAMG, load_pipeline

pipe = load_pipeline(
    "sdxl",
    samg=SAMG(variant="cfg", base_scale=1.0, strong_scale=7.5),
    device="cuda",
)

image = pipe(
    "A red cube to the left of a blue sphere on a white table",
    guidance_scale=7.5,
    num_inference_steps=30,
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]
image.save("result.png")
```

`load_pipeline()` selects an appropriate deterministic scheduler for Z² or CFG++.
Use `model_id="/path/to/model"` to load a compatible local checkpoint.

## Insert a component into an existing pipeline

```python
from manifold_gen import OracleNoise, SAMG, Z2Sampling, plug

oracle_pipe = plug(base_pipe, oracle_noise=OracleNoise(theta=0.02, steps=1))
samg_pipe = plug(base_pipe, samg=SAMG(base_scale=1.0, strong_scale=7.5))
z2_pipe = plug(base_pipe, z2=Z2Sampling(warmup_steps=1, zigzag_steps=10))
```

Wrappers share model weights and preserve native prompt encoding, conditioning,
VAE decoding and output types. Call them sequentially. `plug()` preserves the existing
scheduler; for Z²/CFG++ on SD/SDXL, configure it explicitly:

```python
from manifold_gen import ManifoldDDIMScheduler

base_pipe.scheduler = ManifoldDDIMScheduler.from_config(
    base_pipe.scheduler.config, clip_sample=False, thresholding=False
)
```

Components can also be combined in one `plug()` call. The supported combinations
and solver requirements are listed in the [compatibility guide](docs/compatibility.md).

## SAMG variants

```python
SAMG(variant="cfg",      base_scale=1.0, strong_scale=7.5)
SAMG(variant="cfg++",    base_scale=0.4, strong_scale=0.8)
SAMG(variant="cfg_zero", base_scale=1.0, strong_scale=5.0, zero_init_steps=1)
```

- **CFG:** spatial interpolation between two guidance strengths.
- **CFG++:** spatial interpolation in predicted-clean-sample space, followed by
  unconditional-epsilon renoising.
- **CFG-Zero\*:** zero-initialized steps and optimal unconditional projection.

Configure spatial/temporal smoothing and lambda bounds, or register a custom guidance
backend with `register_guidance()`. See [API reference](docs/api.md).

For SAMG, pass `guidance_scale > 1` to enable native CFG batching. The effective
spatial strength is controlled by the SAMG parameters.

## Models

| Profile | Model | SAMG | Oracle Noise | Z² |
|---|---|---|---|---|
| `sd15` | Stable Diffusion 1.5 | ✓ | — | ✓ |
| `sd21` | Stable Diffusion 2.1 | ✓ | — | ✓ |
| `sdxl` | Stable Diffusion XL | ✓ | ✓ | ✓ |
| `sdxl-turbo` | SDXL-Turbo | — | ✓ | — |
| `sd35` | Stable Diffusion 3.5 Medium | ✓ | ✓ | ✓ |
| `dreamshaper-xl` | DreamShaper XL v2 Turbo | ✓ | ✓ | ✓ |
| `hunyuan-dit` | Hunyuan-DiT | ✓ | — | ✓ |
| `cogvideox` | CogVideoX-2B | ✓ | — | ✓ |
| `modelscope` | ModelScope text-to-video 1.7B | ✓ | — | ✓ |

✓ denotes an implemented adapter. Variant restrictions and architectural extensions are
specified in the [full matrix](docs/compatibility.md). SDXL-Turbo uses conditional-only
Oracle Noise with `guidance_scale=0`. Compatible Diffusion-DPO checkpoints reuse the SDXL
adapter; AYS is available through explicit DDIM timestep grids.

## Command line

```bash
# Standard SAMG
manifold-gen --model sdxl --components samg --prompt "A red cube on a white table"

# SAMG with CFG++
manifold-gen --model sdxl --components samg --samg-variant 'cfg++' \
  --base-scale 0.4 --strong-scale 0.8 --prompt "A red cube"

# SAMG with CFG-Zero*
manifold-gen --model sd35 --components samg --samg-variant cfg_zero \
  --prompt "A blue vase on a wooden table"

# Oracle Noise on SDXL-Turbo
manifold-gen --model sdxl-turbo --components oracle --oracle-cfg off \
  --theta 0.05 --prompt "A small yellow dog"

# Z² video generation
manifold-gen --model cogvideox --components z2 --warmup-steps 10 \
  --zigzag-steps 30 --prompt "A train crosses a stone bridge"

# Batch prompts
manifold-gen --model sdxl --components samg \
  --prompts examples/prompts.jsonl --output outputs/batch
```

Use `manifold-gen --help` or `python -m manifold_gen --help` for all options.
Prompt files may be TXT, JSON, JSONL, CSV or Parquet. Results include PNG/MP4 files
and `run.json` with parameters, seeds and execution metadata.

## Examples

- [Single-component generation](examples/quickstart.py)
- [Video generation](examples/video.py)
- [Custom guidance](examples/custom_guidance.py)
- [AYS and compatible DPO checkpoints](examples/ays_and_dpo.py)

## Development

```bash
python -m pip install -e ".[dev,text]"
python -m pytest -q
python -m ruff check src tests examples
python -m build
```

Tests use small randomized Diffusers models and run on CPU without downloading
checkpoints. Full-model quality and speed benchmarks are not included.

## Citation

Please cite the relevant papers when using these methods. BibTeX entries are available
in [CITATION.bib](CITATION.bib).
