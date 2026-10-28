# Public API

## Insertion

```python
plug(pipe, *, samg=None, oracle_noise=None, z2=None) -> ManifoldPipeline
load_pipeline(model, *, model_id=None, device="cuda", torch_dtype=None,
              samg=None, oracle_noise=None, z2=None, **pretrained_kwargs)
```

`plug` does not load weights or change a scheduler. `load_pipeline` loads one of `MODELS`
and prepares a deterministic scheduler when needed. Both return a callable wrapper that
preserves the native result (`images` or `frames`). `wrapper.pipe` exposes the original
pipeline. `wrapper.to(device)` returns the wrapper; other model-management methods are
forwarded to the underlying pipe. `with_components(...)` creates a fresh wrapper sharing
weights and uses **exactly** the specified component set.

There is no global monkey patch. Per-instance overrides are scoped to one call and restored
in `finally` blocks. Do not concurrently run other code on the same native pipeline.

## SAMG

```python
SAMG(base_scale=1., strong_scale=7.5,
     lambda_min=0., lambda_max=1., smooth_kernel=1, temporal_kernel=1,
     variant="cfg", zero_init_steps=1)
```

The effective coefficient is `base_scale + lambda_map * (strong_scale-base_scale)`.
Lambda is largest at low-energy positions. All energy calculations use FP32. Spatial
min/max normalization is independent for each batch item; for video it spans each whole
clip. Spatial smoothing uses zero-padded average pooling with zero padding;
temporal smoothing is off unless `temporal_kernel > 1`.

`variant` is `cfg`, `cfg++` (`cfgpp` alias), or `cfg_zero` (`cfg-zero` and
`cfg_zero_star` aliases). Custom registered names are accepted. CFG-Zero's first
`zero_init_steps` steps actually bypass denoiser computation.

Low-level `samg(unconditional, conditional, channel_dim=1, step_index=0)` returns a
guided tensor for CFG/CFG-Zero/custom variants. `samg.mix(a, b, channel_dim=...)`
performs spatial interpolation independent of the variant. CFG++ must be used with
the pipeline wrapper because it also changes the scheduler transition.

### Custom backend

```python
from manifold_gen import SAMG, register_guidance

def my_variant(component, unconditional, conditional, *, channel_dim, step_index):
    # This example simply delegates to SAMG's regular spatial interpolation.
    return component.mix(unconditional, conditional, channel_dim=channel_dim)

register_guidance("my_variant", my_variant)
samg = SAMG(variant="my_variant")
```

Backends must return the same shape/device/dtype. They receive native prediction tensors,
cannot request extra network calls, and should not store state across generations.
Custom variants are available in Python; the CLI intentionally lists built-in variants only.

## OracleNoise

```python
OracleNoise(theta=.02, steps=1, cfg_aware=None, use_token_weights=True,
            min_weight=.5, max_weight=3., optimization_timestep=999.,
            query_chunk_size=256, key_chunk_size=256, checkpoint_attention=True,
            sd3_layer_indices=None)
```

`cfg_aware=None` selects conditional-only optimization if `guidance_scale <= 1`, otherwise
CFG-aware optimization. Set it explicitly to `False` for the SDXL-Turbo conditional-only variant.
Oracle adds preprocessing cost and gradient memory; “training-free” does not mean
“no backward pass”. `steps=0` or `theta=0` is a no-op.

```python
optimized = oracle.optimize(
    pipe, prompt, raw_latents,
    negative_prompt="", guidance_scale=5., height=1024, width=1024,
    token_weights=None, max_sequence_length=256,
)
result = pipe(prompt, latents=optimized, ...)
```

Inputs are **unscaled** Gaussian noise, with one prompt per latent sample. The wrapper
handles native scheduler scaling automatically. Batch prompts and multiple images per
prompt are supported. `token_weights` can override weights in the direct `optimize` API;
use `[B, token_count]`, or CLIP-only length for SD3 (T5 positions get unit weights).
SDXL micro-conditioning uses actual size/crop arguments instead of hard-coded 1024/512.

The SDXL adapter captures cross-attention projections in the down, mid and up blocks. SD3 chooses first,
middle and last transformer blocks (0, 11, 23 for 24 blocks), and gets head counts from
the modules. Query/key chunks reduce the loss computation's peak attention memory.

## Z2Sampling

```python
Z2Sampling(zigzag_steps=10, warmup_steps=1, inversion_guidance=0.)
```

`zigzag_steps` is an **exclusive step-index cutoff**.
With warmup=1 and zigzag_steps=10, corrected steps have indices 1 through 9, except
that the final timestep is always an ordinary step. `warmup_steps` must be at least 1.
The previous surrogate is used for both the pre-model shift and post-model correction,
then updated. State is reset on every call.

SD2.1/CogVideoX v-predictions are converted to epsilon at the shifted evaluation point,
then converted back at the original scheduler sample. Flow Euler uses the actual sigma
gap instead of a DDIM alpha coefficient.

## Call counts

After generation, `pipe.last_stats` contains:

- `denoiser_calls`: actual sampling model invocations (excluding Oracle preprocessing).
- `cfg_branch_evaluations`: 2 per batched CFG call when sampling components are active.
  This counter is not populated for an Oracle-only/baseline sampling loop.
- `oracle_denoiser_calls`: additional conditional/unconditional forwards during Oracle.
- `scheduler_steps`, `z2_corrected_steps`, `zero_init_skipped_calls`.

Oracle's text-encoder forwards are not denoiser calls. These counters describe execution,
not measured acceleration.

