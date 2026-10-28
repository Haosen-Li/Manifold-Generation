# Model and scheduler compatibility

All entries below describe implemented interfaces. A check mark does not imply that every
checkpoint/variant combination was evaluated in its associated paper.

| Models | Diffusers pipeline | SAMG CFG | SAMG CFG++ | SAMG CFG-Zero* | Oracle Noise | Z² |
|---|---|---|---|---|---|---|
| SD 1.5 / SD 2.1 | `StableDiffusionPipeline` | ✓ | DDIM | ✓ | — | DDIM |
| SDXL / DreamShaper XL / compatible DPO | `StableDiffusionXLPipeline` | ✓ | DDIM | ✓ | ✓ | DDIM |
| SDXL-Turbo | `StableDiffusionXLPipeline` | — | — | — | Conditional-only | — |
| SD3.5 Medium | `StableDiffusion3Pipeline` | ✓ | — | ✓ | Joint attention | Flow Euler |
| Hunyuan-DiT | `HunyuanDiTPipeline` | ✓ | DDIM | ✓ | — | DDIM |
| CogVideoX-2B | `CogVideoXPipeline` | ✓ | CogVideoX DDIM | ✓ | — | CogVideoX DDIM |
| ModelScope 1.7B | `TextToVideoSDPipeline` | ✓ | DDIM | ✓ | — | DDIM |

## Architectural coverage

- **SAMG:** SD1.5, SDXL, SD3.5, CogVideoX and ModelScope. The SD2.1, DreamShaper,
  DPO and Hunyuan entries reuse the corresponding architecture adapters.
- **Oracle Noise:** SDXL, SDXL-Turbo and SD3.5. DreamShaper/DPO use the SDXL adapter;
  video and Hunyuan Oracle adapters are not provided.
- **Z²:** SD2.1, SDXL, DreamShaper, Hunyuan, CogVideoX and ModelScope. SD1.5 reuses
  the SD adapter; SD3.5 uses a deterministic flow extension.
- **SAMG variants:** CFG++ on SD1.5/SDXL and CFG-Zero* on SD3.5 are the reference
  configurations. Other applicable architecture/variant pairs use the same tensor and
  solver interfaces.

SDXL-Turbo and DreamShaper XL Turbo have different guidance settings. The former uses
`guidance_scale=0`; the latter uses ordinary CFG. Video loading profiles default to 49
frames for CogVideoX and 16 for ModelScope. Override generation settings explicitly as needed.

## Schedulers

| Configuration | Requirement |
|---|---|
| SAMG CFG / CFG-Zero*, without Z² | Native scheduler and separate conditional/unconditional branches |
| Oracle Noise alone | Native scheduler; raw-noise optimization precedes initial scaling |
| Z² on SD / SDXL / Hunyuan / ModelScope | DDIM, eta=0, clipping and thresholding disabled |
| Z² on CogVideoX | `CogVideoXDDIMScheduler`, eta=0, dynamic CFG disabled |
| Z² on SD3.5 | Deterministic `FlowMatchEulerDiscreteScheduler` |
| SAMG CFG++ | DDIM/CogVideoX DDIM with unconditional-epsilon renoising |
| AYS / custom time grid | `ManifoldDDIMScheduler` on SD/SDXL |

`load_pipeline()` selects the required deterministic scheduler for Z²/CFG++.
`plug()` keeps the current scheduler and validates compatibility. Switching a checkpoint's
native scheduler changes its sampling configuration; these are configurable model presets.

## Combinations

Oracle initialization can precede SAMG, Z², or SAMG CFG + Z² where both model adapters
exist. Combined quality and speed have not been benchmarked here.

SAMG CFG + Z² stores the previous guided-minus-inversion prediction as its temporal
surrogate. Z² with CFG++, CFG-Zero* or custom guidance backends is not supported.

## Interface constraints

- **SAMG/Z²:** `guidance_scale > 1`, `guidance_rescale=0`, and `use_dynamic_cfg=False`.
  Guidance-distilled time conditioning, SD3 skip-layer guidance and step callbacks are
  rejected because they alter the two-branch sampling contract.
- **Oracle:** requires prompt text, default text/attention settings and resident model
  weights. Use `.to(device)`, not CPU offload. `torch.no_grad()` is supported;
  `torch.inference_mode()` is not. Precomputed embeddings, distinct prompts per text
  encoder, IP-Adapter inputs and fused QKV are not supported by this adapter.
- **Pipelines:** text-to-image and text-to-video only. Image-to-image, inpainting,
  ControlNet, compiled pipelines and quantized Oracle gradients are outside this interface.
- **Concurrency:** wrappers can share weights for sequential use. Do not call wrappers or
  their underlying pipeline concurrently on the same instance.
