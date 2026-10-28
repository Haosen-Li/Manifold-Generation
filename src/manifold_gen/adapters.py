"""Native Diffusers adapters; all conditioning/rotary/decoding stays upstream."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Adapter:
    family: str
    pipeline_class: str
    model_attr: str
    input_name: str
    channel_dim: int = 1
    learned_variance: bool = False


ADAPTERS = (
    Adapter("sd", "StableDiffusionPipeline", "unet", "sample"),
    Adapter("sdxl", "StableDiffusionXLPipeline", "unet", "sample"),
    Adapter("sd3", "StableDiffusion3Pipeline", "transformer", "hidden_states"),
    Adapter("hunyuan", "HunyuanDiTPipeline", "transformer", "hidden_states", learned_variance=True),
    Adapter("cogvideox", "CogVideoXPipeline", "transformer", "hidden_states", channel_dim=2),
    Adapter("modelscope", "TextToVideoSDPipeline", "unet", "sample"),
)


def adapter_for(pipe):
    classes = {cls.__name__ for cls in type(pipe).__mro__}
    for adapter in ADAPTERS:
        if adapter.pipeline_class in classes:
            return adapter
    raise ValueError(
        f"Unsupported pipeline {type(pipe).__name__}. Supported: "
        + ", ".join(a.pipeline_class for a in ADAPTERS)
    )


@dataclass(frozen=True)
class ModelSpec:
    family: str
    model_id: str
    height: int
    width: int
    steps: int
    guidance: float
    frames: int | None = None


MODELS = {
    "sd15": ModelSpec("sd", "stable-diffusion-v1-5/stable-diffusion-v1-5", 512, 512, 50, 7.5),
    "sd21": ModelSpec("sd", "stabilityai/stable-diffusion-2-1", 768, 768, 50, 7.5),
    "sdxl": ModelSpec("sdxl", "stabilityai/stable-diffusion-xl-base-1.0", 1024, 1024, 50, 5.0),
    "sdxl-turbo": ModelSpec("sdxl", "stabilityai/sdxl-turbo", 512, 512, 1, 0.0),
    "sd35": ModelSpec("sd3", "stabilityai/stable-diffusion-3.5-medium", 1024, 1024, 28, 4.5),
    "dreamshaper-xl": ModelSpec("sdxl", "Lykon/dreamshaper-xl-v2-turbo", 1024, 1024, 8, 2.0),
    "hunyuan-dit": ModelSpec("hunyuan", "Tencent-Hunyuan/HunyuanDiT-Diffusers", 1024, 1024, 50, 5.0),
    "cogvideox": ModelSpec("cogvideox", "THUDM/CogVideoX-2b", 480, 720, 50, 6.0, 49),
    "modelscope": ModelSpec("modelscope", "damo-vilab/text-to-video-ms-1.7b", 256, 256, 50, 9.0, 16),
}


def load_pipeline(
    model,
    *,
    model_id=None,
    device="cuda",
    torch_dtype=None,
    samg=None,
    oracle_noise=None,
    z2=None,
    **pretrained_kwargs,
):
    """Load a documented model profile, then insert independent components.

    model_id may be a local directory or a compatible fine-tuned checkpoint,
    including Diffusion-DPO. No weights are bundled or downloaded on import.
    """
    import diffusers
    import torch

    from .pipeline import plug
    from .schedulers import ManifoldDDIMScheduler

    if model not in MODELS:
        raise ValueError(f"Unknown profile {model!r}; choices: {', '.join(MODELS)}")
    spec = MODELS[model]
    if model == "sdxl-turbo" and (samg is not None or z2 is not None):
        raise ValueError("SDXL-Turbo uses guidance_scale=0: this profile supports Oracle Noise only.")
    if oracle_noise is not None and spec.family not in {"sdxl", "sd3"}:
        raise ValueError("Oracle Noise has SDXL/SDXL-Turbo and SD3.5 adapters only.")
    if torch_dtype is None:
        torch_dtype = (
            torch.float32
            if str(device) == "cpu"
            else (torch.bfloat16 if spec.family == "sd3" else torch.float16)
        )
    cls = getattr(diffusers, next(a.pipeline_class for a in ADAPTERS if a.family == spec.family))
    pipe = cls.from_pretrained(model_id or spec.model_id, torch_dtype=torch_dtype, **pretrained_kwargs)
    if z2 is not None or (samg is not None and samg.variant == "cfg++"):
        if spec.family == "cogvideox":
            pipe.scheduler = diffusers.CogVideoXDDIMScheduler.from_config(pipe.scheduler.config)
        elif spec.family != "sd3":
            pipe.scheduler = ManifoldDDIMScheduler.from_config(
                pipe.scheduler.config, clip_sample=False, thresholding=False
            )
    return plug(pipe.to(device), samg=samg, oracle_noise=oracle_noise, z2=z2)
