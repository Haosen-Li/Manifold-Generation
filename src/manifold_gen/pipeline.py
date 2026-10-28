"""Scoped component insertion around native Diffusers pipelines."""

import inspect
from contextlib import ExitStack, contextmanager
from functools import wraps
from types import SimpleNamespace

import torch
from diffusers import CogVideoXDDIMScheduler, DDIMScheduler, FlowMatchEulerDiscreteScheduler
from diffusers.schedulers.scheduling_ddim import DDIMSchedulerOutput

from .adapters import adapter_for
from .schedulers import ManifoldDDIMScheduler, alpha_pair, from_epsilon, to_epsilon


@contextmanager
def _replace(obj, name, value):
    existed = name in obj.__dict__
    previous = obj.__dict__.get(name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        if existed:
            setattr(obj, name, previous)
        else:
            delattr(obj, name)


def _prediction(output):
    return output[0] if isinstance(output, (tuple, list)) else output.sample


def _replace_prediction(output, value):
    if isinstance(output, tuple):
        return (value,) + output[1:]
    if isinstance(output, list):
        return [value] + output[1:]
    # Diffusers BaseOutput updates its mapping when sample is assigned.
    output.sample = value
    return output


def _flatten_video(value, family):
    if family == "modelscope" and value.ndim == 5:
        b, c, f, h, w = value.shape
        return value.permute(0, 2, 1, 3, 4).reshape(b * f, c, h, w)
    return value


class ManifoldPipeline:
    """Use plug(pipe, samg=..., oracle_noise=..., z2=...), then call normally.

    Underlying weights, safety checker, VAE, conditioning and returned output
    types are unchanged. Overrides exist only inside this wrapper's call.
    Do not concurrently call the underlying native pipe from another thread.
    """

    def __init__(self, pipe, *, samg=None, oracle_noise=None, z2=None):
        self.pipe = pipe
        self.samg = samg
        self.oracle_noise = oracle_noise
        self.z2 = z2
        self.adapter = adapter_for(pipe)
        self.last_stats = {}

    def __getattr__(self, name):
        return getattr(self.pipe, name)

    def to(self, *args, **kwargs):
        self.pipe.to(*args, **kwargs)
        return self

    def with_components(self, *, samg=None, oracle_noise=None, z2=None):
        """Return a new wrapper sharing weights, with exactly these components."""
        return ManifoldPipeline(self.pipe, samg=samg, oracle_noise=oracle_noise, z2=z2)

    def _validate(self, options):
        active = self.samg is not None or self.z2 is not None
        model = getattr(self.pipe, self.adapter.model_attr)
        if active:
            if options.get("guidance_scale", 1.0) <= 1:
                raise ValueError("SAMG/Z² require separate CFG branches: set guidance_scale > 1.")
            if getattr(model.config, "time_cond_proj_dim", None) is not None:
                raise ValueError(
                    "Guidance-distilled time-conditioning models do not expose two CFG branches."
                )
            if options.get("guidance_rescale", 0.0) != 0 or options.get("use_dynamic_cfg", False):
                raise ValueError("Use guidance_rescale=0 and use_dynamic_cfg=False with SAMG/Z².")
            if options.get("skip_guidance_layers"):
                raise ValueError("SD3 skip-layer guidance adds model calls and cannot be combined here.")
            for name in ("callback", "callback_on_step_end"):
                if options.get(name) is not None:
                    raise ValueError(
                        "Callbacks can change CFG branches/latents; they are not supported by this release."
                    )
        affine = self.z2 is not None or (self.samg is not None and self.samg.variant == "cfg++")
        scheduler = self.pipe.scheduler
        if affine:
            if not isinstance(
                scheduler, (DDIMScheduler, CogVideoXDDIMScheduler, FlowMatchEulerDiscreteScheduler)
            ):
                raise ValueError(
                    "Z²/CFG++ require deterministic DDIM/CogVideoXDDIM, or FlowMatchEuler for Z². "
                    "Use load_pipeline() or explicitly choose ManifoldDDIMScheduler."
                )
            if options.get("eta", 0.0) != 0:
                raise ValueError("Z² and CFG++ require eta=0.")
            if isinstance(scheduler, DDIMScheduler) and (
                getattr(scheduler.config, "clip_sample", False)
                or getattr(scheduler.config, "thresholding", False)
            ):
                raise ValueError("Disable scheduler clip_sample/thresholding for affine transitions.")
            if isinstance(scheduler, FlowMatchEulerDiscreteScheduler):
                if getattr(scheduler.config, "stochastic_sampling", False):
                    raise ValueError("Stochastic flow sampling is not supported by Z².")
            if (
                self.samg is not None
                and self.samg.variant == "cfg++"
                and isinstance(scheduler, FlowMatchEulerDiscreteScheduler)
            ):
                raise ValueError(
                    "CFG++ requires an x0/unconditional-epsilon DDIM transition; flow solvers are unsupported."
                )
        if self.z2 is not None and self.samg is not None and self.samg.variant != "cfg":
            raise ValueError(
                "Z² + SAMG currently supports variant='cfg' only; use other variants independently."
            )
        if self.oracle_noise is not None:
            if self.adapter.family not in {"sdxl", "sd3"}:
                raise ValueError("Oracle Noise is implemented for SDXL/Turbo and SD3.5.")
            if options.get("prompt") is None:
                raise ValueError("Oracle Noise needs prompt text for token routing, not only prompt_embeds.")
            for key in (
                "prompt_embeds",
                "negative_prompt_embeds",
                "prompt_2",
                "prompt_3",
                "negative_prompt_2",
                "negative_prompt_3",
                "ip_adapter_image",
                "ip_adapter_image_embeds",
            ):
                if options.get(key) is not None:
                    raise ValueError(
                        f"Oracle Noise does not support {key}; use one shared text prompt per sample."
                    )
            if (
                options.get("clip_skip") is not None
                or options.get("cross_attention_kwargs")
                or options.get("joint_attention_kwargs")
            ):
                raise ValueError("Oracle Noise currently expects default text/attention settings.")

    def __call__(self, *args, **kwargs):
        signature = inspect.signature(self.pipe.__call__)
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        options = bound.arguments
        self._validate(options)
        if getattr(self.pipe, "_manifold_gen_active", False):
            raise RuntimeError("This pipeline already has an active Manifold Gen call.")
        native = self.pipe
        model = getattr(native, self.adapter.model_attr)
        scheduler = native.scheduler
        model_forward = model.forward
        model_signature = inspect.signature(model_forward)
        scheduler_step = scheduler.step
        step_signature = inspect.signature(scheduler_step)
        prepare = native.prepare_latents
        prepare_signature = inspect.signature(prepare)
        samg, z2, oracle = self.samg, self.z2, self.oracle_noise
        guidance = options.get("guidance_scale", 1.0)
        family, channel_dim = self.adapter.family, self.adapter.channel_dim
        is_flow = isinstance(scheduler, FlowMatchEulerDiscreteScheduler)
        affine = z2 is not None or (samg is not None and samg.variant == "cfg++")
        state = {"index": 0, "correction": None, "oracle": False, "x0": None, "eps_u": None}
        stats = {
            "denoiser_calls": 0,
            "cfg_branch_evaluations": 0,
            "oracle_denoiser_calls": 0,
            "scheduler_steps": 0,
            "z2_corrected_steps": 0,
            "zero_init_skipped_calls": 0,
        }
        self.last_stats = stats

        @wraps(prepare)
        def prepared(*a, **kw):
            result = prepare(*a, **kw)
            if oracle is None:
                return result
            data = prepare_signature.bind(*a, **kw)
            data.apply_defaults()
            height, width = data.arguments["height"], data.arguments["width"]
            prompts = options["prompt"]
            prompts = [prompts] if isinstance(prompts, str) else list(prompts)
            repeat = options.get("num_images_per_prompt", 1)
            negative = options.get("negative_prompt")
            negatives = (
                [negative or ""] * len(prompts) if isinstance(negative, (str, type(None))) else list(negative)
            )
            prompts = [p for p in prompts for _ in range(repeat)]
            negatives = [p for p in negatives for _ in range(repeat)]
            sigma = 1.0 if family == "sd3" else scheduler.init_noise_sigma
            supplied_latents = data.arguments.get("latents")
            raw_latents = (
                supplied_latents.to(device=result.device, dtype=result.dtype)
                if supplied_latents is not None
                else result / sigma
            )
            extra = {
                key: options[key]
                for key in (
                    "max_sequence_length",
                    "original_size",
                    "target_size",
                    "crops_coords_top_left",
                    "negative_original_size",
                    "negative_target_size",
                    "negative_crops_coords_top_left",
                )
                if key in options
            }
            state["oracle"] = True
            try:
                optimized = oracle.optimize(
                    native,
                    prompts,
                    raw_latents,
                    negative_prompt=negatives,
                    guidance_scale=guidance,
                    height=height,
                    width=width,
                    **extra,
                )
                return optimized * sigma
            finally:
                state["oracle"] = False

        @wraps(model_forward)
        def forward(*a, **kw):
            if state["oracle"]:
                stats["oracle_denoiser_calls"] += 1
                return model_forward(*a, **kw)
            if samg is None and z2 is None:
                stats["denoiser_calls"] += 1
                return model_forward(*a, **kw)
            data = model_signature.bind(*a, **kw)
            data.apply_defaults()
            sample = data.arguments[self.adapter.input_name]
            if sample.shape[0] % 2:
                raise RuntimeError("Expected unconditional/conditional CFG batches.")
            batch = sample.shape[0] // 2
            original = sample[:batch]
            timestep = data.arguments.get("timestep", data.arguments.get("timesteps"))
            t = timestep.flatten()[0] if isinstance(timestep, torch.Tensor) else timestep
            index = state["index"]
            active = z2 is not None and z2.active(index, len(scheduler.timesteps))
            if samg is not None and samg.variant == "cfg_zero" and index < samg.zero_init_steps:
                value = torch.zeros_like(sample)
                if self.adapter.learned_variance:
                    value = torch.cat([value, value], dim=channel_dim)
                stats["zero_init_skipped_calls"] += 1
                return (
                    (value,) if not data.arguments.get("return_dict", True) else SimpleNamespace(sample=value)
                )
            alpha = None
            if affine and not is_flow:
                alpha, previous = alpha_pair(scheduler, t, index, original)
            evaluation = original
            if active:
                if state["correction"] is None:
                    raise RuntimeError("Missing Z² temporal surrogate after warmup.")
                if is_flow:
                    coefficient = (scheduler.sigmas[index] - scheduler.sigmas[index + 1]).to(original.device)
                else:
                    coefficient = (1 - alpha).sqrt() - (alpha / previous).sqrt() * (1 - previous).sqrt()
                evaluation = original - coefficient * state["correction"]
                data.arguments[self.adapter.input_name] = torch.cat([evaluation, evaluation], dim=0).to(
                    sample.dtype
                )
                stats["z2_corrected_steps"] += 1
            output = model_forward(*data.args, **data.kwargs)
            stats["denoiser_calls"] += 1
            stats["cfg_branch_evaluations"] += 2
            raw = _prediction(output)
            prediction, variance = (
                raw.chunk(2, dim=channel_dim) if self.adapter.learned_variance else (raw, None)
            )
            u, c = prediction.chunk(2)
            if affine and not is_flow:
                u = to_epsilon(u, evaluation, alpha, scheduler.config.prediction_type)
                c = to_epsilon(c, evaluation, alpha, scheduler.config.prediction_type)
            if samg is not None and samg.variant == "cfg++":
                x0u = (original - (1 - alpha).sqrt() * u) / alpha.sqrt()
                x0c = (original - (1 - alpha).sqrt() * c) / alpha.sqrt()
                state["x0"] = samg.mix(x0u, x0c, channel_dim=channel_dim)
                state["eps_u"] = u
                guided = u  # The scoped scheduler transition below consumes x0 and eps_u.
            else:
                guided = (
                    samg(u, c, channel_dim=channel_dim, step_index=index) if samg else u + guidance * (c - u)
                )
            if z2 is not None:
                new_correction = guided - (u + z2.inversion_guidance * (c - u))
                if active:
                    guided = guided + state["correction"]
                # Never replace the cached OLD surrogate before applying the correction.
                state["correction"] = new_correction.detach()
            if affine and not is_flow:
                guided = from_epsilon(guided, original, alpha, scheduler.config.prediction_type)
            value = torch.cat([guided, guided], dim=0).to(prediction.dtype)
            if variance is not None:
                value = torch.cat([value, variance], dim=channel_dim)
            return _replace_prediction(output, value)

        @wraps(scheduler_step)
        def step(*a, **kw):
            if samg is not None and samg.variant == "cfg++":
                data = step_signature.bind(*a, **kw)
                data.apply_defaults()
                sample, t = data.arguments["sample"], data.arguments["timestep"]
                if state["x0"] is None:
                    raise RuntimeError("CFG++ scheduler called before its model prediction.")
                _, prev_alpha = alpha_pair(scheduler, t, state["index"], sample)
                x0 = _flatten_video(state["x0"], family)
                eps = _flatten_video(state["eps_u"], family)
                previous = (prev_alpha.sqrt() * x0 + (1 - prev_alpha).sqrt() * eps).to(sample.dtype)
                output = DDIMSchedulerOutput(prev_sample=previous, pred_original_sample=x0)
                if not data.arguments.get("return_dict", True):
                    output = (previous, x0)
                if isinstance(scheduler, ManifoldDDIMScheduler):
                    scheduler._manifold_step_index += 1
                state["x0"] = None
            else:
                output = scheduler_step(*a, **kw)
            state["index"] += 1
            stats["scheduler_steps"] += 1
            return output

        with ExitStack() as stack:
            stack.enter_context(_replace(native, "_manifold_gen_active", True))
            stack.enter_context(_replace(model, "forward", forward))
            stack.enter_context(_replace(scheduler, "step", step))
            if oracle is not None:
                stack.enter_context(_replace(native, "prepare_latents", prepared))
            return native(*args, **kwargs)


def plug(pipe, *, samg=None, oracle_noise=None, z2=None):
    """Insert the three independent components. None leaves a stage unchanged."""
    if isinstance(pipe, ManifoldPipeline):
        return pipe.with_components(samg=samg, oracle_noise=oracle_noise, z2=z2)
    return ManifoldPipeline(pipe, samg=samg, oracle_noise=oracle_noise, z2=z2)
