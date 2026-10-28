"""Spatial Adaptive Multi Guidance and its explicit backend extension point."""

import math
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F


def spatial_map(delta, *, channel_dim=1, lambda_min=0.0, lambda_max=1.0, smooth_kernel=1, temporal_kernel=1):
    """FP32 energy; normalize per sample (never across a batch).

    BCHW images, BCFHW videos, and BFCHW videos are supported. Spatial pooling
    uses zero padding. Video normalization
    spans one entire clip; temporal smoothing is optional and off by default.
    """
    if delta.ndim not in (4, 5):
        raise ValueError("SAMG expects BCHW, BCFHW or BFCHW tensors.")
    if channel_dim not in ((1,) if delta.ndim == 4 else (1, 2)):
        raise ValueError("Invalid channel_dim for the given tensor layout.")
    energy = delta.float().square().mean(dim=channel_dim, keepdim=True)
    if delta.ndim == 4:
        if temporal_kernel != 1:
            raise ValueError("temporal_kernel is only available for videos.")
        if smooth_kernel > 1:
            energy = F.avg_pool2d(energy, smooth_kernel, stride=1, padding=smooth_kernel // 2)
    else:
        energy = energy.movedim(channel_dim, 1)  # B,1,F,H,W
        if smooth_kernel > 1 or temporal_kernel > 1:
            kernel = (temporal_kernel, smooth_kernel, smooth_kernel)
            energy = F.avg_pool3d(energy, kernel, stride=1, padding=tuple(k // 2 for k in kernel))
        energy = energy.movedim(1, channel_dim)
    dims = tuple(range(1, energy.ndim))
    lo = energy.amin(dim=dims, keepdim=True)
    hi = energy.amax(dim=dims, keepdim=True)
    normalized = (energy - lo) / (hi - lo + 1e-8)
    return lambda_max - (lambda_max - lambda_min) * normalized


def optimal_unconditional(unconditional, conditional):
    """CFG-Zero*: project conditional output onto unconditional output."""
    u, c = unconditional.float(), conditional.float()
    dims = tuple(range(1, u.ndim))
    scale = (u * c).sum(dims, keepdim=True) / (u.square().sum(dims, keepdim=True) + 1e-8)
    return (u * scale).to(unconditional.dtype)


_BACKENDS: dict[str, Callable] = {}


def register_guidance(name: str, function: Callable, *, replace=False):
    """Register a prediction-space SAMG variant.

    function(component, unconditional, conditional, *, channel_dim, step_index)
    must return a prediction tensor of the same shape/dtype/device. Stateful
    samplers or extra model calls belong in a separate adapter, not this hook.
    """
    if not name or name in {"cfg", "cfg++", "cfg_zero", "cfgpp", "cfg-zero", "cfg_zero_star"}:
        raise ValueError("Choose a non-reserved guidance backend name.")
    if name in _BACKENDS and not replace:
        raise ValueError(f"Guidance backend already registered: {name}")
    if not callable(function):
        raise TypeError("function must be callable")
    _BACKENDS[name] = function


@dataclass(frozen=True)
class SAMG:
    base_scale: float = 1.0
    strong_scale: float = 7.5
    lambda_min: float = 0.0
    lambda_max: float = 1.0
    smooth_kernel: int = 1
    temporal_kernel: int = 1
    variant: str = "cfg"
    zero_init_steps: int = 1

    def __post_init__(self):
        aliases = {"cfgpp": "cfg++", "cfg-zero": "cfg_zero", "cfg_zero_star": "cfg_zero"}
        object.__setattr__(self, "variant", aliases.get(self.variant, self.variant))
        if self.variant not in {"cfg", "cfg++", "cfg_zero"} and self.variant not in _BACKENDS:
            raise ValueError(f"Unknown SAMG variant: {self.variant}. Use register_guidance for extensions.")
        if not all(
            math.isfinite(v) for v in (self.base_scale, self.strong_scale, self.lambda_min, self.lambda_max)
        ):
            raise ValueError("SAMG scales must be finite.")
        if self.strong_scale < self.base_scale or self.lambda_max < self.lambda_min:
            raise ValueError("strong_scale >= base_scale and lambda_max >= lambda_min are required.")
        if any(
            not isinstance(k, int) or k < 1 or k % 2 == 0 for k in (self.smooth_kernel, self.temporal_kernel)
        ):
            raise ValueError("Smoothing kernels must be positive odd integers.")
        if not isinstance(self.zero_init_steps, int) or self.zero_init_steps < 0:
            raise ValueError("zero_init_steps must be a nonnegative integer.")

    def mix(self, unconditional, conditional, *, channel_dim=1):
        """Mix two outputs (epsilon/velocity for CFG; x0 for CFG++)."""
        if unconditional.shape != conditional.shape:
            raise ValueError("Conditional and unconditional shapes differ.")
        u, c = unconditional.float(), conditional.float()
        base = u + self.base_scale * (c - u)
        delta = (self.strong_scale - self.base_scale) * (c - u)
        weights = spatial_map(
            delta,
            channel_dim=channel_dim,
            lambda_min=self.lambda_min,
            lambda_max=self.lambda_max,
            smooth_kernel=self.smooth_kernel,
            temporal_kernel=self.temporal_kernel,
        )
        return (base + weights * delta).to(unconditional.dtype)

    def __call__(self, unconditional, conditional, *, channel_dim=1, step_index=0):
        if self.variant == "cfg++":
            raise ValueError("CFG++ needs a scheduler transition. Use plug(..., samg=SAMG(variant='cfg++')).")
        if self.variant in _BACKENDS:
            result = _BACKENDS[self.variant](
                self, unconditional, conditional, channel_dim=channel_dim, step_index=step_index
            )
            if (
                not isinstance(result, torch.Tensor)
                or result.shape != unconditional.shape
                or result.dtype != unconditional.dtype
                or result.device != unconditional.device
            ):
                raise ValueError("A custom guidance backend must preserve shape, dtype and device.")
            return result
        if self.variant == "cfg_zero":
            if step_index < self.zero_init_steps:
                return torch.zeros_like(unconditional)
            unconditional = optimal_unconditional(unconditional, conditional)
        return self.mix(unconditional, conditional, channel_dim=channel_dim)
