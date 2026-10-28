"""Deterministic DDIM with an explicit, possibly nonuniform, timestep grid."""

import torch
from diffusers import DDIMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMSchedulerOutput


class ManifoldDDIMScheduler(DDIMScheduler):
    """DDIM eta=0. Supports AYS/custom grids using the actual next timestep.

    Standard DDIMScheduler.step computes t_prev from a fixed stride. This class
    makes the solver transition and the Z² correction use the same grid.
    """

    def set_timesteps(self, num_inference_steps=None, device=None, timesteps=None):
        if timesteps is None:
            super().set_timesteps(num_inference_steps, device=device)
        else:
            if num_inference_steps is not None and num_inference_steps != len(timesteps):
                raise ValueError("num_inference_steps must equal len(timesteps) when both are supplied.")
            values = list(timesteps)
            if (
                not values
                or any(int(t) != t for t in values)
                or any(t < 0 or t >= self.config.num_train_timesteps for t in values)
                or any(a <= b for a, b in zip(values, values[1:], strict=False))
            ):
                raise ValueError("timesteps must be strictly decreasing integers inside the training grid.")
            self.timesteps = torch.tensor(values, device=device, dtype=torch.long)
            self.num_inference_steps = len(values)
        self._manifold_step_index = 0

    def step(
        self,
        model_output,
        timestep,
        sample,
        eta=0.0,
        use_clipped_model_output=False,
        generator=None,
        variance_noise=None,
        return_dict=True,
    ):
        if eta != 0 or variance_noise is not None:
            raise ValueError("ManifoldDDIMScheduler is deterministic: use eta=0 and no variance_noise.")
        if self.config.clip_sample or self.config.thresholding or use_clipped_model_output:
            raise ValueError("Use clip_sample=False and thresholding=False for the affine DDIM solver.")
        idx = self._manifold_step_index
        if idx >= len(self.timesteps) or int(timestep) != int(self.timesteps[idx]):
            raise ValueError("Unexpected timestep; call set_timesteps() before a new generation.")
        alpha, previous = alpha_pair(self, timestep, idx, sample)
        eps = to_epsilon(model_output, sample, alpha, self.config.prediction_type)
        x0 = (sample - (1 - alpha).sqrt() * eps) / alpha.sqrt()
        prev = previous.sqrt() * x0 + (1 - previous).sqrt() * eps
        self._manifold_step_index += 1
        if not return_dict:
            return (prev,)
        return DDIMSchedulerOutput(prev_sample=prev, pred_original_sample=x0)


def alpha_pair(scheduler, timestep, step_index, reference):
    t = int(timestep)
    alpha = scheduler.alphas_cumprod[t].to(device=reference.device, dtype=torch.float32)
    if isinstance(scheduler, ManifoldDDIMScheduler):
        prev_t = int(scheduler.timesteps[step_index + 1]) if step_index + 1 < len(scheduler.timesteps) else -1
    else:
        # Matches diffusers DDIMScheduler and CogVideoXDDIMScheduler exactly.
        prev_t = t - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
    previous = scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else scheduler.final_alpha_cumprod
    return alpha, previous.to(device=reference.device, dtype=torch.float32)


def to_epsilon(prediction, sample, alpha, prediction_type):
    if prediction_type == "epsilon":
        return prediction
    if prediction_type == "v_prediction":
        return alpha.sqrt() * prediction + (1 - alpha).sqrt() * sample
    if prediction_type == "sample":
        return (sample - alpha.sqrt() * prediction) / (1 - alpha).sqrt().clamp_min(1e-8)
    raise ValueError(f"Unsupported prediction type: {prediction_type}")


def from_epsilon(eps, sample, alpha, prediction_type):
    if prediction_type == "epsilon":
        return eps
    if prediction_type == "v_prediction":
        return (eps - (1 - alpha).sqrt() * sample) / alpha.sqrt().clamp_min(1e-8)
    if prediction_type == "sample":
        return (sample - (1 - alpha).sqrt() * eps) / alpha.sqrt().clamp_min(1e-8)
    raise ValueError(f"Unsupported prediction type: {prediction_type}")
