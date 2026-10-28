"""Z²-Sampling: zero-cost trajectory correction with a temporal semantic surrogate."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Z2Sampling:
    zigzag_steps: int = 10
    warmup_steps: int = 1
    inversion_guidance: float = 0.0

    def __post_init__(self):
        if not isinstance(self.warmup_steps, int) or self.warmup_steps < 1:
            raise ValueError("At least one warmup step is needed to initialize the temporal surrogate.")
        if not isinstance(self.zigzag_steps, int) or self.zigzag_steps < 0:
            raise ValueError("zigzag_steps must be a nonnegative integer (an exclusive step-index cutoff).")
        if not math.isfinite(self.inversion_guidance):
            raise ValueError("inversion_guidance must be finite.")

    def active(self, step_index, total_steps):
        return self.warmup_steps <= step_index < min(self.zigzag_steps, total_steps - 1)

    def shift(self, sample, previous_delta, coefficient, guidance_scale):
        if previous_delta is None:
            raise RuntimeError("Z²-Sampling surrogate is missing; run the warmup first.")
        return sample - coefficient * (guidance_scale - self.inversion_guidance) * previous_delta

    def correct(self, prediction, previous_delta, guidance_scale):
        # The OLD delta is intentionally used, then updated by the caller.
        return prediction + (guidance_scale - self.inversion_guidance) * previous_delta
