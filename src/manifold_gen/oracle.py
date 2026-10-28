"""Oracle Noise: attention-guided latent optimization on a hypersphere."""

import math
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .adapters import adapter_for


def spherical_update(x, gradient, theta):
    """Per-sample exponential-map step. A zero tangent leaves x unchanged."""
    original_dtype = x.dtype
    xf, gf = x.detach().float(), gradient.detach().float()
    dims = tuple(range(1, x.ndim))
    radius = xf.square().sum(dims, keepdim=True).sqrt()
    tangent = gf - (xf * gf).sum(dims, keepdim=True) / radius.square().clamp_min(1e-12) * xf
    tangent_norm = tangent.square().sum(dims, keepdim=True).sqrt()
    valid = (radius > 1e-12) & (tangent_norm > 1e-8)
    result = xf * math.cos(theta) + radius * tangent / tangent_norm.clamp_min(1e-12) * math.sin(theta)
    # Correct only numerical roundoff; no change to the direction of the update.
    result = result * radius / result.square().sum(dims, keepdim=True).sqrt().clamp_min(1e-12)
    return torch.where(valid, result, xf).to(original_dtype).detach()


def _importance(tokenizer, encoder, prompt, device, length):
    ids = tokenizer(
        prompt, padding="max_length", max_length=length, truncation=True, return_tensors="pt"
    ).input_ids.to(device)
    specials = {tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id}
    valid = torch.tensor([[int(t) not in specials for t in ids[0]]], device=device)[0]
    eos = (ids[0] == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
    eos_index = int(eos[0]) if eos.numel() else max(int(valid.sum()) - 1, 0)

    def embedding(tokens):
        out = encoder(tokens)[0]
        return out if out.ndim == 2 else out[:, eos_index]

    with torch.no_grad():
        baseline = embedding(ids)
        importance = torch.zeros(length, device=device)
        for i in valid.nonzero(as_tuple=True)[0].tolist():
            masked = ids.clone()
            masked[0, i] = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            importance[i] = 1 - F.cosine_similarity(baseline.float(), embedding(masked).float(), dim=-1)[0]
    return importance, valid


def token_importance(pipe, prompts, *, device, joint=False, min_weight=0.5, max_weight=3.0):
    """Two-CLIP leave-one-token-out weights for UNet and joint-attention objectives."""
    rows = []
    length = pipe.tokenizer.model_max_length
    for prompt in prompts:
        a, va = _importance(pipe.tokenizer, pipe.text_encoder, prompt, device, length)
        b, vb = _importance(pipe.tokenizer_2, pipe.text_encoder_2, prompt, device, length)
        valid = va | vb if joint else va
        score = torch.nan_to_num((a + b) / 2, nan=0.0, posinf=max_weight, neginf=0.0)
        row = torch.full_like(score, 1.0 if joint else 0.0)
        if valid.any():
            values = score[valid]
            span = values.max() - values.min()
            row[valid] = (
                ((values - values.min()) / span.clamp_min(1e-8) * (max_weight - min_weight) + min_weight)
                if span > 0
                else (min_weight + max_weight) / 2
            )
        rows.append(row)
    return torch.stack(rows)


@contextmanager
def _frozen(model):
    params = list(model.parameters())
    flags = [p.requires_grad for p in params]
    training = [(m, m.training) for m in model.modules()]
    checkpointing = bool(getattr(model, "is_gradient_checkpointing", False))
    try:
        for p in params:
            p.requires_grad_(False)
        model.eval()
        if checkpointing:
            model.disable_gradient_checkpointing()
        yield
    finally:
        for p, flag in zip(params, flags, strict=True):
            p.requires_grad_(flag)
        for m, flag in training:
            m.training = flag
        if checkpointing:
            model.enable_gradient_checkpointing()


def _unet_loss(cq, ck, uq, uk, mask, scale):
    factor = cq.shape[-1] ** -0.5
    logits = torch.matmul(cq.float(), ck.float().transpose(-1, -2)) * factor
    if uq is not None:
        uncond = torch.matmul(uq.float(), uk.float().transpose(-1, -2)) * factor
        logits = uncond + scale * (logits - uncond)
    return (logits.softmax(-1) * mask[:, None, :]).sum()


def _joint_loss(cq, ck, ct, uq, uk, ut, mask, scale, key_chunk):
    """Exact joint-softmax denominator using streaming logsumexp.

    Unlike text-only softmax, this includes image keys. No in-place autograd
    accumulation; query chunk checkpointing limits saved attention matrices.
    The objective uses raw Q/K projections before native QK normalization.
    """
    factor = cq.shape[-1] ** -0.5

    def logits(kc, ku):
        conditional = torch.matmul(cq.float(), kc.float().transpose(-1, -2)) * factor
        if uq is None:
            return conditional
        unconditional = torch.matmul(uq.float(), ku.float().transpose(-1, -2)) * factor
        return unconditional + scale * (conditional - unconditional)

    text_logits = logits(ct, ut)
    log_denominator = torch.logsumexp(text_logits, dim=-1, keepdim=True)
    for start in range(0, ck.shape[-2], key_chunk):
        image_logits = logits(
            ck[..., start : start + key_chunk, :],
            None if uk is None else uk[..., start : start + key_chunk, :],
        )
        log_denominator = torch.logaddexp(log_denominator, torch.logsumexp(image_logits, -1, keepdim=True))
    return ((text_logits - log_denominator).exp() * mask[:, None, None, :]).sum()


@dataclass(frozen=True)
class OracleNoise:
    theta: float = 0.02
    steps: int = 1
    cfg_aware: bool | None = None  # auto: conditional-only when guidance_scale <= 1
    use_token_weights: bool = True
    min_weight: float = 0.5
    max_weight: float = 3.0
    optimization_timestep: float = 999.0
    query_chunk_size: int = 256
    key_chunk_size: int = 256
    checkpoint_attention: bool = True
    sd3_layer_indices: tuple[int, ...] | None = None

    def __post_init__(self):
        if not math.isfinite(self.theta) or not 0 <= self.theta < math.pi:
            raise ValueError("theta must be a finite angle in [0, pi).")
        if not isinstance(self.steps, int) or self.steps < 0:
            raise ValueError("steps must be a nonnegative integer.")
        if self.query_chunk_size < 1 or self.key_chunk_size < 1:
            raise ValueError("Attention chunk sizes must be positive.")
        if not 0 <= self.min_weight <= self.max_weight or not math.isfinite(self.max_weight):
            raise ValueError("Expected 0 <= min_weight <= max_weight < infinity.")

    def optimize(
        self,
        pipe,
        prompt,
        latents,
        *,
        negative_prompt="",
        guidance_scale=5.0,
        height=None,
        width=None,
        token_weights=None,
        max_sequence_length=256,
        original_size=None,
        target_size=None,
        crops_coords_top_left=(0, 0),
        negative_original_size=None,
        negative_target_size=None,
        negative_crops_coords_top_left=(0, 0),
    ):
        """Optimize *unscaled* Gaussian latents; callers may then pass them to pipe.

        Supported: SDXL, SDXL-Turbo (conditional-only), SD3.5 Medium. Batch
        prompts must match the latent batch. Models are frozen temporarily and
        forward hooks/gradient-checkpointing state are restored even on errors.
        """
        adapter = adapter_for(pipe)
        if adapter.family not in {"sdxl", "sd3"}:
            raise ValueError("Oracle Noise currently has adapters for SDXL/Turbo and SD3.5 only.")
        if torch.is_inference_mode_enabled():
            raise RuntimeError("Oracle Noise needs gradients; use no_grad(), not inference_mode().")
        if latents.ndim != 4 or not latents.is_floating_point():
            raise ValueError("Oracle Noise expects floating BCHW latents.")
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        if len(prompts) != latents.shape[0] or not all(isinstance(p, str) for p in prompts):
            raise ValueError("Provide one text prompt for each latent sample.")
        if self.steps == 0 or self.theta == 0:
            return latents.detach().clone()
        negatives = (
            [negative_prompt or ""] * len(prompts)
            if isinstance(negative_prompt, (str, type(None)))
            else list(negative_prompt)
        )
        if len(negatives) != len(prompts):
            raise ValueError("Negative prompt batch must match prompt batch.")
        model = getattr(pipe, adapter.model_attr)
        if any(hasattr(m, "_hf_hook") for m in model.modules()):
            raise ValueError("Oracle Noise needs a resident model. Use pipe.to(device), without CPU offload.")
        device = latents.device
        joint = adapter.family == "sd3"
        cfg = guidance_scale > 1 if self.cfg_aware is None else self.cfg_aware
        height = height or latents.shape[-2] * pipe.vae_scale_factor
        width = width or latents.shape[-1] * pipe.vae_scale_factor
        with torch.no_grad():
            encode_args = dict(
                prompt=prompts,
                negative_prompt=negatives,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=cfg,
            )
            if joint:
                encode_args.update(prompt_2=None, prompt_3=None, max_sequence_length=max_sequence_length)
            pos, neg, pooled, neg_pooled = pipe.encode_prompt(**encode_args)
            if token_weights is None and self.use_token_weights:
                token_weights = token_importance(
                    pipe,
                    prompts,
                    device=device,
                    joint=joint,
                    min_weight=self.min_weight,
                    max_weight=self.max_weight,
                )
            if token_weights is not None:
                mask = token_weights.to(device=device, dtype=torch.float32)
                if mask.ndim != 2 or mask.shape[0] != latents.shape[0]:
                    raise ValueError("token_weights must have shape [batch, text_tokens].")
                if not torch.isfinite(mask).all():
                    raise ValueError("token_weights must be finite.")
            elif joint:
                mask = torch.ones(len(prompts), pos.shape[1], device=device)
            else:
                ids = pipe.tokenizer(
                    prompts,
                    padding="max_length",
                    truncation=True,
                    max_length=pos.shape[1],
                    return_tensors="pt",
                ).input_ids.to(device)
                mask = torch.ones_like(ids, dtype=torch.float32)
                for special in {
                    pipe.tokenizer.bos_token_id,
                    pipe.tokenizer.eos_token_id,
                    pipe.tokenizer.pad_token_id,
                }:
                    if special is not None:
                        mask[ids == special] = 0
            if joint and mask.shape[1] < pos.shape[1]:
                mask = F.pad(mask, (0, pos.shape[1] - mask.shape[1]), value=1.0)
            if mask.shape[1] != pos.shape[1]:
                raise ValueError("Token weights must match the encoded text length (or CLIP length for SD3).")
        if joint:
            n = len(model.transformer_blocks)
            indices = self.sd3_layer_indices or (0, (n - 1) // 2, n - 1)
            if any(i < 0 or i >= n for i in indices):
                raise ValueError("SD3 attention layer index is out of range.")
            layers = [model.transformer_blocks[i].attn for i in indices]
        else:
            try:
                layers = [
                    model.down_blocks[1].attentions[0].transformer_blocks[0].attn2,
                    model.mid_block.attentions[0].transformer_blocks[0].attn2,
                    model.up_blocks[1].attentions[0].transformer_blocks[0].attn2,
                ]
            except (AttributeError, IndexError) as exc:
                raise ValueError(
                    "This UNet does not have the SDXL attention layout required by Oracle Noise."
                ) from exc
        layer_weights = [1.0 + i / max(len(layers) - 1, 1) for i in range(len(layers))]
        captured = {}
        handles = []

        def hook(key):
            def save(module, args, output):
                captured[key] = output

            return save

        def forward(x, text, pool, negative=False):
            captured.clear()
            t = torch.full(
                (x.shape[0],),
                self.optimization_timestep,
                device=device,
                dtype=torch.float32 if joint else torch.long,
            )
            if joint:
                model(
                    hidden_states=x,
                    encoder_hidden_states=text,
                    pooled_projections=pool,
                    timestep=t,
                    return_dict=False,
                )
            else:
                size = (negative_original_size if negative else original_size) or (height, width)
                target = (negative_target_size if negative else target_size) or (height, width)
                crop = negative_crops_coords_top_left if negative else crops_coords_top_left
                time_ids = torch.tensor(
                    [tuple(size) + tuple(crop) + tuple(target)], dtype=text.dtype, device=device
                ).repeat(len(prompts), 1)
                model(
                    x,
                    t,
                    encoder_hidden_states=text,
                    added_cond_kwargs={"text_embeds": pool, "time_ids": time_ids},
                    return_dict=False,
                )
            expected = len(layers) * (3 if joint else 2)
            if len(captured) != expected:
                raise RuntimeError(
                    "Attention projections were not called. Disable fused QKV/custom processors."
                )
            return dict(captured)

        try:
            for i, layer in enumerate(layers):
                for name in ("to_q", "to_k", "add_k_proj") if joint else ("to_q", "to_k"):
                    handles.append(getattr(layer, name).register_forward_hook(hook((i, name))))
            current = latents.detach().clone()
            with _frozen(model), torch.enable_grad():
                for _ in range(self.steps):
                    current = current.detach().requires_grad_(True)
                    conditional = forward(current, pos, pooled)
                    unconditional = forward(current, neg, neg_pooled, True) if cfg else None
                    loss = current.new_zeros((), dtype=torch.float32)
                    for i, (layer, weight) in enumerate(zip(layers, layer_weights, strict=True)):
                        cq, ck = conditional[i, "to_q"], conditional[i, "to_k"]
                        uq = unconditional[i, "to_q"] if cfg else None
                        uk = unconditional[i, "to_k"] if cfg else None
                        if joint:
                            heads = layer.heads

                            def split(value, heads=heads):
                                return (
                                    None
                                    if value is None
                                    else value.unflatten(-1, (heads, -1)).transpose(1, 2)
                                )

                            cq, ck, uq, uk = map(split, (cq, ck, uq, uk))
                            ct = split(conditional[i, "add_k_proj"])
                            ut = split(unconditional[i, "add_k_proj"]) if cfg else None
                            for start in range(0, cq.shape[-2], self.query_chunk_size):
                                args = (
                                    cq[..., start : start + self.query_chunk_size, :],
                                    ck,
                                    ct,
                                    None if uq is None else uq[..., start : start + self.query_chunk_size, :],
                                    uk,
                                    ut,
                                    mask,
                                    guidance_scale,
                                    self.key_chunk_size,
                                )
                                value = (
                                    checkpoint(_joint_loss, *args, use_reentrant=False)
                                    if self.checkpoint_attention
                                    else _joint_loss(*args)
                                )
                                loss = loss + weight * value
                        else:
                            for start in range(0, cq.shape[1], self.query_chunk_size):
                                args = (
                                    cq[:, start : start + self.query_chunk_size],
                                    ck,
                                    None if uq is None else uq[:, start : start + self.query_chunk_size],
                                    uk,
                                    mask,
                                    guidance_scale,
                                )
                                value = (
                                    checkpoint(_unet_loss, *args, use_reentrant=False)
                                    if self.checkpoint_attention
                                    else _unet_loss(*args)
                                )
                                loss = loss + weight * value
                    gradient = torch.autograd.grad(loss, current)[0]
                    if not torch.isfinite(gradient).all():
                        raise FloatingPointError("Oracle Noise produced a nonfinite gradient. Try BF16/FP32.")
                    current = spherical_update(current, gradient, self.theta)
                    captured.clear()
            return current
        finally:
            for handle in handles:
                handle.remove()
            captured.clear()
