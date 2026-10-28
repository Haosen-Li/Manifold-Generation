import json

import pytest
import torch
from diffusers import (
    AutoencoderKL,
    AutoencoderKLCogVideoX,
    CogVideoXDDIMScheduler,
    CogVideoXPipeline,
    CogVideoXTransformer3DModel,
    FlowMatchEulerDiscreteScheduler,
    HunyuanDiT2DModel,
    HunyuanDiTPipeline,
    SD3Transformer2DModel,
    StableDiffusion3Pipeline,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    TextToVideoSDPipeline,
    UNet2DConditionModel,
    UNet3DConditionModel,
)
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from manifold_gen import ManifoldDDIMScheduler

torch.set_num_threads(1)


def vae(blocks=2):
    return AutoencoderKL(
        block_out_channels=(32,) * blocks,
        down_block_types=("DownEncoderBlock2D",) * blocks,
        up_block_types=("UpDecoderBlock2D",) * blocks,
        norm_num_groups=8,
        sample_size=32,
        latent_channels=4,
        shift_factor=0.0,
    )


@pytest.fixture
def tokenizer(tmp_path):
    vocab = {
        "<|startoftext|>": 0,
        "<|endoftext|>": 1,
        "a</w>": 2,
        "b</w>": 3,
        "c</w>": 4,
        "d</w>": 5,
        "e</w>": 6,
    }
    (tmp_path / "vocab.json").write_text(json.dumps(vocab))
    (tmp_path / "merges.txt").write_text("#version: 0.2\n")
    return CLIPTokenizer(str(tmp_path / "vocab.json"), str(tmp_path / "merges.txt"), model_max_length=8)


def text_encoder(projection=False):
    config = CLIPTextConfig(
        vocab_size=16,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=8,
        projection_dim=16,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=1,
    )
    return (CLIPTextModelWithProjection if projection else CLIPTextModel)(config).eval()


def make_pipeline(family, tokenizer=None, prediction_type="epsilon"):
    torch.manual_seed(7)
    scheduler = ManifoldDDIMScheduler(
        num_train_timesteps=1000, clip_sample=False, prediction_type=prediction_type, set_alpha_to_one=True
    )
    common_unet = dict(
        sample_size=8,
        in_channels=4,
        out_channels=4,
        block_out_channels=(32, 32),
        layers_per_block=1,
        norm_num_groups=8,
        cross_attention_dim=32,
        attention_head_dim=4,
    )
    embed = torch.randn(1, 8, 32)
    negative = torch.randn_like(embed)
    options = dict(
        prompt_embeds=embed,
        negative_prompt_embeds=negative,
        height=16,
        width=16,
        num_inference_steps=4,
        guidance_scale=5.0,
        output_type="latent",
    )
    if family in {"sd", "sdxl"}:
        extra = {}
        if family == "sdxl":
            extra = dict(
                addition_embed_type="text_time",
                addition_time_embed_dim=8,
                projection_class_embeddings_input_dim=64,
            )
        unet = UNet2DConditionModel(
            **common_unet,
            down_block_types=("CrossAttnDownBlock2D",) * 2,
            up_block_types=("CrossAttnUpBlock2D",) * 2,
            **extra,
        ).eval()
        if family == "sd":
            pipe = StableDiffusionPipeline(
                vae(), None, None, unet, scheduler, None, None, requires_safety_checker=False
            )
        else:
            pipe = StableDiffusionXLPipeline(
                vae(),
                text_encoder() if tokenizer else None,
                text_encoder(True) if tokenizer else None,
                tokenizer,
                tokenizer,
                unet,
                scheduler,
                add_watermarker=False,
            )
            options.update(
                pooled_prompt_embeds=torch.randn(1, 16), negative_pooled_prompt_embeds=torch.randn(1, 16)
            )
    elif family == "sd3":
        model = SD3Transformer2DModel(
            sample_size=8,
            patch_size=2,
            in_channels=4,
            out_channels=4,
            num_layers=3,
            attention_head_dim=8,
            num_attention_heads=4,
            joint_attention_dim=32,
            caption_projection_dim=32,
            pooled_projection_dim=32,
            pos_embed_max_size=16,
        ).eval()
        pipe = StableDiffusion3Pipeline(
            model,
            FlowMatchEulerDiscreteScheduler(),
            vae(),
            text_encoder(True) if tokenizer else None,
            tokenizer,
            text_encoder(True) if tokenizer else None,
            tokenizer,
            None,
            None,
        )
        options.update(
            pooled_prompt_embeds=torch.randn(1, 32),
            negative_pooled_prompt_embeds=torch.randn(1, 32),
            max_sequence_length=8,
        )
    elif family == "hunyuan":
        model = HunyuanDiT2DModel(
            num_attention_heads=2,
            attention_head_dim=16,
            hidden_size=32,
            in_channels=4,
            sample_size=4,
            patch_size=2,
            num_layers=2,
            cross_attention_dim=32,
            cross_attention_dim_t5=32,
            pooled_projection_dim=16,
            text_len=8,
            text_len_t5=8,
        ).eval()
        pipe = HunyuanDiTPipeline(
            vae(4), None, None, model, scheduler, None, None, requires_safety_checker=False
        )
        options.update(
            height=32,
            width=32,
            use_resolution_binning=False,
            prompt_embeds_2=torch.randn(1, 8, 32),
            negative_prompt_embeds_2=torch.randn(1, 8, 32),
            prompt_attention_mask=torch.ones(1, 8, dtype=torch.long),
            negative_prompt_attention_mask=torch.ones(1, 8, dtype=torch.long),
            prompt_attention_mask_2=torch.ones(1, 8, dtype=torch.long),
            negative_prompt_attention_mask_2=torch.ones(1, 8, dtype=torch.long),
        )
    elif family == "cogvideox":
        model = CogVideoXTransformer3DModel(
            num_attention_heads=2,
            attention_head_dim=16,
            in_channels=4,
            out_channels=4,
            time_embed_dim=32,
            text_embed_dim=32,
            num_layers=1,
            sample_width=8,
            sample_height=8,
            sample_frames=5,
            patch_size=2,
            temporal_compression_ratio=2,
            max_text_seq_length=8,
        ).eval()
        video_vae = AutoencoderKLCogVideoX(
            block_out_channels=(32, 32),
            down_block_types=("CogVideoXDownBlock3D",) * 2,
            up_block_types=("CogVideoXUpBlock3D",) * 2,
            latent_channels=4,
            layers_per_block=1,
            temporal_compression_ratio=2,
            norm_num_groups=8,
        )
        pipe = CogVideoXPipeline(
            None, None, video_vae, model, CogVideoXDDIMScheduler(prediction_type="v_prediction")
        )
        options.update(num_frames=5, max_sequence_length=8)
    elif family == "modelscope":
        model = UNet3DConditionModel(
            **common_unet,
            down_block_types=("CrossAttnDownBlock3D", "DownBlock3D"),
            up_block_types=("UpBlock3D", "CrossAttnUpBlock3D"),
        ).eval()
        pipe = TextToVideoSDPipeline(vae(), None, None, model, scheduler)
        options.update(num_frames=3)
    else:
        raise ValueError(family)
    pipe.set_progress_bar_config(disable=True)
    return pipe, options
