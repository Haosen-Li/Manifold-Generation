# Manifold Gen

**面向图像与视频生成的几何推理优化组件。**

| 组件 | 功能 | 论文 |
|---|---|---|
| **Oracle Noise** | 在球面约束下优化初始噪声，增强语义对齐 | [arXiv](https://arxiv.org/abs/2604.23540) |
| **SAMG** | 空间自适应引导，支持 CFG、CFG++、CFG-Zero* | [arXiv](https://arxiv.org/abs/2604.26503) |
| **Z²-Sampling** | 利用历史语义代理修正采样轨迹，不增加采样模型调用 | [arXiv](https://arxiv.org/abs/2604.23536) |

[English](README.md) · [API 文档](docs/api.md) · [模型兼容表](docs/compatibility.md)

## 安装

需要 Python 3.10+、PyTorch 2.2+、Diffusers 0.35.1。
先安装适合设备的 PyTorch，再在仓库根目录运行：

```bash
python -m pip install -e ".[text,video]"
```

Parquet 数据另装 `.[data]`，开发工具另装 `.[dev]`。
模型权重可通过 Diffusers 下载，也可从本地目录加载。

## 快速开始

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

`load_pipeline()` 在使用 Z² 或 CFG++ 时选择适合的确定性调度器。
本地模型通过 `model_id="/path/to/model"` 指定。

## 接入已有 pipeline

```python
from manifold_gen import OracleNoise, SAMG, Z2Sampling, plug

oracle_pipe = plug(base_pipe, oracle_noise=OracleNoise(theta=0.02, steps=1))
samg_pipe = plug(base_pipe, samg=SAMG(base_scale=1.0, strong_scale=7.5))
z2_pipe = plug(base_pipe, z2=Z2Sampling(warmup_steps=1, zigzag_steps=10))
```

三个 wrapper 共享模型权重，保留原生文本编码、条件输入、VAE 解码和输出格式，
可顺序调用。`plug()` 不替换已有调度器；SD/SDXL 使用 Z² 或 CFG++ 时可配置：

```python
from manifold_gen import ManifoldDDIMScheduler

base_pipe.scheduler = ManifoldDDIMScheduler.from_config(
    base_pipe.scheduler.config, clip_sample=False, thresholding=False
)
```

同一次 `plug()` 可插入多个组件。组合方式和调度器要求见[兼容表](docs/compatibility.md)。

## SAMG 变体

```python
SAMG(variant="cfg",      base_scale=1.0, strong_scale=7.5)
SAMG(variant="cfg++",    base_scale=0.4, strong_scale=0.8)
SAMG(variant="cfg_zero", base_scale=1.0, strong_scale=5.0, zero_init_steps=1)
```

- **CFG**：在两档 guidance 之间进行空间自适应混合。
- **CFG++**：在预测的干净样本空间中混合，再用无条件 epsilon 回加噪声。
- **CFG-Zero\***：初始零输出步骤与最优无条件投影。

支持 `lambda_min`、`lambda_max`、空间／时间平滑参数，以及
`register_guidance()` 自定义变体接口。SAMG 调用时需要 `guidance_scale > 1`
开启原生 CFG 双分支，实际空间引导强度由 SAMG 参数决定。

## 模型

| 预设 | 模型 | SAMG | Oracle Noise | Z² |
|---|---|---|---|---|
| `sd15` | Stable Diffusion 1.5 | ✓ | — | ✓ |
| `sd21` | Stable Diffusion 2.1 | ✓ | — | ✓ |
| `sdxl` | Stable Diffusion XL | ✓ | ✓ | ✓ |
| `sdxl-turbo` | SDXL-Turbo | — | ✓ | — |
| `sd35` | Stable Diffusion 3.5 Medium | ✓ | ✓ | ✓ |
| `dreamshaper-xl` | DreamShaper XL v2 Turbo | ✓ | ✓ | ✓ |
| `hunyuan-dit` | Hunyuan-DiT | ✓ | — | ✓ |
| `cogvideox` | CogVideoX-2B | ✓ | — | ✓ |
| `modelscope` | ModelScope 文生视频 1.7B | ✓ | — | ✓ |

✓ 表示已提供接口适配。各变体的限制、架构扩展范围见[完整兼容表](docs/compatibility.md)。
SDXL-Turbo 使用条件分支 Oracle Noise，设置 `guidance_scale=0`。
同架构的 Diffusion-DPO 权重复用 SDXL 接口；AYS 通过显式 DDIM 时间步接入。

## 命令行

```bash
# SAMG
manifold-gen --model sdxl --components samg --prompt "A red cube on a white table"

# CFG++
manifold-gen --model sdxl --components samg --samg-variant 'cfg++' \
  --base-scale 0.4 --strong-scale 0.8 --prompt "A red cube"

# CFG-Zero*
manifold-gen --model sd35 --components samg --samg-variant cfg_zero \
  --prompt "A blue vase on a wooden table"

# Oracle Noise
manifold-gen --model sdxl-turbo --components oracle --oracle-cfg off \
  --theta 0.05 --prompt "A small yellow dog"

# Z² 视频生成
manifold-gen --model cogvideox --components z2 --warmup-steps 10 \
  --zigzag-steps 30 --prompt "A train crosses a stone bridge"

# 批量生成
manifold-gen --model sdxl --components samg \
  --prompts examples/prompts.jsonl --output outputs/batch
```

完整参数：`manifold-gen --help` 或 `python -m manifold_gen --help`。
支持 TXT、JSON、JSONL、CSV、Parquet。输出包含 PNG/MP4，以及记录参数、种子和
运行信息的 `run.json`。离线使用 `--model-id /path/to/model --offline`。

## 示例与开发

[单组件调用](examples/quickstart.py) · [视频生成](examples/video.py) ·
[自定义 guidance](examples/custom_guidance.py) · [AYS / DPO](examples/ays_and_dpo.py)

```bash
python -m pip install -e ".[dev,text]"
python -m pytest -q
python -m ruff check src tests examples
python -m build
```

测试使用 CPU 上的小型随机 Diffusers 模型，不下载完整权重。
本仓库不附带完整模型的画质或速度基准结果。

使用相关方法时，请引用对应论文，引用条目见 [CITATION.bib](CITATION.bib)。
