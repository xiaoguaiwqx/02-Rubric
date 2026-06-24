# 使用 `train_qwen3vl_bt_reward.py` 训练 Qwen3-VL BT Reward Model

本文档说明如何使用 [`critiq/scripts/train_qwen3vl_bt_reward.py`](../critiq/scripts/train_qwen3vl_bt_reward.py) 训练一个基于 Qwen3-VL 的 Bradley-Terry reward model。该脚本面向 RLHF-V 多模态偏好数据：对同一张图像和同一个问题下的两个候选回答分别打分，并通过 Bradley-Terry loss 学习让 preferred/chosen 回答的 reward 高于 rejected 回答。

脚本默认使用：

- 基座模型：`Qwen/Qwen3-VL-8B-Instruct`
- 服务器本地权重目录：`/media/disk12T/2022-sgh/ModelWeight/Qwen3-VL-8B-Instruct`
- 训练文件：`data/RLHF-V/bt_reward/full_train.jsonl`
- 验证文件：`data/RLHF-V/bt_reward/full_val.jsonl`
- holdout 评测文件：`data/RLHF-V/bt_reward/holdout500_eval.jsonl`
- 训练方式：QLoRA
- 视觉塔：默认冻结
- 注意力实现：`flash_attention_2`

## 1. 环境准备

建议在 Linux GPU 服务器上运行训练。Windows 本地适合阅读代码和准备数据，但 `flash_attn`、`bitsandbytes`、DeepSpeed 等训练依赖在 Linux/CUDA 环境下更稳定。

### 1.1 Python 和 CUDA

推荐环境：

- Python `>=3.10`
- NVIDIA GPU，建议显存 `>=24GB`
- CUDA 版本与 PyTorch wheel 匹配
- Git
- Hugging Face 账号和模型访问权限

创建环境示例：

```bash
conda create -n critiq-qwen3vl-rm python=3.10 -y
conda activate critiq-qwen3vl-rm
python -m pip install --upgrade pip setuptools wheel
```

安装 PyTorch 时请根据服务器 CUDA 版本选择对应命令。以下以 CUDA 12.4 为例：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

如果服务器是 CUDA 12.1，可以改用：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 1.2 安装项目依赖

在仓库根目录执行：

```bash
git clone https://github.com/KYLN24/CritiQ
cd CritiQ
pip install -e ".[train]"
```

`train` extra 会安装训练所需的主要依赖：

- `datasets`：数据集工具，当前脚本不强依赖，但训练环境通常需要。
- `trl>=0.12.0`：reward 训练相关生态依赖，当前脚本自定义了 Trainer，但项目训练 extra 中保留。
- `transformers @ git+https://github.com/huggingface/transformers`：Qwen3-VL 需要较新的 Transformers 类和 processor，建议从源码安装。
- `peft`：LoRA/QLoRA adapter。
- `bitsandbytes`：4-bit QLoRA 量化。
- `pillow`：读取图像。
- `flash_attn`：`flash_attention_2` 注意力实现。
- `deepspeed`：可选的多卡/ZeRO 训练。
- `tensorboardX`：TensorBoard 日志。
- `evaluate`：评测工具依赖。
- `accelerate`：Transformers Trainer 分布式训练依赖。

如果 `flash_attn` 安装失败，先完成其他依赖，然后训练时将参数改成 `--attn_implementation sdpa`：

```bash
pip install -e .
pip install datasets "trl>=0.12.0" "transformers @ git+https://github.com/huggingface/transformers" peft bitsandbytes pillow deepspeed tensorboardX evaluate accelerate
```

### 1.3 模型权重路径

当前服务器已经下载好 Qwen3-VL-8B-Instruct 权重，路径为：

```text
/media/disk12T/2022-sgh/ModelWeight/Qwen3-VL-8B-Instruct
```

因此推荐训练时显式传入：

```bash
--model /media/disk12T/2022-sgh/ModelWeight/Qwen3-VL-8B-Instruct
```

这样 `transformers.from_pretrained()` 会直接从本地目录加载模型和 processor，不需要重新从 Hugging Face 下载。

如果要在新服务器上重新下载模型，可以先登录 Hugging Face：

```bash
huggingface-cli login
```

也可以手动设置缓存目录，避免模型下载到默认 home 目录：

```bash
export HF_HOME=/path/to/hf_cache
export TRANSFORMERS_CACHE=/path/to/hf_cache/transformers
```

## 2. 数据准备

脚本默认读取仓库中的 JSONL 文件：

```text
data/RLHF-V/bt_reward/full_train.jsonl
data/RLHF-V/bt_reward/full_val.jsonl
data/RLHF-V/bt_reward/holdout500_eval.jsonl
```

这些 JSONL 中的 `image_path` 是相对路径，实际图片需要另外准备。根据 `data/RLHF-V/bt_reward/manifest.json`，图片路径应相对于 RLHF-V processed images 根目录，例如：

```text
/media/disk12T/wenqx/datasets/RLHF-V-Dataset/processed/images/
```

训练时用 `--image_root` 指向这个目录。

### 2.1 训练/验证数据格式

`--train_file` 和 `--val_file` 中每一行必须包含：

```json
{
  "sample_id": "rlhfv-000769",
  "image_path": "coco2017/train2017/000000210693.jpg",
  "question": "What is unusual about the English language term used to describe the cutting tool?",
  "chosen": "preferred answer text",
  "rejected": "less preferred answer text"
}
```

脚本会构造两个多模态 conversation：

- chosen conversation：图像 + 问题 + `chosen`
- rejected conversation：图像 + 问题 + `rejected`

训练目标是让：

```text
reward(chosen) > reward(rejected)
```

### 2.2 Holdout 评测数据格式

`--holdout_file` 中每一行必须包含：

```json
{
  "sample_id": "rlhfv-000778",
  "image_path": "sharegpt4v/data/web-celebrity/images/Luke_Evans.jpg",
  "question": "Analyze the image and provide a comprehensive description...",
  "A": "candidate answer A",
  "B": "candidate answer B",
  "answer": "A"
}
```

评测时脚本会分别计算 `score_A` 和 `score_B`，如果 `score_A > score_B` 则预测 `A`，否则预测 `B`。

## 3. 训练前快速检查

正式训练前建议先跑 dry run，确认 processor 能正确读取图像并输出多模态字段：

```bash
python -m critiq.scripts.train_qwen3vl_bt_reward \
  --dry_run_batch \
  --model /media/disk12T/2022-sgh/ModelWeight/Qwen3-VL-8B-Instruct \
  --image_root /media/disk12T/wenqx/datasets/RLHF-V-Dataset/processed/images \
  --train_file data/RLHF-V/bt_reward/full_train.jsonl \
  --batch_size 2 \
  --max_length 4096 \
  --attn_implementation sdpa
```

如果输出里：

```json
"has_required_vision_keys": true
```

说明至少一个 batch 的图像和文本都被 processor 正确编码。

常见失败原因：

- `image_root` 指错，导致图片文件找不到。
- `transformers` 版本太旧，没有 Qwen3-VL 相关类。
- processor 输出缺少 `pixel_values` 或 `image_grid_thw`，通常是模型/processor 版本或图像路径问题。
- 本地 Python 版本低于 3.10，项目代码中的类型标注无法解析。

## 4. 单卡 QLoRA 训练命令

默认训练方式是 QLoRA，适合显存有限的场景：

```bash
CUDA_VISIBLE_DEVICES=1 python -m critiq.scripts.train_qwen3vl_bt_reward \
  --mode train_eval \
  --model /media/disk12T/2022-sgh/ModelWeight/Qwen3-VL-8B-Instruct \
  --image_root /media/disk12T/wenqx/datasets/RLHF-V-Dataset/processed/images \
  --train_file data/RLHF-V/bt_reward/full_train.jsonl \
  --val_file data/RLHF-V/bt_reward/full_val.jsonl \
  --holdout_file data/RLHF-V/bt_reward/holdout500_eval.jsonl \
  --output_dir output/qwen3vl_bt_reward \
  --job_name qwen3vl_bt_full_seed100745534 \
  --train_mode qlora \
  --batch_size 1 \
  --eval_batch_size 1 \
  --accum 32 \
  --epochs 1 \
  --lr 1e-4 \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --eval_steps 50 \
  --logging_steps 1 \
  --max_length 8092 \
  --fp16 \
  --tf32 \
  --gradient_checkpointing \
  --attn_implementation flash_attention_2 \
  --report_to tensorboard
```

如果环境没有 `flash_attn`，改为：

```bash
  --attn_implementation sdpa
```

训练完成后，模型会保存在：

```text
output/qwen3vl_bt_reward/qwen3vl_bt_full_seed100745534/
```

## 5. 多卡训练命令

使用 `torchrun` 启动多卡训练：

```bash
CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc-per-node=2 -m critiq.scripts.train_qwen3vl_bt_reward \
  --mode train_eval \
  --model /media/disk12T/2022-sgh/ModelWeight/Qwen3-VL-8B-Instruct \
  --image_root /media/disk12T/wenqx/datasets/RLHF-V-Dataset/processed/images \
  --train_file data/RLHF-V/bt_reward/full_train.jsonl \
  --val_file data/RLHF-V/bt_reward/full_val.jsonl \
  --holdout_file data/RLHF-V/bt_reward/holdout500_eval.jsonl \
  --output_dir output/qwen3vl_bt_reward \
  --job_name qwen3vl_bt_full_seed100745534 \
  --train_mode qlora \
  --batch_size 2 \
  --eval_batch_size 1 \
  --accum 16 \
  --epochs 1 \
  --lr 1e-4 \
  --max_length 12800 \
  --attn_implementation flash_attention_2
```

有效全局 batch size 约为：

```text
global_batch_size = GPU 数量 * batch_size * accum
```

例如 4 卡、`batch_size=1`、`accum=8` 时，有效 batch size 为 `32` 对 preference pairs。

## 6. 只评测已有 checkpoint

如果已经训练完成，只想跑 holdout 评测：

```bash
CUDA_VISIBLE_DEVICES=1 python -m critiq.scripts.train_qwen3vl_bt_reward \
  --mode eval \
  --model /media/disk12T/2022-sgh/ModelWeight/Qwen3-VL-8B-Instruct \
  --checkpoint output/qwen3vl_bt_reward/qwen3vl_bt_full_seed100745534 \
  --image_root /media/disk12T/wenqx/datasets/RLHF-V-Dataset/processed/images \
  --holdout_file data/RLHF-V/bt_reward/holdout500_eval.jsonl \
  --output_dir output/qwen3vl_bt_reward \
  --job_name qwen3vl_bt_full_seed100745534 \
  --eval_batch_size 1 \
  --max_length 8092 \
  --attn_implementation flash_attention_2
```

如果不提供 `--checkpoint`，脚本默认从：

```text
{output_dir}/{job_name}
```

加载 reward model。

## 7. 从中断处恢复训练

Transformers Trainer 会在 `save_steps` 保存 checkpoint。该脚本里 `save_steps` 等于 `--eval_steps`。

恢复训练示例：

```bash
CUDA_VISIBLE_DEVICES=0 python -m critiq.scripts.train_qwen3vl_bt_reward \
  --mode train_eval \
  --image_root /media/disk12T/wenqx/datasets/RLHF-V-Dataset/processed/images \
  --output_dir output/qwen3vl_bt_reward \
  --job_name qwen3vl_bt_full_seed100745534 \
  --resume_from_checkpoint output/qwen3vl_bt_reward/qwen3vl_bt_full_seed100745534/checkpoint-200
```

恢复训练时，其余关键参数应尽量与原训练保持一致，尤其是 `--model`、`--train_mode`、LoRA 参数、`--max_length` 和精度设置。

## 8. 输出文件说明

训练输出目录为：

```text
{output_dir}/{job_name}/
```

主要文件包括：

- `adapter/`：LoRA/QLoRA adapter 权重。如果 `train_mode=full`，则可能保存完整 backbone 到 `backbone/`。
- `score_head.pt`：reward head，即 hidden state 到标量 reward 的线性层。
- `reward_model_config.json`：reward model 配置，记录基座模型、训练模式、LoRA 参数等。
- `eval_metrics.json`：训练后在 `--val_file` 上的评测指标。
- `holdout500_predictions.jsonl`：holdout 样本逐条预测结果，包括 `score_A`、`score_B`、`pred`、`gold`、`correct`。
- `holdout500_metrics.json`：holdout 总体准确率、平均 margin、near-tie 数量及分组指标。
- `logs/`：TensorBoard 日志目录。
- `checkpoint-*`：Trainer 保存的中间 checkpoint。

查看 TensorBoard：

```bash
tensorboard --logdir output/qwen3vl_bt_reward/qwen3vl_bt_full_seed100745534/logs
```

## 9. 参数详细说明

### 9.1 数据、模型和输出路径

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--model` | `Qwen/Qwen3-VL-8B-Instruct` | Hugging Face 模型名或本地模型路径。当前服务器建议使用 `/media/disk12T/2022-sgh/ModelWeight/Qwen3-VL-8B-Instruct`，避免重复下载。 |
| `--train_file` | `data/RLHF-V/bt_reward/full_train.jsonl` | 训练 JSONL 文件，必须包含 `sample_id`、`image_path`、`question`、`chosen`、`rejected`。 |
| `--val_file` | `data/RLHF-V/bt_reward/full_val.jsonl` | 验证 JSONL 文件，格式同训练文件。 |
| `--holdout_file` | `data/RLHF-V/bt_reward/holdout500_eval.jsonl` | holdout 评测文件，必须包含 `sample_id`、`image_path`、`question`、`A`、`B`、`answer`。 |
| `--image_root` | `None` | 图片根目录。相对 `image_path` 会拼到该目录下；如果 `image_path` 是绝对路径、`http(s)` 或 `file://`，则不拼接。 |
| `--output_dir` | `output/qwen3vl_bt_reward` | 输出根目录。 |
| `--job_name` | `qwen3vl_bt_full_seed100745534` | 当前实验名，最终输出目录是 `{output_dir}/{job_name}`。 |
| `--checkpoint` | `None` | 评测时加载的 reward model checkpoint 目录；不填则使用 `{output_dir}/{job_name}`。 |

### 9.2 运行模式

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--mode` | `train_eval` | `train` 只训练并在 val 上评估；`eval` 只跑 holdout；`train_eval` 先训练再跑 holdout。 |
| `--dry_run_batch` | `False` | 只构造一个 batch 并打印 tensor shape，用于检查数据和 processor，不加载 reward model、不训练。 |

### 9.3 训练超参数

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--seed` | `100745534` | 随机种子，会设置 Python、NumPy 和 Torch seed。 |
| `--batch_size` | `1` | 每张 GPU 上的训练 batch size。一个样本包含 chosen/rejected 两个候选回答。 |
| `--eval_batch_size` | `1` | 每张 GPU 上的验证/评测 batch size。 |
| `--accum` | `8` | 梯度累积步数。增大可提高有效 batch size，但会降低 optimizer 更新频率。 |
| `--epochs` | `1.0` | 训练 epoch 数，可以是小数。 |
| `--lr` | `1e-4` | 学习率。QLoRA 常用 `1e-4` 左右；full finetune 通常应更小。 |
| `--weight_decay` | `0.01` | 权重衰减。 |
| `--warmup_ratio` | `0.03` | warmup 占总训练步数的比例。 |
| `--eval_steps` | `50` | 每多少 step 做一次验证；同时也是 `save_steps`。 |
| `--logging_steps` | `1` | 每多少 step 记录一次训练日志。 |
| `--max_length` | `4096` | 文本 token 最大长度。过大会显著增加显存占用。 |
| `--near_tie_epsilon` | `1e-6` | holdout 评测中判断 near-tie 的 margin 阈值，只影响统计，不影响预测。 |

### 9.4 图像分辨率和 tokenizer padding

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--min_pixels` | `None` | 传给 Qwen3-VL processor 的最小图像像素数。用于控制图像 token 数下限。 |
| `--max_pixels` | `None` | 传给 Qwen3-VL processor 的最大图像像素数。显存不足时可降低该值。 |
| `--padding_side` | `left` | tokenizer padding 方向。脚本默认 left padding，并从 `attention_mask` 找每条样本最后一个有效 token 做 reward pooling。 |

示例：限制图像分辨率以减少显存：

```bash
--min_pixels 3136 --max_pixels 262144
```

### 9.5 训练模式、LoRA 和 QLoRA

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--train_mode` | `qlora` | 可选 `qlora`、`lora`、`full`。`qlora` 为 4-bit 量化后训练 adapter；`lora` 不量化，只训练 adapter；`full` 训练完整模型和 reward head。 |
| `--lora_r` | `16` | LoRA rank。越大可训练容量越强，显存和参数量也更高。 |
| `--lora_alpha` | `32` | LoRA scaling 系数。 |
| `--lora_dropout` | `0.05` | LoRA dropout。 |
| `--lora_target_modules` | `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` | 注入 LoRA 的模块名。默认覆盖 attention 和 MLP 主要投影层。 |
| `--bnb_4bit_quant_type` | `nf4` | QLoRA 的 4-bit 量化类型。常用 `nf4`。 |
| `--bnb_4bit_use_double_quant` | `True` | 是否启用 bitsandbytes double quant。 |
| `--no_bnb_4bit_use_double_quant` | - | 关闭 double quant。 |
| `--freeze_vision` | `True` | 默认冻结名字中包含 `visual`、`vision`、`image`、`vit` 的视觉模块参数。 |
| `--unfreeze_vision` | `False` | 显式解冻视觉模块。设置后即使 `--freeze_vision` 默认开启，也不会冻结视觉塔。 |
| `--center_rewards_coefficient` | `0.01` | reward 居中正则系数。loss 会额外加上 chosen/rejected reward 平方均值，避免 reward 数值无约束漂移。 |

不同训练模式建议：

- 显存有限：使用默认 `--train_mode qlora`。
- 显存较充足且希望避免 4-bit 量化影响：使用 `--train_mode lora`。
- 需要完整微调：使用 `--train_mode full`，同时降低 `--lr`，并准备更大显存或 DeepSpeed。

### 9.6 精度、显存和注意力实现

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--gradient_checkpointing` | `True` | 开启梯度检查点，节省显存但训练更慢。 |
| `--no_gradient_checkpointing` | - | 关闭梯度检查点。 |
| `--fp16` | `True` | 使用 FP16。若同时设置 `--bf16`，脚本会优先使用 BF16。 |
| `--no_fp16` | - | 关闭 FP16。 |
| `--bf16` | `False` | 使用 BF16。A100/H100/L40S 等通常推荐 BF16。 |
| `--tf32` | `True` | 允许 TF32，加速 Ampere 及更新架构上的矩阵运算。 |
| `--no_tf32` | - | 关闭 TF32。 |
| `--attn_implementation` | `flash_attention_2` | 可选 `flash_attention_2`、`sdpa`、`eager`。没有 `flash_attn` 时用 `sdpa`。 |
| `--device_map` | `local` | QLoRA 下默认把模型放到当前 local rank GPU。可选 `local`、`auto`、`none`。 |
| `--deepspeed_config` | `None` | DeepSpeed 配置 JSON 路径。传入后由 Transformers Trainer 使用。 |

BF16 示例：

```bash
--bf16 --no_fp16
```

显存不足时，优先尝试：

```bash
--batch_size 1 \
--accum 16 \
--max_length 2048 \
--max_pixels 262144 \
--attn_implementation sdpa
```

### 9.7 日志、续训和调试

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--resume_from_checkpoint` | `None` | 从 Trainer checkpoint 恢复训练。 |
| `--report_to` | `tensorboard` | 日志后端，逗号分隔。可设为 `tensorboard`、`wandb` 或空字符串。 |
| `--dataloader_num_workers` | `0` | DataLoader worker 数。图像读取较慢时可以增大，例如 `4` 或 `8`。 |
| `--limit_train` | `None` | 只读取前 N 条训练样本，用于快速调试。 |
| `--limit_eval` | `None` | 只读取前 N 条验证样本；如果 `--limit_holdout` 不设置，holdout 也会使用该限制。 |
| `--limit_holdout` | `None` | 只读取前 N 条 holdout 样本。 |

快速 smoke test：

```bash
CUDA_VISIBLE_DEVICES=0 python -m critiq.scripts.train_qwen3vl_bt_reward \
  --mode train_eval \
  --model /media/disk12T/2022-sgh/ModelWeight/Qwen3-VL-8B-Instruct \
  --image_root /media/disk12T/wenqx/datasets/RLHF-V-Dataset/processed/images \
  --limit_train 16 \
  --limit_eval 8 \
  --limit_holdout 8 \
  --eval_steps 4 \
  --logging_steps 1 \
  --job_name smoke_test_qwen3vl_bt_reward \
  --attn_implementation sdpa
```

## 10. 模型结构和训练目标

该脚本没有复用文本 reward model 的训练路径，而是为每个候选回答构造完整的多模态对话：

```text
user: <image> + question
assistant: candidate answer
```

模型结构：

1. 加载 Qwen3-VL backbone。
2. 可选冻结视觉模块。
3. 根据 `--train_mode` 应用 LoRA/QLoRA 或 full finetune。
4. 在最后一层 hidden state 的最后一个有效 token 上接一个线性层 `score: hidden_size -> 1`。
5. 输出标量 reward。

BT loss：

```text
margin = reward(chosen) - reward(rejected)
loss = -logsigmoid(margin)
```

如果 `--center_rewards_coefficient > 0`，还会加入 reward 居中正则：

```text
loss += center_rewards_coefficient * mean([reward(chosen), reward(rejected)]^2)
```

验证指标：

- `eval_reward_accuracy`：`reward(chosen) > reward(rejected)` 的比例。
- `eval_mean_margin`：chosen 与 rejected 的平均 reward 差。
- `eval_mean_chosen_reward`：chosen 平均 reward。
- `eval_mean_rejected_reward`：rejected 平均 reward。

## 11. 常见问题

### 11.1 `Missing training dependencies`

说明训练依赖没有装全。重新安装：

```bash
pip install -e ".[train]"
```

如果是 `flash_attn` 单独失败，可以先用：

```bash
--attn_implementation sdpa
```

### 11.2 `The installed transformers does not expose Qwen3-VL classes`

说明 Transformers 版本太旧。安装源码版：

```bash
pip install --upgrade "transformers @ git+https://github.com/huggingface/transformers"
```

### 11.3 `Processor output is missing multimodal keys`

脚本要求 processor 输出：

```text
pixel_values
image_grid_thw
```

如果缺少这些字段，检查：

- `--image_root` 是否正确。
- 图片文件是否存在。
- `--model` 对应的 processor 是否为 Qwen3-VL processor。
- Transformers 是否为足够新的源码版本。

### 11.4 CUDA 显存不足

优先降低：

```bash
--max_length 2048
--max_pixels 262144
--eval_batch_size 1
--batch_size 1
```

并确认使用：

```bash
--train_mode qlora
--gradient_checkpointing
```

### 11.5 想训练小数据版本

可以改用 `rm90` 文件：

```bash
--train_file data/RLHF-V/bt_reward/rm90_train.jsonl \
--val_file data/RLHF-V/bt_reward/rm90_val.jsonl
```

也可以只用 reserve：

```bash
--train_file data/RLHF-V/bt_reward/reserve_train.jsonl \
--val_file data/RLHF-V/bt_reward/reserve_val.jsonl
```

### 11.6 如何判断训练结果是否正常

重点看：

- `eval_reward_accuracy` 是否高于随机水平 `0.5`。
- `eval_mean_margin` 是否为正。
- holdout 的 `accuracy` 是否高于 `0.5`。
- `near_tie_count` 是否过高；如果大量样本 margin 接近 0，说明 reward model 区分度不足。
- `mean_chosen_reward` 和 `mean_rejected_reward` 是否无限漂移；如果漂移明显，可保留或增大 `--center_rewards_coefficient`。

## 12. 推荐实验配置

### 12.1 稳妥的 QLoRA 配置

```bash
CUDA_VISIBLE_DEVICES=0 python -m critiq.scripts.train_qwen3vl_bt_reward \
  --mode train_eval \
  --model /media/disk12T/2022-sgh/ModelWeight/Qwen3-VL-8B-Instruct \
  --image_root /media/disk12T/wenqx/datasets/RLHF-V-Dataset/processed/images \
  --train_mode qlora \
  --batch_size 1 \
  --eval_batch_size 1 \
  --accum 8 \
  --epochs 1 \
  --lr 1e-4 \
  --max_length 4096 \
  --freeze_vision \
  --center_rewards_coefficient 0.01 \
  --attn_implementation flash_attention_2
```

### 12.2 没有 FlashAttention 的配置

```bash
CUDA_VISIBLE_DEVICES=0 python -m critiq.scripts.train_qwen3vl_bt_reward \
  --mode train_eval \
  --model /media/disk12T/2022-sgh/ModelWeight/Qwen3-VL-8B-Instruct \
  --image_root /media/disk12T/wenqx/datasets/RLHF-V-Dataset/processed/images \
  --train_mode qlora \
  --batch_size 1 \
  --eval_batch_size 1 \
  --accum 8 \
  --epochs 1 \
  --lr 1e-4 \
  --max_length 4096 \
  --attn_implementation sdpa
```

### 12.3 BF16 训练配置

```bash
CUDA_VISIBLE_DEVICES=0 python -m critiq.scripts.train_qwen3vl_bt_reward \
  --mode train_eval \
  --image_root /media/disk12T/wenqx/datasets/RLHF-V-Dataset/processed/images \
  --train_mode qlora \
  --bf16 \
  --no_fp16 \
  --tf32 \
  --batch_size 1 \
  --eval_batch_size 1 \
  --accum 8 \
  --epochs 1 \
  --lr 1e-4 \
  --max_length 4096
```
