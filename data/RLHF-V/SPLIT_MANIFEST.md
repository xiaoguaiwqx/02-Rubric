# RLHF-V DREA Split Manifest

## 划分目标

- 用少量高质量 pairwise preference 做 DREA 准则发现，避免在 5732 条样本上过拟合。
- 当前 discovery train 为 `90` 条，其中显式补入短回答样本。
- Held-out validation 为 `500` 条，用于评估发现出的准则。
- Multi-Crit 后续再作为 criterion-conditioned / routing 实验，不混入当前主线。

## 文件

- 输入 JSONL：`datasets\RLHF-V-Dataset\processed\rlhfv_pairwise.jsonl`
- 图片根目录：`datasets\RLHF-V-Dataset\processed\images`
- Discovery train：`datasets\RLHF-V-Dataset\drea_splits\discovery_train_90.jsonl`
- Held-out validation：`datasets\RLHF-V-Dataset\drea_splits\heldout_validation_500.jsonl`
- Reserve pool：`datasets\RLHF-V-Dataset\drea_splits\reserve_pool.jsonl`

## 抽样策略

- Seed：`42`
- Sampling mode：`constrained_diversity`
- Train/validation image overlap allowed：`False`
- 先过滤缺图片、极短回答、极长回答、A/B 长度差异过大的样本。
- 过滤 identical / near-identical pairs，避免低信号偏好对进入 train/val。
- 对剩余样本打轻量质量分；该分数只在约束桶内排序，不决定整体长度比例。
- 默认先按任务类型分配配额，保持 detailed_description / question_answering 约 1:1。
- 再按 chosen 长度桶分配配额，显式覆盖 `<50` 和 `50-150` 的短回答样本。
- 每个约束内部按 `origin_dataset + task_type + model` 分层轮转抽样。
- 默认保证 discovery train 和 held-out validation 不共享任何 `image_path`。

## 数量

- 原始样本：5732
- 通过质量过滤：5339
- 被过滤：393
- Discovery train：90
- Held-out validation：500
- Reserve pool：4749

## Discovery Train 来源分布

| Value | Count |
|---|---:|
| `vqav2` | 32 |
| `coco` | 21 |
| `sharegpt4v-textvqa` | 11 |
| `sharegpt4v-web-celebrity` | 10 |
| `LCS-558K` | 9 |
| `sharegpt4v-wikiart` | 4 |
| `sharegpt4v-web-landmark` | 3 |

## Discovery Train 任务类型

| Value | Count |
|---|---:|
| `question_answering` | 45 |
| `detailed_description` | 45 |

## Discovery Train chosen 长度桶

| Value | Count |
|---|---:|
| `500-800` | 23 |
| `150-300` | 22 |
| `50-150` | 20 |
| `300-500` | 19 |
| `<50` | 4 |
| `800+` | 2 |

## Held-out Validation 来源分布

| Value | Count |
|---|---:|
| `coco` | 193 |
| `vqav2` | 158 |
| `LCS-558K` | 57 |
| `sharegpt4v-web-landmark` | 27 |
| `sharegpt4v-wikiart` | 26 |
| `sharegpt4v-web-celebrity` | 20 |
| `sharegpt4v-textvqa` | 19 |

## Held-out Validation 任务类型

| Value | Count |
|---|---:|
| `detailed_description` | 250 |
| `question_answering` | 250 |

## Held-out Validation chosen 长度桶

| Value | Count |
|---|---:|
| `500-800` | 129 |
| `150-300` | 122 |
| `50-150` | 111 |
| `300-500` | 110 |
| `800+` | 26 |
| `<50` | 2 |

## 图片隔离

- Discovery train unique images：90
- Held-out validation unique images：500
- Train/validation shared images：0

## 过滤原因

| Value | Count |
|---|---:|
| `near_identical_pair` | 272 |
| `length_ratio_extreme` | 62 |
| `answer_too_short` | 43 |
| `identical_pair` | 16 |
