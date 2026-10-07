# Reasoning70：seed11 训练划分

日期：2026-10-06。状态：数据划分已生成并核对，尚未运行该划分的模型实验。

训练集参考 [Hallucination100 的 seed11 划分](../vlrb-hallucination100-generated-roots/seed11_split.json)，将目标类别改为 Reasoning、训练数量改为 70，沿用固定种子、来源抽样、训练选例时的近重复过滤，以及原始 A/B 映射规则。生成前已复现原 Hallucination100 的训练顺序、留出名单和排除记录，均与参考文件一致。现有 70 条训练样本已经固定；按用户要求，剩余所有 Reasoning 样本均作为留出集，不因与训练图像或文本相同、近似而排除。

## 划分结果

| 项目 | 数量 |
|---|---:|
| Reasoning 全类别 | 317 |
| 与原 Discovery100 重叠而排除 | 0 |
| 训练集 | 70 |
| Reasoning 留出集（全部非训练样本） | 247 |
| 因与训练样本近重复而从留出集排除 | 0 |

训练集包含 MathVerse 35 条、MMMU-Pro 35 条；这是随机抽样和去重后的实际组成，未额外设置两类各 35 条的配额。训练 JSONL 的 A/B 标签分别为 37/33。训练与留出 ID 无交集。

留出集现为全部 247 条非训练样本，包含此前因与训练图像相同或近似而排除的 15 条。`heldout_ids` 已补回这些 ID，`excluded_train_overlap` 为空。此次仅修改留出集规则，训练 ID、顺序、A/B 标签、JSONL 文件内容与哈希均保持不变。

## 抽样方法

1. 使用与参考划分相同的 VLRB Parquet 和原 Discovery100；两份文件的 SHA256 均与参考记录一致。保留完整 VLRB 的稳定样本 ID，重复 benchmark ID 使用已有的 `__row_XXXX` 后缀。
2. 先按原规则排除与原 Discovery100 相同或近似的图像/问答案例。Reasoning 在现有 VLRB 来源分类中统一为 `reasoning_tasks`，配额为 70，不增加子任务分层规则。
3. 将候选样本按 `sample_id` 排序，用 `random.Random("11:reasoning_tasks").shuffle(candidates)` 打乱。按顺序遍历，跳过与已选训练案例近重复的样本，选满 70 条即停止。
4. 固定以上 70 条训练 ID，将 Reasoning 类别中所有其余样本写入 `heldout_ids`，保持原始 Parquet 的遍历顺序；不进行留出集的图像或文本近重复排除。
5. 训练 JSONL 使用完整 VLRB 正式 K=3 日程中的第一次 A/B 顺序，还原对应人类标签。只引用已有共享图片目录，不重新生成或复制图像。

原训练选例的图像规则为 EXIF 校正后转灰度、缩放至 9×8，计算 64 位 dHash，汉明距离不超过 4 判为相同或近似。文本先 casefold 和单词规范化，对两个回答排序以忽略 A/B 位置；规范化问答相同，或问题长度比至少 0.85、问题及两个回答的 `SequenceMatcher` 相似度均至少 0.92 时，判为相同或近似问答。这些规则用于当时的训练选例，当前不用于排除留出样本。

历史实现为 `codex/subtree-local-reflection` 中的 `experiments/evolving_structured_rubrics/vlrb_hallucination_transfer.py`；本次使用的确切 commit、抽样方法和来源哈希写入 [seed11_split.json](seed11_split.json)。未修改旧 runner 的全局配置、当前实验配置或模型提示词。

## 保存位置

- [seed11_split.json](seed11_split.json)：可提交的训练/留出 ID、排除记录、算法与来源哈希。
- [discovery_70.jsonl](../../../data/VL_RewardBench/splits/reasoning70/seed11/discovery_70.jsonl)：本地训练数据，包含图像引用、问题、A/B 回答及人类标签；整个 `data/VL_RewardBench/` 按现有规则保持 Git 忽略。
- `data/VL_RewardBench/dataset_images/`：已有共享图像目录，文件集合保持不变。

目前仅完成数据准备。现有 `prepare` 入口仍专用于原 Hallucination100；本次没有修改其固定 manifest，也没有将新划分接入报告或启动实验。
