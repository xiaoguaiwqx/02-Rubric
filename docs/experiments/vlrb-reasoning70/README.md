# Reasoning70：seed11 准则挖掘与外评

日期：2026-10-06；运行接入更新：2026-10-11；结果归档：2026-10-12。状态：R0 生成、Init Split、五轮演化及三阶段 K=3 外评产物均已保存；Final 的 2 次技术失败仍阻断完整报告汇总，详见下方结果说明。

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

数据准备时未修改 runner。现在 `prepare` 和完整报告已支持在配置中显式指定本划分；未指定时沿用原 Hallucination100 行为。训练数据、旧实验产物和算法提示词保持不变。本次模型运行的产物和结果见下方归档。

## 实验协议

以下是2026-10-11运行前固定的实验约定，与后面的已观察结果分开。

- 在固定训练70上独立挖掘 G5，不复用幻觉实验的 root、初始化材料或演化轨迹。主指标为其余全部 Reasoning247；训练70、Reasoning317与全量1247含学习样本。
- 从训练集按 seed11 抽取10对样例预热，通过 `roots --warmup-count 10` 指定；G5 生成恰好五根，之后固定根名称和职责。全局预热默认值仍为5。
- Init Split 使用英文 V3：signature → cluster → children。三个阶段提供全部根；只有 children 按 root 顺序参考本轮前面生成的子准则。signature/cluster 不读取前序 children，也不读取 Global Arbiter 报告。
- 初始化仍从 Worker 与整体标签不一致的案例开始，签名或聚类不足时补充其余训练案例；每根生成2–5条 children。局部演化使用 Strict ACC 接受，Preserve5 保留最多五条正确训练案例，最多五轮。
- Manager 为 Qwen3.5-27B，temperature=0.2、关闭 thinking；Worker/Arbiter 为 Qwen3-VL-8B-Instruct，temperature=0.5、max_tokens=2048。模型服务地址从本地配置指定。
- `prepare` 保持冻结训练顺序和 A/B 映射，优先复用已有训练 JSONL，缺失时按固定 ID 重建。图像共用 `data/VL_RewardBench/dataset_images/`。
- 先完成初始化和五轮演化、冻结 Final，再执行 Dev150 与外评；Dev150 为 K=1 旁观评测，不用于选择 checkpoint。正式外评使用 K=3、seed42、逐次独立随机换位，先还原回答身份再投票，弃权计错。
- 执行顺序为 `prepare → roots → init → evolve → dev → vlrb-r0 → vlrb-s0 → vlrb → report`。保留预定 Final，不根据训练、Dev 或外评分数改选中间轮次。
- 报告使用显式 `split_manifest`，保留训练70、全部非训练1177、Reasoning留出247和三个类别切片，主留出集计算配对统计。不同训练类别/数量的实验不将差异单独归因于类别。

配置示例为 [reasoning70.example.json](../../../experiments/evolving_structured_rubrics/configs/reasoning70.example.json)。训练 JSONL、实际服务配置、预测、缓存、图像与日志仅本地保存；冻结 ID 及必要结果提交到本目录。主线保留入口、恢复、两种划分兼容及报告的离线测试；完整实施计划与进度保存在 `init-split-prompts` 分支。

## 已观察结果：G5 seed11

### 运行来源与计分口径

本次输出目录为 [`output/reasoning70/seed11/v3_sequential_random_swap_k3/g5/`](../../../output/reasoning70/seed11/v3_sequential_random_swap_k3/g5/)。从固定 Reasoning70 中抽取 10 对样例预热，重新生成五个 root，使用 `init-split-template-en-v3-sequential-children` 初始化，并完成五轮局部演化。Manager 为 Qwen3.5-27B，Worker/Global Arbiter 为 Qwen3-VL-8B-Instruct，采样配置沿用上面的冻结协议。

下表均采用正式 **K=3、seed42、独立随机交换**口径：每次判断先还原原始回答身份，至少两票支持同一个回答才得到最终 A/B 判断；不足两票视为弃权，Strict ACC 将弃权计错。指标由保存的 [`vlrb/r0.json`](../../../output/reasoning70/seed11/v3_sequential_random_swap_k3/g5/vlrb/r0.json)、[`vlrb/initial.json`](../../../output/reasoning70/seed11/v3_sequential_random_swap_k3/g5/vlrb/initial.json) 和 [`vlrb/final.json`](../../../output/reasoning70/seed11/v3_sequential_random_swap_k3/g5/vlrb/final.json) 重算；R0/S0 与其已保存的正式摘要一致。没有使用运行时相对多数的 `metrics.strict_accuracy` 替代正式两票口径。

**报告状态**：Final 在 General 样本 `wildvision-battle-163` 的第一次判断中，Root 3 Worker 和 Global Arbiter 各有一次技术失败，共 2 次。完整 `report` 因 `unresolved technical failure` 中止，`g5/report.json` 尚未包含完整的三阶段外评汇总。本节保留现有预测，失败票按无有效 A/B 票处理，未补跑或改写产物。Reasoning247 中三阶段均无技术失败；全量及 General 的 Final 指标为包含上述失败的已保存结果。

### 正式 K=3 结果

单元格为“正确数/样本数；Strict ACC”。训练70、Reasoning317 和全量1247均包含用于挖掘与演化的样本；主指标为 Reasoning 留出247。

| 阶段 | **Reasoning 留出247** | VLRB 全量1247 | 全部非训练1177 | 训练70 | General181 | Hallucination749 | Reasoning317 |
|---|---:|---:|---:|---:|---:|---:|---:|
| R0：仅 root | **160/247；64.78%** | 809/1247；64.88% | 758/1177；64.40% | 51/70；72.86% | 80/181；44.20% | 518/749；69.16% | 211/317；66.56% |
| S0：Init Split | **156/247；63.16%** | 818/1247；65.60% | 765/1177；65.00% | 53/70；75.71% | 86/181；47.51% | 523/749；69.83% | 209/317；65.93% |
| Final：五轮演化 | **153/247；61.94%** | 801/1247；64.23% | 748/1177；63.55% | 53/70；75.71% | 83/181；45.86% | 512/749；68.36% | 206/317；64.98% |

主留出集的 Coverage 依次为 **246/247（99.60%）、246/247（99.60%）、245/247（99.19%）**，弃权数为 **1、1、2**；Covered ACC 为 **65.04%、63.41%、62.45%**。全量 K=3 弃权数为 **5、7、11**。主指标退化主要来自 A/B 判断错误增加，不能仅用弃权增加解释。

主留出集按任务来源拆分如下。这里的 MathVerse131 与 MMMU-Pro116 是固定留出集的实际组成。

| 留出子集 | R0 | S0 | Final | R0→Final 正确数变化 |
|---|---:|---:|---:|---:|
| MathVerse131 | 78/131；59.54% | 79/131；60.31% | 78/131；59.54% | 0 |
| MMMU-Pro116 | 82/116；70.69% | 77/116；66.38% | 75/116；64.66% | −7 |

### 主留出集配对变化

“纠正”指原来错误、后来正确；“变错”指原来正确、后来错误。置信区间为 10,000 次配对 bootstrap，种子20260925；变化及区间均以百分点表示。

| 比较 | 纠正 | 变错 | 净正确数变化 | Strict ACC 变化 | 95% 配对区间 | 双侧 exact McNemar p |
|---|---:|---:|---:|---:|---:|---:|
| R0→S0 | 10 | 14 | −4 | −1.62 | [−5.67, +2.02] | 0.5413 |
| S0→Final | 10 | 13 | −3 | −1.21 | [−4.86, +2.43] | 0.6776 |
| R0→Final | 9 | 16 | −7 | −2.83 | [−6.88, +1.21] | 0.2295 |

R0→Final 的准确率相对下降 **4.38%**。三个配对区间均包含0；本次单 seed 运行没有观察到主任务提升，也不足以断言方法必然导致显著退化。

### Rubric 与演化轨迹

五个 root 为 `factual_accuracy`、`methodological_correctness`、`computational_precision`、`conceptual_alignment`、`clarity_and_structure`。R0 有5个节点；S0 和 Final 均为 **5个 root＋18条子准则，共23个节点**，五个 root 的名称和职责保持固定。

下表的训练指标为方法阶段保存的 **K=1** 全局判断结果，不能与前面的正式 K=3 训练70指标混用。Root 1–5 对应上述顺序，“接受根组”表示该轮局部竞争后接受了该根的子准则组更新。

| 阶段 | 接受根组 | 训练70正确数 | K=1 Strict ACC |
|---|---|---:|---:|
| R0 | — | 50/70 | 71.43% |
| S0 | — | 49/70 | 70.00% |
| 第1轮 | Root 4、5 | 47/70 | 67.14% |
| 第2轮 | 无 | 47/70 | 67.14% |
| 第3轮 | Root 4、5 | 51/70 | 72.86% |
| 第4轮 | Root 2、3 | 50/70 | 71.43% |
| 第5轮 / Final | Root 3 | 48/70 | 68.57% |

五轮合计接受7次根组更新，最终按预定 `max_epochs=5` 停止。保留第5轮为 Final，没有根据训练或外评分数改选第3轮。局部接受更新没有保证 Global Arbiter 的整体训练判断单调提高。Dev150 旁观评测为 S0 **107/150（71.33%）**、Final **113/150（75.33%）**，采用 K=1，未用于选择 Final。

保存产物中的初始化耗时约 **41.8分钟**，五轮演化累计约 **75.6分钟**；三阶段全量外评耗时依次约 **97.9、141.6、131.9分钟**，合计约 **6小时11分钟**。这些是产物记录的阶段时间，不等同于完整实验从启动到结束的总耗时。初始化与演化记录了739次 Manager 逻辑调用、793次尝试，其中46次缺少 token usage；已记录的 token 数不代表完整成本。

### 主要观察

1. **主任务未提升。** R0→S0→Final 的 Reasoning247 正确数为160→156→153，下降集中在 MMMU-Pro 留出子集；MathVerse 最终与 R0 持平。
2. **全量小幅提升不能替代主任务结果。** S0 全量比 R0 多9个正确判断，来自训练70增加2个、General增加6个、Hallucination增加5个，同时 Reasoning 留出减少4个。仅看全量65.60%会掩盖初始化对子任务的退化。
3. **后续演化没有恢复主任务表现。** S0→Final 的训练70正式 K=3 正确数维持53个，但留出减少3个、General减少3个、Hallucination减少11个。训练表现与留出表现分离值得进一步检查；单次运行不能据此直接证明过拟合。
4. **局部更新的收益需要在整体判断中验证。** 五轮有7次根组更新被接受，但主任务和全量 Final 均低于 R0。本次结果支持继续检查局部准则更新如何影响跨根报告与 Global Arbiter；尚不支持“增加子准则或局部演化改善 Reasoning 留出判断”的结论。

## 完整运行命令

在 PowerShell 中执行。第0步从示例创建本地配置，执行后先修改新 `input_config.json` 的模型服务地址及 `.env` 路径，再继续第1步。它只复制配置，不依赖既有幻觉运行，也不复制其他实验的 root、准则、预测或缓存。

`input_config.json` 只在首次运行时生成；重新执行以下命令将保留它，并按已有阶段产物和缓存继续运行。若要重新生成一套独立 Rubric，请更换 `$runRoot`。确保 `.env` 中已经配置 Manager 的密钥。

```powershell
Set-Location "D:\3-Work\02-DD-LLM\02-Rubric"
conda activate critiq

$runRoot = "output/reasoning70/seed11/v3_main_reproduction"
$templateConfig = "experiments/evolving_structured_rubrics/configs/reasoning70.example.json"
$inputConfig = "$runRoot/input_config.json"

New-Item -ItemType Directory -Force -Path $runRoot | Out-Null

# 0. 首次创建本地运行配置；随后填写实际服务地址和凭据路径
if (-not (Test-Path -LiteralPath $inputConfig)) {
    $settings = Get-Content -Raw -Encoding UTF8 -LiteralPath $templateConfig | ConvertFrom-Json
    $configJson = $settings | ConvertTo-Json -Depth 30
    $configPath = Join-Path (Get-Location).Path $inputConfig
    [System.IO.File]::WriteAllText($configPath, $configJson, [System.Text.UTF8Encoding]::new($false))
}

# 先在此处修改新 input_config.json 的模型服务地址，再执行后续命令

# 1. 准备固定 Reasoning70，复用共享图片和原训练文件；不调用模型
python -u -m experiments.evolving_structured_rubrics.run_subtree_experiment prepare `
    --config $inputConfig --output-root $runRoot --variant g5 --seed 11 `
    2>&1 | Tee-Object -FilePath "$runRoot/prepare.log" -Append
if ($LASTEXITCODE -ne 0) { throw "数据准备失败，请检查 prepare.log" }

$config = "$runRoot/config.json"

# 2. 从 Reasoning70 抽取10对预热样例，重新生成五个 root
python -u -m experiments.evolving_structured_rubrics.run_subtree_experiment roots `
    --config $config --output-root $runRoot --variant g5 --seed 11 `
    --warmup-count 10 --root-attempt-limit 10 `
    2>&1 | Tee-Object -FilePath "$runRoot/roots.log" -Append
if ($LASTEXITCODE -ne 0) { throw "R0 生成失败，请检查 roots.log" }

# 3. 完整执行 signature、cluster、children，得到新的 S0
python -u -m experiments.evolving_structured_rubrics.run_subtree_experiment init `
    --config $config --output-root $runRoot --variant g5 --attempt-limit 10 `
    2>&1 | Tee-Object -FilePath "$runRoot/init.log" -Append
if ($LASTEXITCODE -ne 0) { throw "Init Split 失败，请检查 init.log" }

# 4. 从该 S0 开始最多五轮演化，冻结 Final
python -u -m experiments.evolving_structured_rubrics.run_subtree_experiment evolve `
    --config $config --output-root $runRoot --variant g5 --attempt-limit 10 `
    2>&1 | Tee-Object -FilePath "$runRoot/evolve.log" -Append
if ($LASTEXITCODE -ne 0) { throw "演化失败，请检查 evolve.log" }

# 5. Dev150 旁观评测，保持原 K=1 设置，不选择 checkpoint
python -u -m experiments.evolving_structured_rubrics.run_subtree_experiment dev `
    --config $config --output-root $runRoot --variant g5 --attempt-limit 10 `
    2>&1 | Tee-Object -FilePath "$runRoot/dev.log" -Append
if ($LASTEXITCODE -ne 0) { throw "Dev 评测失败，请检查 dev.log" }

# 6. R0：完整 VLRB，K=3，每次独立随机交换回答
python -u -m experiments.evolving_structured_rubrics.run_subtree_experiment vlrb-r0 `
    --config $config --output-root $runRoot --variant g5 --attempt-limit 10 `
    2>&1 | Tee-Object -FilePath "$runRoot/vlrb-r0.log" -Append
if ($LASTEXITCODE -ne 0) { throw "R0 外评失败，请检查 vlrb-r0.log" }

# 7. S0：同一份 VLRB 和随机展示日程
python -u -m experiments.evolving_structured_rubrics.run_subtree_experiment vlrb-s0 `
    --config $config --output-root $runRoot --variant g5 --attempt-limit 10 `
    2>&1 | Tee-Object -FilePath "$runRoot/vlrb-s0.log" -Append
if ($LASTEXITCODE -ne 0) { throw "S0 外评失败，请检查 vlrb-s0.log" }

# 8. Final：复用已有 S0 外评，评测演化后的最终 Rubric
python -u -m experiments.evolving_structured_rubrics.run_subtree_experiment vlrb `
    --config $config --output-root $runRoot --variant g5 --attempt-limit 10 `
    2>&1 | Tee-Object -FilePath "$runRoot/vlrb-final.log" -Append
if ($LASTEXITCODE -ne 0) { throw "Final 外评失败，请检查 vlrb-final.log" }

# 9. 离线生成完整报告，包括 Reasoning247 和 R0/S0/Final 配对比较
python -u -m experiments.evolving_structured_rubrics.run_subtree_experiment report `
    --config $config --output-root $runRoot --variant g5 `
    2>&1 | Tee-Object -FilePath "$runRoot/report.log" -Append
if ($LASTEXITCODE -ne 0) { throw "报告生成失败，请检查 report.log" }

Write-Host "实验完成，报告位于 $runRoot/g5/report.json"
```
