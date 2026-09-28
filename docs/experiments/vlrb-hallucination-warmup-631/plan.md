# Hallucination 分层十样例预热实验

状态：**seed11 全流程已完成**（2026-09-26 18:21，Asia/Shanghai）。十条预热、初始 Split、五轮演化、1247 条 VLRB 外部评估及只读对照报告均已落盘。新入口的 `signature` 和 `case_reflection` 阶段并发均设为 10。本实验是在已完成的 VLRB 三类别专属 Rubric 实验之后，单独诊断 Hallucination root 归纳的预热样例选择。现有实验的冻结协议和产物保持原样。

## 研究问题

在同一批 Hallucination 150 条演化样本、同一套 GN 自选 2–7 root、Split、五轮局部反思和 VLRB 推理协议下，把 Manager 预热从随机 5 条改为按来源抽取 10 条（6／3／1），是否改善生成 root 的职责覆盖与未见 Hallucination 样本上的表现？

这是**来源分层与样例数同时改变**的探索实验。若结果提高，不能单独归因于分层或增加到十条；若要拆分因素，后续再增加按来源分层的五条预热对照。生成 root 的实际数量仍由 Manager 自选，记录其变化，不根据测试集改数量。

## 冻结数据与样例选择

- 直接读取 `output/vlrb_category_specialized_rubrics/seed11/split.json` 和 `hallucination/discovery.jsonl`，使用其中原有的 150 条 Hallucination 演化样本；留出 599 条及完整 VLRB 1247 条均不重新划分。
- 预热名额固定为 POVID **6**、RLAIF-V **3**、RLHF-V **1**。各来源内先按 `sample_id` 排序，再用固定 seed 11 的、含来源名称的独立随机流无放回抽取；随后用独立固定随机流打乱十条的输入顺序。原有 `random_reverse` 还会以固定 `REVERSE_SEED` 对十条进行打乱和 A/B 翻转，最终呈现顺序以 `warmup/transcript.json` 为准。不要依据测试成绩挑选或替换样例。
- 把十个样本 ID、来源、顺序、翻转后的 A/B、原始偏好和图像哈希冻结在新目录的 `warmup/transcript.json`。只允许这十条进入 Manager 预热；之后 Split 和演化仍使用全部 150 条。
- 新实验使用独立协议名及输出目录，例如 `output/vlrb_hallucination_warmup631_10/seed11/`，不能写入旧三类别实验目录；已完成的旧模型调用与预测只读复用。

## 最小代码修改

1. 在 `generated_root_initialization.generate_r0_pair` 增加可选的 `warmup_examples` 参数。未传时保留现有随机五条路径及默认参数；传入时只校验十条 ID 属于给定训练集且互不重复，然后沿用现有 `random_reverse`、图像校验、Agent 对话和缓存逻辑。新入口负责 6／3／1 抽样，不把来源配额规则写入通用生成器。
2. 把生成提示词首句的样例数变成变量：旧五条仍精确输出 `From the five comparisons above, ...`，新十条输出 `From the ten comparisons above, ...`。除该数量词外，`WARMUP_PROMPT`、root 职责要求、GN 数量指令、重试提示词均不改。
3. 将 `_warmup_request` 的 `sample_count` 和 `_run_warmup` 的轮次上限改为传入样例的实际长度；生成请求继续记录完整提示词与预热历史哈希，保证断点恢复读取的是同一批十条。
4. 增加一个薄的 Hallucination 专用入口：复用现有 150 条训练配置和 split、`generate_r0_pair(..., variants=("gn",))`、`subtree_local_reflection.run`、`framework_v6.evaluate` 及既有只读预测统计函数。只运行这一条新分支，不复制 Worker、Manager、Split、接受或计分实现，不重跑 General/Reasoning 与历史五根。

## 执行与比较

1. **准备**：冻结十条预热 ID 与新运行配置，确认来源配额为 6／3／1，训练与留出 ID 和上一实验一致。
2. **生成**：运行十条预热并归纳 R0；记录 root 数、名称、描述与旧 Hallucination 三根和历史五根的职责差异。
3. **演化**：用完整 150 条完成初始 Split 和最多五轮反思；报告训练集 R0→S0→Final，不用外部测试成绩选轮次。
4. **外部评估**：新 Final 仅推理一次全量 VLRB 1247 条，沿用原 K=3、A/B 日程、Worker/Arbiter 配置和多数票统计；从同一份预测切出 Hallucination 留出 599、全量 Hallucination 749 和 VLRB 留出 998。其他组的新推理不需要运行。
5. **只读对照**：用现有逐样本预测比较本次新 Final、三类别实验中的 Hallucination Final，以及历史 Hallucination100 偏好归纳五根 Final。主指标是同一批留出 599 条上的 Strict ACC 与配对纠正/伤害数；全量 1247、留出 998、Covered ACC 和训练表现作辅助读数。历史五根使用旧 100 条演化集，须标明训练集不同，不能作为单变量对照。

若 6／3／1 十条方案仍显著落后于历史五根，先检查生成 root 的职责覆盖及逐样本失误；不按 599 条测试结果反复重抽预热样例。由于该留出集已被多次分析，本次结果定位为探索性证据。

## 实现入口与运行命令

入口为 `experiments.evolving_structured_rubrics.vlrb_hallucination_warmup_631`。在 `critiq` 环境、仓库根目录按阶段执行；`prepare` 冻结十条样例的来源与 ID、执行既有配置校验，不发起模型请求。每个阶段都会读取并核对同一份既有 150 条演化数据，成功缓存留在独立输出目录。seed11 已从清空的运行目录重新执行 `all`。

```powershell
conda activate critiq
python -m experiments.evolving_structured_rubrics.vlrb_hallucination_warmup_631 prepare --seed 11
python -m experiments.evolving_structured_rubrics.vlrb_hallucination_warmup_631 generate --seed 11
python -m experiments.evolving_structured_rubrics.vlrb_hallucination_warmup_631 evolve --seed 11
python -m experiments.evolving_structured_rubrics.vlrb_hallucination_warmup_631 vlrb --seed 11
python -m experiments.evolving_structured_rubrics.vlrb_hallucination_warmup_631 report --seed 11
```

`all --seed 11` 可顺序执行完整流程；产物位于 `output/vlrb_hallucination_warmup631_10/seed11/`，最终摘要为 `report.json`。`report` 只读对照产物，不发起模型请求。

本次启动器位于运行目录下的 `launch_native.cmd`，stdout 实时追加至 `experiment.log`，stderr 追加至 `experiment.stderr.log`。`process.json` 留存启动时 PID；完成状态以阶段产物和 `report.json` 为准。

## seed11 结果（2026-09-26）

十条预热归纳出 **4 个 root**。演化集 150 条上的 Strict ACC 为 R0 **122/150（81.33%）**、Split 后 **112/150（74.67%）**、五轮 Final **121/150（80.67%）**。Final 比 R0 少 1 个正确样本。

下表以 `report.json` 对三组既有逐样本预测采用相同的只读统计口径；历史生成五根使用不同的 Hallucination100 演化集，属于参照结果。

| Rubric | VLRB 全量 1247 | VLRB 留出 998 | Hallucination 全量 749 | Hallucination 留出 599 |
| --- | ---: | ---: | ---: | ---: |
| 本次 6／3／1 十样例，4 root | 925/1247（74.18%） | 734/998（73.55%） | 619/749（82.64%） | 492/599（82.14%） |
| 三类别实验随机五样例，3 root | 891/1247（71.45%） | 712/998（71.34%） | 590/749（78.77%） | 475/599（79.30%） |
| 历史 Hallucination100 偏好归纳五根 | 934/1247（74.90%） | 747/998（74.85%） | 624/749（83.31%） | 501/599（83.64%） |

主指标 Hallucination 留出 599 条：本次比随机五样例多 **17** 个正确样本（+2.84 pp；配对纠正 38、伤害 21；McNemar 精确双侧 p=0.036）；比历史五根少 **9** 个（−1.50 pp；纠正 16、伤害 25；p=0.211）。因此，分层十样例方案改善了本次同训练集的随机五样例结果，但未超过历史五根。来源分层、样例数和实际 root 数一起变化，不能把提升单独归因于其中一项。留出集已被多次查看，此处为探索性比较。

`gn/vlrb/final.json` 自带的全量 Strict ACC 为 **74.34%**，而 `report.json` 对逐样本预测重新按统一口径统计为 **74.18%**；上表统一使用后者，避免把两种口径混用。
