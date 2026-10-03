# 当前方法与代码边界

```mermaid
flowchart LR
    W[偏好预热：默认五样例] --> G[偏好归纳 G5/GN]
    G --> R[R0：根准则]
    H[人工固定 F5] --> R
    R --> S[S0：错误签名、聚类、整组 children]
    S --> E[每棵完整子树一次 Worker 判断]
    E --> A[Global Arbiter 系统偏好]
    E --> F[逐 root、逐案例反思]
    F --> C[生成候选 children 组]
    C --> L[冻结 Current 上的局部竞争]
    L --> U[轮末提交获胜组]
    U --> E
    U --> V[冻结 Final 与独立外评]
```

主要框架图见 [assets/framework.png](../assets/framework.png)，方法概览与运行说明见仓库 [README](../README.md)。图以五根为例，GN 支持 2–7 根。每轮冻结同一份 Rubric，各 root 遍历发现集逐例反思并独立验证候选，轮末组装获胜组。

`structured_rubrics/` 提供 Agent、JSON 解析、Rubric schema、验证、后端规格和调用遥测。`experiments/evolving_structured_rubrics/` 中 `generated_root_initialization.py` 负责预热及 R0，`rubric_pipeline.py` 负责 S0 与外评，`subtree_local_reflection.py` 负责逐例反思和局部接受，`aligned_system_runtime.py` 负责 Worker/Arbiter 判断与报告复用。`aligned_prompts.py` 固定 prompt 和解析器，`model_call_support.py` 固定请求身份与多模态内容，`vlrb_official.py` 固定正式 K=3 日程和指标。`run_subtree_experiment.py` 是这三个阶段的命令入口。

局部接受可配置为 Covered ACC 或 Strict ACC；Preserve5 是独立的正确案例提示干预。配置区别见 [README 配置表](../README.md#配置与数据)，当前方案的对比表见[最终实验结果](experiments/subtree-local-reflection/results.md)。正式 VLRB 采用至少两票一致的 K=3 多数口径；`aligned_system_runtime.metrics` 的相对多数仅用于运行时诊断。两者不能混作同一结果。

配置示例的区别见 [README 配置表](../README.md#配置与数据)：Hallucination100 的 `generated_roots.example.json` 使用 Strict + Preserve5；原 Discovery100 的两个示例分别使用 Covered 和 Strict，均不加入保留案例。它们的预热、发现集和局部候选验证是不同阶段；Preserve5 不改变预热数量，后者通过 `roots --warmup-count` 设置，默认五条。

`manager_runtime.py` 提供 Manager 调用、解析、缓存、并发、429 冷却及最多 10 次尝试，并保留初始化的 `signature`、`cluster`、`children` 三个阶段。`subtree_local_reflection_manager.py` 注入演化的 `case_reflection`、`subtree_split` prompt 和校验。旧 `system`、`root` 阶段已移出当前代码；历史实现保留在实验分支。

`rubric_pipeline.py` 的 S0 初始化先对各 root 判断错误的案例生成 signature；不足 4 条有效 signature 或 2 个支持 cluster 时，补充其余发现集案例后再尝试生成 children。初始化请求继续保留 `revision_goal=""`、`initial=True`、固定的初始 Split 描述，以及目标 root 报告和完整 Arbiter 理由；旧候选/旧基线输入和非初始化分支已移除。演化的逐例反思使用独立的局部 payload。外评、配置冻结及成本统计沿用原实现。

`generated_root_initialization.py` 默认抽取五个预热样例，可通过 `roots --warmup-count` 改变抽样数量；默认行为保留冻结的抽样、A/B 随机翻转和顺序，并恢复全局随机状态。G5/GN 从同一份预热对话分叉，生成请求只改变 root 数量要求；恢复时重建已成功的对话历史并跳过成功调用。预热数量、样例及历史仍由现有请求缓存记录，改变数量须使用新输出目录。辅助函数均有实际调用，显式预热样例、协议参数及数量提示词变体继续保留，以支持已有实验复现。固定 F5 对照仍由 `rubric_pipeline.build_multicrit_open_ended_init_rubric` 创建。

`aligned_system_runtime.py` 保留三条核心路径：`evaluate` 为每棵完整子树生成报告后调用 Global Arbiter；增量评测复用冻结基线中未改动的子树报告，并在新报告组合上重新执行 Arbiter；`root_from_system` 和 `evaluate_root` 分别提取局部基线、仅调用 Worker 验证候选，局部接受仍由 `subtree_local_reflection.compare` 决定。报告中的 A/B 标签按每次调用的展示顺序还原。请求缓存键、prompt、解析器、最多 10 次尝试及正式计分规则沿用冻结实现。无调用的 `paired_root`、`paired_root_all_samples`、`attributed_error_ids` 已移出当前代码；历史实现仍可在实验分支查看。

原 Gate/Cascade、joint、分层反思和历史 runner 的完整实现保存在 `codex/subtree-local-reflection` 分支及阶段标签。当前入口不会导入那些模块。
