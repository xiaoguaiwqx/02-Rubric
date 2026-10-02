# CritiQ：结构化 Rubric 的子树局部演化

当前主线从多模态偏好样例生成或指定 root，利用错误签名和语义聚类构造初始 children；Worker 对每棵完整子树分别给出 A/B/None 与理由，Global Arbiter 综合完整报告。随后 Manager 按 `(root, sample)` 逐例反思，为每个 root 生成一整组候选 children；程序在冻结的当前 Rubric 上进行局部竞争，轮末提交获胜组。最终 Rubric 在独立数据上评测。

完整的早期 Gate/Cascade、joint、递归投票、模型与 prompt 对照留在 `codex/subtree-local-reflection` 分支和阶段标签。原 CritiQ-V 代码保存在 `CritiQ-V` 分支。

## 安装与离线验证

```powershell
conda activate critiq
python -m pip install -e ".[data]"
python -m unittest discover -s tests -p "test_*.py"
python -m experiments.evolving_structured_rubrics.run_subtree_experiment --help
```

`[data]` 安装读取 VL-RewardBench parquet 所需的 pandas 和 pyarrow。模型调用另需配置 Worker 端点和 Manager 的 API key；配置示例不包含凭据。数据、模型响应和缓存写入本地 `output/`，不提交 Git。

## 当前入口

G5/GN 的冻结 Hallucination100 seed11 split ID 与来源哈希位于 [split manifest](docs/experiments/vlrb-hallucination100-generated-roots/seed11_split.json)。准备阶段从本地 VL-RewardBench parquet 还原图像和发现集，之后分别创建 root、演化、外评。下面的示例以 G5 为例：

```powershell
python -m experiments.evolving_structured_rubrics.run_subtree_experiment prepare --config experiments/evolving_structured_rubrics/configs/generated_roots.example.json --output-root output/subtree_reflection/seed11
python -m experiments.evolving_structured_rubrics.run_subtree_experiment roots --config output/subtree_reflection/seed11/config.json --output-root output/subtree_reflection/seed11 --variant g5
python -m experiments.evolving_structured_rubrics.run_subtree_experiment vlrb-r0 --config output/subtree_reflection/seed11/config.json --output-root output/subtree_reflection/seed11 --variant g5
python -m experiments.evolving_structured_rubrics.run_subtree_experiment evolve --config output/subtree_reflection/seed11/config.json --output-root output/subtree_reflection/seed11 --variant g5
python -m experiments.evolving_structured_rubrics.run_subtree_experiment dev --config output/subtree_reflection/seed11/config.json --output-root output/subtree_reflection/seed11 --variant g5
python -m experiments.evolving_structured_rubrics.run_subtree_experiment vlrb --config output/subtree_reflection/seed11/config.json --output-root output/subtree_reflection/seed11 --variant g5
python -m experiments.evolving_structured_rubrics.run_subtree_experiment report --config output/subtree_reflection/seed11/config.json --output-root output/subtree_reflection/seed11 --variant g5
```

可将 `--variant` 改为 `gn`（同一预热历史、可变 root 数）或 `f5`（固定五根对照）。每个变体使用独立目录。R0 生成默认最多重试 10 次，演化中的 Manager 与 Worker 默认最多重试 4 次；可分别用 `--root-attempt-limit` 和 `--attempt-limit` 调整。旧 Discovery100 的 Covered、Strict 配置分别在 [`subtree_local_reflection.example.json`](experiments/evolving_structured_rubrics/configs/subtree_local_reflection.example.json) 和 [`subtree_local_reflection_strict.example.json`](experiments/evolving_structured_rubrics/configs/subtree_local_reflection_strict.example.json)；G5/GN 参考配置显式使用 Strict + Preserve5。换数据、接受指标、prompt 或模型时使用新运行目录。

当前方法的 [架构](docs/architecture.md)、[实验索引](docs/experiments/README.md)、[冻结协议](docs/experiments/subtree-local-reflection/plan.md)、[已观察结果](docs/experiments/subtree-local-reflection/results.md) 和 [框架 PPT](docs/experiments/subtree-local-reflection/framework.pptx) 分开维护。已保存结果显示 Discovery 的提升没有自动转化为外部泛化收益；请按正式 VLRB K=3 计分口径比较。
