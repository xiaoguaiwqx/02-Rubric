# Split 与 Refine 算子算法框架

这张模块图从算法接口的角度展示当前 Local Unified-Subtree 演化协议。左侧输入在每个 epoch 开始时被冻结；中间两列分别是 Split 和 Refine 的独立候选生成与评估模块；右侧统一比较并同步提交。每个模块都标出了主要输入、模型调用、评估输出，以及失败后的归因和重试状态。图片只在代表性 Refine 证据和 Pairwise Worker 评估阶段实际参与推理。

```mermaid
block-beta
  columns 5
  input["Epoch t 输入<br/>当前 Rubric、Discovery rows、Worker 预测、历史"]
  block:splitmodule
    columns 1
    splitin["输入：root 错误样本"]
    splitclusters["1. ErrorSignature<br/>+ 语义聚类"]
    splitproposal["2. Split Manager<br/>生成 child 候选"]
    splitworker["3. Pairwise Worker<br/>评估 child（看图）"]
    splitguard["4. child 指标<br/>+ root Unified-Subtree guard"]
    splitresult["输出：接受候选<br/>或拒绝归因 / 重试"]
    splitin --> splitclusters
    splitclusters --> splitproposal
    splitproposal --> splitworker
    splitworker --> splitguard
    splitguard --> splitresult
  end
  block:refinemodule
    columns 1
    refinein["输入：目标节点证据"]
    refineevidence["1. 选择错误、正确、None<br/>代表样本与图片"]
    refinmanager["2. Refine Manager<br/>修改同一 criterion（看代表图）"]
    refineworker["3. Pairwise Worker<br/>重新评估候选（看图）"]
    refineguard["4. node gate<br/>+ root Unified-Subtree guard"]
    refineresult["输出：接受新描述<br/>或拒绝归因 / 重试"]
    refinein --> refineevidence
    refineevidence --> refinmanager
    refinmanager --> refineworker
    refineworker --> refineguard
    refineguard --> refineresult
  end
  block:commitmodule
    columns 1
    compare["统一比较<br/>检查候选是否真正改善"]
    merge["同步 Epoch Commit<br/>只合并通过的候选"]
    artifacts["写出 Rubric、预测、反馈<br/>与失败归因历史"]
    compare --> merge
    merge --> artifacts
  end
  output["Epoch t+1 输出<br/>更新后的 Rubric 与下一轮状态"]

  input --> splitmodule
  input --> refinemodule
  splitmodule --> commitmodule
  refinemodule --> commitmodule
  commitmodule --> output

  classDef inputStyle fill:#D1FAE5,stroke:#047857,color:#064E3B,stroke-width:2px;
  classDef splitStyle fill:#FEF3C7,stroke:#B45309,color:#78350F,stroke-width:1.5px;
  classDef refineStyle fill:#EDE9FE,stroke:#6D28D9,color:#4C1D95,stroke-width:1.5px;
  classDef commitStyle fill:#DBEAFE,stroke:#1D4ED8,color:#1E3A8A,stroke-width:1.5px;
  classDef outputStyle fill:#FFEDD5,stroke:#C2410C,color:#7C2D12,stroke-width:2px;
  class input inputStyle;
  class splitmodule splitStyle;
  class refinemodule refineStyle;
  class commitmodule commitStyle;
  class output outputStyle;
```
