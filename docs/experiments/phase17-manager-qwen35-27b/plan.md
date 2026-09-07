# Phase17：本地 Qwen3.5-27B Manager No-Thinking 实验

状态：运行配置已确定，待按本文命令重新生成本地配置并正式运行。

## 目的

在已经完成的本地 Qwen3.5-27B Manager 实验基础上，关闭 Manager 的思考模式，并加入两项针对本地小模型结构化输出的轻量配置调整，检验其演化效果。

本实验不修改 Phase17 的 Split/Refine 接受条件、同步提交方式、epoch 数量或数据划分。相对原设置，解析失败最多重试3次，将每个错误 cluster 的最小样本数从5降为2，并仅简化 Manager 接口中的样本标识。

## 配置变化

| 配置 | Phase17 | 本实验 |
| --- | --- | --- |
| Manager | `Qwen/Qwen3.5-397B-A17B` | `Qwen/Qwen3.5-27B` |
| Manager endpoint | SiliconFlow | 本地 vLLM，使用 `$env:PHASE17_MANAGER_BASE_URL` |
| Pairwise Worker | `Qwen/Qwen3-VL-8B-Instruct` | 不变 |
| Worker endpoint | 两个端点，总并发40 | 使用 `$env:PHASE17_WORKER_BASE_URL`，总并发100 |
| Manager并发 | ErrorSignature 30，其余1 | ErrorSignature 30，其余1 |
| Manager总输出上限 | 未单独设置 | `max_completion_tokens=16384` |
| 客户端单次请求超时 | 180秒 | 900秒（保持上一轮本地 Manager 实验设置） |
| Manager结构化解析重试 | 最多1次 | 最多3次，即最多4次生成尝试 |
| 错误cluster最小样本数 | 5 | 2 |
| Manager样本标识 | 完整 `sample_id` | ErrorSignature 不回传ID；聚类使用 `S001` 等短键 |

本实验使用独立输出目录，避免覆盖 Phase17 结果。内部地址只写入 `.local/` 下的本地配置，不提交到仓库。

本实验将本地 vLLM Manager 的 `chat_template_kwargs.enable_thinking` 设为 `false`，将 `structured_max_retries` 设为3，并将 `N_min_cluster` 设为2；`max_completion_tokens=16384` 保持不变。这是一个包含少量稳定性调整的本地27B实验，不将结果严格归因于单一变量。

## 最小代码改动

- Phase17 从 Manager 配置中读取统一模型名，不再把397B写死在校验中。
- Phase17 从 `discovery_v2_prompt_v2_evolution.worker_endpoints` 读取 Worker 端点列表，允许本实验使用单端点。
- VL-RewardBench 从当前 `backend_pool` 读取端点，不再要求端点必须命名为 `vllm-8000` 和 `vllm-8001`。
- ErrorSignature 的真实 ID 由程序注入；聚类短键在解析后映射回真实 ID，下游逻辑不变。
- 本实验的 VL-RewardBench 结果独立写入 `output/evolving_structured_rubrics/vl_rewardbench_phase17_manager_qwen35_27b_no_thinking_v1`，不覆盖 thinking-on 结果。
- 其他演化代码与评测代码继续复用 Phase17 原实现。

## 生成本地配置

在 PowerShell 中执行，将以下服务地址占位符替换为本地部署地址；真实内网地址仅保存在本地配置中，不提交到仓库：

```powershell
conda activate critiq
$env:PHASE17_MANAGER_BASE_URL = "http://<manager-host>:8000/v1"
$env:PHASE17_WORKER_BASE_URL = "http://<worker-host>:8000/v1"
$ManagerCheckpointRoot = "/media/disk12T/2022-sgh/ModelWeight/Qwen3.5-27B"
$WorkerCheckpointRoot = "/media/oem/12T1/WQX/02_DD_LLM/local-llm-inference/ModelWeight/Qwen3-VL-8B-Instruct"
$Base = "experiments/evolving_structured_rubrics/configs/rubric_evolution_phase5.example.json"
$ManagerUrl = $env:PHASE17_MANAGER_BASE_URL
$WorkerUrl = $env:PHASE17_WORKER_BASE_URL
if (-not $ManagerUrl -or -not $WorkerUrl) { throw "Set PHASE17_MANAGER_BASE_URL and PHASE17_WORKER_BASE_URL first" }
$ConfigDir = ".local/phase17_manager_qwen35_27b"
$Config = "$ConfigDir/config.json"
New-Item -ItemType Directory -Force $ConfigDir | Out-Null
$c = Get-Content $Base -Raw | ConvertFrom-Json
$c.vllm_version = "0.17.0"
$c | Add-Member -NotePropertyName discovery_v2_heldout_reference_output `
    -NotePropertyValue "output/evolving_structured_rubrics/rubric_evolution_phase5" -Force
$c.structured_max_retries = 3
$c.discovery_v2_prompt_v2_evolution |
    Add-Member -NotePropertyName split_min_cluster_size -NotePropertyValue 2 -Force
$c.discovery_v2_prompt_v2_evolution |
    Add-Member -NotePropertyName compact_manager_sample_ids -NotePropertyValue $true -Force

$c.backend_pool.pool_id = "qwen3vl8b-local-single-v1"
$c.backend_pool.common_checkpoint_id = "Qwen/Qwen3-VL-8B-Instruct"
$c.backend_pool.global_request_concurrency = 100
$c.backend_pool.endpoints = @([pscustomobject]@{
    endpoint_id = "qwen3vl8b-local"
    base_url = $WorkerUrl
    checkpoint_root = $WorkerCheckpointRoot
    max_concurrency = 100
})
$c.discovery_v2_prompt_v2_evolution.worker_endpoints = @("qwen3vl8b-local")
$c.vlrb_discovery_v2.endpoint_ids = @("qwen3vl8b-local")
$c.vlrb_prompt_v2_transfer.endpoint_ids = @("qwen3vl8b-local")

foreach ($name in @("error_signature", "semantic_cluster", "child_generation")) {
    $m = $c.specialize_managers.$name
    $concurrency = if ($name -eq "error_signature") { 30 } else { 1 }
    $m.model = "Qwen/Qwen3.5-27B"
    $m.api_key_env = $null
    $m.vllm_identity = $null
    $m.backend_pool.pool_id = "qwen35-27b-local-manager-$name"
    $m.backend_pool.common_checkpoint_id = "Qwen/Qwen3.5-27B"
    $m.backend_pool.global_request_concurrency = $concurrency
    $m.backend_pool.endpoints = @([pscustomobject]@{
        endpoint_id = "qwen35-27b-local"
        base_url = $ManagerUrl
        checkpoint_root = $ManagerCheckpointRoot
        max_concurrency = $concurrency
    })
    $m.request_kwargs | Add-Member -NotePropertyName max_completion_tokens -NotePropertyValue 16384 -Force
    $m.request_kwargs.extra_body = [pscustomobject]@{
        chat_template_kwargs = [pscustomobject]@{ enable_thinking = $false }
    }
}

$c.refine_manager.model = "Qwen/Qwen3.5-27B"
$c.refine_manager.api_key_env = $null
$c.refine_manager.backend_pool.pool_id = "qwen35-27b-local-refine-manager"
$c.refine_manager.backend_pool.common_checkpoint_id = "Qwen/Qwen3.5-27B"
$c.refine_manager.backend_pool.global_request_concurrency = 1
$c.refine_manager.backend_pool.endpoints = @([pscustomobject]@{
    endpoint_id = "qwen35-27b-local"
    base_url = $ManagerUrl
    checkpoint_root = $ManagerCheckpointRoot
    max_concurrency = 1
})
$refineExtra = [pscustomobject]@{
    chat_template_kwargs = [pscustomobject]@{ enable_thinking = $false }
}
$c.refine_manager.generation_request_kwargs | Add-Member -NotePropertyName max_completion_tokens -NotePropertyValue 16384 -Force
$c.refine_manager.attribution_request_kwargs | Add-Member -NotePropertyName max_completion_tokens -NotePropertyValue 16384 -Force
$c.refine_manager.generation_request_kwargs.extra_body = $refineExtra
$c.refine_manager.attribution_request_kwargs.extra_body = $refineExtra

$json = $c | ConvertTo-Json -Depth 100
[System.IO.File]::WriteAllText(
    [System.IO.Path]::GetFullPath($Config), $json,
    [System.Text.UTF8Encoding]::new($false))

$generated = Get-Content $Config -Raw | ConvertFrom-Json
@(
    $generated.specialize_managers.error_signature.request_kwargs.max_completion_tokens
    $generated.specialize_managers.semantic_cluster.request_kwargs.max_completion_tokens
    $generated.specialize_managers.child_generation.request_kwargs.max_completion_tokens
    $generated.refine_manager.generation_request_kwargs.max_completion_tokens
    $generated.refine_manager.attribution_request_kwargs.max_completion_tokens
)
$generated.structured_max_retries
$generated.discovery_v2_prompt_v2_evolution.split_min_cluster_size
$generated.discovery_v2_prompt_v2_evolution.compact_manager_sample_ids
$generated.discovery_v2_heldout_reference_output
```

最后应依次输出五个 `16384`、一个 `3`、一个 `2`、一个 `True`，以及历史对照目录 `output/evolving_structured_rubrics/rubric_evolution_phase5`。前面的 Manager 与演化参数属于冻结实验配置，修改后必须使用空的实验输出目录从 `freeze` 重新开始，不能与旧输出混用。历史对照目录仅用于 heldout 定位既有 Phase10/Phase16 产物，不改变本次实验协议，因此修正该路径后可以直接重跑 heldout。

本实验使用的 checkpoint 路径为：

- Manager：`/media/disk12T/2022-sgh/ModelWeight/Qwen3.5-27B`
- Pairwise Worker：`/media/oem/12T1/WQX/02_DD_LLM/local-llm-inference/ModelWeight/Qwen3-VL-8B-Instruct`

如果运行机器上的挂载路径不同，只需修改上面两个 PowerShell 变量；不要修改演化代码。

## 完整运行命令

```powershell
$Config = ".local/phase17_manager_qwen35_27b/config.json"
$Output = "output/evolving_structured_rubrics/phase17_manager_qwen35_27b_no_thinking_compact_ids/rubric_evolution_phase5"
$ErrorActionPreference = "Stop"

$runConfig = Get-Content $Config -Raw | ConvertFrom-Json
$managerTokenLimits = @(
    $runConfig.specialize_managers.error_signature.request_kwargs.max_completion_tokens
    $runConfig.specialize_managers.semantic_cluster.request_kwargs.max_completion_tokens
    $runConfig.specialize_managers.child_generation.request_kwargs.max_completion_tokens
    $runConfig.refine_manager.generation_request_kwargs.max_completion_tokens
    $runConfig.refine_manager.attribution_request_kwargs.max_completion_tokens
)
if ($managerTokenLimits.Count -ne 5 -or
    @($managerTokenLimits | Where-Object { $_ -ne 16384 }).Count -ne 0) {
    throw "Manager max_completion_tokens is not uniformly set to 16384; regenerate the local config first."
}
if ($runConfig.structured_max_retries -ne 3) {
    throw "Manager structured_max_retries must be 3; regenerate the local config first."
}
if ($runConfig.discovery_v2_prompt_v2_evolution.split_min_cluster_size -ne 2) {
    throw "Phase17 split_min_cluster_size must be 2; regenerate the local config first."
}
if (-not $runConfig.discovery_v2_prompt_v2_evolution.compact_manager_sample_ids) {
    throw "Phase17 compact_manager_sample_ids must be true; regenerate the local config first."
}

function Run-Stage([string]$Stage) {
    python -m experiments.evolving_structured_rubrics.run_rubric_evolution $Stage --config $Config --output-dir $Output
    if ($LASTEXITCODE -ne 0) { throw "Stage failed: $Stage" }
}

Run-Stage "discovery-v2-evolution-freeze"
Run-Stage "discovery-v2-evolution-audit"
Run-Stage "discovery-v2-evolution-smoke"
Run-Stage "discovery-v2-evolution-run"
Run-Stage "discovery-v2-evolution-report"
Run-Stage "discovery-v2-evolution-heldout"
Run-Stage "discovery-v2-evolution-final-report"

# VL-RewardBench：用同一个 Worker 配置评测本次最终 Rubric
Run-Stage "vlrb-discovery-v2-freeze"
Run-Stage "vlrb-discovery-v2-audit"
Run-Stage "vlrb-discovery-v2-smoke"
Run-Stage "vlrb-discovery-v2-run"
Run-Stage "vlrb-discovery-v2-retry"
Run-Stage "vlrb-discovery-v2-report"
```

## 分析原则

比较本实验与 Phase17 的演化轨迹、结构化失败率、接受的 Split/Refine 操作、Discovery100、heldout500 和后续 VL-RewardBench 指标。由于 Manager规模与部署、Worker部署与并发、思考模式、解析重试次数和最小cluster样本数均有变化，应把结论表述为“本地27B Manager调整后运行配置的整体效果”，不能将差异严格归因于某个单一变量。

## 后续诊断：最后一轮完整候选 Rubric，不按 Split 竞争筛选

使用已完成的 `phase17_manager_qwen35_27b_no_thinking_compact_ids` 实验：Completeness、Visual Grounding、Creativity、Clarity 各取 Epoch 5 最后一次 `candidate_rubric.json` 中对应 root 的完整子树；Factuality 保留最终已接受子树及 Refine。保留原始节点、描述和边，不累加历史候选，不纳入无效 Refine。五根的 children 数依次为 5、5、5、5、3，共 28 个节点。

组装脚本是本地 `.local/phase17_27b_full/prepare.py`，生成同目录的 `rubric.json` 和 `config.json`。Rubric SHA256 为 `fa5286cbe828af4527ddd239bf9a522ad96d83451ff1a1c24338d5056474fc7f`。两台 Worker 各 100 并发、总并发 200，模型均为 Qwen3-VL-8B-Instruct。地址与各服务器 checkpoint 路径仅保存在本地配置。其余 Prompt、temperature=0.5、max_tokens=2048、K=3 和投票逻辑沿用原评测。

入口复用 `vlrb-discovery-v2-*`，仅在 `vlrb_discovery_v2` 中增加 `rubric_path` 和 `output_dir`。显式 Rubric 不要求伪造 evolution/heldout 报告；评测仍记录 Rubric 哈希。结果独立存入 `output/evolving_structured_rubrics/vlrb_27b_full`，不覆盖正常竞争版。报告中的 treatment 系统名称沿用旧入口，应结合源 Rubric 路径区分本次实验。终端 Control 仍指历史 Phase10；同次初始五根及各 subtree 预测保存在 `retry/combined/logical_votes.json`，正常竞争版也可从相同节点预测离线重算。

这是看过 VL-RB 结果后提出的事后诊断，用于分析候选与竞争机制，不用于宣称未见数据上的确认性提升。

配置和 Rubric 已生成，直接从仓库根目录运行下面整段代码。脚本块在首个失败阶段停止；重跑使用现有缓存，不删除输出目录。

```powershell
conda activate critiq
$Config = ".local/phase17_27b_full/config.json"
$Output = "output/evolving_structured_rubrics/vlrb_27b_full"
& {
    $ErrorActionPreference = "Stop"
    foreach ($Stage in @(
        "vlrb-discovery-v2-freeze",
        "vlrb-discovery-v2-audit",
        "vlrb-discovery-v2-smoke",
        "vlrb-discovery-v2-run",
        "vlrb-discovery-v2-retry",
        "vlrb-discovery-v2-report"
    )) {
        python -m experiments.evolving_structured_rubrics.run_rubric_evolution $Stage --config $Config --output-dir $Output
        if ($LASTEXITCODE -ne 0) { throw "Stage failed: $Stage" }
    }
}
```

## 后续诊断：Epoch 5 完整 Rubric + Clean S5-v2

固定上面的28节点 Rubric，只将显式递归投票替换为第17章的 Clean S5-v2：五个 Unified Subtree Worker 各生成一份完整子树报告，再由 Global Arbiter 形成判断。复用 `aligned_system_runtime.evaluate()` 及已有 Prompt、parser、缓存和指标函数，不修改任何推理或演化实现；本地调用脚本为 `.local/phase17_27b_full/run_clean_s5.py`（与本地配置一起保留，不是新增的公共实验框架）。

- 参数不变：Qwen3-VL-8B-Instruct、temperature=0.5、max_tokens=2048、最多10次解析重试、不设置 generation seed，原生 `None` 是有效弃权，不强制补成 A/B。
- 两个服务端、checkpoint 和各100并发读取既有 `.local/phase17_27b_full/config.json`。仅在内存中将端点名称映射为运行时已有的 `vllm-8000`、`vllm-8001`，不改服务地址或原配置文件，不修改模块全局变量。
- 复用 `vlrb_27b_full/order_schedule.json` 的1,247条样本、三次 A/B 顺序。逻辑请求为18,705次子树推理 + 3,741次 Arbiter推理，共22,446次。旧 Pairwise 输出不能充当子树报告；本次报告重新生成。
- 主对照是同一28节点 Rubric的显式递归结果（Overall ACC 71.39%、Macro ACC 66.09%、Strict ACC 71.05%），不是 Phase10 或旧 Phase17 E4。报告包含配对纠正/改错数，结果独立写入 `output/evolving_structured_rubrics/vlrb_27b_full_s5/final_report.json`。
- `check` 为离线检查；`smoke` 测20条，`run` 正式推理，`retry` 复用成功缓存并补失败请求，`report` 只做离线统计。进度按样本报告：一个完整样本包含三次重复的18个请求，不再显示 `prompt_v2_replicate_01`。同一样本的调用沿用现有端点亲和调度，因此完成首条进度前需要等待多个请求。

从仓库根目录运行整段命令；遇错立即停止，恢复时保留输出和缓存，不删除旧实验：

```powershell
conda activate critiq
$Config = ".local/phase17_27b_full/config.json"
$Rubric = ".local/phase17_27b_full/rubric.json"
$Control = "output/evolving_structured_rubrics/vlrb_27b_full"
$Output = "output/evolving_structured_rubrics/vlrb_27b_full_s5"
& {
    $ErrorActionPreference = "Stop"
    foreach ($Stage in @("check", "smoke", "run", "retry", "report")) {
        python .local/phase17_27b_full/run_clean_s5.py $Stage --config $Config --rubric $Rubric --control-dir $Control --output-dir $Output
        if ($LASTEXITCODE -ne 0) { throw "Stage failed: $Stage" }
    }
}
```

实现验证：已通过实际28节点/1,247条数据的离线 `check`，两个 System Prompt 与历史 Clean S5 对应文本一致，parser 接受原生 `None`。本地假后端测试覆盖 smoke→run→retry→report；尚未启动真实 smoke 或正式推理。本实验是已见测试集上的聚合诊断，不将结果用于反向选择 Rubric。

## 后续模型替换：完整 Rubric + Clean S5-v2 + Qwen3.5-27B

等8B实验全部完成（包括 retry/report）后，再将两台服务部署为 `Qwen/Qwen3.5-27B`，服务端默认设置 `--default-chat-template-kwargs '{"enable_thinking":false}'`，确保 served model name 与配置一致。地址和端口沿用本地配置；不要在8B运行中切换模型。

仅使用新配置 `.local/phase17_27b_full/config_qwen35_27b.json`：模型及公共 checkpoint 标识改为27B，每个端点并发20、总并发40。端点 `checkpoint_root` 暂用模型标识 `Qwen/Qwen3.5-27B`（不是已验证的磁盘路径），现有本地脚本仅用其作缓存身份，实际API按 `model` 选模型。其余复制原配置，Manager字段不参与本次评测。Rubric、Prompt、temperature=0.5、max_tokens=2048、K=3顺序、原生None及重试规则不变，算法代码与调用脚本均不修改。关闭思考由服务端负责，离线 check 不验证服务端设置。

新结果和缓存独立写入 `vlrb_27b_full_s5_qwen35`。脚本内置报告仍对比8B显式递归；模型替换的主要科学对照应是已完成的8B Clean S5-v2，待运行后离线配对比较，不把控制目录换成S5目录（其产物格式不同）。并发变化同时影响性能，耗时不能只归因于模型规模。

**已完成运行的恢复提醒：** 该27B Clean S5-v2实验已完成，最终报告技术失败为0，包含解析恢复与两条4096补救；不要对其已有输出直接重跑下列整段命令。结果与限制见主文档第19章。本段命令保留作为最初运行协议，下一项递归实验使用独立目录。

部署完成后，从仓库根目录运行：

```powershell
conda activate critiq
$Config = ".local/phase17_27b_full/config_qwen35_27b.json"
$Rubric = ".local/phase17_27b_full/rubric.json"
$Control = "output/evolving_structured_rubrics/vlrb_27b_full"
$Output = "output/evolving_structured_rubrics/vlrb_27b_full_s5_qwen35"
& {
    $ErrorActionPreference = "Stop"
    foreach ($Stage in @("check", "smoke", "run", "retry", "report")) {
        python .local/phase17_27b_full/run_clean_s5.py $Stage --config $Config --rubric $Rubric --control-dir $Control --output-dir $Output
        if ($LASTEXITCODE -ne 0) { throw "Stage failed: $Stage" }
    }
}
```

## 新对照：Qwen3.5-27B + Epoch 5 完整 Rubric + 子准则多数投票递归

目的：固定同一27B与28节点Rubric，对比“逐节点判断、子准则多数投票递归、五根等权投票”和“完整子树报告 → Clean S5-v2 Arbiter”。这是两种完整推理方式的比较，不是只改最后一次投票公式。主要对照为已完成的27B Clean S5-v2（Overall 83.08%、Macro 78.79%、Strict 82.68%），不是历史Phase10。

### 最小改动与固定设置

- 复用8B完整Rubric的 `vlrb-discovery-v2-*` 入口，不改节点Prompt、递归多数规则、父节点回退、五根聚合、样本或解析器。
- 新配置：`.local/phase17_27b_full/config_recursive_qwen35_27b.json`。沿用两个本地API地址，模型为 `Qwen/Qwen3.5-27B`，各20并发、总40；两台服务端默认 `enable_thinking=false`。配置本身不覆盖服务端thinking设置。
- 同一 `.local/phase17_27b_full/rubric.json`，SHA-256 为 `fa5286cbe828af4527ddd239bf9a522ad96d83451ff1a1c24338d5056474fc7f`；1,247样本，K=3、seed=42生成原有平衡A/B顺序，temperature=0.5、max_tokens=2048。
- 独立输出：`output/evolving_structured_rubrics/vlrb_27b_full_recursive_qwen35`。模型与输出目录不复用8B缓存，也不覆盖27B Clean S5-v2结果。
- 原入口要求与历史8B对照的模型完全一致，会阻止模型替换。仅新增 `vlrb_discovery_v2.allow_model_change=true` 配置开关：默认关闭，开启时只放行模型字段差异，其他请求参数仍严格匹配；完整模型身份照常写入manifest。此处改变的是参数入口校验，不是算法。
- 主体逻辑调用数为 $1247\times3\times28=104748$，Clean S5-v2为22,446；递归版调用数约为其4.67倍，但单次输入/输出长度不同，不能据此直接推断耗时比。

### 解析、补救和比较规则

保留原节点解析与现有重试设置，技术retry最多10次，不把格式失败当作有效 `None`，不按答案正确性选择重试结果。本次**不预先把失败预算提高到4096**，也不套用子树/Arbiter专用解析补救；若原重试后仍失败，停下检查，再单独记录补救范围和参数。不要删除输出后重跑。

原 `report` 仍会显示历史Phase10对照。新增本地**离线分析脚本** `.local/phase17_27b_full/compare_recursive_qwen35.py`，不参与推理：检查Rubric、样本和A/B顺序一致且技术失败为0，复用已有指标及配对函数，生成 `comparison_clean_s5.json`。比较方向固定为“27B Clean S5-v2 → 27B递归”：报告Overall/Macro/Strict、覆盖率、分类结果、纠正/改错、净收益和McNemar检验。

效率同时记录逻辑调用数、API尝试数、输入/输出tokens和实际阶段耗时。递归报告主运行telemetry与额外retry消耗分列；Clean S5-v2从唯一缓存键汇总已持久化调用，包括失败重试和4096补救，排除备份，避免重复计数。请求延迟求和不是并发墙钟时间；Clean S5-v2的 `full.json.wall_seconds` 是最后一次31秒retry，不是总耗时，**不能拿它计算速度倍数**。下列命令逐阶段追加本次实际耗时，恢复执行也保留记录；历史S5若没有完整计时，不编造端到端速度比。

### 运行命令

在仓库根目录、两台服务均为27B且关闭thinking后运行。freeze/audit只做冻结与检查；smoke起才发模型推理请求。配置已生成，不要再执行旧的8B `prepare.py` 来覆盖它。

```powershell
conda activate critiq
$Config = ".local/phase17_27b_full/config_recursive_qwen35_27b.json"
$Output = "output/evolving_structured_rubrics/vlrb_27b_full_recursive_qwen35"
& {
    $ErrorActionPreference = "Stop"
    New-Item -ItemType Directory -Force -Path $Output | Out-Null
    foreach ($Stage in @(
        "vlrb-discovery-v2-freeze",
        "vlrb-discovery-v2-audit",
        "vlrb-discovery-v2-smoke",
        "vlrb-discovery-v2-run",
        "vlrb-discovery-v2-retry",
        "vlrb-discovery-v2-report"
    )) {
        $startedAt = (Get-Date).ToString("o")
        $timer = [System.Diagnostics.Stopwatch]::StartNew()
        python -m experiments.evolving_structured_rubrics.run_rubric_evolution $Stage --config $Config --output-dir $Output
        $stageExit = $LASTEXITCODE
        $timer.Stop()
        @{stage=$Stage; started_at=$startedAt; wall_seconds=$timer.Elapsed.TotalSeconds; exit_code=$stageExit} |
            ConvertTo-Json -Compress | Add-Content -Encoding UTF8 -LiteralPath "$Output/stage_times.jsonl"
        if ($stageExit -ne 0) { throw "Stage failed: $Stage" }
    }
    python .local/phase17_27b_full/compare_recursive_qwen35.py
    if ($LASTEXITCODE -ne 0) { throw "Offline comparison failed; inspect failures and artifacts." }
}
```

本节是待运行协议，不填入预期分数。新配置及分析脚本属于本地产物，公共实现仅增加上述显式跨模型参数开关。
