"""在 RLHF-V 视觉问答/图文回答偏好数据上运行 CritiQ Flow。

这个脚本是基于 `demo.py` 改出来的 RLHF-V 专用入口，主要完成几件事：

1. 读取已经转换成 CritiQ pair 格式的数据：
   {"A": "...", "B": "...", "answer": "A"}
2. RLHF-V 的 A/B 是同一个视觉问题下的两个候选回答，所以脚本会保留顶层
   `question` 和 `image_path`，并交给 MultiModalPairEvaluator 传给 VLM。
3. 定义 RLHF-V 专用的 manager / worker / warmup prompt。
4. 用 90 条 discovery pair 数据优化 criteria；从 heldout validation pair 数据中随机抽 100 条做过程观察。
5. 最后用完整 heldout validation 500 条样本做最终评估。

注意：manager 仍然只处理文本 prompt；真正需要看图的是 worker evaluator。
图片本体由 MultiModalPairEvaluator 读取 `image_path` 并编码成 VLM 可接收的
`image_url` content。
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

from critiq import (
    Agent,
    Criterion,
    MultiModalPairEvaluator,
    Workflow,
    print_score_changes,
)


# 任务名会用在输出目录里，例如 ./output/rlhfv。
TASK_NAME = os.getenv(
    "CRITIQ_TASK_NAME",
    "rlhfv_exp6_mmmanager-reflection-noquestion_dis90_val100_n10_wp-final-heldout500_e10",
)

# manager 最终要维护多少条评价标准。exp3 使用 10 条来测试更细粒度 criteria 是否有帮助。
N_CRITERIA = int(os.getenv("CRITIQ_N_CRITERIA", "10"))

# workflow.optimize 的迭代轮数。每一轮都会在 train_set 上评估并改写 criteria。
NUM_EPOCHS = int(os.getenv("CRITIQ_NUM_EPOCHS", "10"))

# Agent 调用失败或输出 JSON 解析失败时，Evaluator 会最多重试这么多次。
MAX_RETRIES = int(os.getenv("CRITIQ_MAX_RETRIES", "10"))

# 固定随机种子，让 train/valid 切分、warmup 抽样等步骤可复现。
SEED = 42

# RLHF-V 数据所在目录。
DATA_DIR = Path("./data/RLHF-V")

# 由 data/RLHF-V/convert_to_pair.py 生成的 CritiQ pair 数据。
# discovery 90 条只用于 criteria 优化，避免把 heldout 数据混进训练循环。
TRAIN_PAIR_DATA_PATH = DATA_DIR / "discovery_train_90_pair.jsonl"

# heldout 500 条用于观察验证和最终评估：
# - 优化过程中只随机抽 VALID_SIZE 条作为 valid_set，降低每轮 API 成本；
# - 最后再用完整 500 条做一次 final evaluation。
HELDOUT_PAIR_DATA_PATH = DATA_DIR / "heldout_validation_500_pair.jsonl"

# 从 heldout validation 中固定随机抽取多少条作为优化过程中的观察集。
VALID_SIZE = int(os.getenv("CRITIQ_VALID_SIZE", "100"))

# 可选的数据量限制，便于在调试器中单步跟踪完整流程，避免发送数千次请求。
# 设置为 0 时不限制数据量，仍然使用对应数据集中的全部样本。
TRAIN_LIMIT = int(os.getenv("CRITIQ_TRAIN_LIMIT", "0"))
HELDOUT_LIMIT = int(os.getenv("CRITIQ_HELDOUT_LIMIT", "0"))

# 当前任务的所有日志、checkpoint、criteria 都会写到这个目录。
OUTPUT_DIR = Path(os.getenv("CRITIQ_OUTPUT_DIR", str(Path("./output") / TASK_NAME)))

# worker 并发数。默认 20，可以通过环境变量 CRITIQ_MAX_CONCURRENT 调低，避免 API 限流。
MAX_CONCURRENT = int(os.getenv("CRITIQ_MAX_CONCURRENT", "20"))


def load_local_env(path: str = ".env") -> None:
    """读取本地 .env 文件，把 KEY=VALUE 写入 os.environ。

    这里不依赖 python-dotenv，避免增加额外依赖。只处理最简单的 `.env` 行：
    CRITIQ_WORKER_BASE_URL=https://...
    GUIJI_API_KEY=...
    """
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def parse_api_keys(list_var: str, single_var: str | None = None) -> list[str]:
    """从环境变量中解析 API key 列表。

    CritiQ 的 Agent 支持传多个 key 轮换使用，所以这里优先读逗号分隔的
    `list_var`，例如 GUIJI_API_KEYS=key1,key2,key3。
    如果没有配置列表，就回退到单 key 变量 `single_var`。
    """
    raw = (os.getenv(list_var) or "").strip()
    keys = [k.strip() for k in raw.split(",") if k.strip()] if raw else []
    if not keys and single_var:
        single = (os.getenv(single_var) or "").strip()
        if single:
            keys = [single]
    return keys or ["EMPTY_KEY"]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL 文件，返回每一行解析后的 dict。"""
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_rlhfv_pair_data(pair_path: Path) -> list[dict[str, Any]]:
    """加载 RLHF-V pair 数据，并确保顶层带有 question/image_path。

    CritiQ 训练/评估代码只要求每条数据有：
        {"A": "...", "B": "...", "answer": "A" 或 "B"}

    但 RLHF-V 是视觉问答/图文回答偏好任务，worker 需要看到原问题和图片。
    新版转换脚本会直接输出顶层 `question` 和 `image_path`。
    """
    if not pair_path.exists():
        raise FileNotFoundError(
            f"Converted pair data not found: {pair_path}. "
            "Run `python data\\RLHF-V\\convert_to_pair.py` first."
        )

    pairs = load_jsonl(pair_path)
    if not pairs:
        raise ValueError(f"Empty pair data: {pair_path}")

    enriched = []
    for idx, pair in enumerate(pairs):
        answer = pair.get("answer")
        if answer not in ("A", "B"):
            raise ValueError(f"Invalid pair answer: {answer}")
        if not isinstance(pair.get("A"), str) or not isinstance(pair.get("B"), str):
            raise ValueError(f"A/B must be strings at row {idx + 1}")

        question = pair.get("question")
        image_path = pair.get("image_path")

        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Missing question for row {idx + 1}")
        if not isinstance(image_path, str) or not image_path.strip():
            raise ValueError(f"Missing image_path for row {idx + 1}")

        item = {
            "sample_id": pair.get("sample_id"),
            "image_path": image_path,
            "question": question,
            "A": pair["A"],
            "B": pair["B"],
            "answer": answer,
        }
        enriched.append({k: v for k, v in item.items() if v is not None})

    return enriched


# 先加载 .env，再读取下面的模型/API 配置。
load_local_env()
random.seed(SEED)

# manager 使用 OpenAI 兼容接口。这里沿用 demo.py 的变量名，默认从 OPENAI_API_KEY 读。
OPENAI_API_KEYS = [os.getenv("OPENAI_API_KEY", "EMPTY_KEY")]

# worker 是真正逐条比较 A/B 的模型。它会被 evaluator 大量调用，所以通常可以
# 选便宜一点、吞吐高一点的模型；manager 则负责生成和改写 criteria，通常用强模型。
WORKER_ARGS = {
    "model": os.getenv("CRITIQ_WORKER_MODEL", "Qwen/Qwen3-VL-8B-Instruct"),
    "base_url": os.getenv("CRITIQ_WORKER_BASE_URL", "http://10.102.137.255:8000/v1"),
    "api_keys": 'EMPTY',
    # "api_keys": parse_api_keys("GUIJI_API_KEYS", "GUIJI_API_KEY"),
    "request_kwargs": {
        # worker 需要相对稳定地执行判断，温度不宜太高。
        "temperature": 0.5,
    },
}

# manager 负责“提出 criteria”和“根据错误案例改写 criteria”。这个角色更偏生成式，
# 所以温度可以高一些，让它探索更多可能的评价维度。
MANAGER_ARGS = {
    "model": os.getenv("CRITIQ_MANAGER_MODEL", "Qwen/Qwen3-VL-8B-Instruct"),
    "api_keys": 'EMPTY',
    # "api_keys": OPENAI_API_KEYS,
    "base_url": os.getenv("CRITIQ_MANAGER_BASE_URL", "http://10.102.137.255:8000/v1"),
    "request_kwargs": {
        "temperature": 1.0,
    },
}

# manager prompt 的目标：让 manager 生成“人类如何比较两个视觉问答回答质量”的标准。
#
# Exp6 保持 manager_multimodal：warm-up 阶段接收图片、Question 和候选回答；
# 逐错误样例 reflection 接收图片但不显式输入 Question；汇总 suggestions 的
# revise 阶段仍使用文本上下文。生成的 criteria 随后交给多模态 worker 执行判断。
MANAGER_PROMPT = f"""List and describe {N_CRITERIA} criteria for how human annotators compare the quality of two answers to the same visual question.

The task is RLHF-V style preference comparison for visual question answering and image-text responses. The worker model will receive:
- the image,
- the source question,
- candidate answer A,
- candidate answer B.

Focus on human preference factors such as visual grounding, question relevance, factual consistency with the image, avoidance of hallucinated visual details, completeness, specificity, calibrated uncertainty, and clear language."""

# worker prompt 的目标：给定一条 criterion，判断 A/B 哪个回答更符合人类偏好。
#
# MultiModalPairEvaluator 会把下面这些占位符替换成真实内容：
# - {criterion}: criterion 名称
# - {description}: criterion 详细描述
# - {question}: 原始视觉问题
# - {A}: 候选回答 A
# - {B}: 候选回答 B
#
# 末尾的 JSON 输出格式不是这里写的，而是 evaluator 自动拼接
# local_prompts.PAIR_WORKER_PROMPT_POSTFIX，里面要求模型返回 thought 和 answer。
WORKER_PROMPT = """## Instruction
You are judging an RLHF-V visual QA / image-text preference pair under one criterion. You are given the image, the source question, and two candidate answers.

Use the image when the criterion depends on visual evidence. If the criterion is not applicable to this pair, answer None.

## Question
{question}

## Criterion
**{criterion}**: {description}

## Candidate A
{A}

## Candidate B
{B}

Which candidate better matches the criterion and is more likely to align with human preference?"""

# warmup prompt 用在 workflow.get_init_criteria() 的预热阶段。
#
# 这一步不是让 manager 直接输出 criteria，而是先给它少量有标签的 A/B 样本：
# - 如果 answer=A，就用第一个模板，告诉 manager 人类偏好 A；
# - 如果 answer=B，就用第二个模板，告诉 manager 人类偏好 B。
#
# manager 会先解释这些偏好原因。解释完成后，它再根据这些“刚看过的例子”
# 生成更贴近 RLHF-V 的 criteria。
WARMUP_PROMPT = (
    """There are two candidate answers to the same visual question.

The worker will later receive the image, question, and answers. For this warm-up, explain the human preference from the question and answer texts, and focus on what visual evidence would matter.

## Question
{question}

## Candidate A
{A}

## Candidate B
{B}

Human annotators prefer A over B. Explain why A is better in this RLHF-V visual QA / image-text answer preference setting.""",
    """There are two candidate answers to the same visual question.

The worker will later receive the image, question, and answers. For this warm-up, explain the human preference from the question and answer texts, and focus on what visual evidence would matter.

## Question
{question}

## Candidate A
{A}

## Candidate B
{B}

Human annotators prefer B over A. Explain why B is better in this RLHF-V visual QA / image-text answer preference setting.""",
)


def _criterion_vote(vote_counts: dict[str, int]) -> str | None:
    """把单条 criterion 的原始计数转换成 A/B/None 判定。

    evaluator 为了兼容投票聚合，保存的是 {"A": n, "B": n, "U": n}
    这种计数结构。当前 RLHF-V 设置里，每个 sample-criterion pair 通常只会
    调用一次 worker，所以这些计数大多是 0/1。

    `U` 表示 worker 返回 None / unsure / not applicable。它代表这条
    criterion 对当前样本弃权，因此不能算作 A 或 B 的有效票。
    """
    if vote_counts.get("U", 0) > 0:
        return None
    if vote_counts.get("A", 0) > vote_counts.get("B", 0):
        return "A"
    if vote_counts.get("B", 0) > vote_counts.get("A", 0):
        return "B"
    return None


def _ensemble_vote(vote_counts: dict[str, int]) -> str | None:
    """复现 PairEvaluator 默认的最终 A/B 聚合规则。

    默认规则会忽略 None/U 票，只比较 A 和 B 的票数：
    - A 票更多，最终预测 A；
    - B 票更多，最终预测 B；
    - A/B 平票，最终预测 None。

    在 evaluator.eval() 里，最终预测 None 会被计为该样本预测错误。
    """
    if vote_counts["A"] > vote_counts["B"]:
        return "A"
    if vote_counts["B"] > vote_counts["A"]:
        return "B"
    return None


def build_final_eval_diagnostics(
    dataset: list[dict[str, Any]],
    criteria: list[Criterion],
    eval_output: Any,
) -> dict[str, Any]:
    """基于已有 heldout 评估结果构造诊断信息，不额外调用模型。

    这里完全复用 `eval_output.prediction` 里的原始投票结果，只做本地统计。
    目标是解释一个常见现象：为什么某些单条 criterion 的准确率很高，但多
    criterion ensemble 后的最终准确率反而更低。

    返回值包含三类信息：
    - criterion_stats：每条 criterion 在 heldout500 上的拒答数、覆盖率、
      correct/applicable 计数和 accuracy。
    - sample_vote_counts：每个样本有多少条 criterion 投 A、投 B、投 None，
      以及默认投票规则得到的最终答案。
    - ensemble_wrong_with_correct_criteria：最终 ensemble 判错，但至少有
      一条单独 criterion 判对的样本。这类样本最适合用来分析投票冲突、
      低质量 criterion 抢票、或者 tie-breaker 使用不当的问题。
    """
    criterion_names = [criterion.name for criterion in criteria]
    total = len(dataset)

    # 每条 criterion 的统计容器。
    # `applicable` 不包含 U/None，因为 evaluator 计算 per_criterion_acc 时
    # 也只在该 criterion 实际给出 A/B 判断的样本上统计正确率。
    criterion_stats = {
        name: {
            "total": total,
            "refuse": 0,
            "applicable": 0,
            "correct": 0,
            "incorrect": 0,
            "coverage": 0.0,
            "accuracy": 0.0,
        }
        for name in criterion_names
    }
    sample_votes = []
    ensemble_wrong_with_correct_criteria = []

    for index, (data, prediction, final_correct) in enumerate(
        zip(dataset, eval_output.prediction, eval_output.is_correct)
    ):
        gold = data["answer"]

        # 当前样本的投票形态。
        # `None` 记录 criterion 拒答/不适用的次数；默认 A/B majority vote
        # 不会把 None 计入 A 或 B 的票数。
        vote_counts = {"A": 0, "B": 0, "None": 0}

        # 保留 criterion 名称，而不是只保留数量。
        # 这样后续分析可以直接看到：哪些标准在某个样本上救回了正确答案，
        # 哪些标准把最终 ensemble 带偏了。
        correct_criteria = []
        incorrect_criteria = []
        refused_criteria = []

        for name in criterion_names:
            criterion_prediction = prediction[name]
            vote = _criterion_vote(criterion_prediction)
            if vote is None:
                # U/None 表示该 criterion 对这个样本弃权。
                # 它会增加 refuse，并降低 coverage；但不会进入该 criterion
                # 的 correct/incorrect 统计。
                vote_counts["None"] += 1
                criterion_stats[name]["refuse"] += 1
                refused_criteria.append(name)
                continue

            vote_counts[vote] += 1
            criterion_stats[name]["applicable"] += 1
            if vote == gold:
                # 单条 criterion 的 A/B 投票与人工偏好标签一致时，记为该
                # criterion 在这个 applicable 样本上判断正确。
                criterion_stats[name]["correct"] += 1
                correct_criteria.append(name)
            else:
                criterion_stats[name]["incorrect"] += 1
                incorrect_criteria.append(name)

        final_vote = _ensemble_vote(vote_counts)

        # 保存紧凑的样本级投票摘要。
        # 这些信息足够定位平票、低覆盖、以及多条 criterion 一起投错的样本。
        sample_summary = {
            "index": index,
            "sample_id": data.get("sample_id"),
            "gold": gold,
            "final_vote": final_vote,
            "final_correct": bool(final_correct),
            "votes": vote_counts,
        }
        sample_votes.append(sample_summary)

        if not final_correct and correct_criteria:
            # 这类失败样本对改进 aggregation 最有价值：
            # 至少有一条 criterion 判对，但最终 ensemble 仍然错了。
            # 这通常说明其他 criterion 抢票、投票平局、或当前等权投票策略
            # 没有正确利用高质量 criterion。
            ensemble_wrong_with_correct_criteria.append(
                {
                    **sample_summary,
                    "correct_criteria": correct_criteria,
                    "incorrect_criteria": incorrect_criteria,
                    "refused_criteria": refused_criteria,
                }
            )

    for stats in criterion_stats.values():
        applicable = stats["applicable"]
        # coverage 回答的问题是：这条 criterion 在 heldout500 上有多大比例
        # 的样本真正给出了 A/B 判断？
        # accuracy 只在 applicable 样本上计算，与 EvaluationOutput 里的
        # per_criterion_acc 口径保持一致。
        stats["coverage"] = applicable / total if total else 0.0
        stats["accuracy"] = stats["correct"] / applicable if applicable else 0.0

    return {
        "criterion_stats": criterion_stats,
        "sample_vote_counts": sample_votes,
        "ensemble_wrong_with_correct_criteria": (
            ensemble_wrong_with_correct_criteria
        ),
    }


def print_final_eval_diagnostics(
    output_dir: Path,
    diagnostics: dict[str, Any],
    max_error_examples: int = 50,
) -> None:
    """打印可读的 heldout 诊断摘要，并把完整明细保存为 JSON。

    实验日志需要能快速扫读，所以这里只打印摘要表格和少量典型错误样本。
    完整的 500 条样本级诊断会写入 JSON 文件，后续可以用脚本继续分析。
    """
    diagnostics_path = output_dir / "final_heldout_diagnostics.json"
    diagnostics_path.write_text(
        json.dumps(diagnostics, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    criterion_stats = diagnostics["criterion_stats"]
    sample_vote_counts = diagnostics["sample_vote_counts"]
    ensemble_errors = diagnostics["ensemble_wrong_with_correct_criteria"]

    # 先打印完整 JSON 文件路径和总体失败样本数量。
    # 这样 log.txt 不会被 500 条样本明细撑爆，但需要追样本时仍然有入口。
    print("### Final 500-heldout diagnostics summary")
    print(f"Full diagnostics JSON: {diagnostics_path}")
    print(f"Samples with ensemble error but at least one correct criterion: {len(ensemble_errors)}")

    # criterion 诊断表：
    # - acc = correct / applicable；
    # - correct/app 展示原始分子和分母；
    # - refuse 是 U/None 的数量；
    # - coverage = applicable / total。
    # 这个表可以快速发现“高准确率但低覆盖率”的 criterion。
    print("\n### Criterion diagnostics")
    print(
        f"{'criterion':45} {'acc':>8} {'correct/app':>13} "
        f"{'refuse':>8} {'coverage':>9}"
    )
    print("-" * 90)
    for name, stats in sorted(
        criterion_stats.items(),
        key=lambda item: item[1]["accuracy"],
        reverse=True,
    ):
        correct_applicable = f"{stats['correct']}/{stats['applicable']}"
        print(
            f"{name[:45]:45} "
            f"{stats['accuracy']:8.3f} "
            f"{correct_applicable:>13} "
            f"{stats['refuse']:8d} "
            f"{stats['coverage']:9.3f}"
        )

    vote_distribution: dict[tuple[int, int, int, str | None, bool], int] = {}
    for sample in sample_vote_counts:
        votes = sample["votes"]
        # 按投票形态聚合样本。例如 N_CRITERIA=10 时：
        # - A=5, B=5, None=0 表示最终平票；
        # - A=4, B=3, None=3 表示虽然 A 赢，但有 3 条 criterion 弃权，
        #   覆盖率可能正在影响最终决策。
        key = (
            votes["A"],
            votes["B"],
            votes["None"],
            sample["final_vote"],
            sample["final_correct"],
        )
        vote_distribution[key] = vote_distribution.get(key, 0) + 1

    # 这个分布用来判断错误主要来自平票、接近票数，还是明显错误多数。
    print("\n### Sample vote-count distribution")
    print(f"{'A':>3} {'B':>3} {'None':>5} {'final':>7} {'correct':>8} {'count':>7}")
    print("-" * 45)
    for (a_votes, b_votes, none_votes, final_vote, final_correct), count in sorted(
        vote_distribution.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(
            f"{a_votes:3d} {b_votes:3d} {none_votes:5d} "
            f"{str(final_vote):>7} {str(final_correct):>8} {count:7d}"
        )

    # 展示有限数量的可行动错误样本。
    # 完整列表已经写入 JSON；日志里的预览主要用于跑完后快速定性检查。
    print("\n### Ensemble errors with at least one correct criterion")
    print(f"Showing first {min(max_error_examples, len(ensemble_errors))} / {len(ensemble_errors)}")
    for error in ensemble_errors[:max_error_examples]:
        sample_label = error["sample_id"] if error["sample_id"] is not None else error["index"]
        print(
            f"- sample={sample_label} gold={error['gold']} final={error['final_vote']} "
            f"votes={error['votes']} correct_criteria={error['correct_criteria']} "
            f"incorrect_criteria={error['incorrect_criteria']} refused={error['refused_criteria']}"
        )


def save_final_eval_raw_prediction(
    output_dir: Path,
    dataset: list[dict[str, Any]],
    criteria: list[Criterion],
    eval_output: Any,
) -> None:
    """保存 final heldout eval 的原始投票矩阵，方便后续离线分析。

    `final_heldout_diagnostics.json` 更偏向人读的诊断摘要，会把每个样本压缩成
    A/B/None 的总票数；但做 ablation、重算 voting 或排查单条 criterion 行为时，
    需要保留 `eval_output.prediction` 中每个 sample-criterion 的原始 A/B/U 票。

    这个文件不保存 worker 的完整 thought，避免 final eval 后额外产生一个很大的
    JSON。需要看模型解释时仍然可以去 workflow/agent log 里追。
    """
    raw_prediction_path = output_dir / "final_heldout_raw_prediction.json"
    raw_prediction = {
        "criteria": [criterion.to_dict() for criterion in criteria],
        "samples": [
            {
                "index": index,
                "sample_id": data.get("sample_id"),
                "gold": data["answer"],
            }
            for index, data in enumerate(dataset)
        ],
        "prediction": eval_output.prediction,
        "is_correct": eval_output.is_correct,
        "accuracy": eval_output.accuracy,
        "per_criterion_acc": eval_output.per_criterion_acc,
    }
    raw_prediction_path.write_text(
        json.dumps(raw_prediction, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"### Final heldout raw prediction JSON: {raw_prediction_path}")


def ask_agent(criterion: Criterion) -> bool:
    """用 worker 过滤可选知识库里的 criterion。

    原始 demo.py 会读取 `./data/kb.json`，再筛选适合代码质量的 criteria。
    RLHF-V 和代码质量差别很大，所以本脚本默认不使用知识库。
    只有当用户显式设置 CRITIQ_RLHFV_KB 时，才会调用这个函数判断某条 criterion
    是否适合“图片 + question + answers”的 RLHF-V 多模态偏好比较。
    """
    prompt = (
        "# Instruction\n"
        "Is this criterion applicable for multimodal RLHF-V visual QA answer "
        "preference comparison, where the worker can inspect the image, the "
        "source question, and both candidate answers?\n\n"
        f"# Criterion\n{criterion.name}: {criterion.description}\n\n"
        "Reply only 'yes' or 'no'."
    )
    response = Agent(**WORKER_ARGS)(prompt, stream=False)
    return response is not None and "yes" in response.lower()


def main() -> None:
    """串起 RLHF-V 版 CritiQ Flow 的完整流程。"""
    # Agent 会把原始请求/响应记录到这个日志文件，方便之后追踪某个 criterion
    # 为什么生成、为什么被改写、某条样本为什么判断错。
    agent_logfile = os.getenv(
        "WORKFLOW_AGENT_LOGFILE", str(OUTPUT_DIR / "workflow_agent.log")
    )
    os.environ["WORKFLOW_AGENT_LOGFILE"] = agent_logfile

    # critiq.agent 会在模块导入时读取日志路径。
    # 这里同步更新已经导入模块中的变量，确保直接运行和调试运行的日志行为一致。
    import critiq.agent as agent_module

    agent_module.WORKFLOW_AGENT_LOGFILE = agent_logfile
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 读取 pair 数据，并把 question/image_path 保留下来。最后每条样本仍然符合 CritiQ 的格式：
    # {"A": "...", "B": "...", "answer": "A" 或 "B"}
    train_set = load_rlhfv_pair_data(TRAIN_PAIR_DATA_PATH)
    heldout_set = load_rlhfv_pair_data(HELDOUT_PAIR_DATA_PATH)

    if TRAIN_LIMIT > 0:
        train_set = train_set[:TRAIN_LIMIT]
    if HELDOUT_LIMIT > 0:
        heldout_set = heldout_set[:HELDOUT_LIMIT]

    if len(heldout_set) < VALID_SIZE:
        raise ValueError(
            f"Need at least {VALID_SIZE} heldout samples, got {len(heldout_set)}"
        )

    # valid_set 只用于每轮优化后的过程观察，不参与 criteria 改写。
    # 使用局部 RNG 可以保证抽样稳定，同时不影响 workflow 内部可能使用的 random 状态。
    valid_set = random.Random(SEED).sample(heldout_set, VALID_SIZE)

    print(
        "Loaded RLHF-V pairs: "
        f"train={len(train_set)}, valid={len(valid_set)}, heldout={len(heldout_set)}"
    )

    # evaluator 负责把一组 criteria 应用到 valid_set 上，并计算准确率。
    # MultiModalPairEvaluator 会把 image_path 对应图片编码成 image_url 发给 VLM。
    evaluator = MultiModalPairEvaluator(
        WORKER_ARGS,
        dataset=valid_set,
        max_concurrent=MAX_CONCURRENT,
        max_retries=MAX_RETRIES,
        worker_prompt=WORKER_PROMPT,
    )

    final_evaluator = MultiModalPairEvaluator(
        WORKER_ARGS,
        dataset=heldout_set,
        max_concurrent=MAX_CONCURRENT,
        max_retries=MAX_RETRIES,
        worker_prompt=WORKER_PROMPT,
    )

    # Workflow 是 CritiQ 的调度器。它不直接定义 RLHF-V 任务，只接收 manager/worker
    # 参数和 prompt。通过 load_state_dict 注入这些配置后，同一套优化逻辑就能复用。
    workflow = Workflow()
    workflow.load_state_dict(
        {
            "manager_args": MANAGER_ARGS,
            "worker_args": WORKER_ARGS,
            "worker_max_concurrent": MAX_CONCURRENT,
            "n_criteria": N_CRITERIA,
            "manager_prompt": MANAGER_PROMPT,
            "worker_prompt": WORKER_PROMPT,
            "evaluator_type": "multimodal_pair",
            "manager_multimodal": True,
            "manager_reflection_include_question": False,
            "evaluator_kwargs": {
                "image_field": "image_path",
                "question_field": "question",
                "encode_local_image": True,
            },
        }
    )

    # 默认不接知识库。原因是通用数据质量 KB 或代码质量 KB 里可能有很多不适合 RLHF-V
    # 的标准，直接注入会误导 manager。需要时可以手动设置：
    #   CRITIQ_RLHFV_KB=./data/kb.json
    # 然后脚本会先用 ask_agent 做一层适用性过滤。
    kb = []
    kb_path = os.getenv("CRITIQ_RLHFV_KB")
    if kb_path:
        from critiq import load_criteria_from_json

        raw_kb = load_criteria_from_json(kb_path)
        kb = [criterion for criterion in raw_kb if ask_agent(criterion)]
        print(f"Retrieved {len(kb)} RLHF-V applicable criteria from {kb_path}")

    # 生成初始 criteria：
    # 1. 如果 kb 非空，先从知识库里评估并取高分 criteria；
    # 2. 如果数量不足 N_CRITERIA，则 manager 先看 n_shot=5 个 warmup 样本；
    # 3. manager 根据 MANAGER_PROMPT 输出剩余的新 criteria。
    workflow.get_init_criteria(
        train_set,
        prompt_template=WARMUP_PROMPT,
        knowledge_base=kb,
        n_shot=5,
        max_retrived=None,
    )

    # warmup 后先在 500-final-heldout_set 上观察初始 criteria 的表现,最后看提高多少。
    # 这里 update_score=False，
    eval_output = final_evaluator.eval(workflow.current_criteria, update_score=False)
    print("### After warm up (500-heldout evaluation):", eval_output.accuracy, eval_output.is_correct)
    print(
        "### After warm up (500-heldout per-criterion accuracy):",
        json.dumps(eval_output.per_criterion_acc, indent=4, ensure_ascii=False),
    )

    # 保存初始化后的状态，文件名类似 ./output/rlhfv/epoch_init.json。
    workflow.save(str(OUTPUT_DIR), "init", None)

    # 进入主优化循环。每一轮大致做：
    # 1. 在 train_set 上评估每条 criterion；
    # 2. score >= 0.8 的认为 good，保留；
    # 3. 0.6 < score < 0.8 的认为 mid，让 manager 看错例后改写；
    # 4. score <= 0.6 的认为 low，移除并生成新 criterion；
    # 5. 可选地在 valid_set 上观察一轮。
    workflow.optimize(
        train_set,
        valid_set,
        output_dir=str(OUTPUT_DIR),
        num_epochs=NUM_EPOCHS,
        threshold=(0.6, 0.8),
        max_retries=MAX_RETRIES,
    )

    # 打印 init / 每个 epoch / final 的 criterion 分数变化，方便看哪些标准稳定有效。
    print_score_changes(
        str(OUTPUT_DIR),
        [
            "epoch_init.json",
            *[f"epoch_{i}.json" for i in range(NUM_EPOCHS)],
            "epoch_final.json",
        ],
    )

    # 最终评估只使用历史上 score >= 0.6 的 best criteria。
    # 这里不再复用只含 100 条样本的 valid evaluator，而是在完整 heldout 500 条上做 final evaluation。
    # 如果这里为空，说明阈值太严或训练样本太少，可以临时调低 get_best_criteria 的阈值。
    best_criteria = workflow.get_best_criteria(0.6)
    print(
        "### Final eval criteria:",
        [criterion.name for criterion in best_criteria],
    )
    eval_output = final_evaluator.eval(best_criteria, update_score=False)
    print("### Final 500-heldout evaluation:", eval_output.accuracy, eval_output.is_correct)
    print(
        "### Final 500-heldout per-criterion accuracy:",
        json.dumps(eval_output.per_criterion_acc, indent=4, ensure_ascii=False),
    )
    save_final_eval_raw_prediction(OUTPUT_DIR, heldout_set, best_criteria, eval_output)
    final_diagnostics = build_final_eval_diagnostics(
        heldout_set, best_criteria, eval_output
    )
    print_final_eval_diagnostics(OUTPUT_DIR, final_diagnostics)


if __name__ == "__main__":
    main()
