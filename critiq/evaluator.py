"""成对比较与二分类数据质量判断的评估逻辑。

这个文件是 CritiQ Flow 里“用 criterion 做打分”的核心位置。

整体思路可以理解成三层：

1. Workflow 负责维护一批自然语言 criterion。
   例如：“回答是否忠实于图片内容”、“代码是否具有可读性”。

2. Evaluator 负责把这些 criterion 应用到带标签数据上。
   对 pair 数据来说，一条样本形如：
       {"A": "...", "B": "...", "answer": "A"}
   evaluator 会问 worker：“在某个 criterion 下，A/B 哪个更好？”

3. Evaluator 再把 worker 的判断和人工标签 `answer` 对比，得到：
   - 整体投票准确率：多条 criterion 聚合后，最终 A/B 判断是否正确；
   - criterion 级别准确率：单独使用某条 criterion 时，它是否能逼近人工偏好。

Workflow 后续会根据 criterion 级别准确率决定哪些 criterion 保留、改写或替换。
"""

import base64
import json
import mimetypes
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from tqdm import tqdm

from .agent import Agent, AgentCallMetrics
from .i18n import local_prompts
from .router import RouterManager, RoutingDecision
from .structured import (
    FinalPreference,
    StructuredCriterionSnapshot,
    StructuredEvaluationOutput,
    StructuredNodeOutput,
    StructuredOutputParseError,
    StructuredPredictionOutput,
    StructuredWorkerRequestSpec,
    Vote,
    aggregate_selected_roots,
    make_parse_failure_judgement,
    parse_structured_worker_response,
    structured_worker_prompt_sha256,
    structured_input_fingerprint,
)
from .structured_prompts import (
    STRUCTURED_MULTIMODAL_WORKER_PROMPT,
    STRUCTURED_WORKER_PROMPT_POSTFIX,
)
from .structured.telemetry import ModelCallMetrics, TokenPricing
from .utils import (
    USE_TQDM,
    Criterion,
    PairData,
    ZeroOneData,
    is_pair_dataset,
    is_zero_one_dataset,
    parse_json,
    print_debug,
    reverse_ab,
)

PredictionOutput = list[
    dict[str, dict[Literal["A", "B", "U"] | Literal[0, 1], int]]
]
PairAnswer = Literal["A", "B", "Tie", None]
ZeroOneAnswer = Literal[0, 1, None]
# PredictionOutput 的结构是：
# prediction[data_idx][criterion_name] = {"A": 1, "B": 0, "U": 0}
#
# 其中 data_idx 是样本序号，criterion_name 是某条 criterion 的名字。
# A/B 表示 worker 选择了哪个候选；U 表示 None/Unsure/不适用。
# ZeroOneEvaluator 会使用 0/1 键，但整体容器结构相同。


@dataclass
class PredictionOutputWithAnswer:
    """每个 criterion 的原始投票结果，以及最终聚合后的答案。

    `prediction` 保存所有 worker/criterion 的原始票数。
    `answer` 是 voting_fn 聚合后的最终预测，例如 ["A", None, "B"]。
    `thoughts` 保存模型返回的解释，主要用于 workflow 反思错例。
    """

    prediction: PredictionOutput
    answer: list[PairAnswer | ZeroOneAnswer]
    thoughts: list[dict[str, str]] | None = None
    routing: list[dict[str, Any]] | None = None


@dataclass
class EvaluationOutput:
    """`eval()` 返回的完整评估结果。

    - prediction: 原始投票矩阵。
    - is_correct: 每条样本最终聚合判断是否命中人工标签。
    - per_criterion_acc: 每条 criterion 自己的准确率。
    - accuracy: 多条 criterion 聚合后的整体准确率。
    - thoughts: worker 的分析文本，供 workflow 生成 critique。
    """

    prediction: PredictionOutput
    is_correct: list[bool]  # True if the voting result is correct
    per_criterion_acc: dict[str, float]  # accuracy for each criterion
    accuracy: float
    thoughts: list[dict[str, str]] | None = None
    routing: list[dict[str, Any]] | None = None

    def __str__(self):
        """以便于命令行查看的格式打印评估细节。"""
        criteria = list(self.per_criterion_acc.keys())
        output_json = {}
        for c in criteria:
            n_refuse = sum([p[c]["U"] for p in self.prediction])
            output_json[c] = {
                "Accuracy": self.per_criterion_acc[c],
                "Refuse to Respond": f"{n_refuse / len(self.prediction)} ({n_refuse})",
            }
        return f"Accuracy: {self.accuracy}\nCorrect: {self.is_correct}\n{json.dumps(output_json, ensure_ascii=False, indent=4)}"


@dataclass(frozen=True)
class CriterionPerformance:
    """Historical performance statistics used for static criterion weighting."""

    train_acc: float = 1.0
    coverage: float = 1.0


def _unit_float(value: object, default: float = 1.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return min(1.0, max(0.0, result))


def static_weight(
    stat: CriterionPerformance, alpha: float = 0.7, beta: float = 0.3
) -> float:
    """Combine historical accuracy and coverage into one static prior weight."""
    return alpha * stat.train_acc + beta * stat.coverage


def _performance_from_mapping(data: Mapping[str, Any]) -> CriterionPerformance:
    train_acc = data.get(
        "train_acc",
        data.get("accuracy", data.get("acc", data.get("score", 1.0))),
    )
    return CriterionPerformance(
        train_acc=_unit_float(train_acc, 1.0),
        coverage=_unit_float(data.get("coverage", 1.0), 1.0),
    )


def _performance_from_value(value: object) -> CriterionPerformance:
    if isinstance(value, CriterionPerformance):
        return value
    if isinstance(value, Mapping):
        return _performance_from_mapping(value)
    return CriterionPerformance(train_acc=_unit_float(value, 1.0), coverage=1.0)


def _load_json_if_path(value: object) -> object:
    if isinstance(value, (str, Path)):
        with Path(value).open("r", encoding="utf-8") as f:
            return json.load(f)
    return value


def load_criterion_performance(
    criterion_stats: Mapping[str, Any] | str | Path | None,
) -> dict[str, CriterionPerformance]:
    """Load criterion stats from explicit maps, diagnostics, caches, or checkpoints."""
    loaded = _load_json_if_path(criterion_stats)
    if loaded is None:
        return {}
    if not isinstance(loaded, Mapping):
        raise ValueError("criterion_stats must be a mapping or a JSON path")

    for stats_key in ("criterion_stats", "per_criterion_stats"):
        nested_stats = loaded.get(stats_key)
        if isinstance(nested_stats, Mapping):
            return {
                str(name): _performance_from_value(value)
                for name, value in nested_stats.items()
            }

    per_criterion_acc = loaded.get("per_criterion_acc")
    if isinstance(per_criterion_acc, Mapping):
        return {
            str(name): CriterionPerformance(
                train_acc=_unit_float(value, 1.0), coverage=1.0
            )
            for name, value in per_criterion_acc.items()
        }

    checkpoint_stats: dict[str, CriterionPerformance] = {}
    for criteria_key in ("all_criteria", "current_criteria"):
        raw_criteria = loaded.get(criteria_key)
        if not isinstance(raw_criteria, (list, tuple)):
            continue
        for raw_criterion in raw_criteria:
            if not isinstance(raw_criterion, Mapping):
                continue
            name = raw_criterion.get("name")
            if not name:
                continue
            checkpoint_stats[str(name)] = CriterionPerformance(
                train_acc=_unit_float(raw_criterion.get("score", 1.0), 1.0),
                coverage=_unit_float(raw_criterion.get("coverage", 1.0), 1.0),
            )
    if checkpoint_stats:
        return checkpoint_stats

    return {
        str(name): _performance_from_value(value)
        for name, value in loaded.items()
        if isinstance(name, str)
    }


class Evaluator:
    """不同数据格式共享的抽象 evaluator 接口。

    子类至少要实现两个入口：
    - pred(): 只跑预测，返回原始投票和聚合答案；
    - eval(): 在 pred() 基础上对比人工标签，计算准确率。
    """

    @abstractmethod
    def pred(
        self, criteria: Sequence[Criterion | dict[Literal["name", "description"], str]]
    ) -> PredictionOutputWithAnswer: ...

    @abstractmethod
    def eval(
        self,
        criteria: Sequence[Criterion | dict[Literal["name", "description"], str]],
        update_score=False,
    ) -> EvaluationOutput: ...


class BaselinePairEvaluator(Evaluator):
    """基线版本：一次 prompt 同时评估全部 criterion，而不是逐个评估。"""

    worker_prompt_postfix = local_prompts.BASELINE_WORKER_PROMPT_POSTFIX

    def __init__(
        self,
        worker_args: dict,
        dataset: Sequence[PairData],
        max_concurrent: int = 1,
        max_retries: int = 3,
        worker_prompt: str | None = None,
    ) -> None:
        self.worker_args = worker_args
        self.dataset = dataset
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries

        self.worker_prompt = worker_prompt or local_prompts.BASELINE_WORKER_PROMPT
        assert all([k in self.worker_prompt for k in ["{C}", "{A}", "{B}"]])

    def voting_fn(self, prediction: PredictionOutput) -> list[Literal["A", "B", None]]:
        """将单个样本上的所有 worker 投票聚合成一个 A/B/None 标签。"""
        result = []
        for d in prediction:
            stat = {"A": 0, "B": 0}
            for c in d.values():
                stat["A"] += c["A"]
                stat["B"] += c["B"]
            if stat["A"] > stat["B"]:
                result.append("A")
            elif stat["B"] > stat["A"]:
                result.append("B")
            else:
                result.append(None)
        return result

    def _make_prompt(self, data, criteria):
        """将整组 criterion 和一条 A/B 样本序列化进 prompt。"""
        c = "\n".join([f"- {c.name}: {c.description}" for c in criteria])
        prompt = (
            self.worker_prompt.replace("{C}", c)
            .replace("{A}", data["A"])
            .replace("{B}", data["B"])
        )
        prompt += self.worker_prompt_postfix
        return prompt

    def _pred_one_openai(
        self, data: PairData, criteria: Criterion, ttl: int
    ) -> tuple[Literal["A", "B", "U"] | None, str | None]:
        """执行一次 baseline worker 调用，并解析结构化 JSON 返回。"""
        worker = Agent(**self.worker_args)
        prompt = self._make_prompt(data, criteria)
        response = worker(prompt, stream=False)
        thought = None
        try:
            response = parse_json(response)
            answer = response["answer"].strip()[0].upper()
            thought = response["thought"].strip()
        except Exception as e:  # pylint: disable=W0718:broad-exception-caught
            if ttl > 0:
                print_debug(
                    f"Failed to parse worker response, retrying {ttl=}", response, e
                )
                return self._pred_one_openai(data, criteria, ttl - 1)
            else:
                print_debug("Failed to parse worker response", response, e)
                return None, thought
        if answer == "N":  # None
            answer = "U"
        if answer not in ("A", "B", "U"):
            answer = None
        return answer, thought

    def _pred_openai_one_worker(
        self, args: tuple[PairData, Sequence[Criterion]]
    ) -> tuple[dict[dict[Literal["A", "B", "U"], int]], dict[str, str]]:
        """用 baseline 的多 criterion prompt 评估一条数据对。"""
        data, criteria = args
        prediction = {"all": {"A": 0, "B": 0, "U": 0}}
        thoughts = {"all": None}

        one_pred, thought = self._pred_one_openai(data, criteria, self.max_retries)
        if one_pred in ("A", "B", "U"):
            prediction["all"][one_pred] += 1
            thoughts["all"] = thought

        return prediction, thoughts

    def pred_openai(self, criteria: Sequence[Criterion]):
        """在数据集维度并行执行 baseline 推理。"""
        with ThreadPoolExecutor(max_workers=self.max_concurrent) as t:
            prediction = list(
                tqdm(
                    t.map(
                        self._pred_openai_one_worker,
                        [(data, criteria) for data in self.dataset],
                    ),
                    total=len(self.dataset),
                    dynamic_ncols=True,
                    disable=not USE_TQDM,
                )
            )
            thoughts = [p[1] for p in prediction]
            prediction = [p[0] for p in prediction]
        return prediction, thoughts

    def pred(
        self,
        criteria: Sequence[Criterion | dict[Literal["name", "description"], str]],
        **voting_fn_kwargs,
    ) -> PredictionOutputWithAnswer:
        """规范化输入 criterion，并返回预测结果与最终答案。"""
        for i, c in enumerate(criteria):
            if isinstance(c, dict):
                criteria[i] = Criterion.from_dict(c)
        prediction, thoughts = self.pred_openai(criteria)

        return PredictionOutputWithAnswer(
            prediction=prediction,
            answer=self.voting_fn(prediction, **voting_fn_kwargs),
            thoughts=thoughts,
        )

    def eval(
        self,
        criteria: Sequence[Criterion | dict[Literal["name", "description"], str]],
        **voting_fn_kwargs,
    ) -> EvaluationOutput:
        """在全部 criterion 打包评估时，计算数据集准确率。"""
        for i, c in enumerate(criteria):
            if isinstance(c, dict):
                criteria[i] = Criterion.from_dict(c)

        prediction_with_answer = self.pred(criteria, **voting_fn_kwargs)
        prediciton = prediction_with_answer.prediction
        answer = prediction_with_answer.answer

        is_correct = []
        for d, a in zip(self.dataset, answer):
            is_correct.append(a is not None and a == d["answer"])

        per_criterion_acc = {"all": 0}  # All criteria are evaluated together

        n_correct = 0
        n_total = len(self.dataset)
        for d, p in zip(self.dataset, prediciton):
            if p["all"][d["answer"]] > p["all"][reverse_ab(d["answer"])]:
                n_correct += 1
        per_criterion_acc["all"] = n_correct / n_total

        return EvaluationOutput(
            prediction=prediciton,
            is_correct=is_correct,
            per_criterion_acc=per_criterion_acc,
            accuracy=len(list(filter(None, is_correct))) / len(is_correct),
            thoughts=prediction_with_answer.thoughts,
        )


class PairEvaluator(Evaluator):
    """用于成对偏好数据集的 evaluator。

    每个 criterion 都会被独立询问一次，因此可以统计 criterion 级别的准确率，
    而这正是 workflow 优化时依赖的核心信号。

    典型输入数据：
        {"A": "候选文本 1", "B": "候选文本 2", "answer": "A"}

    典型调用链：
        eval(criteria)
          -> pred(criteria)
             -> pred_openai(criteria)
                -> _pred_one_openai(data, criterion)
                   -> _make_prompt(data, criterion)

    `_make_prompt` 只负责拼 prompt；`_pred_one_openai` 只负责单次 worker
    调用和 JSON 解析；`pred_openai` 负责把所有样本和所有 criterion 的组合并发
    跑完；`eval` 负责把预测和人工标签对比成分数。
    """

    worker_prompt_postfix = local_prompts.PAIR_WORKER_PROMPT_POSTFIX

    def __init__(
        self,
        worker_args: dict,
        dataset: Sequence[PairData],
        max_concurrent: int = 1,
        max_retries: int = 3,
        worker_prompt: str | None = None,
        max_data_chars: int | None = None,
    ) -> None:
        # worker_args 会原样传给 Agent，例如 model/base_url/api_keys/temperature。
        self.worker_args = worker_args

        # dataset 是已经带人工偏好标签的数据。这里不复制，调用者需要避免后续乱改。
        self.dataset = dataset

        # 同时向 worker 模型发起的最大请求数。
        self.max_concurrent = max_concurrent

        # 单个 worker 输出无法解析时的重试次数。
        self.max_retries = max_retries

        # 可选截断 A/B，防止超长样本撑爆上下文窗口。
        self.max_data_chars = max_data_chars

        # worker_prompt 可以由任务脚本传入；不传则使用 i18n 默认 pair prompt。
        self.worker_prompt = worker_prompt or local_prompts.PAIR_WORKER_PROMPT

        # PairEvaluator 至少需要这四个占位符，否则没法把 criterion 和 A/B 填进去。
        assert all(
            [
                k in self.worker_prompt
                for k in ["{criterion}", "{description}", "{A}", "{B}"]
            ]
        )

    def voting_fn(
        self, prediction: PredictionOutput, threshold: int = 0
    ) -> list[Literal["A", "B", None]]:
        """将各个 criterion 的 A/B 投票聚合成最终成对偏好结果。

        对一条样本来说，每个 criterion 会贡献一票：
        - 选 A：stat["A"] += 1
        - 选 B：stat["B"] += 1
        - 返回 None/Unsure/不适用：记为 U，不参与 A/B 总票差

        threshold 控制需要领先多少票才算最终选择某一边。默认 threshold=0，
        也就是 A 多一票选 A，B 多一票选 B，平票返回 None。
        """
        result = []
        for d in prediction:
            stat = {"A": 0, "B": 0}
            for c in d.values():
                stat["A"] += c["A"]
                stat["B"] += c["B"]
            if stat["A"] - stat["B"] > threshold:
                result.append("A")
            elif stat["B"] - stat["A"] > threshold:
                result.append("B")
            else:
                result.append(None)
        return result

    def _make_prompt(self, data, criterion):
        """为一条 A/B 样本构造单 criterion prompt。

        文本版 PairEvaluator 只读取 data["A"] 和 data["B"]。如果样本还有
        question、image_path 等顶层字段，它们不会自动进入 prompt。
        需要图片或顶层 question 时，请使用 MultiModalPairEvaluator。
        """
        a = data["A"][: self.max_data_chars] if self.max_data_chars else data["A"]
        b = data["B"][: self.max_data_chars] if self.max_data_chars else data["B"]
        prompt = (
            self.worker_prompt.replace("{criterion}", criterion.name)
            .replace("{description}", criterion.description)
            .replace("{A}", a)
            .replace("{B}", b)
        )
        prompt += self.worker_prompt_postfix
        return prompt

    def _pred_one_openai(
        self, data: PairData, criterion: Criterion, ttl: int
    ) -> tuple[Literal["A", "B", "U"] | None, str | None]:
        """让一个 worker 基于一个 criterion 判断一条样本对。

        这是最小的模型调用单元：一条样本 + 一条 criterion -> 一次 worker 请求。
        返回值里的 answer 有四种语义：
        - "A": worker 认为 A 更符合 criterion；
        - "B": worker 认为 B 更符合 criterion；
        - "U": worker 明确返回 None/不确定/不适用；
        - None: worker 输出无法解析，或重试后仍失败。
        """
        worker = Agent(**self.worker_args)
        prompt = self._make_prompt(data, criterion)
        response = worker(prompt, stream=False)
        thought = None
        try:
            response = parse_json(response)
            answer = response["answer"].strip()[0].upper()
            thought = response["thought"].strip()
        except Exception as e:  # pylint: disable=W0718:broad-exception-caught
            if ttl > 0:
                print_debug(
                    f"Failed to parse worker response, retrying {ttl=}", response, e
                )
                return self._pred_one_openai(data, criterion, ttl - 1)
            else:
                print_debug("Failed to parse worker response", response, e)
                return None, thought
        if answer == "N":  # None
            answer = "U"
        if answer not in ("A", "B", "U"):
            answer = None
        return answer, thought

    def pred_openai(self, criteria: Sequence[Criterion]):
        """执行完整的笛卡尔积评估：每条样本对都要过每个 criterion。

        如果有 30 条样本、5 条 criterion，这里会发起 30 * 5 = 150 次 worker
        调用。所有 future 先扁平地放进列表，完成后再按“样本 -> criterion”的
        顺序组装回 PredictionOutput。
        """
        with ThreadPoolExecutor(max_workers=self.max_concurrent) as t:
            futures = []
            for data in self.dataset:
                for criterion in criteria:
                    # 提交顺序很重要：后面会按同样的 data/criterion 嵌套顺序取回结果。
                    future = t.submit(
                        self._pred_one_openai, data, criterion, self.max_retries
                    )
                    futures.append(future)

            # as_completed 本身会按完成顺序返回。
            # as_completed 只用于进度条等待所有任务完成；真正读取结果时仍按 futures，futures 就是原始顺序的扁平列表。
            # 原始顺序读取，从而保持结果能准确还原到对应样本和 criterion。
            for _ in tqdm(
                as_completed(futures),
                total=len(futures),
                dynamic_ncols=True,
                disable=not USE_TQDM,
            ):
                pass

            prediction = []
            thoughts = []
            results = (future.result() for future in futures)
            for data in self.dataset:
                # 将扁平的 future 结果重新组装成按样本索引的字典结构。
                _prediction = {
                    criterion.name: {"A": 0, "B": 0, "U": 0} for criterion in criteria
                }
                _thoughts = {criterion.name: None for criterion in criteria}
                for criterion in criteria:
                    one_pred, thought = next(results)
                    if one_pred in ("A", "B", "U"):
                        # 每个 data/criterion 只有一次 worker 调用，所以这里通常是 0/1 票。
                        # 保持计数结构，是为了和 baseline/投票聚合逻辑统一。
                        _prediction[criterion.name][one_pred] += 1
                        _thoughts[criterion.name] = thought
                prediction.append(_prediction)
                thoughts.append(_thoughts)

        return prediction, thoughts

    def pred(
        self,
        criteria: Sequence[Criterion | dict[Literal["name", "description"], str]],
        **voting_fn_kwargs,
    ) -> PredictionOutputWithAnswer:
        """提供给 workflow 和 CLI 工具使用的公共预测入口。"""
        for i, c in enumerate(criteria):
            if isinstance(c, dict):
                criteria[i] = Criterion.from_dict(c)
        prediction, thoughts = self.pred_openai(criteria)
        return PredictionOutputWithAnswer(
            prediction=prediction,
            answer=self.voting_fn(prediction, **voting_fn_kwargs),
            thoughts=thoughts,
        )

    def eval(
        self,
        criteria: Sequence[Criterion | dict[Literal["name", "description"], str]],
        update_score=False,
        **voting_fn_kwargs,
    ) -> EvaluationOutput:
        """同时评估最终投票准确率和 criterion 级别有效性。

        这里会计算两种分数：

        1. `accuracy`
           多条 criterion 经过 voting_fn 聚合后的最终预测，与 data["answer"]
           对比得到的整体准确率。

        2. `per_criterion_acc`
           每条 criterion 单独判断时的准确率。Workflow 优化 criterion 时主要看
           这个分数：高分保留，中等分数改写，低分替换。
        """
        for i, c in enumerate(criteria):
            if isinstance(c, dict):
                criteria[i] = Criterion.from_dict(c)

        prediction_with_answer = self.pred(criteria, **voting_fn_kwargs)
        prediciton = prediction_with_answer.prediction
        answer = prediction_with_answer.answer

        is_correct = []
        for d, a in zip(self.dataset, answer):
            # a 为 None 表示最终投票无法决定 A/B，视为该样本预测不正确。
            is_correct.append(a is not None and a == d["answer"])

        per_criterion_acc = {criterion.name: 0 for criterion in criteria}
        for criterion in criteria:
            n_correct = 0
            n_total = 0
            for d, p in zip(self.dataset, prediciton):
                # "U" 表示该 criterion 不适用，或模型选择弃权。
                # 这类情况不会被计入 criterion 自身准确率的统计。
                if p[criterion.name]["U"] > 0:
                    # 只在 worker 确信 criterion 适用时，才统计该 criterion 的准确率。
                    # 这样可以避免“好 criterion 在不适用样本上主动弃权”被错误惩罚。
                    continue
                n_total += 1
                if (
                    p[criterion.name][d["answer"]]
                    > p[criterion.name][reverse_ab(d["answer"])]
                ):
                    n_correct += 1
            per_criterion_acc[criterion.name] = (
                0 if n_correct == 0 else n_correct / n_total
            )
            if update_score:
                criterion.score = per_criterion_acc[criterion.name]

        return EvaluationOutput(
            prediction=prediciton,
            is_correct=is_correct,
            per_criterion_acc=per_criterion_acc,
            accuracy=len(list(filter(None, is_correct))) / len(is_correct),
            thoughts=prediction_with_answer.thoughts,
        )


class MultiModalPairEvaluator(PairEvaluator):
    """用于视觉问答/图文回答偏好的多模态 PairEvaluator。

    这个类继承 PairEvaluator，复用它的投票、并发、pred/eval 逻辑，只改两件事：

    1. prompt 里除了 {criterion}/{description}/{A}/{B}，还可以使用：
       - {question}: 原始视觉问题。

    2. 当样本带有 image_path 时，worker 收到的 user content 不再是纯字符串，
       而是 OpenAI 兼容的多模态内容：
           [
               {"type": "text", "text": "...prompt..."},
               {"type": "image_url", "image_url": {"url": "data:image/..."}}
           ]

    之所以不用大改 Agent，是因为 Agent.__call__ 会把 prompt 原样放进
    {"role": "user", "content": prompt}。OpenAI chat completion API 的 content
    既可以是字符串，也可以是上面的多模态 list。

    期望输入样本至少仍满足 CritiQ pair 格式：
        {"A": "...", "B": "...", "answer": "A"}

    多模态任务可额外包含：
        {"question": "...", "image_path": "..."}
    """

    def __init__(
        self,
        worker_args: dict,
        dataset: Sequence[PairData],
        max_concurrent: int = 1,
        max_retries: int = 3,
        worker_prompt: str | None = None,
        max_data_chars: int | None = None,
        image_field: str = "image_path",
        question_field: str = "question",
        encode_local_image: bool = True,
    ) -> None:
        # 验证数据格式：每条样本必须有非空的 image_field 和 question_field。
        for idx, data in enumerate(dataset):
            image_value = data.get(image_field)
            question_value = data.get(question_field)
            if not isinstance(image_value, str) or not image_value.strip():
                raise ValueError(
                    f"multimodal_pair requires a non-empty {image_field!r} at row {idx + 1}"
                )
            if not isinstance(question_value, str) or not question_value.strip():
                raise ValueError(
                    f"multimodal_pair requires a non-empty {question_field!r} at row {idx + 1}"
                )

        # 先初始化父类。父类会检查 worker_prompt 至少包含
        # {criterion}/{description}/{A}/{B} 四个占位符。
        super().__init__(
            worker_args=worker_args,
            dataset=dataset,
            max_concurrent=max_concurrent,
            max_retries=max_retries,
            worker_prompt=worker_prompt,
            max_data_chars=max_data_chars,
        )

        # image_field/question_field 允许适配不同数据集字段名，例如 image 或 images。
        self.image_field = image_field
        self.question_field = question_field

        # True: 把本地图片读成 data URL，适合 OpenAI 兼容 VLM API。
        # False: 直接把 image_path 当 URL 传入，适合已经是 http(s) 图片 URL 的数据。
        self.encode_local_image = encode_local_image

    @staticmethod
    def _image_path_to_data_url(image_path: str) -> str:
        """读取本地图片，并编码成 VLM API 可接收的 data URL。

        OpenAI 兼容多模态接口通常支持：
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}

        这里根据文件后缀猜 MIME 类型。猜不到时默认使用 image/jpeg。
        如果图片路径不存在，直接抛错，让调用方尽早发现数据路径问题。
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        mime_type, _ = mimetypes.guess_type(str(path))
        if mime_type is None:
            mime_type = "image/jpeg"

        with path.open("rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("ascii")
        return f"data:{mime_type};base64,{image_b64}"

    def _make_prompt(self, data, criterion):
        """构造多模态 worker 的文本部分 prompt。

        与 PairEvaluator._make_prompt 相比，这里多替换了：
        - {question}

        图片本体不会拼进文本；真正的图片会在 _make_user_content() 中作为
        image_url content 传给模型。image_path 只用于读取/发送图片，不作为文本
        提示词暴露给 worker。
        """
        a = data["A"][: self.max_data_chars] if self.max_data_chars else data["A"]
        b = data["B"][: self.max_data_chars] if self.max_data_chars else data["B"]
        question = data.get(self.question_field, "")

        # 避免 question 为 None、Path 或其他类型时 replace 报错。
        question = "" if question is None else str(question)

        prompt = (
            self.worker_prompt.replace("{criterion}", criterion.name)
            .replace("{description}", criterion.description)
            .replace("{A}", a)
            .replace("{B}", b)
            .replace("{question}", question)
            .replace("{image_path}", "")
        )
        prompt += self.worker_prompt_postfix
        # postfix 会要求 worker 按 JSON 返回 thought 和 answer，供 parse_json 解析。
        return prompt

    def _make_user_content(self, data, criterion):
        """构造传给 Agent 的 user content。

        返回值有两种形态：

        1. 样本没有 image_path：
           返回纯文本 prompt，行为和 PairEvaluator 一致。

        2. 样本有 image_path：
           返回 OpenAI 兼容的多模态 list。Agent 会把这个 list 原样作为
           message["content"] 发送给后端 VLM。
        """
        prompt = self._make_prompt(data, criterion)
        image_path = data.get(self.image_field)
        image_path = "" if image_path is None else str(image_path)

        if not image_path:
            # 没有图片时退化成纯文本评估，避免强制要求所有样本都有图片。
            return prompt

        # 本地文件需要转成 data URL；远程 URL 可以通过 encode_local_image=False 直传。
        image_url = (
            self._image_path_to_data_url(image_path)
            if self.encode_local_image
            else str(image_path)
        )
        return [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]

    def _pred_one_openai(
        self, data: PairData, criterion: Criterion, ttl: int
    ) -> tuple[Literal["A", "B", "U"] | None, str | None]:
        """让一个多模态 worker 基于图片、问题、A/B 和 criterion 做判断。

        解析逻辑和 PairEvaluator 保持一致：仍然要求 worker 返回 JSON，并从
        response["answer"] 中读取 A/B/None。这样 MultiModalPairEvaluator 可以无缝
        复用父类 pred_openai/pred/eval 的统计逻辑。
        """
        worker = Agent(**self.worker_args)
        prompt = self._make_user_content(data, criterion)
        response = worker(prompt, stream=False)
        thought = None
        try:
            # worker prompt 的 postfix 要求返回 JSON。这里从模型输出中抽取 JSON，
            # 再读取 answer 和 thought。
            response = parse_json(response)
            answer = response["answer"].strip()[0].upper()
            thought = response["thought"].strip()
        except Exception as e:  # pylint: disable=W0718:broad-exception-caught
            # JSON 解析失败时递归重试。ttl 递减到 0 后，返回 None 给上层统计。
            if ttl > 0:
                print_debug(
                    f"Failed to parse worker response, retrying {ttl=}", response, e
                )
                return self._pred_one_openai(data, criterion, ttl - 1)
            else:
                print_debug("Failed to parse worker response", response, e)
                return None, thought
        if answer == "N":  # None
            # 模型有时会把 None 简写成 N，这里统一映射成 U。
            answer = "U"
        if answer not in ("A", "B", "U"):
            answer = None
        return answer, thought


class StructuredMultiModalPairEvaluator(MultiModalPairEvaluator):
    """Multimodal evaluator that preserves a full NodeJudgement per criterion."""

    worker_prompt_postfix = STRUCTURED_WORKER_PROMPT_POSTFIX

    def __init__(
        self,
        worker_args: dict,
        dataset: Sequence[PairData],
        worker_backend_id: str,
        max_concurrent: int = 1,
        max_retries: int = 3,
        worker_prompt: str | None = None,
        max_data_chars: int | None = None,
        image_field: str = "image_path",
        question_field: str = "question",
        sample_id_field: str = "sample_id",
        encode_local_image: bool = True,
        worker_pricing: TokenPricing | None = None,
    ) -> None:
        if not dataset:
            raise ValueError("structured multimodal dataset must not be empty")
        if (
            isinstance(max_concurrent, bool)
            or not isinstance(max_concurrent, int)
            or max_concurrent < 1
        ):
            raise ValueError("max_concurrent must be a positive integer")
        if (
            isinstance(max_retries, bool)
            or not isinstance(max_retries, int)
            or max_retries < 0
        ):
            raise ValueError("max_retries must be a non-negative integer")
        if not isinstance(worker_backend_id, str) or not worker_backend_id.strip():
            raise ValueError("worker_backend_id must be a non-empty string")
        if max_data_chars is not None and (
            isinstance(max_data_chars, bool)
            or not isinstance(max_data_chars, int)
            or max_data_chars < 1
        ):
            raise ValueError("max_data_chars must be None or a positive integer")
        if not isinstance(encode_local_image, bool):
            raise TypeError("encode_local_image must be bool")
        if not isinstance(sample_id_field, str) or not sample_id_field.strip():
            raise ValueError("sample_id_field must be a non-empty string")
        if not isinstance(worker_args, dict):
            raise TypeError("worker_args must be a dict")
        if worker_pricing is not None and not isinstance(worker_pricing, TokenPricing):
            raise TypeError("worker_pricing must be TokenPricing or None")

        sample_ids: list[str] = []
        for index, data in enumerate(dataset):
            if not isinstance(data, Mapping):
                raise ValueError(f"dataset row {index + 1} must be an object")
            sample_id = data.get(sample_id_field)
            if not isinstance(sample_id, str) or not sample_id.strip():
                raise ValueError(
                    f"structured multimodal evaluator requires a non-empty "
                    f"{sample_id_field!r} at row {index + 1}"
                )
            if not isinstance(data.get("A"), str) or not isinstance(
                data.get("B"), str
            ):
                raise ValueError(f"A/B must be strings at row {index + 1}")
            if data.get("answer") not in ("A", "B"):
                raise ValueError(f"answer must be A or B at row {index + 1}")
            sample_ids.append(sample_id)
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("sample IDs must not contain duplicates")

        structured_prompt = worker_prompt or STRUCTURED_MULTIMODAL_WORKER_PROMPT
        required_placeholders = {
            "{criterion}",
            "{description}",
            "{question}",
            "{A}",
            "{B}",
        }
        missing_placeholders = {
            placeholder
            for placeholder in required_placeholders
            if placeholder not in structured_prompt
        }
        if missing_placeholders:
            raise ValueError(
                "structured worker prompt missing placeholders: "
                f"{sorted(missing_placeholders)}"
            )
        self.worker_backend_id = worker_backend_id
        self.worker_pricing = worker_pricing

        super().__init__(
            worker_args=worker_args,
            dataset=dataset,
            max_concurrent=max_concurrent,
            max_retries=max_retries,
            worker_prompt=structured_prompt,
            max_data_chars=max_data_chars,
            image_field=image_field,
            question_field=question_field,
            encode_local_image=encode_local_image,
        )
        self.sample_id_field = sample_id_field

    @staticmethod
    def _normalize_structured_criteria(
        criteria: Sequence[Criterion | dict[Literal["name", "description"], str]],
    ) -> tuple[Criterion, ...]:
        if not criteria:
            raise ValueError("criteria must not be empty")
        normalized: list[Criterion] = []
        for index, criterion in enumerate(criteria):
            if isinstance(criterion, dict):
                try:
                    criterion = Criterion.from_dict(dict(criterion))
                except Exception as exc:
                    raise ValueError(f"invalid criterion at index {index}") from exc
            if not isinstance(criterion, Criterion):
                raise TypeError(f"criterion at index {index} must be Criterion or dict")
            if not isinstance(criterion.name, str) or not criterion.name.strip():
                raise ValueError(f"criterion name at index {index} must be non-empty")
            if (
                not isinstance(criterion.description, str)
                or not criterion.description.strip()
            ):
                raise ValueError(
                    f"criterion description at index {index} must be non-empty"
                )
            normalized.append(criterion)
        names = [criterion.name for criterion in normalized]
        if len(set(names)) != len(names):
            raise ValueError("criterion names must not contain duplicates")
        return tuple(normalized)

    @staticmethod
    def _structured_agent_metrics(worker: object) -> AgentCallMetrics:
        metrics = getattr(worker, "last_call_metrics", None)
        if isinstance(metrics, AgentCallMetrics):
            return metrics
        return AgentCallMetrics(
            api_attempts=1,
            input_tokens=None,
            output_tokens=None,
            total_tokens=None,
            usage_complete=False,
        )

    def infer_one(
        self,
        data: PairData,
        criterion: Criterion,
    ) -> tuple[StructuredNodeOutput, ModelCallMetrics]:
        """Infer one sample-node lazily and return current-run API telemetry."""

        if not isinstance(data, Mapping):
            raise TypeError("data must be a mapping")
        if not isinstance(criterion, Criterion):
            raise TypeError("criterion must be Criterion")

        last_raw_response: str | None = None
        last_parse_error = "worker did not return a parseable response"
        last_inconsistent: StructuredNodeOutput | None = None
        total_attempts = self.max_retries + 1
        agent_metrics: list[AgentCallMetrics] = []

        for attempt_count in range(1, total_attempts + 1):
            worker = Agent(**self.worker_args)
            prompt = self._make_user_content(data, criterion)
            raw_response = worker(prompt, stream=False)
            agent_metrics.append(self._structured_agent_metrics(worker))
            last_raw_response = (
                raw_response if isinstance(raw_response, str) else None
            )
            try:
                judgement = parse_structured_worker_response(raw_response)
            except StructuredOutputParseError as exc:
                last_parse_error = str(exc)
                print_debug(
                    "Failed to parse structured worker response",
                    f"attempt={attempt_count}/{total_attempts}",
                    raw_response,
                    exc,
                )
                continue

            output = StructuredNodeOutput(
                judgement=judgement,
                raw_response=raw_response,
                parse_error=None,
                attempt_count=attempt_count,
            )
            if judgement.consistency_ok:
                final_output = output
                break
            last_inconsistent = output
            print_debug(
                "Structured worker response failed consistency validation",
                f"attempt={attempt_count}/{total_attempts}",
                judgement.consistency_errors,
            )

        else:
            if last_inconsistent is not None:
                final_output = StructuredNodeOutput(
                    judgement=last_inconsistent.judgement,
                    raw_response=last_inconsistent.raw_response,
                    parse_error=None,
                    attempt_count=total_attempts,
                )
            else:
                final_output = StructuredNodeOutput(
                    judgement=make_parse_failure_judgement(),
                    raw_response=last_raw_response,
                    parse_error=last_parse_error,
                    attempt_count=total_attempts,
                )

        metrics = ModelCallMetrics.from_agent_calls(
            agent_metrics,
            logical_evaluations=1,
            parse_retries=max(0, len(agent_metrics) - 1),
            pricing=self.worker_pricing,
        )
        return final_output, metrics

    def _pred_one_openai(
        self,
        data: PairData,
        criterion: Criterion,
        ttl: int,
    ) -> StructuredNodeOutput:
        """Compatibility wrapper used by existing batch prediction."""

        if ttl != self.max_retries:
            raise ValueError("ttl must equal evaluator max_retries")
        return self.infer_one(data, criterion)[0]

    def pred_openai(
        self,
        criteria: Sequence[Criterion],
    ) -> tuple[dict[str, StructuredNodeOutput], ...]:
        """Evaluate the dataset/criterion Cartesian product in stable input order."""

        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            futures = [
                executor.submit(
                    self._pred_one_openai,
                    data,
                    criterion,
                    self.max_retries,
                )
                for data in self.dataset
                for criterion in criteria
            ]
            for _ in tqdm(
                as_completed(futures),
                total=len(futures),
                dynamic_ncols=True,
                disable=not USE_TQDM,
            ):
                pass

            results = (future.result() for future in futures)
            outputs: list[dict[str, StructuredNodeOutput]] = []
            for _data in self.dataset:
                outputs.append(
                    {
                        criterion.name: next(results)
                        for criterion in criteria
                    }
                )
        return tuple(outputs)

    def _structured_sample_fingerprints(self) -> tuple[str, ...]:
        return tuple(
            self.sample_fingerprint(data)
            for data in self.dataset
        )

    def sample_fingerprint(self, data: PairData) -> str:
        return structured_input_fingerprint(
            data,
            image_field=self.image_field,
            question_field=self.question_field,
            sample_id_field=self.sample_id_field,
            max_data_chars=self.max_data_chars,
            encode_local_image=self.encode_local_image,
        )

    def request_spec(self) -> StructuredWorkerRequestSpec:
        decoding_config = self.worker_args.get("request_kwargs") or {}
        try:
            decoding_config = json.loads(
                json.dumps(decoding_config, ensure_ascii=False, allow_nan=False)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("worker request_kwargs must be JSON serializable") from exc
        return StructuredWorkerRequestSpec(
            model=str(self.worker_args.get("model", "gpt-4o-mini")),
            worker_backend_id=self.worker_backend_id,
            prompt_sha256=structured_worker_prompt_sha256(
                self.worker_prompt,
                self.worker_prompt_postfix,
            ),
            max_data_chars=self.max_data_chars,
            encode_local_image=self.encode_local_image,
            image_field=self.image_field,
            question_field=self.question_field,
            sample_id_field=self.sample_id_field,
            decoding_config=decoding_config,
        )


    def pred(
        self,
        criteria: Sequence[Criterion | dict[Literal["name", "description"], str]],
    ) -> StructuredPredictionOutput:
        """Return versioned structured outputs plus a flat-uniform sanity vote."""

        sample_fingerprints = self._structured_sample_fingerprints()
        normalized = self._normalize_structured_criteria(criteria)
        node_outputs = self.pred_openai(normalized)
        if self._structured_sample_fingerprints() != sample_fingerprints:
            raise RuntimeError(
                "structured worker inputs changed while prediction was running"
            )

        criterion_names = tuple(criterion.name for criterion in normalized)
        flat_answers = tuple(
            aggregate_selected_roots(
                {
                    name: outputs[name].local_decision.vote
                    for name in criterion_names
                },
                criterion_names,
            )
            for outputs in node_outputs
        )
        return StructuredPredictionOutput(
            sample_ids=tuple(
                str(data[self.sample_id_field]) for data in self.dataset
            ),
            sample_fingerprints=sample_fingerprints,
            criteria=tuple(
                StructuredCriterionSnapshot(
                    name=criterion.name,
                    description=criterion.description,
                )
                for criterion in normalized
            ),
            node_outputs=node_outputs,
            flat_answers=flat_answers,
            request_spec=self.request_spec(),
        )

    def eval(
        self,
        criteria: Sequence[Criterion | dict[Literal["name", "description"], str]],
        update_score: bool = False,
    ) -> StructuredEvaluationOutput:
        """Compute flat sanity accuracy and decisive-vote coverage."""

        normalized = self._normalize_structured_criteria(criteria)
        prediction = self.pred(normalized)
        is_correct = tuple(
            answer in (FinalPreference.A, FinalPreference.B)
            and answer.value == data["answer"]
            for data, answer in zip(self.dataset, prediction.flat_answers)
        )
        decisive_answers = sum(
            answer in (FinalPreference.A, FinalPreference.B)
            for answer in prediction.flat_answers
        )
        accuracy = sum(is_correct) / len(is_correct)
        coverage = decisive_answers / len(prediction.flat_answers)

        per_criterion_accuracy: dict[str, float] = {}
        per_criterion_coverage: dict[str, float] = {}
        for criterion in normalized:
            decisive = 0
            correct = 0
            for data, outputs in zip(self.dataset, prediction.node_outputs):
                vote = outputs[criterion.name].local_decision.vote
                if vote not in (Vote.A, Vote.B):
                    continue
                decisive += 1
                if vote.value == data["answer"]:
                    correct += 1
            per_criterion_accuracy[criterion.name] = (
                correct / decisive if decisive else 0.0
            )
            per_criterion_coverage[criterion.name] = decisive / len(self.dataset)
            if update_score:
                criterion.score = per_criterion_accuracy[criterion.name]

        return StructuredEvaluationOutput(
            prediction=prediction,
            is_correct=is_correct,
            accuracy=accuracy,
            coverage=coverage,
            per_criterion_accuracy=per_criterion_accuracy,
            per_criterion_coverage=per_criterion_coverage,
        )


class MoEPairEvaluator(MultiModalPairEvaluator):
    """Multimodal pair evaluator with static priors and dynamic soft routing."""

    def __init__(
        self,
        worker_args: dict,
        dataset: Sequence[PairData],
        max_concurrent: int = 1,
        max_retries: int = 3,
        worker_prompt: str | None = None,
        max_data_chars: int | None = None,
        image_field: str = "image_path",
        question_field: str = "question",
        encode_local_image: bool = True,
        router_args: dict | None = None,
        criterion_stats: Mapping[str, Any] | str | Path | None = None,
        static_alpha: float = 0.7,
        static_beta: float = 0.3,
        routing_threshold: float = 0.2,
        fallback_criteria: Sequence[str] = (
            "visual_grounding",
            "factual_consistency",
        ),
        fallback_weight: float = 0.5,
        tie_epsilon: float = 0.05,
    ) -> None:
        super().__init__(
            worker_args=worker_args,
            dataset=dataset,
            max_concurrent=max_concurrent,
            max_retries=max_retries,
            worker_prompt=worker_prompt,
            max_data_chars=max_data_chars,
            image_field=image_field,
            question_field=question_field,
            encode_local_image=encode_local_image,
        )
        self.router_args = router_args or worker_args
        self.criterion_performance = load_criterion_performance(criterion_stats)
        self.static_alpha = static_alpha
        self.static_beta = static_beta
        self.routing_threshold = routing_threshold
        self.fallback_criteria = tuple(fallback_criteria)
        self.fallback_weight = fallback_weight
        self.tie_epsilon = tie_epsilon
        self.router = RouterManager(
            router_args=self.router_args,
            image_field=self.image_field,
            question_field=self.question_field,
            encode_local_image=self.encode_local_image,
            routing_threshold=self.routing_threshold,
            fallback_criteria=self.fallback_criteria,
            fallback_weight=self.fallback_weight,
            max_retries=self.max_retries,
        )
        self.routing: list[dict[str, Any]] = []
        self._last_dynamic_weights: list[dict[str, float]] = []

    def _static_weight_for(self, criterion_name: str) -> float:
        stat = self.criterion_performance.get(criterion_name)
        if stat is None:
            return 1.0
        return static_weight(stat, self.static_alpha, self.static_beta)

    def _active_criteria(
        self, decision: RoutingDecision, criteria: Sequence[Criterion]
    ) -> list[Criterion]:
        return [
            criterion
            for criterion in criteria
            if decision.routing_weights.get(criterion.name, 0.0)
            >= self.routing_threshold
        ]

    def _route_dataset(self, criteria: Sequence[Criterion]) -> list[RoutingDecision]:
        with ThreadPoolExecutor(max_workers=self.max_concurrent) as t:
            futures = [
                t.submit(self.router.route, data, criteria) for data in self.dataset
            ]
            for _ in tqdm(
                as_completed(futures),
                total=len(futures),
                dynamic_ncols=True,
                disable=not USE_TQDM,
            ):
                pass
            return [future.result() for future in futures]

    def pred_openai(self, criteria: Sequence[Criterion]):
        """Route each sample first, then evaluate only active criteria."""
        routing_decisions = self._route_dataset(criteria)
        self._last_dynamic_weights = [
            decision.routing_weights for decision in routing_decisions
        ]
        self.routing = []

        prediction = [
            {criterion.name: {"A": 0, "B": 0, "U": 0} for criterion in criteria}
            for _ in self.dataset
        ]
        thoughts = [
            {criterion.name: None for criterion in criteria} for _ in self.dataset
        ]

        tasks: list[tuple[int, Criterion]] = []
        for data_idx, decision in enumerate(routing_decisions):
            active = self._active_criteria(decision, criteria)
            self.routing.append(
                {
                    "scene_analysis": decision.scene_analysis,
                    "routing_weights": decision.routing_weights,
                    "active_criteria": [criterion.name for criterion in active],
                    "source": decision.source,
                }
            )
            for criterion in active:
                tasks.append((data_idx, criterion))

        if not tasks:
            return prediction, thoughts

        with ThreadPoolExecutor(max_workers=self.max_concurrent) as t:
            futures = [
                t.submit(
                    self._pred_one_openai,
                    self.dataset[data_idx],
                    criterion,
                    self.max_retries,
                )
                for data_idx, criterion in tasks
            ]
            for _ in tqdm(
                as_completed(futures),
                total=len(futures),
                dynamic_ncols=True,
                disable=not USE_TQDM,
            ):
                pass

            for (data_idx, criterion), future in zip(tasks, futures):
                one_pred, thought = future.result()
                if one_pred in ("A", "B", "U"):
                    prediction[data_idx][criterion.name][one_pred] += 1
                    thoughts[data_idx][criterion.name] = thought

        return prediction, thoughts

    def voting_fn(self, prediction: PredictionOutput, **_: Any) -> list[PairAnswer]:
        """Aggregate A/B votes with static and dynamic MoE weights."""
        result: list[PairAnswer] = []
        for data_idx, row in enumerate(prediction):
            score = {"A": 0.0, "B": 0.0}
            active_weight_sum = 0.0
            dynamic_weights = (
                self._last_dynamic_weights[data_idx]
                if data_idx < len(self._last_dynamic_weights)
                else {}
            )
            for criterion_name, vote_counts in row.items():
                dynamic_weight = dynamic_weights.get(criterion_name, 1.0)
                combined_weight = (
                    self._static_weight_for(criterion_name) * dynamic_weight
                )
                if combined_weight <= 0:
                    continue
                if vote_counts.get("A", 0) > vote_counts.get("B", 0):
                    score["A"] += combined_weight
                    active_weight_sum += combined_weight
                elif vote_counts.get("B", 0) > vote_counts.get("A", 0):
                    score["B"] += combined_weight
                    active_weight_sum += combined_weight

            if active_weight_sum <= 0:
                result.append("Tie")
                continue

            score_a = score["A"] / active_weight_sum
            score_b = score["B"] / active_weight_sum
            if abs(score_a - score_b) < self.tie_epsilon:
                result.append("Tie")
            elif score_a > score_b:
                result.append("A")
            else:
                result.append("B")
        return result

    def pred(
        self,
        criteria: Sequence[Criterion | dict[Literal["name", "description"], str]],
        **voting_fn_kwargs,
    ) -> PredictionOutputWithAnswer:
        """Run routed multimodal prediction and weighted aggregation."""
        normalized_criteria = [
            c if isinstance(c, Criterion) else Criterion.from_dict(c) for c in criteria
        ]
        prediction, thoughts = self.pred_openai(normalized_criteria)
        return PredictionOutputWithAnswer(
            prediction=prediction,
            answer=self.voting_fn(prediction, **voting_fn_kwargs),
            thoughts=thoughts,
            routing=self.routing,
        )

    def eval(
        self,
        criteria: Sequence[Criterion | dict[Literal["name", "description"], str]],
        update_score=False,
        **voting_fn_kwargs,
    ) -> EvaluationOutput:
        """Evaluate MoE predictions while excluding skipped criteria from stats."""
        normalized_criteria = [
            c if isinstance(c, Criterion) else Criterion.from_dict(c) for c in criteria
        ]
        prediction_with_answer = self.pred(
            normalized_criteria, **voting_fn_kwargs
        )
        prediction = prediction_with_answer.prediction
        answer = prediction_with_answer.answer

        is_correct = []
        for data, predicted_answer in zip(self.dataset, answer):
            is_correct.append(
                predicted_answer in ("A", "B")
                and predicted_answer == data["answer"]
            )

        per_criterion_acc = {criterion.name: 0.0 for criterion in normalized_criteria}
        for criterion in normalized_criteria:
            n_correct = 0
            n_total = 0
            for data, row in zip(self.dataset, prediction):
                vote_counts = row[criterion.name]
                if vote_counts.get("U", 0) > 0:
                    continue
                if vote_counts.get("A", 0) == 0 and vote_counts.get("B", 0) == 0:
                    continue
                n_total += 1
                if (
                    vote_counts[data["answer"]]
                    > vote_counts[reverse_ab(data["answer"])]
                ):
                    n_correct += 1
            per_criterion_acc[criterion.name] = (
                n_correct / n_total if n_total else 0.0
            )
            if update_score:
                criterion.score = per_criterion_acc[criterion.name]

        return EvaluationOutput(
            prediction=prediction,
            is_correct=is_correct,
            per_criterion_acc=per_criterion_acc,
            accuracy=len(list(filter(None, is_correct))) / len(is_correct),
            thoughts=prediction_with_answer.thoughts,
            routing=prediction_with_answer.routing,
        )


class ZeroOneEvaluator(Evaluator):
    """用于二分类好/坏数据集的 evaluator。"""

    worker_prompt_postfix = local_prompts.ZERO_ONE_WORKER_PROMPT_POSTFIX

    def __init__(
        self,
        worker_args: dict,
        dataset: Sequence[ZeroOneData],
        max_concurrent: int = 1,
        max_retries: int = 3,
        worker_prompt: str | None = None,
        max_data_chars: int | None = None,
    ) -> None:
        self.worker_args = worker_args
        self.dataset = dataset
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.max_data_chars = max_data_chars

        self.worker_prompt = worker_prompt or local_prompts.ZERO_ONE_WORKER_PROMPT
        assert all(
            [
                k in self.worker_prompt
                for k in ["{criterion}", "{description}", "{text}"]
            ]
        )

    def voting_fn(
        self, prediction: PredictionOutput, threshold: int = 0
    ) -> list[Literal[0, 1, None]]:
        """将多个 criterion 的投票聚合成二分类标签。"""
        result = []
        for d in prediction:
            stat = {0: 0, 1: 0}
            for c in d.values():
                stat[1] += c[1]
                stat[0] += c[0]
            if stat[1] - stat[0] > threshold:
                result.append(1)
            elif stat[1] - stat[0] < threshold:
                result.append(0)
            else:
                result.append(None)
        return result

    def _make_prompt(self, data, criterion):
        """为单条样本在单个 criterion 下构造 prompt。"""
        text = (
            data["text"][: self.max_data_chars] if self.max_data_chars else data["text"]
        )
        prompt = (
            self.worker_prompt.replace("{criterion}", criterion.name)
            .replace("{description}", criterion.description)
            .replace("{text}", text)
        )
        prompt += self.worker_prompt_postfix
        return prompt

    def _pred_one_openai(
        self, data: PairData, criterion: Criterion, ttl: int
    ) -> tuple[Literal[0, 1] | None, str | None]:
        """让一个 worker 判断样本是否满足该 criterion。"""
        worker = Agent(**self.worker_args)
        prompt = self._make_prompt(data, criterion)
        response = worker(prompt, stream=False)
        thought = None
        try:
            response = parse_json(response)
            answer = response["answer"].strip()[0].upper()
            thought = response["thought"].strip()
        except Exception as e:  # pylint: disable=W0718:broad-exception-caught
            if ttl > 0:
                print_debug(
                    f"Failed to parse worker response, retrying {ttl=}", response, e
                )
                return self._pred_one_openai(data, criterion, ttl - 1)
            else:
                print_debug("Failed to parse worker response", response, e)
                return None
        if answer not in ("Y", "N"):
            return None, thought
        return {"Y": 1, "N": 0}[answer], thought

    def pred_openai(self, criteria: Sequence[Criterion]):
        """对数据集和 criterion 列表进行并行二分类预测。"""
        with ThreadPoolExecutor(max_workers=self.max_concurrent) as t:
            futures = []
            for data in self.dataset:
                for criterion in criteria:
                    future = t.submit(
                        self._pred_one_openai, data, criterion, self.max_retries
                    )
                    futures.append(future)

            for _ in tqdm(
                as_completed(futures),
                total=len(futures),
                dynamic_ncols=True,
                disable=not USE_TQDM,
            ):
                pass

            prediction = []
            thoughts = []
            results = (future.result() for future in futures)
            for data in self.dataset:
                _prediction = {criterion.name: {0: 0, 1: 0} for criterion in criteria}
                _thoughts = {criterion.name: None for criterion in criteria}
                for criterion in criteria:
                    one_pred, thought = next(results)
                    if one_pred in (0, 1):
                        _prediction[criterion.name][one_pred] += 1
                        _thoughts[criterion.name] = thought
                prediction.append(_prediction)
                thoughts.append(_thoughts)

        return prediction, thoughts

    def pred(
        self,
        criteria: Sequence[Criterion | dict[Literal["name", "description"], str]],
        **voting_fn_kwargs,
    ) -> PredictionOutputWithAnswer:
        """规范化 criterion 输入并执行二分类预测。"""
        for i, c in enumerate(criteria):
            if isinstance(c, dict):
                criteria[i] = Criterion.from_dict(c)
        prediction, thoughts = self.pred_openai(criteria)
        return PredictionOutputWithAnswer(
            prediction=prediction,
            answer=self.voting_fn(prediction, **voting_fn_kwargs),
            thoughts=thoughts,
        )

    def eval(
        self,
        criteria: Sequence[Criterion | dict[Literal["name", "description"], str]],
        update_score=False,
        **voting_fn_kwargs,
    ) -> EvaluationOutput:
        """计算聚合后的准确率和 criterion 级别的二分类准确率。"""
        for i, c in enumerate(criteria):
            if isinstance(c, dict):
                criteria[i] = Criterion.from_dict(c)

        prediction_with_answer = self.pred(criteria, **voting_fn_kwargs)
        prediciton = prediction_with_answer.prediction
        answer = prediction_with_answer.answer

        is_correct = []
        for d, a in zip(self.dataset, answer):
            is_correct.append(a is not None and a == d["label"])

        per_criterion_acc = {criterion.name: 0 for criterion in criteria}
        for criterion in criteria:
            n_correct = 0
            n_total = len(self.dataset)
            for d, p in zip(self.dataset, prediciton):
                if p[criterion.name][d["label"]] > p[criterion.name][1 - d["label"]]:
                    n_correct += 1
            per_criterion_acc[criterion.name] = (
                0 if n_correct == 0 else n_correct / n_total
            )
            if update_score:
                criterion.score = per_criterion_acc[criterion.name]

        return EvaluationOutput(
            prediction=prediciton,
            is_correct=is_correct,
            per_criterion_acc=per_criterion_acc,
            accuracy=len(list(filter(None, is_correct))) / len(is_correct),
        )


def get_evaluator_cls_from_dataset(dataset: Sequence[dict]):
    """根据数据集结构推断应该使用哪种 evaluator。

    这个函数只做“传统文本格式”的自动分发：
    - zero-one 数据 -> ZeroOneEvaluator
    - pair 数据 -> PairEvaluator

    注意：即使 pair 数据里额外带有 image_path/question，这个函数仍然会返回
    PairEvaluator，因为 `is_pair_dataset()` 只检查 A/B/answer。
    多模态任务请在脚本中显式实例化 MultiModalPairEvaluator。
    """
    if is_zero_one_dataset(dataset):
        return ZeroOneEvaluator
    elif is_pair_dataset(dataset):
        return PairEvaluator
    else:
        raise ValueError("Invalid validset format")


def _demo_multimodal_pair_evaluator():
    """本地自测 MultiModalPairEvaluator 的 prompt/content 构造。
    1. 能从 RLHF-V pair JSONL 里读到一条样本；
    2. 能把 {question}/{A}/{B} 填进文本 prompt；
    3. 如果 image_path 存在，能生成 OpenAI 兼容的 text + image_url content。
    4. 使用 vllm 服务本地模型时，能正确把 content 传给 Agent，并得到响应。

    运行方式：
        python -m critiq.evaluator
    """
    candidate_paths = [
        Path("data/RLHF-V/discovery_train_90_pair.jsonl"),
        Path("output/discovery_train_90_review_tmp.jsonl"),
    ]
    data_path = next((path for path in candidate_paths if path.exists()), None)
    if data_path is None:
        print("No RLHF-V pair JSONL found for demo.")
        print("Please run data/RLHF-V/convert_to_pair.py first.")
        return

    with data_path.open("r", encoding="utf-8") as f:
        data = json.loads(next(line for line in f if line.strip()))

    prompt = """## Instruction
Compare two candidate answers to the visual question under one criterion.

## Question
{question}

## Criterion
{criterion}: {description}

## A
{A}

## B
{B}"""

    criterion = Criterion(
        name="Visual Grounding",
        description=(
            "Prefer the answer that is better grounded in the image and avoids "
            "unsupported visual hallucinations."
        ),
    )
    import os
    WORKER_ARGS = {
        "model": os.getenv("CRITIQ_WORKER_MODEL", "Qwen/Qwen3-VL-8B-Instruct"),
        "base_url": os.getenv("CRITIQ_WORKER_BASE_URL", "http://10.102.137.255:8000/v1"),
        "api_keys": 'EMPTY',
        "request_kwargs": {
            "temperature": 0.5,
        },
    }
    evaluator = MultiModalPairEvaluator(
        worker_args=WORKER_ARGS,
        dataset=[data],
        worker_prompt=prompt,
    )
    content = evaluator._make_user_content(data, criterion)
    # 调用模型，查看真实响应
    eval_output = evaluator.eval([criterion])
    print("Worker response:")
    print(f"Prediction: {eval_output.prediction}")
    print(f"Thoughts: {eval_output.thoughts}")
    print(f"Is correct: {eval_output.is_correct}")
    print()
    print(f"Loaded sample from: {data_path}")
    print(f"Sample keys: {sorted(data.keys())}")
    print(f"Answer label: {data.get('answer')}")
    print(f"Content type: {type(content).__name__}")

    if isinstance(content, list):
        print(f"Message parts: {[part['type'] for part in content]}")
        print("Text preview:")
        print(content[0]["text"][:2048])
        print("Image URL preview:")
        print(content[1]["image_url"]["url"][:120])
    else:
        print("Text-only prompt preview:")
        print(content[:2048])


if __name__ == "__main__":
    _demo_multimodal_pair_evaluator()
