"""成对比较与二分类数据质量判断的评估逻辑。

Workflow 会让 worker agent 将每个 criterion 应用于带标签数据，
再通过这里的评估逻辑衡量哪些 criterion 更贴近人工标注。
"""

import json
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Literal, Sequence

from tqdm import tqdm

from .agent import Agent
from .i18n import local_prompts
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
    dict[str, dict[Literal["A", "B"] | Literal[0, 1], int]]
]  # output[data_idx][criterion_name] = {"A": ..., "B": ...}


@dataclass
class PredictionOutputWithAnswer:
    """每个 criterion 的原始投票结果，以及最终聚合后的答案。"""

    prediction: PredictionOutput
    answer: list[Literal["A", "B", None] | Literal[0, 1, None]]
    thoughts: list[dict[str, str]] | None = None


@dataclass
class EvaluationOutput:
    """`eval()` 返回的完整评估结果。"""

    prediction: PredictionOutput
    is_correct: list[bool]  # True if the voting result is correct
    per_criterion_acc: dict[str, float]  # accuracy for each criterion
    accuracy: float
    thoughts: list[dict[str, str]] | None = None

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


class Evaluator:
    """不同数据格式共享的抽象 evaluator 接口。"""

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
        self.worker_args = worker_args
        self.dataset = dataset
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.max_data_chars = max_data_chars

        self.worker_prompt = worker_prompt or local_prompts.PAIR_WORKER_PROMPT
        assert all(
            [
                k in self.worker_prompt
                for k in ["{criterion}", "{description}", "{A}", "{B}"]
            ]
        )

    def voting_fn(
        self, prediction: PredictionOutput, threshold: int = 0
    ) -> list[Literal["A", "B", None]]:
        """将各个 criterion 的 A/B 投票聚合成最终成对偏好结果。"""
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
        """为一条 A/B 样本构造单 criterion prompt。"""
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
        """让一个 worker 基于一个 criterion 判断一条样本对。"""
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
        """执行完整的笛卡尔积评估：每条样本对都要过每个 criterion。"""
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
                # 将扁平的 future 结果重新组装成按样本索引的字典结构。
                _prediction = {
                    criterion.name: {"A": 0, "B": 0, "U": 0} for criterion in criteria
                }
                _thoughts = {criterion.name: None for criterion in criteria}
                for criterion in criteria:
                    one_pred, thought = next(results)
                    if one_pred in ("A", "B", "U"):
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
        """同时评估最终投票准确率和 criterion 级别有效性。"""
        for i, c in enumerate(criteria):
            if isinstance(c, dict):
                criteria[i] = Criterion.from_dict(c)

        prediction_with_answer = self.pred(criteria, **voting_fn_kwargs)
        prediciton = prediction_with_answer.prediction
        answer = prediction_with_answer.answer

        is_correct = []
        for d, a in zip(self.dataset, answer):
            is_correct.append(a is not None and a == d["answer"])

        per_criterion_acc = {criterion.name: 0 for criterion in criteria}
        for criterion in criteria:
            n_correct = 0
            n_total = 0
            for d, p in zip(self.dataset, prediciton):
                # "U" 表示该 criterion 不适用，或模型选择弃权。
                # 这类情况不会被计入 criterion 自身准确率的统计。
                if p[criterion.name]["U"] > 0:
                    # Only count when the worker is sure and crriterion is applicable.
                    # If the answer is None, which means the output is invalid to resolve.
                    # It will not be passed and will be regarded as incorrect.
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
    """根据数据集结构推断应该使用哪种 evaluator。"""
    if is_zero_one_dataset(dataset):
        return ZeroOneEvaluator
    elif is_pair_dataset(dataset):
        return PairEvaluator
    else:
        raise ValueError("Invalid validset format")
