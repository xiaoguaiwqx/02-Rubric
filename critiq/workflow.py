"""演化评估 criterion 的高层优化循环。

`Workflow` 是 CritiQ Flow 的调度核心。它本身不负责直接执行 LLM 推理，
而是把不同角色的组件组织起来：

- manager agent:
  负责“提出标准”和“改写标准”，也就是生产 criterion 文本的人。
- worker evaluator:
  负责“拿标准去判样本”，也就是测这些标准到底是否能逼近人工偏好的人。

这个模块围绕下面这条闭环工作：
1. 从知识库检索一批初始 criterion，或让 manager 直接生成；
2. 用 evaluator 在带标签样本上测每条 criterion 的有效性；
3. 把 criterion 按准确率分成 good / mid / low；
4. 保留 good，反思并改写 mid，淘汰 low 并补充新标准；
5. 保存每轮状态，最后从历史上挑出最强的一组 criterion。

理解这个文件时，可以把它看成“criterion 的训练器 / 演化器”，
而不是一个普通的 prompt 调用包装器。
"""

import json
import os
import random
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from typing import Any, Literal, Sequence

from prettytable import PrettyTable
from tqdm import tqdm

from .agent import Agent
from .evaluator import (
    MultiModalPairEvaluator,
    MoEPairEvaluator,
    PairEvaluator,
    ZeroOneEvaluator,
    get_evaluator_cls_from_dataset,
)
from .i18n import local_prompts
from .utils import (
    MANAGER_MAX_CONCURRENT,
    USE_TQDM,
    Criterion,
    PairData,
    ZeroOneData,
    criteria_list_to_dict,
    is_pair_dataset,
    is_zero_one_dataset,
    parse_json,
    print_debug,
    random_reverse,
    reverse_ab,
)


EVALUATOR_REGISTRY = {
    "pair": PairEvaluator,
    "zero_one": ZeroOneEvaluator,
    "multimodal_pair": MultiModalPairEvaluator,
    "moe_multimodal_pair": MoEPairEvaluator,
}


class Workflow:
    """负责协调 criterion 的生成、评估、改写与持久化。

    这个类维护三份核心状态：

    - `current_criteria`
      当前这一轮真正参与评估和优化的 criterion 集合。
    - `all_criteria`
      历史上出现过的全部 criterion 的最佳版本，用于最后筛选最强标准。
    - `banned_criteria`
      已经被判定太差、不应再次生成同名项的 criterion 名称集合。

    这三者共同定义了 workflow 当前“知道什么、正在试什么、以后不要再试什么”。
    """

    def __init__(
        self,
        manager_args: dict[str, Any] | None = None,
        worker_args: dict[str, Any] | None = None,
        worker_max_concurrent: int = 1,
        init_criteria: (
            Sequence[Criterion | dict[Literal["name", "description"], str]] | None
        ) = None,
        n_criteria: int = 10,
        manager_prompt: str | None = None,
        manager_prompt_postfix: str | None = None,
        worker_prompt: str | None = None,
        evaluator_type: str | None = None,
        evaluator_kwargs: dict[str, Any] | None = None,
        manager_multimodal: bool = False,
        manager_reflection_include_question: bool = True,
    ) -> None:
        """初始化一个 criterion 演化 workflow。

        参数大致分成三类：

        - agent 配置：
          `manager_args` 控制负责生成/改写标准的模型，`worker_args` 控制负责执行评估的模型。
          两者都允许只传局部配置，剩余字段会沿用这里的默认值。
        - criterion 配置：
          `init_criteria` 是外部预置的起始标准，`n_criteria` 是 workflow 希望长期维持的标准数量。
          后续每轮优化会尽量通过“保留 + 改写 + 补新”把数量维持在这个目标附近。
        - prompt 配置：
          manager prompt 决定“怎样生成 criterion”，worker prompt 决定“怎样使用 criterion 判样本”。
          这些模板可覆写，因此同一个 workflow 可以迁移到不同数据质量定义或偏好任务上。
        """
        # manager 负责提出或改写 criterion，worker 负责将 criterion 用到数据上。
        # 默认给 manager 更强、温度更高的配置，是因为它承担“发散式生成/改写”任务；
        # worker 更像执行器，通常更需要稳定而不是发散。
        self.manager_args = {
            "model": "gpt-4o",
            "request_kwargs": {
                "temperature": 1.0,
            },
        }
        if manager_args is not None:
            self.manager_args.update(manager_args)

        self.worker_args = {"model": "gpt-4o-mini"}
        if worker_args is not None:
            self.worker_args.update(worker_args)

        self.worker_max_concurrent = worker_max_concurrent

        # `current_criteria` 是“本轮候选池”。
        # 支持直接传 Criterion，也支持从序列化 dict 恢复，方便读取 checkpoint。
        self.current_criteria: list[Criterion] = (
            [
                c if isinstance(c, Criterion) else Criterion.from_dict(c)
                for c in init_criteria
            ]
            if init_criteria
            else []
        )
        self.banned_criteria: set[str] = set()

        # `all_criteria` 保存截至目前见过的最好版本“”。
        # 它和 current_criteria 不同：后者会在优化中被覆盖，前者更像历史排行榜。
        self.all_criteria: list[Criterion] = deepcopy(self.current_criteria)

        self.n_criteria = n_criteria

        # prompt 模板允许外部覆写，这样下游任务无需改 workflow 逻辑，
        # 就能定制优化目标和表达方式。
        self.manager_prompt = (
            manager_prompt
            or local_prompts.MANAGER_PROMPT_TEMPLATE.format(n_criteria=self.n_criteria)
        )
        self.manager_prompt_postfix = (
            manager_prompt_postfix or local_prompts.MANAGER_PROMPT_POSTFIX
        )

        self.worker_prompt = worker_prompt
        self.evaluator_type = (evaluator_type or "auto").lower()
        self.evaluator_kwargs = evaluator_kwargs or {}
        # 默认保持原始 CritiQ 的纯文本 manager 行为。多模态任务可显式开启，
        # 让 warm-up 和错误案例 reflection 同时接收当前样本的图片。
        self.manager_multimodal = manager_multimodal
        # 默认在逐错误样例 reflection 中显式提供 Question。消融实验可以关闭
        # 这一项，同时保留 warm-up 的 Question 和 reflection 的图片输入。
        self.manager_reflection_include_question = (
            manager_reflection_include_question
        )
        # 预留的思维轨迹缓存。当前保存逻辑主要通过 `save(..., thought=...)` 传入，
        # 这个字段保留给未来需要在对象级别累积中间反思的场景。
        self.thoughts = []

    def _update_criteria(
        self,
        old: Sequence[Criterion],
        new: Sequence[Criterion],
        only_higher_score: bool = True,
    ) -> None:
        """将新的 criterion 列表原地合并到已有列表中。

        合并规则按名称去重。

        当 `only_higher_score=True` 时：
        - 如果新旧 criterion 同名，只保留 score 更高的那个版本；
        - 如果名称不存在，则直接加入。

        这个行为对 `all_criteria` 很关键，因为它的职责就是保留历史最优版本。

        注意：`old` 会被原地修改。调用方如果传入 `self.current_criteria` 或
        `self.all_criteria`，对应的 workflow 状态会立即发生变化。
        """
        if only_higher_score:
            _old = {c.name: c for c in deepcopy(old)}
            for i in deepcopy(new):
                if i.name in _old:
                    if i.score >= _old[i.name].score:
                        _old[i.name] = i
                else:
                    _old[i.name] = i
            old[:] = list(_old.values())
        else:
            old[:] = list(criteria_list_to_dict(deepcopy(old) + deepcopy(new)).values())

    def _get_evaluator_cls(self, dataset: Sequence[dict]):
        """根据 workflow 配置和数据格式选择 evaluator 类。

        默认 `evaluator_type="auto"` 时保持旧行为：根据 A/B pair 或 zero-one
        数据格式自动选择 evaluator。显式配置时会做数据格式校验，避免把
        multimodal_pair 用到 zero-one 数据上这类难排查错误。
        """
        evaluator_type = self.evaluator_type or "auto"
        if evaluator_type == "auto":
            return get_evaluator_cls_from_dataset(dataset)

        if evaluator_type not in EVALUATOR_REGISTRY:
            valid_types = ", ".join(["auto", *sorted(EVALUATOR_REGISTRY)])
            raise ValueError(
                f"Unknown evaluator_type={evaluator_type!r}. "
                f"Expected one of: {valid_types}"
            )

        if evaluator_type in ("pair", "multimodal_pair", "moe_multimodal_pair"):
            if not is_pair_dataset(dataset):
                raise ValueError(
                    f"evaluator_type={evaluator_type!r} requires pair data"
                )
        elif evaluator_type == "zero_one":
            if not is_zero_one_dataset(dataset):
                raise ValueError("evaluator_type='zero_one' requires zero-one data")

        return EVALUATOR_REGISTRY[evaluator_type]

    def _make_evaluator(self, dataset: Sequence[dict], max_retries: int = 3):
        """统一创建 worker evaluator，供初始化、优化和验证阶段复用。"""
        evaluator_cls = self._get_evaluator_cls(dataset)
        return evaluator_cls(
            worker_args=self.worker_args,
            dataset=dataset,
            max_concurrent=self.worker_max_concurrent,
            worker_prompt=self.worker_prompt,
            max_retries=max_retries,
            **self.evaluator_kwargs,
        )

    def _make_manager_content(
        self, prompt: str, data: dict[str, Any]
    ) -> str | list[dict[str, Any]]:
        """为 manager 构造纯文本或单样例多模态 user content。

        多模态模式复用 evaluator_kwargs 中的图片、问题字段和本地图片编码配置。
        这里只负责附加当前样本的一张图片；Question 由调用阶段写入文本 prompt。
        """
        if not self.manager_multimodal:
            return prompt

        image_field = self.evaluator_kwargs.get("image_field", "image_path")
        question_field = self.evaluator_kwargs.get("question_field", "question")
        encode_local_image = self.evaluator_kwargs.get("encode_local_image", True)

        image_value = data.get(image_field)
        question_value = data.get(question_field)
        if not isinstance(image_value, str) or not image_value.strip():
            raise ValueError(
                f"multimodal manager requires a non-empty {image_field!r}"
            )
        if not isinstance(question_value, str) or not question_value.strip():
            raise ValueError(
                f"multimodal manager requires a non-empty {question_field!r}"
            )

        image_url = (
            MultiModalPairEvaluator._image_path_to_data_url(image_value)
            if encode_local_image
            else image_value
        )
        return [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]

    @staticmethod
    def _revise_mid_criterion(
        manager: Agent,
        criterion_name: str,
        prompt: str,
        max_retries: int,
    ) -> str | None:
        """生成并校验一个 Mid Criterion，解析失败时重新生成。

        每次尝试都从原 manager 创建独立分支，避免失败响应进入下一次尝试的
        上下文。只有返回 JSON 中包含当前准则名，且描述为非空字符串时才成功。
        """
        max_retries = max(0, max_retries)
        for attempt in range(max_retries + 1):
            # Agent 自身负责网络/API 异常重试；这里仅针对已经返回的响应做
            # JSON 解析和字段校验重试，与 worker 的结构化输出重试保持一致。
            response = manager.fork()(prompt, stream=False)
            try:
                parsed = parse_json(response)
                revised_description = parsed[criterion_name]
                if (
                    not isinstance(revised_description, str)
                    or not revised_description.strip()
                ):
                    raise ValueError(
                        f"Revised description for {criterion_name!r} "
                        "must be a non-empty string"
                    )
                return revised_description.strip()
            except Exception as e:  # pylint: disable=W0718:broad-exception-caught
                if attempt < max_retries:
                    print_debug(
                        f"Failed to parse revised criterion {criterion_name}; "
                        f"retrying ({attempt + 1}/{max_retries})",
                        response,
                        e,
                    )
                else:
                    print_debug(
                        f"Failed to parse revised criterion {criterion_name} after "
                        f"{max_retries} retries; keeping the old criterion",
                        response,
                        e,
                    )
        return None

    @staticmethod
    def _warmup_zero_one(
        manager: Agent,
        dataset: Sequence[ZeroOneData],
        prompt_template: tuple[str, str] | None = None,
    ) -> None:
        """在生成 criterion 前，用带标签的二分类样本先给 manager 预热。

        这里的 warm-up 不直接产出结构化结果，目的只是把“高质量/低质量样本长什么样”
        先灌进 manager 的对话历史，让它后面生成的 criterion 更贴近当前任务。
        """
        if prompt_template is None:
            prompt_template = (
                local_prompts.WARMUP_ZERO_ONE_PROMPT_TEMPLATE_0,
                local_prompts.WARMUP_ZERO_ONE_PROMPT_TEMPLATE_1,
            )
        assert "{text}" in prompt_template[0] and "{text}" in prompt_template[1]

        prompts = []
        for data in dataset:
            prompts.append(
                prompt_template[data["label"]].replace("{text}", data["text"])
            )
        random.shuffle(prompts)

        for prompt in tqdm(
            prompts,
            desc="Warming up",
            disable=not USE_TQDM,
        ):
            manager(prompt, stream=False)

    def _warmup_pair(
        self,
        manager: Agent,
        dataset: Sequence[PairData],
        prompt_template: tuple[str, str] | None = None,
    ) -> None:
        """在生成 criterion 前，用带标签的成对样本先给 manager 预热。

        相比 zero-one warm-up，这里输入的是“为什么 A 比 B 好”这类偏好解释。
        它更贴近 pairwise 任务，尤其适合代码比较、回答比较、偏好排序这类场景。
        """
        if prompt_template is None:
            prompt_template = (
                local_prompts.WARMUP_PAIR_PROMPT_TEMPLATE_AB,
                local_prompts.WARMUP_PAIR_PROMPT_TEMPLATE_BA,
            )
        assert all(["{A}" in p and "{B}" in p for p in prompt_template])

        prompts = []
        for data in dataset:
            question_field = (
                self.evaluator_kwargs.get("question_field", "question")
                if self.manager_multimodal
                else "question"
            )
            prompt = (
                prompt_template[0 if data["answer"] == "A" else 1]
                .replace("{A}", data["A"])
                .replace("{B}", data["B"])
            )
            prompt = prompt.replace(
                "{question}", str(data.get(question_field, ""))
            )
            # image_path 不作为文本暴露；多模态模式会把图片作为 image_url content
            # 单独发送给 manager。
            prompt = prompt.replace("{image_path}", "")
            # 保留 prompt 与原始样本的对应关系，多模态模式下需要从当前样本
            # 读取 image_path；文本模式下仍然只发送原始 prompt。
            prompts.append((prompt, data))
        random.shuffle(prompts)

        for prompt, data in tqdm(
            prompts,
            desc="Warming up",
            disable=not USE_TQDM,
        ):
            manager(self._make_manager_content(prompt, data), stream=False)

    def get_init_criteria(
        self,
        dataset: Sequence[ZeroOneData] | Sequence[PairData],
        prompt_template: tuple[str, str] | None = None,
        knowledge_base: Sequence[Criterion] | None = None,
        max_retries: int = 3,
        n_shot: int = -1,
        max_retrived: int = None,
        retrieval_threashold: float = 0.5,
    ) -> None:
        """通过“检索 + 新生成”的方式构建初始 criterion 集合。

        初始化分两段：

        1. retrieval
           如果给了知识库，先把知识库里的 criterion 放到当前数据上粗评一遍，
           选出分数高、和当前任务相匹配的那部分。

        2. generation
           如果检索出的 criterion 数量不够，再让 manager 在 warm-up 之后补齐。

        这样做的动机是：
        - 知识库能提供强先验，降低冷启动成本；
        - 新生成能补足知识库的覆盖盲区，避免完全受限于已有标准。

        参数里几个容易混淆的点：
        - `knowledge_base` 不是直接并入，而是先在当前数据集上评估，再按分数筛选；
        - `n_shot` 只影响 manager warm-up 用多少条样本，不改变 evaluator 对知识库的评估数据；
        - `max_retrived` 限制最多从知识库拿多少条，剩余缺口由 manager 生成补齐；
        - `retrieval_threashold` 是知识库 criterion 的最低通过分数。
        """
        init_criteria = []
        if knowledge_base is not None:
            # 最多取多少条知识库 criterion 进入初始集合。
            n_to_retrieve = (
                min(max_retrived, self.n_criteria)
                if max_retrived is not None
                else self.n_criteria
            )
            knowledge_base = deepcopy(knowledge_base)

            evaluator = self._make_evaluator(dataset, max_retries=max_retries)

            # 这里直接复用 evaluator 给知识库标准打分，而不是让 manager 主观挑选。
            # 好处是：是否适用当前任务，由“在数据上表现如何”决定，而不是由另一层 prompt 判断。
            evaluator.eval(knowledge_base, update_score=True)
            knowledge_base = list(
                filter(lambda x: x.score > retrieval_threashold, knowledge_base)
            )
            knowledge_base = sorted(knowledge_base, key=lambda x: x.score, reverse=True)

            # 取前 n_to_retrieve 个 criterion。遇到同分时随机抽样，
            # 避免多次运行总是偏向知识库中的原始顺序。
            buffer = []
            for c in knowledge_base:
                if len(buffer) == 0:
                    buffer.append(c)
                elif c.score == buffer[0].score:
                    buffer.append(c)
                elif len(init_criteria) + len(buffer) < n_to_retrieve:
                    init_criteria.extend(buffer)
                    buffer = [c]
                else:
                    break
            random.shuffle(buffer)
            init_criteria.extend(buffer[: n_to_retrieve - len(init_criteria)])
            print(f"Retrieved {len(init_criteria)} criteria from knowledge base.")

        num_new_criteria = self.n_criteria - len(init_criteria)
        if num_new_criteria > 0:
            print(f"Generating {num_new_criteria} new criteria")
            manager = Agent(**self.manager_args)

            if len(dataset) <= 0:
                raise ValueError("Empty dataset")
            if n_shot > 0:
                # 用少量样本预热 manager，成本更低，也能减小提示长度。
                assert n_shot <= len(dataset)
                dataset = random.sample(dataset, n_shot)
            if is_zero_one_dataset(dataset):  # ZeroOneData
                self._warmup_zero_one(manager, dataset, prompt_template)
            elif is_pair_dataset(dataset):  # PairData
                # 随机翻转 pair 顺序，降低 “A 往往更好” 这类伪偏置。
                random_reversed_dataset = random_reverse(dataset)
                self._warmup_pair(manager, random_reversed_dataset, prompt_template)
            else:
                raise ValueError("Invalid dataset format")

            try:
                # 通过把 “生成多少条 criterion” 写进 prompt，
                # manager 会输出一个 JSON: {criterion_name: description, ...}
                prompt = self.manager_prompt.replace(
                    str(self.n_criteria), str(num_new_criteria)
                )
                if len(init_criteria) > 0:
                    # 已检索出的 criterion 会显式塞回 prompt，避免 manager 生成重复标准。
                    prompt += (
                        "\n\nThe new criteria should be different from the following:\n"
                    )
                    prompt += "\n".join(
                        [f"{c.name}: {c.description}" for c in init_criteria]
                    )
                prompt += self.manager_prompt_postfix
                response = manager(prompt, stream=False)
                print(response)
                init_criteria += [
                    Criterion(name=name, description=description, score=0.0)
                    for name, description in parse_json(response).items()
                ]
            except Exception as e:
                raise ValueError("Failed to parse initial criteria") from e

        # 最终截断到 n_criteria，避免 retrieval + generation 总数超出目标。
        init_criteria = init_criteria[: self.n_criteria]
        self._update_criteria(self.current_criteria, init_criteria)
        self._update_criteria(self.all_criteria, init_criteria)

    def _optimize_loop_pair_data(
        self,
        train_set: Sequence[PairData],
        threshold: tuple[float, float],
        max_retries: int = 3,
    ):
        """在成对数据集上执行一轮 criterion 优化。

        单轮 pairwise 优化的主流程：

        1. 在 train_set 上评估当前 criterion 集合；
        2. 按阈值分桶为 good / mid / low；
        3. 保留 good；
        4. 对 mid 做基于错误案例的反思与改写；
        5. 对 low 直接淘汰，并生成等量新 criterion；
        6. 用新集合替换 `current_criteria`，供下一轮继续优化。
        """
        assert 0 <= threshold[0] < threshold[1] <= 1
        manager = Agent(**self.manager_args)

        # 先把当前 criterion 集合放进 manager 历史中，后续改写时就不用每次
        # 都重新解释完整任务背景。
        manager.history = [
            {
                "role": "user",
                "content": self.manager_prompt + self.manager_prompt_postfix,
            },
            {
                "role": "assistant",
                "content": f"```json\n{json.dumps({c.name: c.description for c in self.current_criteria}, indent=4, ensure_ascii=False)}\n```",
            },
        ]
        evaluator = self._make_evaluator(train_set, max_retries=max_retries)

        # `update_score=True` 会把每条 criterion 的 score 更新成其在 train_set 上的准确率。
        eval_output = evaluator.eval(self.current_criteria, update_score=True)
        self._update_criteria(self.all_criteria, self.current_criteria)
        print("Train:", eval_output.accuracy, eval_output.is_correct)

        criteria = criteria_list_to_dict(self.current_criteria)
        prompt = local_prompts.ACCURACY_PROMPT + "\n\n"
        good_criteria: dict[str, Criterion] = {}
        mid_criteria: dict[str, Criterion] = {}
        low_criteria: list[str] = []

        acc_table = PrettyTable()
        acc_table.field_names = ["Criterion", "Type", "Accuracy"]
        for criterion_name in sorted(eval_output.per_criterion_acc):
            acc = eval_output.per_criterion_acc[criterion_name]
            prompt += f"{criterion_name}: {acc}\n"

            # 高于上阈值：说明这个 criterion 已经足够稳定，直接保留。
            if acc >= threshold[1]:
                good_criteria[criterion_name] = criteria[criterion_name].description
                acc_table.add_row([criterion_name, "Good", acc])

            # 落在中间：说明方向对，但描述还不够好，值得基于错误案例细化。
            elif acc > threshold[0]:
                mid_criteria[criterion_name] = criteria[criterion_name].description
                acc_table.add_row([criterion_name, "Mid", acc])

            # 低于下阈值：视为弱标准，后续直接淘汰。
            else:
                low_criteria.append(criterion_name)
                acc_table.add_row([criterion_name, "Low", acc])
        print(acc_table)

        pred_result = eval_output.prediction
        thoughts = eval_output.thoughts
        # `new_criteria` 是下一轮要使用的完整集合，最后会重建成 Criterion 对象。
        # good 会原样放入，mid 会被改写后放入，low 会被新生成的 criterion 替换。
        new_criteria = {}
        if len(good_criteria) > 0:
            prompt += "\n\n"
            prompt += local_prompts.GOOD_CRITERIA_PROMPT_TEMPLATE.format(
                criteria=", ".join(good_criteria.keys()), threshold=threshold[1]
            )

            # good criterion 直接进入下一轮，不再改写。
            new_criteria.update(good_criteria)
            print("\n\n#### Good")
            print(", ".join(good_criteria.keys()))

        if len(mid_criteria) > 0:
            print("\n\n#### Mid")
            mid_table = PrettyTable()
            mid_table.field_names = ["Criterion", "Old", "New"]
            mid_table.align["Old"] = "l"
            mid_table.align["New"] = "l"
            mid_table.max_width["Old"] = 50
            mid_table.max_width["New"] = 50

            def _optimize_get_critic(
                criterion_name,
                data,
                answer,
                wrong_answer,
                thought,
                manager_for_critique,
            ):
                """让 manager 分析一个具体失败案例。

                输入是一条“该 criterion 判断错了”的 pair 数据，
                输出是一段 critique，解释这个 criterion 描述为什么会误导 worker。
                这一步不是直接重写 criterion，而是先收集失败模式。
                """
                # Var `prompt`, `mid_criteria` and `local_prompts` are captured from context
                prompt_parts = [
                    prompt,
                    local_prompts.MID_CRITERIA_PROMPT_TEMPLATE.format(
                        criterion_name=criterion_name,
                        threshold_0=threshold[0],
                        threshold_1=threshold[1],
                    ),
                    local_prompts.CRITERION_NAME_DESC_FORMAT_TEMPLAT.format(
                        name=criterion_name, desc=mid_criteria[criterion_name]
                    ),
                    local_prompts.MID_CRITIQUE_PROMPT,
                ]
                if (
                    self.manager_multimodal
                    and self.manager_reflection_include_question
                ):
                    question_field = self.evaluator_kwargs.get(
                        "question_field", "question"
                    )
                    question = data.get(question_field, "")
                    prompt_parts.append(
                        f"[BEGIN_OF_QUESTION]\n{question}\n[/END_OF_QUESTION]"
                    )
                prompt_parts.extend(
                    (
                        local_prompts.MID_A_PROMPT_TEMPLATE.format(data["A"]),
                        local_prompts.MID_B_PROMPT_TEMPLATE.format(data["B"]),
                        local_prompts.MID_HOWEVER_PROMPT_TEMPLATE.format(
                            wrong=wrong_answer,
                            correct=answer,
                            thought=thought,
                        ),
                        local_prompts.MID_REFLECTION_PROMPT,
                    )
                )
                prompt_for_critique = "\n\n".join(prompt_parts)
                try:
                    response = manager_for_critique(
                        self._make_manager_content(prompt_for_critique, data),
                        stream=False,
                    )
                    return parse_json(response)["critique"]
                except Exception as e:
                    print_debug("Failed to parse critique", e)
                    return None

            with ThreadPoolExecutor(
                max_workers=MANAGER_MAX_CONCURRENT  # TODO: by config instead of env var
            ) as executor:
                # futures 的结构是：
                #   criterion_name -> [每个失败样本对应的 critique future]
                # 这样做能让不同 criterion 的反思结果分开收集，后面分别汇总改写。
                futures = {}
                for criterion_name in mid_criteria:
                    f_c = []
                    for idx, data in enumerate(train_set):
                        stat = pred_result[idx][criterion_name]
                        thought = thoughts[idx][criterion_name]
                        answer = data["answer"]
                        wrong_answer = reverse_ab(data["answer"])
                        # 只对错误案例做 reflection。正确案例说明该 criterion 至少在这个样本上可用，
                        # 对改写的帮助通常不如失败案例直接。
                        if stat[wrong_answer] > stat[answer]: # 如果stat[wrong_answer]=1，stat[answer]=0，则这是一个错误样本，不考虑None的情况
                            # 每个失败案例都在 manager 的独立 fork 分支里分析，
                            # 避免多个错误案例相互污染上下文。
                            f_c.append(
                                executor.submit(
                                    _optimize_get_critic,
                                    criterion_name,
                                    data,
                                    answer,
                                    wrong_answer,
                                    thought,
                                    manager.fork(),
                                )
                            )
                    futures[criterion_name] = f_c

                all_futures = sum(futures.values(), [])
                # 第一段并发只负责“找问题”：把每个失败样本转成一句 critique。
                # tqdm 包住 as_completed，是为了让长批量 LLM 调用时能看到反思进度。
                for _ in tqdm(
                    as_completed(all_futures),
                    desc="Reflection",
                    total=len(all_futures),
                    disable=not USE_TQDM,
                ):
                    pass
                critiques = {
                    c: [f.result() for f in futures_c]
                    for c, futures_c in futures.items()
                }

                # 第二段并发负责“改标准”：每个 mid criterion 汇总自己的全部 critique，
                # 让 manager 生成一个更清晰、更能区分偏好的新描述。
                revision_futures = {}
                for criterion_name in mid_criteria:
                    # 将多个具体错误案例的 critique 汇总，生成同一 criterion 的
                    # 一个统一改写版本。
                    critique_prompt = "\n".join(
                        f"{i}: {c}" for i, c in enumerate(critiques[criterion_name])
                    )
                    _prompt = "\n\n".join(
                        (
                            prompt,
                            local_prompts.MID_CRITERIA_PROMPT_TEMPLATE.format(
                                criterion_name=criterion_name,
                                threshold_0=threshold[0],
                                threshold_1=threshold[1],
                            ),
                            local_prompts.CRITERION_NAME_DESC_FORMAT_TEMPLAT.format(
                                name=criterion_name, desc=mid_criteria[criterion_name]
                            ),
                            local_prompts.MID_REFINE_PROMPT_TEMPLATE.format(
                                critique_prompt
                            ),
                            local_prompts.MID_FORMAT_PROMPT_TEMPLATE.format(
                                criterion_name=criterion_name
                            ),
                        )
                    )
                    # future 与 criterion_name 显式绑定，不能依赖并发任务的完成顺序。
                    # 单个任务内部会为每次生成创建独立 fork，并在解析失败时重试。
                    future = executor.submit(
                        self._revise_mid_criterion,
                        manager,
                        criterion_name,
                        _prompt,
                        max_retries,
                    )
                    revision_futures[future] = criterion_name

                # 按实际完成顺序收集任务，但通过映射找回各自的 criterion_name。
                revision_results: dict[str, str | None] = {}
                for future in tqdm(
                    as_completed(revision_futures),
                    desc="Optimization",
                    total=len(revision_futures),
                    disable=not USE_TQDM,
                ):
                    criterion_name = revision_futures[future]
                    old_description = mid_criteria[criterion_name]
                    try:
                        revised_description = future.result()
                    except Exception as e:  # 防御 future 中未预期的异常
                        print_debug(
                            f"Failed to revise criterion {criterion_name}; "
                            "keeping the old criterion",
                            e,
                        )
                        revised_description = None

                    revision_results[criterion_name] = revised_description

                # 按原始 Mid Criterion 顺序写回，避免并发完成顺序改变准则排列。
                # 重试仍失败时保留旧描述，保证 Mid Criterion 不会凭空消失。
                for criterion_name, old_description in mid_criteria.items():
                    revised_description = revision_results[criterion_name]
                    if revised_description is None:
                        revised_description = old_description
                    new_criteria[criterion_name] = revised_description
                    mid_table.add_row(
                        [criterion_name, old_description, revised_description]
                    )

            print(mid_table)

        if len(low_criteria) > 0:
            print("\n#### Low")
            low_table = PrettyTable()
            low_table.field_names = ["Criterion", "Description"]
            low_table.align["Description"] = "l"
            low_table.max_width["Description"] = 100
            print(", ".join(low_criteria))

            # 进入 banned_criteria 后，后面 prompt 会明确要求 manager 不要再生成这些名字。
            self.banned_criteria.update(low_criteria)
            if len(low_criteria) != self.n_criteria - len(new_criteria):
                warnings.warn(
                    f"Num of low_criteria({len(low_criteria)}) != n_criteria - #new_criteria ({self.n_criteria - len(new_criteria)})"
                )
            prompt += (
                "\n\n"
                + local_prompts.LOW_PROMPT_TEMPLATE.format(
                    criteria=", ".join(self.banned_criteria),
                    threshold_0=threshold[0],
                    num=len(low_criteria),
                )
                + "\n\n"
                + local_prompts.LOW_FORMAT_PROMPT
            )
            response = manager(prompt, stream=False)
            if response is not None:
                try:
                    _new = parse_json(response)
                    new_criteria.update(_new)
                    for name, desc in _new.items():
                        low_table.add_row([name, desc])
                except Exception as e:
                    print_debug("Failed to parse new criteria", e)
            print(low_table)

        # 注意：这里会重建 current_criteria，而不是原地修改原对象。
        # 因而中间轮次保存的 current_criteria 可能 score 还是默认值 0；
        # 历史最佳分数真正保存在 all_criteria 里，最终还会再统一打一次分。
        self.current_criteria = [
            Criterion.from_dict({"name": name, "description": desc})
            for name, desc in new_criteria.items()
        ]

    def _optimize_loop_zero_one_data(
        self,
        train_set: Sequence[ZeroOneData],
        threshold: tuple[float, float],
        max_retries: int = 3,
    ):
        """在二分类数据集上执行一轮 criterion 优化。

        整体思想与 pair 版本一致，只是错误案例的组织形式不同：
        pair 任务用 A/B 错判来反思，zero-one 任务用高/低质量误判来反思。
        """
        assert 0 <= threshold[0] < threshold[1] <= 1
        manager = Agent(**self.manager_args)

        manager.history = [
            {
                "role": "user",
                "content": self.manager_prompt + self.manager_prompt_postfix,
            },
            {
                "role": "assistant",
                "content": f"```json\n{json.dumps({c.name: c.description for c in self.current_criteria}, indent=4, ensure_ascii=False)}\n```",
            },
        ]
        evaluator = self._make_evaluator(train_set, max_retries=max_retries)

        eval_output = evaluator.eval(self.current_criteria, update_score=True)
        self._update_criteria(self.all_criteria, self.current_criteria)
        print("Train:", eval_output.accuracy, eval_output.is_correct)

        criteria = criteria_list_to_dict(self.current_criteria)
        prompt = local_prompts.ACCURACY_PROMPT + "\n\n"
        good_criteria: dict[str, Criterion] = {}
        mid_criteria: dict[str, Criterion] = {}
        low_criteria: list[str] = []
        # zero-one 版本没有 PrettyTable 汇总，但分桶逻辑和 pair 版本相同：
        # 高分保留，中分改写，低分淘汰。
        for criterion_name in sorted(eval_output.per_criterion_acc):
            acc = eval_output.per_criterion_acc[criterion_name]
            print(f"{criterion_name}:\t{acc}")
            prompt += f"{criterion_name}: {acc}\n"
            if acc >= threshold[1]:
                good_criteria[criterion_name] = criteria[criterion_name].description
            elif acc > threshold[0]:
                mid_criteria[criterion_name] = criteria[criterion_name].description
            else:
                low_criteria.append(criterion_name)
        prompt += "\n"

        pred_result = eval_output.prediction
        # 和 pair 版本一样，new_criteria 表示下一轮完整候选池。
        # 这里保存的是 name -> description，最后统一转换成 Criterion。
        new_criteria = {}
        if len(good_criteria) > 0:
            prompt += local_prompts.GOOD_CRITERIA_PROMPT_TEMPLATE.format(
                criteria=", ".join(good_criteria.keys()), threshold=threshold[1]
            )
            new_criteria.update(good_criteria)
            print("\n===== Good")
            print(", ".join(good_criteria.keys()))

        if len(mid_criteria) > 0:
            print("\n===== Mid")
            for criterion_name in mid_criteria:
                # 每个中等质量 criterion 都使用独立分支改写，避免不同 criterion
                # 的 critique 上下文互相污染。
                _manager = manager.fork()
                _prompt = prompt[:]
                _prompt += (
                    local_prompts.MID_CRITERIA_PROMPT_TEMPLATE.format(
                        criterion_name=criterion_name,
                        threshold_0=threshold[0],
                        threshold_1=threshold[1],
                    )
                    + "\n\n"
                )
                for idx, data in enumerate(train_set):
                    stat = pred_result[idx][criterion_name]
                    if stat[0] != stat[1]:
                        # evaluator 对二分类样本会返回该 criterion 对 0/1 两类的支持强度。
                        # 两边不一致时，取更高的一边作为 worker 的实际判断；
                        # 如果实际判断和数据标签冲突，就把该样本加入改写 prompt。
                        answer = 1 if stat[1] > stat[0] else 0
                        _prompt += (
                            local_prompts.MID_01_PROMPT_TEMPLATE.format(
                                text=data["text"]
                            )
                            + "\n\n"
                            + (
                                local_prompts.MID_0_FOR_1_PROMPT
                                if answer == 0
                                else local_prompts.MID_1_FOR_0_PROMPT
                            )
                    )

                _prompt += (
                    "\n\n"
                    + local_prompts.MID_REFINE_PROMPT_TEMPLATE
                    + "\n\n"
                    + local_prompts.MID_FORMAT_PROMPT_TEMPLATE.format(
                        criterion_name=criterion_name
                    )
                )

                response = _manager(_prompt, stream=False)
                if response is not None:
                    try:
                        _new = parse_json(response)
                        new_criteria.update(_new)
                        print(criterion_name, "\t->\t", _new[criterion_name])
                    except Exception as e:
                        print_debug("Failed to parse new criteria", e)

        if len(low_criteria) > 0:
            print("\n===== Low")
            print(", ".join(low_criteria))
            # 低质量 criterion 的名字会进入 ban list。下一次补新时，manager 会被提示
            # 避免复用这些名字，从而减少“换汤不换药”的循环。
            self.banned_criteria.update(low_criteria)
            prompt += (
                local_prompts.LOW_PROMPT_TEMPLATE.format(
                    criteria=", ".join(self.banned_criteria),
                    threshold_0=threshold[0],
                    num=self.n_criteria - len(new_criteria),
                )
                + "\n\n"
                + local_prompts.LOW_FORMAT_PROMPT
            )
            response = manager(prompt, stream=False)
            if response is not None:
                prompt = ""
                try:
                    _new = parse_json(response)
                    new_criteria.update(parse_json(response))
                    print("NEW:")
                    print(json.dumps(_new, indent=4, ensure_ascii=False))
                except Exception as e:
                    print_debug("Failed to parse new criteria", e)

        self.current_criteria = [
            Criterion.from_dict({"name": name, "description": desc})
            for name, desc in new_criteria.items()
        ]

    def optimize(
        self,
        train_set: Sequence[PairData | ZeroOneData],
        valid_set: Sequence[PairData | ZeroOneData] | None = None,
        output_dir: str | None = None,
        num_epochs: int = 1,
        threshold: tuple[float, float] = (0.5, 0.75),
        save_thought: bool = False,
        max_retries: int = 3,
    ) -> None:
        """运行完整的多轮 criterion 优化循环。

        这是 workflow 暴露给 demo / 脚本层的主入口。

        典型调用顺序是：
        1. 先调用 `get_init_criteria()` 建立起始标准集；
        2. 再调用 `optimize()` 跑若干轮演化；
        3. 最后用 `get_best_criteria()` 从历史里取出最强标准。

        参数说明里最重要的是 `threshold=(low, high)`：
        - `acc >= high`    -> good，直接保留
        - `low < acc < high` -> mid，反思后重写
        - `acc <= low`     -> low，删除并补新

        `valid_set` 只用于观察泛化效果，不会更新 criterion 的训练分数。
        `output_dir` 如果提供，会保存每一轮 checkpoint，方便中断后检查或复现实验。
        `save_thought` 目前保留为接口参数，实际保存思维轨迹的代码仍在下方以注释形式保留。
        """
        do_valid = valid_set is not None and len(valid_set) != 0
        if len(self.current_criteria) <= 0:
            raise ValueError(
                "No initial criteria, please give some or call `get_init_criteria`"
            )

        if output_dir is None:
            warnings.warn("`output_dir` is not set. Results will not be saved.")
        elif not os.path.exists(output_dir):
            os.mkdir(output_dir)

        if not is_zero_one_dataset(train_set) and not is_pair_dataset(train_set):
            raise ValueError("Invalid dataset format")
        if do_valid:
            valid_evaluator = self._make_evaluator(
                valid_set, max_retries=max_retries
            )

        if is_pair_dataset(train_set):
            # 在训练阶段评估前先打乱并随机翻转 pair 顺序。
            train_set = random_reverse(train_set)
        for epoch in range(num_epochs):
            print(f"## Iteration {epoch}")
            # In _optimize_loop:
            # 1. Evaluate `current_criteria` on the training set.
            # 2. update `all_critieria` by `current_criteria` if the
            #    score is higher.
            # 3. Optimize the criteria by the scores, `current_criteria` will
            #    be updated. New `current_criteria` will be used in the next iter.
            #
            # After this loop, you'll get a new **unmerged** `current_criteria`
            # and an updated `all_criteria`.
            if is_pair_dataset(train_set):
                self._optimize_loop_pair_data(
                    train_set, threshold, max_retries=max_retries
                )
            elif is_zero_one_dataset(train_set):
                self._optimize_loop_zero_one_data(
                    train_set, threshold, max_retries=max_retries
                )
            else:
                raise ValueError("Invalid trainset format")
            if do_valid:
                # valid 集仅做观测，不更新 criterion.score。
                eval_output = valid_evaluator.eval(
                    self.current_criteria, update_score=False
                )
                print(eval_output)
            # thoughts = eval_output.thoughts if save_thought else None
            if output_dir is not None:
                self.save(output_dir, epoch=epoch, thought=None)

        # The scores of final criteria after optimization is not updated on the
        # training set while not merged to `all_critieria`. So we need to update
        # the scores and update the `all_criteria` here.
        #
        # 也就是说，循环内部保存出来的 `epoch_i.json` 更像“这一轮生成出了哪些标准”，
        # 而最后这一步才负责给最终 current_criteria 补上正式 score 并合并进 all_criteria。
        evaluator = self._make_evaluator(train_set, max_retries=max_retries)
        eval_output = evaluator.eval(self.current_criteria, update_score=True)
        self._update_criteria(self.all_criteria, self.current_criteria)
        print("Final Train Acc:", eval_output.accuracy, eval_output.is_correct)
        if output_dir is not None:
            self.save(output_dir, epoch="final", thought=None)

    def get_best_criteria(self, threshold: int = 0.75) -> list[Criterion]:
        """返回到当前为止发现的最强 criterion。

        注意这里筛选的是 `all_criteria`，不是 `current_criteria`。
        因此返回的是“历史最佳版本”，而不是“最后一轮正在使用的版本”。
        """
        return [c for c in self.all_criteria if c.score >= threshold]

    def get_state_dict(self) -> dict[str, Any]:
        """将 workflow 状态序列化，用于保存 checkpoint。

        这里导出的不仅是 criterion 列表，也包括 manager/worker 配置和 prompt 模板。
        所以一个 checkpoint 可以近似看作“完整复现实验状态”的快照。
        """
        return {
            "manager_args": self.manager_args,
            "worker_args": self.worker_args,
            "worker_max_concurrent": self.worker_max_concurrent,
            "current_criteria": [c.to_dict() for c in self.current_criteria],
            "banned_criteria": list(self.banned_criteria),
            "all_criteria": [c.to_dict() for c in self.all_criteria],
            "n_criteria": self.n_criteria,
            "manager_prompt": self.manager_prompt,
            "manager_prompt_postfix": self.manager_prompt_postfix,
            "worker_prompt": self.worker_prompt,
            "evaluator_type": self.evaluator_type,
            "evaluator_kwargs": self.evaluator_kwargs,
            "manager_multimodal": self.manager_multimodal,
            "manager_reflection_include_question": (
                self.manager_reflection_include_question
            ),
        }

    def load_state_dict(self, state: dict[str, Any]):
        """用 checkpoint 中的内容部分覆盖当前 workflow 配置。

        加载策略是：
        - 先取当前默认状态；
        - 再用传入 state 覆盖。

        这样即便 checkpoint 缺少某些较新的字段，也能回退到当前默认值。
        """
        current_state = self.get_state_dict()
        current_state.update(state)
        self.manager_args = current_state["manager_args"]
        self.worker_args = current_state["worker_args"]
        self.worker_max_concurrent = current_state["worker_max_concurrent"]

        self.current_criteria = [
            Criterion.from_dict(c) for c in current_state["current_criteria"]
        ]
        self.banned_criteria.update(current_state["banned_criteria"])
        self.all_criteria = [
            Criterion.from_dict(c) for c in current_state["all_criteria"]
        ]

        self.n_criteria = current_state["n_criteria"]
        self.manager_prompt = current_state["manager_prompt"]
        self.manager_prompt_postfix = current_state["manager_prompt_postfix"]
        self.worker_prompt = current_state["worker_prompt"]
        self.evaluator_type = (current_state["evaluator_type"] or "auto").lower()
        self.evaluator_kwargs = current_state["evaluator_kwargs"] or {}
        self.manager_multimodal = current_state["manager_multimodal"]
        self.manager_reflection_include_question = current_state[
            "manager_reflection_include_question"
        ]

    def save(self, path, epoch, thought) -> None:
        """保存 workflow 状态，以及可选的思维轨迹。

        默认主文件名是 `epoch_{epoch}.json`。
        如果提供了 `thought`，还会额外保存 `thought_{epoch}.json`。
        """
        with open(os.path.join(path, f"epoch_{epoch}.json"), "w", encoding="utf8") as f:
            json.dump(
                self.get_state_dict(),
                f,
                ensure_ascii=False,
                indent=4,
            )
        if thought:
            with open(
                os.path.join(path, f"thought_{epoch}.json"), "w", encoding="utf8"
            ) as f:
                json.dump(
                    thought,
                    f,
                    ensure_ascii=False,
                    indent=4,
                )

    def load(self, path: str) -> None:
        """从 JSON checkpoint 文件中加载 workflow 状态。"""
        with open(path, "r", encoding="utf8") as f:
            state = json.load(f)
        self.load_state_dict(state)
