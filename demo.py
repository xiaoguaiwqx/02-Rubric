"""在代码质量任务上运行 CritiQ 的最小端到端示例。

这个脚本不是库源码中的“核心算法实现”，而是一个把多个组件串起来的示例入口。
它展示了 CritiQ 在 pairwise 代码质量比较任务上的完整工作流：

1. 读取原始数据，并转换成 A/B 成对比较格式；
2. 配置 manager 与 worker 两类模型；
3. 用知识库 + warm-up 生成初始 criteria；
4. 在 train / valid 上循环评估并优化 criteria；
5. 导出每一轮结果，并查看最后保留下来的高分 criteria。

如果你第一次读这个项目，可以把它当成“Workflow / Evaluator / Agent 三者如何协作”
的总控脚本。
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import datasets
from critiq import (
    Agent,
    Criterion,
    PairEvaluator,
    Workflow,
    load_criteria_from_json,
    print_score_changes,
    zero_one_dataset_to_pair_dataset,
)
from tqdm import tqdm


def load_local_env(path: str = ".env") -> None:
    """Load simple KEY=VALUE pairs from a local .env file if present."""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


load_local_env()


def parse_api_keys(list_var: str, single_var: str | None = None) -> list[str]:
    """Parse comma-separated API keys from env, with optional single-key fallback."""
    raw = (os.getenv(list_var) or "").strip()
    keys = [k.strip() for k in raw.split(",") if k.strip()] if raw else []
    if not keys and single_var:
        single = (os.getenv(single_var) or "").strip()
        if single:
            keys = [single]
    return keys or ["EMPTY_KEY"]

TASK_NAME = "code"

# manager 负责“提出/改写 criterion”，通常用更强的模型。
# 这里优先从环境变量读取 API key；如果环境变量不存在，就退回到代码中的默认值。
# 从工程实践看，真实使用时更推荐只保留环境变量方案，而不要把 key 写死在源码里。
OPENAI_API_KEYS = [os.getenv("OPENAI_API_KEY", "EMPTY_KEY")]

# 最大并发数同时控制两类事情：
# 1. 过滤知识库时，同时发起多少个“这个 criterion 是否适用于代码质量”的请求；
# 2. evaluator 在验证集上并发多少个 worker 请求。
# 这里默认保守一些，避免首次运行时就被远端 API 限流。
MAX_CONCURRENT = int(os.getenv("CRITIQ_MAX_CONCURRENT", "20"))

OUTPUT_DIR = f"./output/{TASK_NAME}"

# worker prompt 定义了“单个 criterion 下，如何比较两个候选代码文件”。
# PairEvaluator 会把其中的占位符替换成：
# - {criterion}: criterion 名称
# - {description}: criterion 描述
# - {A}/{B}: 两段待比较的代码
WORKER_PROMPT = "## Instruction\nGiven criterion **{criterion}**, compare two Python code files and determine which one human annotators will consider to be of higher quality.\n\n## A\n{A}\n\n## B\n{B}\n\n# Criterion\n**{criterion}**: {description}"

# manager 要生成多少条 criterion。
N_CRITERIA = 30

# manager prompt 的职责是告诉强模型：
# “请给出人类比较两段 Python 代码整体质量时会用到的标准列表。”
MANAGER_PROMPT = f"List and describe {N_CRITERIA} criteria on how human compare the overall quality of two Python code files."

# warm-up 阶段不会直接让 manager 输出 criterion，
# 而是先给它若干有答案的 A/B 样本，让它“感受”人类为什么偏好 A 或 B。
# 这样后续生成 criterion 时，通常会更贴近具体任务。
WARMUP_PROMPT = (
    "There are two python code files.\n\n## A\n{A}\n\n## B\n{B}\n\nHuman annotators are asked to compare the quality of A and B. They report that A is better than B. Please explain why they think A is better than B.",
    "There are two python code files.\n\n## A\n{A}\n\n## B\n{B}\n\nHuman annotators are asked to compare the quality of A and B. They report that B is better than A. Please explain why they think B is better than A.",
)

# worker 是真正跑在每条样本上的“执行者”：
# 给它一个 criterion、两段代码和 prompt，它需要判断 A/B 哪个更符合该 criterion。
WORKER_ARGS = {
    "model": "Qwen/Qwen2.5-72B-Instruct",
    # "model": "Qwen/Qwen3.5-122B-A10B",   # 比2.5-72B 便宜一半，默认开启思考模式
    # OpenAI SDK 需要的是 API 根路径，而不是具体的 /chat/completions 路径。
    "base_url": os.getenv("CRITIQ_WORKER_BASE_URL", "https://api.siliconflow.cn/v1"),
    # 优先读取 GUIJI_API_KEYS（逗号分隔），没有时回退到 GUIJI_API_KEY。
    "api_keys": parse_api_keys("GUIJI_API_KEYS", "GUIJI_API_KEY"),
    "request_kwargs": {
        "temperature": 0.5,
        # "enable_thinking": False,
    },
}

NUM_EPOCHS = 3
MAX_RETRIES = 3
SEED = 196705814

##################################################################################################

# 下面开始进入“脚本执行区”。
# 这部分代码的主要作用是把前面定义好的配置项拼装起来，驱动整个 workflow 跑通。

# Agent 在写日志时会读取这个环境变量。
# 所有 manager / worker 的原始请求与回复都会被记录到输出目录，便于回溯问题。
os.environ["WORKFLOW_AGENT_LOGFILE"] = OUTPUT_DIR + "/workflow_agent.log"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 这个 demo 假设原始数据先以 zero-one 形式落盘在 `./data/code`：
# 每条样本通常是“一个代码片段 + 一个 0/1 标签”。
# 但当前示例走的是 PairEvaluator / pairwise workflow，
# 所以会先把 zero-one 数据转成 A/B 成对偏好数据。
# 转换后的每条样本都形如：
# - A: 代码候选 1
# - B: 代码候选 2
# - answer: 人类更偏好 "A" 还是 "B"
try:
    dataset = datasets.load_from_disk("./data/code").to_list()
except FileNotFoundError:
    print("Error: Dataset not found at ./data/code")
    print("Please prepare your dataset or modify the path")
    raise
dataset = zero_one_dataset_to_pair_dataset(dataset, seed=SEED)

# 这里用最简单的切片方式划分 train / valid。
# 前 40 个 pair 用于优化 criterion，剩余 pair 用于观察泛化效果。
# train_set = dataset[:40]
# valid_set = dataset[40:]

train_set = dataset[:30]
valid_set = dataset[30:45]

# evaluator 负责“拿一组 criterion 去验证集上打分”。
# 在这个 demo 里，它主要被用来：
# 1. warm-up 后先看一下初始 criteria 的效果；
# 2. 最后只拿最佳 criteria 再做一次最终评估。
evaluator = PairEvaluator(
    WORKER_ARGS,
    dataset=valid_set,
    max_concurrent=MAX_CONCURRENT,
    max_retries=MAX_RETRIES,
    worker_prompt=WORKER_PROMPT,
)

# Workflow 是整个系统的调度核心。
# 这里没有直接把参数传进 `Workflow(...)`，而是先构造一个 state dict，
# 再通过 `load_state_dict()` 注入配置。这样做的好处是：
# 1. 与 workflow.save()/load() 的存档结构保持一致；
# 2. 后续更容易把这份配置保存成 checkpoint 再恢复。
workflow_state_dict = {
    "manager_args": {
        "model": "gpt-4o-2024-11-20",
        "api_keys": OPENAI_API_KEYS,
        "base_url": os.getenv("CRITIQ_MANAGER_BASE_URL", "https://api.aiiai.top/v1"),
        "request_kwargs": {
            "temperature": 1.0,
        },
    },
    "worker_args": WORKER_ARGS,
    "worker_max_concurrent": MAX_CONCURRENT,
    "n_criteria": N_CRITERIA,
    "manager_prompt": MANAGER_PROMPT,
    "worker_prompt": WORKER_PROMPT,
}

workflow = Workflow()
workflow.load_state_dict(workflow_state_dict)

# 如果存在 criterion 知识库，就优先把它作为“已有经验”拿来复用。
# 但知识库里的 criterion 可能来自更泛化的任务，不一定适用于“Python 代码质量”。
# 所以这里不会直接全量使用，而是先做一层相关性过滤。
try:
    kb = load_criteria_from_json("./data/kb.json")
except FileNotFoundError:
    print("Warning: Knowledge base not found at ./data/kb.json")
    print("Starting without pre-existing criteria knowledge base")
    kb = []


def ask_agent(criterion: Criterion):
    """询问 worker 模型：某个 criterion 是否适用于 Python 代码质量评估。

    这里复用了 worker 模型做一个非常轻量的 yes/no 分类器。
    如果返回 yes，就把该 criterion 当作“代码质量相关的候选知识”。
    """
    prompt = "# Instruction\nIs this criterion applicable for evaluating the quality of Python code? \n\n# Criterion\n{}: {}".format(
        criterion.name, criterion.description
    )
    prompt = prompt + "\n\nYou should simply reply 'yes' or 'no'."
    kb_agent = Agent(**WORKER_ARGS)
    response = kb_agent(prompt, stream=False)
    if response is None:
        return False
    return "yes" in response.lower()


# 在 warm-up 之前，先把通用知识库过滤成仅与代码质量相关的 criterion。
# 这里使用线程池并发执行，是因为每个 criterion 的 yes/no 判断彼此独立。
kb_code = []
if kb:  # Only run if knowledge base is available
    # 线程数量不超过配置的并发上限，也不超过知识库条目数，避免资源浪费。
    with ThreadPoolExecutor(max_workers=min(MAX_CONCURRENT, len(kb))) as executor:
        futures = []
        for c in kb:
            futures.append(executor.submit(ask_agent, c))
        for _ in tqdm(
            as_completed(futures),
            total=len(futures),
            dynamic_ncols=True,
            desc="Filtering knowledge base"
        ):
            pass
        for f, c in zip(futures, kb):
            if f.result():
                kb_code.append(c)
    print("Retrieved {} criteria related to code from knowledge base.".format(len(kb_code)))
else:
    print("No knowledge base available, starting with empty criteria set")

# `get_init_criteria()` 是 workflow 的初始化入口，内部做两件事：
# 1. 如果提供 knowledge_base，就先在训练集上粗评这些现有 criterion，取高分项；
# 2. 不足的数量再交给 manager 通过 warm-up + prompt 生成新的 criterion。
# 这样得到的 `workflow.current_criteria` 就是后续优化循环的起点。
workflow.get_init_criteria(
    train_set,
    prompt_template=WARMUP_PROMPT,
    knowledge_base=kb_code,
    # n_shot=5,
    n_shot=1,
    max_retrived=None,
)

# 先看 warm-up 后的初始 criteria 在验证集上的效果。
# 这里不更新 score，只做观测，因为当前更关心的是“整体投票表现”。
eval_output = evaluator.eval(workflow.current_criteria, update_score=False)
print("After warm up:", eval_output.accuracy, eval_output.is_correct)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 保存初始化后的 criterion 集合。
# 后面的 `print_score_changes()` 会把 init / 每一轮 / final 的结果串起来对比。
workflow.save(OUTPUT_DIR, "init", None)

# 进入主优化循环。
# optimize() 内部会在每个 epoch 做如下事情：
# 1. 用当前 criteria 在 train_set 上逐条评估；
# 2. 按准确率把 criterion 分成 good / mid / low；
# 3. 保留 good、改写 mid、替换 low；
# 4. 可选地在 valid_set 上做一次评估；
# 5. 把每一轮结果保存到 output_dir。
# threshold=(0.8, 0.9) 的含义是：
# - >= 0.9: good，直接保留
# - (0.8, 0.9): mid，需要反思并改写
# - <= 0.8: low，需要淘汰并生成新的
workflow.optimize(
    train_set,
    valid_set,
    output_dir=OUTPUT_DIR,
    num_epochs=NUM_EPOCHS,
    threshold=(0.8, 0.9),
    max_retries=MAX_RETRIES,
)

# 读取输出目录中的 checkpoint，打印每一轮 criterion 分数变化。
# 这一步主要用于分析“哪些标准稳定保留、哪些被替换、哪些越改越强”。
print_score_changes(
    f"./output/{TASK_NAME}",
    [
        "epoch_init.json",
        *[f"epoch_{i}.json" for i in range(NUM_EPOCHS)],
        "epoch_final.json",
    ],
)

# 最终评估只使用 workflow 历史上得分最高的那部分 criterion。
# `get_best_criteria(0.9)` 会从 `all_criteria` 中筛出 score >= 0.9 的项目，
# 相当于只保留整个优化过程中真正表现稳定、强势的标准。
eval_output = evaluator.eval(workflow.get_best_criteria(0.9), update_score=False)
print("Final:", eval_output.accuracy, eval_output.is_correct)
