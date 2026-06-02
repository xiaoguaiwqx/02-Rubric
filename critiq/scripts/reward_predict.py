"""使用 Hugging Face reward model checkpoint 做离线 reward 预测。"""

import argparse
import json
import os
import random
import time

import datasets
import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def set_random_seed(seed):
    """为 Python、NumPy 和 PyTorch 设置随机种子，保证结果可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """将原始标量 logits 转为概率值。"""
    return 1 / (1 + np.exp(1e-6 - x))


def main(args):
    """加载 reward model，为 JSONL 数据集打分，并保存 `reward` 字段。"""
    set_random_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "right"

    def preprocess_function(examples):
        """对指定文本字段做分词，并顺带记录 token 数用于吞吐统计。"""
        tokenized = tokenizer(examples[args.field])
        return {
            **tokenized,
            "num_tokens": [len(ids) for ids in tokenized["input_ids"]],
        }

    with open(args.data, "r", encoding="utf8") as f:
        data = [json.loads(line) for line in f]

    tokenized_data = (
        datasets.Dataset.from_list(data).map(
            preprocess_function,
            batched=True,
            num_proc=os.cpu_count() // int(os.environ.get("WORLD_SIZE", "1")) // 4,
        )
        .shuffle(seed=args.seed)
    )
    n_tokens = sum(tokenized_data["num_tokens"])

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=1,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    if not model.config.pad_token_id:
        model.config.pad_token_id = model.config.eos_token_id

    training_args = TrainingArguments(
        output_dir=f"{args.output_dir}",
        per_device_eval_batch_size=args.eval_batch_size,
        bf16=True,
        tf32=True,
        seed=args.seed,
        deepspeed=(
            {
                "train_batch_size": "auto",
                "train_micro_batch_size_per_gpu": "auto",
                "gradient_accumulation_steps": "auto",
                "zero_optimization": {"stage": args.zero_stage},
                "bf16": {"enabled": "auto"},
            }
            if args.zero_stage
            else None
        ),
    )

    trainer = Trainer(model=model, args=training_args, tokenizer=tokenizer)
    start_time = time.time()
    prediction_output = trainer.predict(tokenized_data)
    total_time = time.time() - start_time

    tgs = n_tokens / total_time / int(os.environ.get("WORLD_SIZE", "1"))
    print(f"#Tokens: {n_tokens}\tTGS: {tgs:.2f}")

    # `Trainer.predict` 返回的是原始 logits，这里将其转换成标量 reward 分数。
    prediction_output = (
        sigmoid(np.array(prediction_output.predictions)).flatten().tolist()
    )

    file_name = args.data.split("/")[-1]
    with open(os.path.join(args.output_dir, file_name), "w", encoding="utf8") as f:
        for d, p in zip(data, prediction_output):
            f.write(json.dumps({**d, "reward": p}, ensure_ascii=False) + "\n")


def cli():
    """解析 CLI 参数并执行 reward 预测。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--data", type=str)
    parser.add_argument("--seed", type=int, default=100745534)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--zero_stage", type=int, default=0)
    parser.add_argument("--field", type=str, default="text")
    main(parser.parse_args())


if __name__ == "__main__":
    cli()
