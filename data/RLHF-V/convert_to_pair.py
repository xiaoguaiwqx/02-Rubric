"""Convert RLHF-V chosen/rejected JSONL records to CritiQ pair JSONL.

Input records are expected to contain:
    {"chosen": "...", "rejected": "..."}

Output records contain the core multimodal CritiQ pair fields by default:
    {
        "sample_id": "...", 
        "image_path": "...", 
        "question": "...", 
        "A": "...", 
        "B": "...", 
        "answer": "A"
    }

指定输出目录并随机打乱 A/B：
python data/RLHF-V/convert_to_pair.py data/RLHF-V/discovery_train_90.jsonl `
  --output-dir data/RLHF-V/ `
  --shuffle-ab `
  --seed 42


python data/RLHF-V/convert_to_pair.py data\RLHF-V\heldout_validation_500.jsonl `
  --output-dir data/RLHF-V/ `
  --shuffle-ab `
  --seed 1008 `
  --overwrite `
  --image-root D:\3-Work\02-DD-LLM\drea_multicrit_agent\datasets\RLHF-V-Dataset\processed\images

  
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


DEFAULT_INPUTS = ("discovery_train_90.jsonl", "heldout_validation_500.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert RLHF-V chosen/rejected JSONL to CritiQ A/B pair JSONL."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help=(
            "Input JSONL file(s). Defaults to the original RLHF-V files in this "
            "directory."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for converted files. Defaults to each input file's directory.",
    )
    parser.add_argument(
        "--suffix",
        default="_pair",
        help="Suffix inserted before .jsonl for auto-generated output names.",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default= None,
        help="Root directory for absolute image paths.",
    )
    parser.add_argument(
        "--shuffle-ab",
        action="store_true",
        help="Randomly swap A/B per example to reduce position bias.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=100745534,
        help="Random seed used with --shuffle-ab.",
    )
    parser.add_argument(
        "--keep-metadata",
        action="store_true",
        help="Keep non chosen/rejected fields alongside A/B/answer.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files if they already exist.",
    )
    return parser.parse_args()


def default_inputs(script_dir: Path) -> list[Path]:
    return [script_dir / name for name in DEFAULT_INPUTS]


def output_path(input_path: Path, output_dir: Path | None, suffix: str) -> Path:
    directory = output_dir or input_path.parent
    return directory / f"{input_path.stem}{suffix}{input_path.suffix}"


def response_text(record: dict[str, Any], key: str) -> str:
    """获取回答文本并校验类型。"""
    response = record[key]
    if not isinstance(response, str):
        raise ValueError(f"{key!r} must be a string")
    return response


def convert_record(
    record: dict[str, Any],
    *,
    shuffle_ab: bool,
    image_root: Path | None,
) -> dict[str, Any]:
    """核心处理逻辑：将单条 chosen/rejected 记录转换为包含多模态信息的 A/B 选择题形式。"""
    if "chosen" not in record or "rejected" not in record:
        raise ValueError("record must contain 'chosen' and 'rejected'")

    # 1. 提取 chosen(偏好) 和 rejected(拒绝) 的文本
    chosen = response_text(record, "chosen")
    rejected = response_text(record, "rejected")

    # 2. 随机打乱 A 和 B 的顺序，防止模型在一边倒的选项位置（如永远选A）上产生偏置(位置偏差)
    if shuffle_ab and random.choice([True, False]):
        # 50% 概率：把 rejected 放 A，chosen 放 B，正确答案指向 B
        pair = {"A": rejected, "B": chosen, "answer": "B"}
    else:
        # 另外 50% 概率：把 chosen 放 A，rejected 放 B，正确答案指向 A
        pair = {"A": chosen, "B": rejected, "answer": "A"}

    # 3. 按照多模态模型需要的结构顺序，重新组装最终的字典
    res = {}
    
    # 提取样本ID
    if "sample_id" in record:
        res["sample_id"] = record["sample_id"]
        
    # 提取并拼接完整的图像绝对路径
    if "image_path" in record:
        if image_root:
            # 使用 image_root 拼接出绝对路径，并将 Windows 的反斜杠 \ 统一替换为正斜杠 /，防止引发路径解析错误
            res["image_path"] = str(image_root / record["image_path"]).replace("\\", "/")
        else:
            res["image_path"] = record["image_path"]
            
    # 提取原始问题内容
    if "question" in record:
        res["question"] = record["question"]

    # 4. 把构造好的 A/B 结构和正确的 answer 答案追加到字典末尾
    res.update(pair)
    return res


def convert_file(
    input_path: Path,
    output_path_: Path,
    *,
    shuffle_ab: bool,
    keep_metadata: bool,
    overwrite: bool,
    image_root: Path | None,
) -> tuple[int, int]:
    """按行读取输入 JSONL 文件，转换每一行后写入输出文件。"""
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if output_path_.exists() and not overwrite:
        raise FileExistsError(f"{output_path_} already exists; pass --overwrite")

    # 确保输出目录及其父目录存在
    output_path_.parent.mkdir(parents=True, exist_ok=True)

    converted = 0
    skipped = 0
    with input_path.open("r", encoding="utf-8") as src, output_path_.open(
        "w", encoding="utf-8"
    ) as dst:
        # 逐行读取 jsonl 文件
        for line_no, line in enumerate(src, start=1):
            if not line.strip():
                skipped += 1
                continue

            try:
                # 将这一行的 JSON 字符串解析为字典
                record = json.loads(line)
                # 调用核心处理函数提取并转换 A/B 格式
                pair = convert_record(
                    record,
                    shuffle_ab=shuffle_ab,
                    image_root=image_root,
                )
            except Exception as exc:
                raise ValueError(f"{input_path}:{line_no}: {exc}") from exc

            # 如果要求保留其他所有的额外字段（元数据）
            if keep_metadata:
                metadata = {
                    k: v
                    for k, v in record.items()
                    # 过滤掉已经处理过的核心字段，剩下的都作为元数据保留
                    if k not in {"chosen", "rejected", "A", "B", "answer", "sample_id", "image_path", "question"}
                }
                # 将转换后的核心数据和原来的元数据合并
                pair = {**pair, **metadata}

            # 将处理后的新条目作为 JSON 字符串写回到输出文件中，确保不会转义非 ASCII 字符（如中文）
            dst.write(json.dumps(pair, ensure_ascii=False) + "\n")
            converted += 1

    return converted, skipped


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    inputs = args.inputs or default_inputs(script_dir)

    # 设置随机种子，确保每次开启 --shuffle-ab 时 A/B 打乱的结果是一致可复现的
    random.seed(args.seed)

    for input_path in inputs:
        input_path = input_path.resolve()
        target = output_path(input_path, args.output_dir, args.suffix).resolve()
        
        # 开始处理每个文件
        converted, skipped = convert_file(
            input_path,
            target,
            shuffle_ab=args.shuffle_ab,
            keep_metadata=args.keep_metadata,
            overwrite=args.overwrite,
            image_root=args.image_root,
        )
        print(f"{input_path} -> {target}: {converted} converted, {skipped} skipped")


if __name__ == "__main__":
    main()
