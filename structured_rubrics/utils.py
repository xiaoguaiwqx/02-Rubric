"""Criterion representation and frozen JSON/A-B helpers."""
from __future__ import annotations
import json
import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, Sequence, TypedDict


@dataclass
class Criterion:
    """自然语言评价准则。"""

    name: str
    description: str
    score: float = 0  # highest valid accuracy

    def to_dict(self):
        return {
            "name": self.name,
            "description": self.description,
            "score": self.score,
        }

    @staticmethod
    def from_dict(d):
        return Criterion(**d)


class PairData(TypedDict):
    """成对偏好比较样本。"""

    A: str
    B: str
    answer: Literal["A", "B"]


def parse_json(text: str, *, allow_invalid_escapes: bool = False) -> dict:
    """从带噪声的模型输出中提取最外层 JSON 对象。"""
    try:
        text = "{" + text.split("{", 1)[-1].strip().rsplit("}", 1)[0].strip() + "}"
        while True:
            try:
                return json.loads(text, strict=False)
            except json.JSONDecodeError as error:
                if not allow_invalid_escapes or error.msg != "Invalid \\escape":
                    raise
                # Preserve the literal backslash; leave valid JSON escapes alone.
                text = text[:error.pos] + "\\" + text[error.pos:]
    except Exception as e:
        raise ValueError(f"Failed to parse JSON: {text}") from e


def reverse_ab(x):
    """交换成对标签 A 与 B。"""
    x = x[0].upper()
    return {"A": "B", "B": "A"}[x]


def random_reverse(
    pair_dataset: Sequence[PairData], seed: int = 100745534
) -> list[PairData]:
    """随机翻转 A/B 顺序，以减少位置偏置。"""
    pair_dataset = deepcopy(pair_dataset)
    random.seed(seed)
    random.shuffle(pair_dataset)
    for d in pair_dataset:
        if random.choice([True, False]):
            d["A"], d["B"] = d["B"], d["A"]
            d["answer"] = reverse_ab(d["answer"])
    return pair_dataset
