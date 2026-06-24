"""Train and evaluate a Qwen3-VL Bradley-Terry reward model for RLHF-V.

This script intentionally does not reuse the text-only RewardTrainer path in
`train_reward.py`: each candidate answer is scored with image + question +
answer as a multimodal chat conversation.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_TRAIN_FILE = "data/RLHF-V/bt_reward/full_train.jsonl"
DEFAULT_VAL_FILE = "data/RLHF-V/bt_reward/full_val.jsonl"
DEFAULT_HOLDOUT_FILE = "data/RLHF-V/bt_reward/holdout500_eval.jsonl"
DEFAULT_JOB_NAME = "qwen3vl_bt_full_seed100745534"
DEFAULT_LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
VISION_NAME_HINTS = ("visual", "vision", "image", "vit")
REQUIRED_VISION_KEYS = ("pixel_values", "image_grid_thw")


def require_training_deps():
    """Import heavy training deps lazily so `--help` works on light envs."""
    try:
        import numpy as np
        import torch
        import torch.nn.functional as F
        from PIL import Image
        from torch import nn
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoProcessor, Trainer, TrainingArguments
    except ImportError as exc:
        raise SystemExit(
            "Missing training dependencies. Install the `train` extra on the "
            "remote GPU environment, including torch, transformers, peft, "
            "bitsandbytes, accelerate, and pillow."
        ) from exc

    return {
        "np": np,
        "torch": torch,
        "F": F,
        "Image": Image,
        "nn": nn,
        "DataLoader": DataLoader,
        "Dataset": Dataset,
        "AutoProcessor": AutoProcessor,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
    }


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def set_random_seed(seed: int) -> None:
    deps = require_training_deps()
    np = deps["np"]
    torch = deps["torch"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def parse_csv(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def resolve_run_dir(args: argparse.Namespace) -> Path:
    return Path(args.output_dir) / args.job_name


def resolve_image_path(image_root: str | None, image_path: str) -> str:
    normalized = str(image_path).replace("\\", "/")
    if normalized.startswith(("http://", "https://", "file://")):
        return normalized
    path = Path(normalized)
    if path.is_absolute():
        return str(path)
    if image_root:
        return str(Path(image_root) / normalized)
    return normalized


def candidate_messages(image_path: str, question: str, answer: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": str(question)},
            ],
        },
        {"role": "assistant", "content": str(answer)},
    ]


def tensor_shapes(batch: dict[str, Any]) -> dict[str, Any]:
    shapes = {}
    for key, value in batch.items():
        if hasattr(value, "shape"):
            shapes[key] = list(value.shape)
        elif isinstance(value, list):
            shapes[key] = f"list[{len(value)}]"
        else:
            shapes[key] = type(value).__name__
    return shapes


def get_hidden_size(config: Any) -> int:
    for obj in (config, getattr(config, "text_config", None), getattr(config, "llm_config", None)):
        if obj is not None and getattr(obj, "hidden_size", None):
            return int(obj.hidden_size)
    raise ValueError("Cannot infer hidden_size from Qwen3-VL config")


def create_training_args(args: argparse.Namespace):
    deps = require_training_deps()
    TrainingArguments = deps["TrainingArguments"]
    run_dir = resolve_run_dir(args)
    report_to = parse_csv(args.report_to) if args.report_to else []
    kwargs = {
        "output_dir": str(run_dir),
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": args.warmup_ratio,
        "gradient_accumulation_steps": args.accum,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "num_train_epochs": args.epochs,
        "save_strategy": "steps",
        "save_steps": args.eval_steps,
        "eval_steps": args.eval_steps,
        "logging_dir": str(run_dir / "logs"),
        "logging_steps": args.logging_steps,
        "logging_first_step": True,
        "logging_strategy": "steps",
        "fp16": bool(args.fp16 and not args.bf16),
        "bf16": bool(args.bf16),
        "tf32": bool(args.tf32),
        "greater_is_better": True,
        "metric_for_best_model": "eval_reward_accuracy",
        "load_best_model_at_end": False,
        "seed": args.seed,
        "deepspeed": args.deepspeed_config or None,
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "gradient_checkpointing_kwargs": (
            {"use_reentrant": False} if args.gradient_checkpointing else None
        ),
        "remove_unused_columns": False,
        "report_to": report_to,
        "dataloader_num_workers": args.dataloader_num_workers,
        "dataloader_pin_memory": True,
    }
    sig = inspect.signature(TrainingArguments)
    if "eval_strategy" in sig.parameters:
        kwargs["eval_strategy"] = "steps"
    else:
        kwargs["evaluation_strategy"] = "steps"
    return TrainingArguments(**kwargs)


def load_processor(args: argparse.Namespace):
    deps = require_training_deps()
    AutoProcessor = deps["AutoProcessor"]
    processor_kwargs = {"trust_remote_code": True}
    if args.min_pixels is not None:
        processor_kwargs["min_pixels"] = args.min_pixels
    if args.max_pixels is not None:
        processor_kwargs["max_pixels"] = args.max_pixels
    processor = AutoProcessor.from_pretrained(args.model, **processor_kwargs)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = args.padding_side
        tokenizer.truncation_side = "right"
        tokenizer.model_max_length = args.max_length
    return processor


def load_backbone(args: argparse.Namespace, *, for_training: bool):
    deps = require_training_deps()
    torch = deps["torch"]

    try:
        from transformers import Qwen3VLForConditionalGeneration as ModelClass
    except ImportError:
        try:
            from transformers import AutoModelForMultimodalLM as ModelClass
        except ImportError as exc:
            raise SystemExit(
                "The installed transformers does not expose Qwen3-VL classes. "
                "Install the latest transformers from source as recommended by Qwen."
            ) from exc

    dtype = torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "attn_implementation": args.attn_implementation,
    }
    if args.train_mode == "qlora":
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise SystemExit("QLoRA requires bitsandbytes support in transformers.") from exc
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
        )
        if args.device_map == "local" and torch.cuda.is_available():
            kwargs["device_map"] = {"": int(os.environ.get("LOCAL_RANK", "0"))}
        elif args.device_map == "auto":
            kwargs["device_map"] = "auto"
    elif args.device_map == "auto":
        kwargs["device_map"] = "auto"

    if "AutoModel" not in ModelClass.__name__:
        loader = ModelClass.from_pretrained
    else:
        loader = lambda model_name, **kw: ModelClass.from_pretrained(model_name, **kw)

    try:
        model = loader(args.model, dtype=dtype, **kwargs)
    except TypeError:
        model = loader(args.model, torch_dtype=dtype, **kwargs)

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if for_training and args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def freeze_vision_modules(model: Any) -> None:
    for name, param in model.named_parameters():
        lower = name.lower()
        if any(hint in lower for hint in VISION_NAME_HINTS):
            param.requires_grad = False


def apply_lora(args: argparse.Namespace, backbone: Any, *, for_training: bool) -> Any:
    if args.train_mode not in {"qlora", "lora"}:
        return backbone
    try:
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    except ImportError as exc:
        raise SystemExit("LoRA/QLoRA training requires the peft package.") from exc

    if args.train_mode == "qlora" and for_training:
        backbone = prepare_model_for_kbit_training(
            backbone,
            use_gradient_checkpointing=args.gradient_checkpointing,
        )

    target_modules = parse_csv(args.lora_target_modules)
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )
    backbone = get_peft_model(backbone, config)
    if for_training and is_main_process():
        backbone.print_trainable_parameters()
    return backbone


def load_lora_adapter(backbone: Any, adapter_dir: Path, *, is_trainable: bool) -> Any:
    try:
        from peft import PeftModel
    except ImportError as exc:
        raise SystemExit("Loading a LoRA checkpoint requires the peft package.") from exc
    return PeftModel.from_pretrained(backbone, adapter_dir, is_trainable=is_trainable)


def build_reward_model_class():
    deps = require_training_deps()
    torch = deps["torch"]
    nn = deps["nn"]

    class Qwen3VLRewardModel(nn.Module):
        def __init__(
            self,
            backbone: Any,
            *,
            base_model_name_or_path: str,
            train_mode: str,
            reward_config: dict[str, Any],
        ) -> None:
            super().__init__()
            self.backbone = backbone
            self.base_model_name_or_path = base_model_name_or_path
            self.train_mode = train_mode
            self.reward_config = reward_config
            for attr in (
                "is_loaded_in_4bit",
                "is_loaded_in_8bit",
                "hf_device_map",
                "quantization_method",
            ):
                if hasattr(backbone, attr):
                    setattr(self, attr, getattr(backbone, attr))
            hidden_size = get_hidden_size(backbone.config)
            self.score = nn.Linear(hidden_size, 1)
            device = self._infer_device()
            dtype = self._infer_dtype()
            self.score.to(device=device, dtype=dtype)

        def _infer_device(self):
            for param in self.backbone.parameters():
                return param.device
            return torch.device("cpu")

        def _infer_dtype(self):
            for param in self.backbone.parameters():
                if param.dtype.is_floating_point:
                    return param.dtype
            return torch.float32

        @property
        def input_device(self):
            return self._infer_device()

        def forward(self, **inputs):
            attention_mask = inputs.get("attention_mask")
            outputs = self.backbone(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
            hidden = outputs.hidden_states[-1]
            if attention_mask is None:
                pooled = hidden[:, -1]
            else:
                positions = torch.arange(hidden.shape[1], device=hidden.device)
                positions = positions.unsqueeze(0).expand_as(attention_mask)
                last_indices = (positions * attention_mask.long()).max(dim=1).values
                pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), last_indices]
            reward = self.score(pooled).squeeze(-1)
            return {"reward": reward}

        def save_pretrained(self, output_dir: str | Path, **_: Any) -> None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            adapter_dir = output_dir / "adapter"
            backbone_dir = output_dir / "backbone"
            if hasattr(self.backbone, "peft_config"):
                self.backbone.save_pretrained(adapter_dir)
                saved_backbone = {"type": "peft_adapter", "path": "adapter"}
            elif hasattr(self.backbone, "save_pretrained"):
                self.backbone.save_pretrained(backbone_dir)
                saved_backbone = {"type": "full_backbone", "path": "backbone"}
            else:
                saved_backbone = {"type": "external", "path": None}
            torch.save(self.score.state_dict(), output_dir / "score_head.pt")
            config = {
                "base_model_name_or_path": self.base_model_name_or_path,
                "train_mode": self.train_mode,
                "reward_config": self.reward_config,
                "saved_backbone": saved_backbone,
            }
            write_json(output_dir / "reward_model_config.json", config)

    return Qwen3VLRewardModel


def build_reward_model(
    args: argparse.Namespace,
    *,
    for_training: bool,
    checkpoint_dir: Path | None = None,
):
    deps = require_training_deps()
    torch = deps["torch"]
    Qwen3VLRewardModel = build_reward_model_class()

    saved_config = None
    if checkpoint_dir is not None and (checkpoint_dir / "reward_model_config.json").exists():
        saved_config = json.loads((checkpoint_dir / "reward_model_config.json").read_text(encoding="utf-8"))
        args.model = saved_config.get("base_model_name_or_path", args.model)
        args.train_mode = saved_config.get("train_mode", args.train_mode)

    backbone = load_backbone(args, for_training=for_training)
    if args.freeze_vision and not args.unfreeze_vision:
        freeze_vision_modules(backbone)

    adapter_dir = checkpoint_dir / "adapter" if checkpoint_dir is not None else None
    if adapter_dir is not None and adapter_dir.exists():
        backbone = load_lora_adapter(backbone, adapter_dir, is_trainable=for_training)
    elif for_training:
        backbone = apply_lora(args, backbone, for_training=for_training)
    if for_training and args.freeze_vision and not args.unfreeze_vision:
        freeze_vision_modules(backbone)

    reward_config = {
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_target_modules": parse_csv(args.lora_target_modules),
        "freeze_vision": bool(args.freeze_vision and not args.unfreeze_vision),
        "center_rewards_coefficient": args.center_rewards_coefficient,
    }
    if saved_config and saved_config.get("reward_config"):
        reward_config.update(saved_config["reward_config"])

    model = Qwen3VLRewardModel(
        backbone,
        base_model_name_or_path=args.model,
        train_mode=args.train_mode,
        reward_config=reward_config,
    )
    score_path = checkpoint_dir / "score_head.pt" if checkpoint_dir is not None else None
    if score_path is not None and score_path.exists():
        state = torch.load(score_path, map_location="cpu")
        model.score.load_state_dict(state)
    elif checkpoint_dir is not None and not for_training:
        raise FileNotFoundError(f"Missing reward head checkpoint: {score_path}")
    return model


def move_tensor_batch(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if hasattr(value, "to") else value
    return moved


def strip_prefixed_inputs(batch: dict[str, Any], prefix: str) -> dict[str, Any]:
    marker = f"{prefix}_"
    return {
        key[len(marker) :]: value
        for key, value in batch.items()
        if key.startswith(marker) and hasattr(value, "to")
    }


def compute_pair_metrics(eval_pred):
    deps = require_training_deps()
    np = deps["np"]
    logits = eval_pred.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    logits = np.asarray(logits)
    chosen = logits[:, 0]
    rejected = logits[:, 1]
    margin = chosen - rejected
    return {
        "reward_accuracy": float((margin > 0).mean()) if len(margin) else 0.0,
        "mean_margin": float(margin.mean()) if len(margin) else 0.0,
        "mean_chosen_reward": float(chosen.mean()) if len(chosen) else 0.0,
        "mean_rejected_reward": float(rejected.mean()) if len(rejected) else 0.0,
    }


def build_trainer_class():
    deps = require_training_deps()
    torch = deps["torch"]
    F = deps["F"]
    Trainer = deps["Trainer"]

    class Qwen3VLBTTrainer(Trainer):
        def __init__(self, *args, center_rewards_coefficient: float = 0.0, **kwargs):
            super().__init__(*args, **kwargs)
            self.center_rewards_coefficient = center_rewards_coefficient

        def compute_loss(
            self,
            model,
            inputs,
            return_outputs: bool = False,
            num_items_in_batch=None,
        ):
            del num_items_in_batch
            chosen_inputs = strip_prefixed_inputs(inputs, "chosen")
            rejected_inputs = strip_prefixed_inputs(inputs, "rejected")
            chosen_rewards = model(**chosen_inputs)["reward"]
            rejected_rewards = model(**rejected_inputs)["reward"]
            margin = chosen_rewards - rejected_rewards
            loss = -F.logsigmoid(margin).mean()
            if self.center_rewards_coefficient:
                center_loss = torch.cat([chosen_rewards, rejected_rewards]).pow(2).mean()
                loss = loss + self.center_rewards_coefficient * center_loss
            outputs = {
                "loss": loss,
                "logits": torch.stack([chosen_rewards, rejected_rewards], dim=-1),
                "chosen_rewards": chosen_rewards,
                "rejected_rewards": rejected_rewards,
            }
            return (loss, outputs) if return_outputs else loss

        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            del ignore_keys
            inputs = self._prepare_inputs(inputs)
            with torch.no_grad():
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            logits = outputs["logits"].detach()
            labels = torch.ones(logits.shape[0], dtype=torch.long, device=logits.device)
            if prediction_loss_only:
                return loss.detach(), None, None
            return loss.detach(), logits, labels

        def save_model(self, output_dir: str | None = None, _internal_call: bool = False):
            del _internal_call
            output_dir = output_dir or self.args.output_dir
            self.model.save_pretrained(output_dir)
            processing_class = getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)
            if processing_class is not None and hasattr(processing_class, "save_pretrained"):
                processing_class.save_pretrained(output_dir)

    return Qwen3VLBTTrainer


class Qwen3VLBTPairDataset:
    def __init__(
        self,
        path: Path,
        *,
        required_fields: tuple[str, ...],
        limit: int | None = None,
    ) -> None:
        rows = read_jsonl(path, limit=limit)
        clean_rows = []
        for idx, row in enumerate(rows):
            missing = [field for field in required_fields if not row.get(field)]
            if missing:
                raise ValueError(f"{path}:{idx + 1} missing fields: {', '.join(missing)}")
            clean_rows.append(row)
        self.path = path
        self.rows = clean_rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


class Qwen3VLPairCollator:
    def __init__(
        self,
        *,
        processor: Any,
        image_root: str | None,
        left_field: str,
        right_field: str,
        left_prefix: str,
        right_prefix: str,
        max_length: int,
        require_vision: bool = True,
        include_metadata: bool = False,
    ) -> None:
        self.processor = processor
        self.image_root = image_root
        self.left_field = left_field
        self.right_field = right_field
        self.left_prefix = left_prefix
        self.right_prefix = right_prefix
        self.max_length = max_length
        self.require_vision = require_vision
        self.include_metadata = include_metadata

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        left_messages = [self._messages(feature, self.left_field) for feature in features]
        right_messages = [self._messages(feature, self.right_field) for feature in features]
        batch = {}
        batch.update(self._prefixed(self._encode(left_messages), self.left_prefix))
        batch.update(self._prefixed(self._encode(right_messages), self.right_prefix))
        if self.include_metadata:
            batch["metadata"] = [
                {
                    "sample_id": feature.get("sample_id"),
                    "answer": feature.get("answer"),
                    "source_split": feature.get("source_split"),
                    "origin_dataset": feature.get("origin_dataset"),
                }
                for feature in features
            ]
        return batch

    def _messages(self, feature: dict[str, Any], answer_field: str) -> list[dict[str, Any]]:
        image_path = resolve_image_path(self.image_root, feature["image_path"])
        return candidate_messages(image_path, feature["question"], feature[answer_field])

    def _encode(self, messages_batch: list[list[dict[str, Any]]]) -> dict[str, Any]:
        conversation = messages_batch[0] if len(messages_batch) == 1 else messages_batch
        kwargs = {
            "tokenize": True,
            "add_generation_prompt": False,
            "return_dict": True,
            "return_tensors": "pt",
            "padding": True,
            "truncation": True,
            "max_length": self.max_length,
        }
        try:
            encoded = self.processor.apply_chat_template(conversation, **kwargs)
        except TypeError as first_error:
            for dropped_keys in (
                ("truncation", "max_length"),
                ("padding", "truncation", "max_length"),
            ):
                retry_kwargs = dict(kwargs)
                for key in dropped_keys:
                    retry_kwargs.pop(key, None)
                try:
                    encoded = self.processor.apply_chat_template(
                        conversation, **retry_kwargs
                    )
                    break
                except TypeError:
                    encoded = None
            if encoded is None:
                raise first_error
        encoded = dict(encoded)
        if self.require_vision:
            missing = [key for key in REQUIRED_VISION_KEYS if key not in encoded]
            if missing:
                raise ValueError(
                    "Processor output is missing multimodal keys "
                    f"{missing}. Check image_root and Qwen3-VL processor version."
                )
        return encoded

    @staticmethod
    def _prefixed(encoded: dict[str, Any], prefix: str) -> dict[str, Any]:
        return {
            f"{prefix}_{key}": value
            for key, value in encoded.items()
            if hasattr(value, "to")
        }


def train(args: argparse.Namespace):
    set_random_seed(args.seed)
    processor = load_processor(args)
    train_dataset = Qwen3VLBTPairDataset(
        Path(args.train_file),
        required_fields=("sample_id", "image_path", "question", "chosen", "rejected"),
        limit=args.limit_train,
    )
    val_dataset = Qwen3VLBTPairDataset(
        Path(args.val_file),
        required_fields=("sample_id", "image_path", "question", "chosen", "rejected"),
        limit=args.limit_eval,
    )
    collator = Qwen3VLPairCollator(
        processor=processor,
        image_root=args.image_root,
        left_field="chosen",
        right_field="rejected",
        left_prefix="chosen",
        right_prefix="rejected",
        max_length=args.max_length,
    )
    model = build_reward_model(args, for_training=True)
    training_args = create_training_args(args)
    Qwen3VLBTTrainer = build_trainer_class()
    trainer = Qwen3VLBTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        compute_metrics=compute_pair_metrics,
        center_rewards_coefficient=args.center_rewards_coefficient,
        processing_class=processor,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)
    trainer.save_model(str(resolve_run_dir(args)))
    metrics = trainer.evaluate()
    if is_main_process():
        write_json(resolve_run_dir(args) / "eval_metrics.json", metrics)
        print(json.dumps(metrics, indent=4, ensure_ascii=False))
    return model, processor


def dry_run_batch(args: argparse.Namespace) -> None:
    processor = load_processor(args)
    dataset = Qwen3VLBTPairDataset(
        Path(args.train_file),
        required_fields=("sample_id", "image_path", "question", "chosen", "rejected"),
        limit=args.limit_train or 2,
    )
    collator = Qwen3VLPairCollator(
        processor=processor,
        image_root=args.image_root,
        left_field="chosen",
        right_field="rejected",
        left_prefix="chosen",
        right_prefix="rejected",
        max_length=args.max_length,
    )
    batch = collator([dataset[idx] for idx in range(min(len(dataset), args.batch_size))])
    summary = {
        "dataset": str(dataset.path),
        "examples": min(len(dataset), args.batch_size),
        "shapes": tensor_shapes(batch),
        "has_required_vision_keys": all(
            f"chosen_{key}" in batch and f"rejected_{key}" in batch
            for key in REQUIRED_VISION_KEYS
        ),
    }
    print(json.dumps(summary, indent=4, ensure_ascii=False))


def score_batch(model: Any, batch: dict[str, Any], prefix: str):
    deps = require_training_deps()
    torch = deps["torch"]
    inputs = strip_prefixed_inputs(batch, prefix)
    inputs = move_tensor_batch(inputs, model.input_device)
    with torch.no_grad():
        return model(**inputs)["reward"].detach().float().cpu()


def summarize_holdout(rows: list[dict[str, Any]], near_tie_epsilon: float) -> dict[str, Any]:
    correct = sum(1 for row in rows if row["correct"])
    margins = [row["margin"] for row in rows]
    groups: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    for row in rows:
        for key in ("source_split", "origin_dataset"):
            value = row.get(key)
            if value is None:
                continue
            group_key = f"{key}:{value}"
            groups[group_key]["total"] += 1
            groups[group_key]["correct"] += int(row["correct"])
    group_metrics = {
        key: {
            "accuracy": value["correct"] / value["total"] if value["total"] else 0.0,
            **value,
        }
        for key, value in sorted(groups.items())
    }
    return {
        "accuracy": correct / len(rows) if rows else 0.0,
        "correct": correct,
        "total": len(rows),
        "mean_margin": sum(margins) / len(margins) if margins else 0.0,
        "near_tie_epsilon": near_tie_epsilon,
        "near_tie_count": sum(1 for margin in margins if abs(margin) <= near_tie_epsilon),
        "group_metrics": group_metrics,
    }


def evaluate_holdout(
    args: argparse.Namespace,
    *,
    model: Any | None = None,
    processor: Any | None = None,
) -> dict[str, Any] | None:
    deps = require_training_deps()
    DataLoader = deps["DataLoader"]
    torch = deps["torch"]
    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()

    if not is_main_process():
        if distributed:
            torch.distributed.barrier()
        return None

    if processor is None:
        processor = load_processor(args)
    if model is None:
        checkpoint_dir = Path(args.checkpoint) if args.checkpoint else resolve_run_dir(args)
        model = build_reward_model(args, for_training=False, checkpoint_dir=checkpoint_dir)
    model.eval()

    dataset = Qwen3VLBTPairDataset(
        Path(args.holdout_file),
        required_fields=("sample_id", "image_path", "question", "A", "B", "answer"),
        limit=args.limit_holdout or args.limit_eval,
    )
    collator = Qwen3VLPairCollator(
        processor=processor,
        image_root=args.image_root,
        left_field="A",
        right_field="B",
        left_prefix="left",
        right_prefix="right",
        max_length=args.max_length,
        include_metadata=True,
    )
    loader = DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collator)
    predictions = []
    for batch in loader:
        metadata = batch.pop("metadata")
        score_a = score_batch(model, batch, "left").tolist()
        score_b = score_batch(model, batch, "right").tolist()
        for item, a_reward, b_reward in zip(metadata, score_a, score_b):
            pred = "A" if a_reward > b_reward else "B"
            gold = item.get("answer")
            predictions.append(
                {
                    "sample_id": item.get("sample_id"),
                    "gold": gold,
                    "score_A": a_reward,
                    "score_B": b_reward,
                    "margin": a_reward - b_reward,
                    "pred": pred,
                    "correct": pred == gold,
                    "source_split": item.get("source_split"),
                    "origin_dataset": item.get("origin_dataset"),
                }
            )
    metrics = summarize_holdout(predictions, args.near_tie_epsilon)
    out_dir = resolve_run_dir(args)
    write_jsonl(out_dir / "holdout500_predictions.jsonl", predictions)
    write_json(out_dir / "holdout500_metrics.json", metrics)
    print(json.dumps(metrics, indent=4, ensure_ascii=False))
    if distributed:
        torch.distributed.barrier()
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/evaluate a Qwen3-VL Bradley-Terry reward model on RLHF-V."
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--train_file", type=str, default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--val_file", type=str, default=DEFAULT_VAL_FILE)
    parser.add_argument("--holdout_file", type=str, default=DEFAULT_HOLDOUT_FILE)
    parser.add_argument("--image_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="output/qwen3vl_bt_reward")
    parser.add_argument("--job_name", type=str, default=DEFAULT_JOB_NAME)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--mode", choices=("train", "eval", "train_eval"), default="train_eval")
    parser.add_argument("--dry_run_batch", action="store_true")

    parser.add_argument("--seed", type=int, default=100745534)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--accum", type=int, default=8)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--min_pixels", type=int, default=None)
    parser.add_argument("--max_pixels", type=int, default=None)
    parser.add_argument("--padding_side", choices=("left", "right"), default="left")
    parser.add_argument("--near_tie_epsilon", type=float, default=1e-6)

    parser.add_argument("--train_mode", choices=("qlora", "lora", "full"), default="qlora")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default=",".join(DEFAULT_LORA_TARGETS),
    )
    parser.add_argument("--bnb_4bit_quant_type", type=str, default="nf4")
    parser.add_argument("--bnb_4bit_use_double_quant", action="store_true", default=True)
    parser.add_argument("--no_bnb_4bit_use_double_quant", dest="bnb_4bit_use_double_quant", action="store_false")
    parser.add_argument("--freeze_vision", action="store_true", default=True)
    parser.add_argument("--unfreeze_vision", action="store_true")
    parser.add_argument("--center_rewards_coefficient", type=float, default=0.01)

    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--no_fp16", dest="fp16", action="store_false")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--tf32", action="store_true", default=True)
    parser.add_argument("--no_tf32", dest="tf32", action="store_false")
    parser.add_argument(
        "--attn_implementation",
        choices=("flash_attention_2", "sdpa", "eager"),
        default="flash_attention_2",
    )
    parser.add_argument("--device_map", choices=("local", "auto", "none"), default="local")
    parser.add_argument("--deepspeed_config", type=str, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--dataloader_num_workers", type=int, default=0)

    parser.add_argument("--limit_train", type=int, default=None)
    parser.add_argument("--limit_eval", type=int, default=None)
    parser.add_argument("--limit_holdout", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device_map == "none":
        args.device_map = None
    if args.dry_run_batch:
        dry_run_batch(args)
        return

    trained_model = None
    processor = None
    if args.mode in {"train", "train_eval"}:
        trained_model, processor = train(args)
    if args.mode in {"eval", "train_eval"}:
        evaluate_holdout(args, model=trained_model, processor=processor)


if __name__ == "__main__":
    main()
