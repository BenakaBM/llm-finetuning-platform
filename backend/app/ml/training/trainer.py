"""Real LoRA / QLoRA fine-tuning using Hugging Face Transformers + PEFT.

This module performs actual gradient-descent training - there is no
simulated or fabricated metric anywhere in this file. On CPU-only machines,
use a tiny model (e.g. sshleifer/tiny-gpt2) and a handful of examples via
`max_train_samples` so a real, complete training run finishes in seconds
for demo/testing purposes. The same code path runs full LoRA/QLoRA
training on a GPU with a larger model.
"""
from __future__ import annotations

import inspect
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from datasets import Dataset as HFDataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from app.ml.datasets.loader import format_as_prompt
from app.ml.training.target_modules import resolve_target_modules


@dataclass
class TrainingConfigDTO:
    method: str  # LORA | QLORA
    learning_rate: float
    batch_size: int
    gradient_accumulation_steps: int
    epochs: int
    max_seq_length: int
    warmup_ratio: float
    weight_decay: float
    optimizer: str
    scheduler: str
    seed: int
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: list[str]
    use_4bit: bool
    quant_type: str
    compute_dtype: str
    double_quant: bool


def _load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


class MetricStreamCallback(TrainerCallback):
    """Forwards each logged training/eval step to an external sink
    (e.g. persisted to the DB and/or pushed over a websocket)."""

    def __init__(self, on_log: Callable[[dict], None]):
        self.on_log = on_log

    def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: N802
        if logs is None:
            return
        payload = {
            "step": state.global_step,
            "epoch": round(state.epoch or 0, 4),
            "train_loss": logs.get("loss"),
            "eval_loss": logs.get("eval_loss"),
            "learning_rate": logs.get("learning_rate"),
        }
        self._sink(payload)

    def _sink(self, payload: dict) -> None:
        pass  # overridden dynamically below


def _build_training_arguments(
    output_dir: str,
    config: "TrainingConfigDTO",
    steps_per_epoch: int,
) -> TrainingArguments:
    """Builds TrainingArguments defensively across transformers versions.

    transformers 5.x removed a few kwargs that existed in 4.x (notably
    `warmup_ratio` -> only `warmup_steps` remains, and `no_cuda` -> renamed
    to `use_cpu`). Rather than crash with a raw TypeError on whichever
    version happens to be installed, this inspects the installed
    TrainingArguments signature and adapts:
      - warmup_ratio is converted to an equivalent warmup_steps count if
        the installed version no longer accepts warmup_ratio directly.
      - no_cuda / use_cpu: whichever the installed version supports is used.
      - Any other kwarg not present in the installed signature is dropped
        rather than raising, so minor future API drift degrades gracefully
        instead of failing the whole training job.
    """
    accepted = set(inspect.signature(TrainingArguments.__init__).parameters.keys())

    kwargs: dict = dict(
        output_dir=output_dir,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        optim=config.optimizer,
        lr_scheduler_type=config.scheduler,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=1,
        report_to=[],
        seed=config.seed,
        fp16=torch.cuda.is_available(),
        save_total_limit=1,
        # Reduces peak memory during backward pass at some compute cost -
        # particularly helpful on memory-constrained CPU machines.
        gradient_checkpointing=True,
    )

    # warmup_ratio -> warmup_steps fallback
    if "warmup_ratio" in accepted:
        kwargs["warmup_ratio"] = config.warmup_ratio
    elif "warmup_steps" in accepted:
        total_steps = max(1, steps_per_epoch * config.epochs)
        kwargs["warmup_steps"] = int(total_steps * config.warmup_ratio)

    # no_cuda -> use_cpu fallback
    if "no_cuda" in accepted:
        kwargs["no_cuda"] = not torch.cuda.is_available()
    elif "use_cpu" in accepted:
        kwargs["use_cpu"] = not torch.cuda.is_available()

    # Drop anything the installed version doesn't recognize instead of
    # letting TrainingArguments(...) raise a TypeError.
    filtered = {k: v for k, v in kwargs.items() if k in accepted}
    return TrainingArguments(**filtered)


def _resolve_max_seq_length(model, requested: int) -> int:
    """Clamps the requested max_seq_length to the model's actual supported
    context window. Without this, requesting a sequence length longer than
    the model's position-embedding table produces a cryptic
    'IndexError: index out of range in self' deep inside the forward pass
    instead of a clear, actionable limit."""
    model_max = getattr(getattr(model, "config", None), "max_position_embeddings", None)
    if not model_max or model_max <= 0:
        return requested
    return min(requested, model_max)


def build_model_and_tokenizer(base_model_id: str, config: TrainingConfigDTO, cache_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    torch_dtype = torch.float32
    if config.use_4bit and torch.cuda.is_available():
        from transformers import BitsAndBytesConfig

        compute_dtype = getattr(torch, config.compute_dtype, torch.float16)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=config.double_quant,
        )
        torch_dtype = compute_dtype
    elif torch.cuda.is_available():
        torch_dtype = torch.float16
    else:
        # CPU: load base weights in bfloat16 instead of float32. This halves
        # the base model's memory footprint (float32 is 4 bytes/param;
        # bfloat16 is 2) and, unlike float16, bfloat16 is numerically safe
        # on CPU. LoRA adapter parameters are kept in float32 by PEFT
        # regardless, so training stability isn't affected - only the
        # frozen base weights get smaller. This matters even for "small"
        # ~0.5B parameter models on memory-constrained machines.
        torch_dtype = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        cache_dir=cache_dir,
        quantization_config=quantization_config,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        # Without this, from_pretrained materializes a full float32 copy of
        # the model before converting to the target dtype/device, briefly
        # requiring roughly double the final memory footprint. This is a
        # very common, otherwise-invisible cause of "memory exhausted"
        # errors even for models that comfortably fit once loaded.
        low_cpu_mem_usage=True,
    )

    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=resolve_target_modules(model, config.lora_target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # Gradient checkpointing (enabled via TrainingArguments below) trades
    # some compute time for a large reduction in activation memory during
    # the backward pass - important on memory-constrained machines. With a
    # frozen base model + trainable LoRA adapters, gradient checkpointing
    # additionally requires enable_input_require_grads(), otherwise the
    # backward pass can fail with "none of the outputs have requires_grad"
    # since the frozen embedding layer's output doesn't track gradients by
    # default.
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model, "config"):
        model.config.use_cache = False  # required alongside gradient checkpointing

    return model, tokenizer


def prepare_dataset(records: list[dict], tokenizer, max_seq_length: int) -> HFDataset:
    texts = [format_as_prompt(r) for r in records]

    def tokenize_fn(batch):
        out = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
        )
        out["labels"] = out["input_ids"].copy()
        return out

    ds = HFDataset.from_dict({"text": texts})
    return ds.map(tokenize_fn, batched=True, remove_columns=["text"])


def run_training(
    base_model_id: str,
    train_path: str,
    val_path: str,
    output_dir: str,
    config: TrainingConfigDTO,
    cache_dir: str,
    max_train_samples: int | None = None,
    on_metric: Callable[[dict], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> dict:
    """Executes a real LoRA/QLoRA fine-tuning run. Returns a summary dict
    with real, measured values (final loss, duration, step count, etc.)."""
    torch.manual_seed(config.seed)

    train_records = _load_jsonl(train_path)
    val_records = _load_jsonl(val_path) or train_records[: max(1, len(train_records) // 5)]

    if max_train_samples:
        train_records = train_records[:max_train_samples]
        val_records = val_records[: max(1, max_train_samples // 4)]

    model, tokenizer = build_model_and_tokenizer(base_model_id, config, cache_dir)
    effective_max_seq_length = _resolve_max_seq_length(model, config.max_seq_length)
    train_ds = prepare_dataset(train_records, tokenizer, effective_max_seq_length)
    val_ds = prepare_dataset(val_records, tokenizer, effective_max_seq_length)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    steps_per_epoch = max(1, len(train_records) // (config.batch_size * config.gradient_accumulation_steps))
    training_args = _build_training_arguments(output_dir, config, steps_per_epoch)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    class StreamingCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: N802
            if logs is None or on_metric is None:
                return
            on_metric(
                {
                    "step": state.global_step,
                    "epoch": round(state.epoch or 0, 4),
                    "train_loss": logs.get("loss"),
                    "eval_loss": logs.get("eval_loss"),
                    "learning_rate": logs.get("learning_rate"),
                }
            )

        def on_epoch_end(self, args, state, control, **kwargs):
            if should_stop and should_stop():
                control.should_training_stop = True

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        callbacks=[StreamingCallback()],
    )

    start = time.time()
    train_result = trainer.train()
    duration = time.time() - start

    final_eval = trainer.evaluate()

    final_dir = str(Path(output_dir) / "final")
    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    return {
        "output_dir": final_dir,
        "duration_seconds": round(duration, 2),
        "global_step": trainer.state.global_step,
        "final_train_loss": train_result.training_loss,
        "final_eval_loss": final_eval.get("eval_loss"),
        "train_examples": len(train_records),
        "val_examples": len(val_records),
    }


def load_finetuned_model(base_model_id: str, adapter_path: str, cache_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, cache_dir=cache_dir)
    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        cache_dir=cache_dir,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    return model, tokenizer