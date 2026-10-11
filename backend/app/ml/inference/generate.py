"""Real text generation used by both the evaluation engine and the playground.
No response is ever fabricated - every call here invokes model.generate()."""
from __future__ import annotations

import time
from functools import lru_cache

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from app.core.config import get_settings

settings = get_settings()

_MODEL_CACHE: dict[str, tuple] = {}


def _inference_dtype():
    """Same reasoning as trainer.py: bfloat16 on CPU halves memory vs the
    float32 default and is numerically safe (unlike float16) on CPU."""
    if torch.cuda.is_available():
        return torch.float16
    return torch.bfloat16


def load_base_model(model_id: str) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    if model_id in _MODEL_CACHE:
        return _MODEL_CACHE[model_id]

    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=settings.MODEL_CACHE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=settings.MODEL_CACHE,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=_inference_dtype(),
        low_cpu_mem_usage=True,
    )
    model.eval()
    _MODEL_CACHE[model_id] = (model, tokenizer)
    return model, tokenizer


def load_adapter_model(base_model_id: str, adapter_path: str) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    cache_key = f"adapter::{adapter_path}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(adapter_path, cache_dir=settings.MODEL_CACHE)
    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        cache_dir=settings.MODEL_CACHE,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=_inference_dtype(),
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    _MODEL_CACHE[cache_key] = (model, tokenizer)
    return model, tokenizer


def generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 128,
) -> dict:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    input_tokens = inputs["input_ids"].shape[1]

    if model.device.type != "cpu":
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    start = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-4),
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    latency_ms = (time.time() - start) * 1000

    generated_ids = output_ids[0][input_tokens:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return {
        "response": text.strip(),
        "input_tokens": input_tokens,
        "output_tokens": len(generated_ids),
        "latency_ms": round(latency_ms, 2),
    }