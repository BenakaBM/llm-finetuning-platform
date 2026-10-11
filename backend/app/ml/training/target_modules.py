"""Resolves which module names LoRA adapters should attach to.

Different model architectures name their attention projection layers
differently (LLaMA/Mistral/Qwen2 use q_proj/v_proj, GPT-2 uses a fused
c_attn, Falcon uses query_key_value, Phi uses Wqkv, etc.). A hardcoded
target-module list that works for one architecture will make PEFT raise
`ValueError: Target modules {...} not found in the base model` for any
other architecture. This module makes target-module selection adapt to
whatever model is actually loaded instead.
"""
from __future__ import annotations

import torch.nn as nn

# Known-good target modules per architecture family, used as a first-choice
# suggestion. This is a convenience default, not a hard requirement - if
# the requested modules exist on the loaded model, they're always honored.
ARCHITECTURE_DEFAULTS: dict[str, list[str]] = {
    "gpt2": ["c_attn"],
    "gpt_neo": ["q_proj", "v_proj"],
    "gptj": ["q_proj", "v_proj"],
    "gpt_neox": ["query_key_value"],
    "llama": ["q_proj", "v_proj"],
    "mistral": ["q_proj", "v_proj"],
    "qwen2": ["q_proj", "v_proj"],
    "phi": ["q_proj", "v_proj"],
    "phi3": ["qkv_proj"],
    "falcon": ["query_key_value"],
    "bloom": ["query_key_value"],
    "mpt": ["Wqkv"],
    "gemma": ["q_proj", "v_proj"],
    "gemma2": ["q_proj", "v_proj"],
}

# Fallback: common projection-layer suffixes to look for by scanning the
# model's actual module names, used only if neither the requested modules
# nor an architecture-specific default match.
_COMMON_SUFFIXES = (
    "q_proj", "v_proj", "k_proj", "o_proj",
    "c_attn", "c_proj",
    "query_key_value", "Wqkv", "qkv_proj",
)


def _existing_leaf_module_names(model: nn.Module) -> set[str]:
    """The final path segment of every leaf module name, e.g. for
    'transformer.h.0.attn.c_attn' this includes 'c_attn'."""
    names = set()
    for full_name, module in model.named_modules():
        if list(module.children()):
            continue  # not a leaf module
        names.add(full_name.rsplit(".", 1)[-1])
    return names


def resolve_target_modules(model: nn.Module, requested: list[str]) -> list[str]:
    """Returns a target-module list that actually exists on `model`.

    Resolution order:
      1. If every requested module name exists on the model, use it as-is
         (respects an explicit user choice).
      2. Otherwise, try the architecture's known-good default.
      3. Otherwise, fall back to whichever common projection-layer names
         are actually present on the model.
      4. If nothing matches, raise a clear, actionable error instead of
         letting PEFT's cryptic "not found in the base model" surface.
    """
    existing = _existing_leaf_module_names(model)

    if requested and all(name in existing for name in requested):
        return requested

    architecture = getattr(getattr(model, "config", None), "model_type", "") or ""
    architecture = architecture.lower()
    default = ARCHITECTURE_DEFAULTS.get(architecture)
    if default and all(name in existing for name in default):
        return default

    detected = [name for name in _COMMON_SUFFIXES if name in existing]
    # De-duplicate paired q/k/v/o projections predictably before fused variants.
    if detected:
        return detected

    raise ValueError(
        f"Could not find LoRA-compatible attention projection layers on this "
        f"model (architecture: '{architecture or 'unknown'}'). Requested target "
        f"modules {requested} were not found, and no common attention "
        f"projection layer names were detected. Please specify target_modules "
        f"that match this model's actual layer names."
    )
