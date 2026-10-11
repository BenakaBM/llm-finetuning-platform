import torch.nn as nn

from app.ml.training.target_modules import resolve_target_modules


class _FakeLlamaAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 8)
        self.v_proj = nn.Linear(8, 8)


class _FakeConfig:
    def __init__(self, model_type: str):
        self.model_type = model_type


class _FakeLlama(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = _FakeLlamaAttn()
        self.config = _FakeConfig("llama")


class _FakeGPT2Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_attn = nn.Linear(8, 24)


class _FakeGPT2(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = _FakeGPT2Attn()
        self.config = _FakeConfig("gpt2")


class _FakeUnknownArch(nn.Module):
    def __init__(self):
        super().__init__()
        self.mystery_layer = nn.Linear(8, 8)
        self.config = _FakeConfig("some_custom_arch")


def test_requested_modules_honored_when_they_exist():
    """LLaMA-family models: q_proj/v_proj exist, so the explicit request
    should be used as-is."""
    model = _FakeLlama()
    result = resolve_target_modules(model, ["q_proj", "v_proj"])
    assert result == ["q_proj", "v_proj"]


def test_falls_back_to_architecture_default_when_requested_missing():
    """Regression test for the exact bug: selecting a GPT-2 architecture
    model with the default LLaMA-style target_modules (q_proj/v_proj)
    must NOT crash - it should fall back to GPT-2's actual c_attn module."""
    model = _FakeGPT2()
    result = resolve_target_modules(model, ["q_proj", "v_proj"])
    assert result == ["c_attn"]


def test_raises_clean_error_when_nothing_matches():
    """When neither the requested modules nor any known architecture
    default nor any common projection layer name is present, this must
    raise a clear, actionable ValueError instead of letting PEFT's cryptic
    internal error surface."""
    model = _FakeUnknownArch()
    try:
        resolve_target_modules(model, ["q_proj", "v_proj"])
        assert False, "expected a ValueError"
    except ValueError as e:
        assert "Could not find LoRA-compatible" in str(e)


def test_common_suffix_fallback_detects_fused_qkv():
    """Models with a fused query_key_value layer (Falcon/Bloom-style)
    should be detected even without an exact architecture match."""

    class FusedAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.query_key_value = nn.Linear(8, 24)

    class FusedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = FusedAttn()
            self.config = _FakeConfig("some_unlisted_arch")

    model = FusedModel()
    result = resolve_target_modules(model, ["q_proj", "v_proj"])
    assert "query_key_value" in result
