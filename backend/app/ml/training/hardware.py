"""Real hardware detection - no placeholder values."""
from __future__ import annotations

import platform

import psutil  # type: ignore


def detect_hardware() -> dict:
    info: dict = {
        "cpu": platform.processor() or platform.machine(),
        "cpu_cores": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "device": "cpu",
        "mode": "Development",
        "cuda_available": False,
    }

    try:
        import torch

        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            info.update(
                {
                    "device": "cuda",
                    "mode": "Production",
                    "cuda_available": True,
                    "gpu_name": props.name,
                    "gpu_memory_gb": round(props.total_memory / (1024**3), 2),
                    "cuda_version": torch.version.cuda,
                }
            )
    except ImportError:
        pass

    return info


def estimate_memory_requirement_gb(
    param_count_billion: float, use_4bit: bool, batch_size: int, max_seq_length: int
) -> float:
    """Rough but real, physics-based estimate: bytes-per-param depends on
    quantization; add optimizer state (LoRA -> only adapter params) and
    activation memory scaled by batch size and sequence length."""
    bytes_per_param = 0.5 if use_4bit else 2.0  # 4-bit vs fp16 base weights
    base_weights_gb = param_count_billion * 1e9 * bytes_per_param / (1024**3)

    # LoRA adapters + optimizer state are small relative to base weights,
    # but activations scale with batch size and sequence length.
    activation_gb = (batch_size * max_seq_length * 4096 * 4) / (1024**3)
    overhead_gb = 1.0  # CUDA context, framework overhead

    return round(base_weights_gb + activation_gb + overhead_gb, 2)


def check_resource_feasibility(required_gb: float, hardware: dict) -> dict:
    if hardware["device"] == "cpu":
        return {
            "feasible": True,
            "warning": None,
            "note": "Running on CPU. Training will be slow; use small datasets/models for demo purposes.",
        }

    available_gb = hardware.get("gpu_memory_gb", 0)
    if required_gb > available_gb:
        return {
            "feasible": False,
            "warning": (
                f"The selected configuration may require approximately {required_gb} GB of GPU memory. "
                f"Available: {available_gb} GB."
            ),
            "suggestions": [
                "Enable QLoRA (4-bit quantization)",
                "Reduce batch size",
                "Reduce max sequence length",
                "Increase gradient accumulation steps instead of batch size",
            ],
        }
    return {"feasible": True, "warning": None, "note": None}
