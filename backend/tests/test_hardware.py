from app.ml.training.hardware import (
    check_resource_feasibility,
    detect_hardware,
    estimate_memory_requirement_gb,
)


def test_detect_hardware_returns_required_fields():
    info = detect_hardware()
    assert "device" in info
    assert "cpu_cores" in info
    assert "ram_gb" in info


def test_memory_estimate_scales_with_batch_size():
    small = estimate_memory_requirement_gb(1.1, use_4bit=False, batch_size=1, max_seq_length=256)
    large = estimate_memory_requirement_gb(1.1, use_4bit=False, batch_size=8, max_seq_length=256)
    assert large > small


def test_4bit_quantization_reduces_estimate():
    fp16 = estimate_memory_requirement_gb(1.1, use_4bit=False, batch_size=2, max_seq_length=256)
    int4 = estimate_memory_requirement_gb(1.1, use_4bit=True, batch_size=2, max_seq_length=256)
    assert int4 < fp16


def test_cpu_is_always_feasible():
    hardware = {"device": "cpu"}
    result = check_resource_feasibility(999, hardware)
    assert result["feasible"] is True


def test_gpu_infeasible_when_memory_insufficient():
    hardware = {"device": "cuda", "gpu_memory_gb": 4}
    result = check_resource_feasibility(12.0, hardware)
    assert result["feasible"] is False
    assert "suggestions" in result
