#!/usr/bin/env python3
"""
End-to-end demo of the LLM Fine-Tuning & Evaluation Platform's core ML
pipeline, runnable on CPU with no GPU required.

This script does NOT go through the FastAPI/Celery/Postgres stack - it
calls the same underlying ML modules directly so it can run standalone
in a couple of minutes on a laptop, using:

  - A tiny example instruction dataset (examples/demo_instruction_dataset.jsonl)
  - A tiny HF model (sshleifer/tiny-gpt2) so full LoRA training completes fast

Every step performs REAL work: real dataset validation, a real LoRA
fine-tuning run with real gradient updates, real ROUGE/BLEU/exact-match
evaluation of both the base and fine-tuned model, and a real generated
HTML report. Nothing here is mocked or hard-coded.

Usage:
    python scripts/demo.py
"""
import sys
import time
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.ml.datasets.loader import load_and_normalize  # noqa: E402
from app.ml.datasets.validation import (  # noqa: E402
    compute_quality_report,
    compute_statistics,
    split_records,
    validate_records,
)
from app.ml.training.trainer import TrainingConfigDTO, run_training  # noqa: E402
from app.ml.evaluation.metrics import run_metrics  # noqa: E402
from app.ml.inference.generate import generate, load_adapter_model, load_base_model  # noqa: E402
from app.ml.datasets.loader import format_prompt_only  # noqa: E402

DEMO_MODEL = "sshleifer/tiny-gpt2"
ROOT = Path(__file__).resolve().parents[1]
DEMO_DATASET = ROOT / "examples" / "demo_instruction_dataset.jsonl"
DEMO_STORAGE = ROOT / "storage" / "demo_run"
CACHE_DIR = str(ROOT / "data" / "hf_cache")


def banner(text: str) -> None:
    print(f"\n{'=' * 70}\n{text}\n{'=' * 70}")


def main() -> None:
    DEMO_STORAGE.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load & validate dataset ----------------------------------
    banner("STEP 1/9: Loading and validating dataset")
    fmt, normalized, dropped = load_and_normalize(str(DEMO_DATASET), "jsonl")
    errors = validate_records(normalized)
    if errors:
        print("Dataset validation FAILED:", errors)
        sys.exit(1)
    stats = compute_statistics(normalized)
    quality = compute_quality_report(normalized, stats, dropped)
    print(f"Format detected: {fmt.value}")
    print(f"Records: {stats['records']}  |  Duplicates: {stats['duplicates']}  |  Avg tokens: {stats['average_tokens']}")
    print(f"Quality score: {quality['score']}/100")
    for w in quality["warnings"]:
        print(f"  - warning: {w}")

    # ---- 2. Split dataset ----------------------------------------------
    banner("STEP 2/9: Splitting into train / validation / test")
    train, val, test = split_records(normalized)
    print(f"train={len(train)}  validation={len(val)}  test={len(test)}")

    import json

    train_path = DEMO_STORAGE / "train.jsonl"
    val_path = DEMO_STORAGE / "validation.jsonl"
    test_path = DEMO_STORAGE / "test.jsonl"
    for path, records in ((train_path, train), (val_path, val), (test_path, test)):
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    # ---- 3. Load lightweight model --------------------------------------
    banner("STEP 3/9: Loading lightweight base model")
    print(f"Model: {DEMO_MODEL} (tiny model chosen so this demo runs on CPU in ~1-2 minutes)")

    # ---- 4. Run a short REAL LoRA fine-tuning job ------------------------
    banner("STEP 4/9: Running a real LoRA fine-tuning job")
    config = TrainingConfigDTO(
        method="LORA",
        learning_rate=5e-4,
        batch_size=2,
        gradient_accumulation_steps=1,
        epochs=1,
        max_seq_length=64,
        warmup_ratio=0.0,
        weight_decay=0.0,
        optimizer="adamw_torch",
        scheduler="constant",
        seed=42,
        lora_rank=4,
        lora_alpha=8,
        lora_dropout=0.0,
        lora_target_modules=["c_attn"],  # GPT-2-style attention projection
        use_4bit=False,
        quant_type="nf4",
        compute_dtype="float32",
        double_quant=False,
    )

    def on_metric(payload):
        print(f"  step={payload['step']} epoch={payload['epoch']} loss={payload['train_loss']}")

    start = time.time()
    result = run_training(
        base_model_id=DEMO_MODEL,
        train_path=str(train_path),
        val_path=str(val_path),
        output_dir=str(DEMO_STORAGE / "checkpoints"),
        config=config,
        cache_dir=CACHE_DIR,
        max_train_samples=None,
        on_metric=on_metric,
    )
    print(f"\nTraining finished in {result['duration_seconds']}s, {result['global_step']} steps.")
    print(f"Final train loss: {result['final_train_loss']}  |  Final eval loss: {result['final_eval_loss']}")

    adapter_dir = result["output_dir"]

    # ---- 5. Evaluate base model -------------------------------------------
    banner("STEP 5/9: Evaluating BASE model on held-out test set")
    base_model, base_tok = load_base_model(DEMO_MODEL)
    test_records = test or val
    base_preds, refs = [], []
    for r in test_records:
        out = generate(base_model, base_tok, format_prompt_only(r), temperature=0.0, max_new_tokens=32)
        base_preds.append(out["response"])
        refs.append(r["output"])
    base_scores = run_metrics(["rouge", "bleu", "exact_match"], base_preds, refs)
    print("Base model scores:", base_scores)

    # ---- 6. Evaluate fine-tuned model --------------------------------------
    banner("STEP 6/9: Evaluating FINE-TUNED model on the same test set")
    ft_model, ft_tok = load_adapter_model(DEMO_MODEL, adapter_dir)
    ft_preds = []
    for r in test_records:
        out = generate(ft_model, ft_tok, format_prompt_only(r), temperature=0.0, max_new_tokens=32)
        ft_preds.append(out["response"])
    ft_scores = run_metrics(["rouge", "bleu", "exact_match"], ft_preds, refs)
    print("Fine-tuned model scores:", ft_scores)

    # ---- 7. Compare -----------------------------------------------------
    banner("STEP 7/9: Base vs Fine-Tuned comparison")
    print(f"{'Metric':<12}{'Base':<10}{'Fine-Tuned':<12}{'Change'}")
    comparison = []
    for metric, base_val in base_scores.items():
        ft_val = ft_scores.get(metric)
        change = None
        if base_val and ft_val is not None:
            change = round(((ft_val - base_val) / abs(base_val)) * 100, 1)
        comparison.append({"metric": metric, "base": base_val, "fine_tuned": ft_val, "improvement_pct": change})
        change_str = f"{change:+.1f}%" if change is not None else "n/a"
        print(f"{metric:<12}{base_val:<10}{ft_val if ft_val is not None else 'n/a':<12}{change_str}")

    # ---- 8. Save experiment artifacts --------------------------------------
    banner("STEP 8/9: Saving experiment artifacts")
    artifact_summary = {
        "model": DEMO_MODEL,
        "method": "LORA",
        "duration_seconds": result["duration_seconds"],
        "train_examples": result["train_examples"],
        "val_examples": result["val_examples"],
        "base_scores": base_scores,
        "fine_tuned_scores": ft_scores,
        "comparison": comparison,
    }
    summary_path = DEMO_STORAGE / "experiment_summary.json"
    summary_path.write_text(json.dumps(artifact_summary, indent=2))
    print(f"Saved: {summary_path}")

    # ---- 9. Generate HTML report --------------------------------------------
    banner("STEP 9/9: Generating HTML report")
    samples = [
        {"prompt": format_prompt_only(r), "reference": r["output"], "prediction": p}
        for r, p in zip(test_records[:5], ft_preds[:5])
    ]
    report_html = _render_report(artifact_summary, comparison, samples)
    report_path = DEMO_STORAGE / "report.html"
    report_path.write_text(report_html, encoding="utf-8")
    print(f"Report saved to: {report_path}")

    banner("DEMO COMPLETE")
    print(f"Open the report in a browser:\n  file://{report_path.resolve()}")


def _render_report(summary: dict, comparison: list, samples: list) -> str:
    import html

    rows = "".join(
        f"<tr><td>{c['metric']}</td><td>{c['base']}</td><td>{c['fine_tuned']}</td>"
        f"<td>{c['improvement_pct']}%</td></tr>"
        for c in comparison
    )
    sample_html = "".join(
        f"<pre>PROMPT:\n{html.escape(s['prompt'])}\n\nREFERENCE:\n{html.escape(s['reference'])}\n\n"
        f"PREDICTION:\n{html.escape(s['prediction'])}</pre>"
        for s in samples
    )
    return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Demo Experiment Report</title>
<style>
body {{ font-family: sans-serif; background:#0b0f14; color:#e6edf3; padding:40px; }}
table {{ width:100%; border-collapse:collapse; }}
th,td {{ padding:8px; border-bottom:1px solid #30363d; text-align:left; }}
pre {{ background:#0d1117; padding:12px; border-radius:6px; white-space:pre-wrap; }}
.card {{ background:#161b22; border:1px solid #30363d; border-radius:10px; padding:20px; margin-bottom:24px; }}
</style></head><body>
<h1>Demo Experiment Report</h1>
<div class="card"><h2>Summary</h2>
<p>Model: {summary['model']} | Method: {summary['method']} | Duration: {summary['duration_seconds']}s</p>
<p>Train examples: {summary['train_examples']} | Val examples: {summary['val_examples']}</p>
</div>
<div class="card"><h2>Base vs Fine-Tuned</h2>
<table><tr><th>Metric</th><th>Base</th><th>Fine-Tuned</th><th>Change</th></tr>{rows}</table>
</div>
<div class="card"><h2>Sample Predictions</h2>{sample_html}</div>
</body></html>"""


if __name__ == "__main__":
    main()
