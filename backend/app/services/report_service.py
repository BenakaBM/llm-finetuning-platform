from __future__ import annotations

import html
import json
from pathlib import Path

from app.core.config import get_settings
from app.models.evaluation import Experiment

settings = get_settings()

_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<title>Experiment Report - {name}</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; padding: 40px;
          background: #0b0f14; color: #e6edf3; }}
  h1, h2 {{ color: #f0f6fc; }}
  .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 10px;
           padding: 20px; margin-bottom: 24px; }}
  table {{ width: 100%; border-collapse: collapse; }}
  th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid #30363d; font-size: 14px; }}
  th {{ color: #8b949e; font-weight: 600; }}
  .badge {{ display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 12px;
            background: #1f6feb33; color: #58a6ff; }}
  .improve {{ color: #3fb950; font-weight: 600; }}
  .worsen {{ color: #f85149; font-weight: 600; }}
  pre {{ white-space: pre-wrap; word-break: break-word; background: #0d1117; padding: 12px;
         border-radius: 6px; font-size: 13px; }}
</style>
</head>
<body>
  <h1>{name}</h1>
  <span class="badge">{status}</span>

  <div class="card">
    <h2>Experiment Summary</h2>
    <table>
      <tr><th>Base Model</th><td>{base_model}</td></tr>
      <tr><th>Fine-Tuning Method</th><td>{method}</td></tr>
      <tr><th>Duration</th><td>{duration} seconds</td></tr>
      <tr><th>Seed</th><td>{seed}</td></tr>
      <tr><th>Created</th><td>{created_at}</td></tr>
    </table>
  </div>

  <div class="card">
    <h2>Training Configuration</h2>
    <pre>{hyperparameters}</pre>
  </div>

  <div class="card">
    <h2>Hardware</h2>
    <pre>{hardware}</pre>
  </div>

  <div class="card">
    <h2>Base vs Fine-Tuned Comparison</h2>
    <table>
      <tr><th>Metric</th><th>Base</th><th>Fine-Tuned</th><th>Improvement</th></tr>
      {comparison_rows}
    </table>
  </div>

  <div class="card">
    <h2>Sample Predictions (Fine-Tuned)</h2>
    {sample_predictions}
  </div>
</body>
</html>
"""


def _comparison_rows_html(comparison: list[dict]) -> str:
    rows = []
    for row in comparison:
        cls = "improve" if (row["improvement_pct"] or 0) >= 0 else "worsen"
        sign = "+" if (row["improvement_pct"] or 0) >= 0 else ""
        rows.append(
            f"<tr><td>{html.escape(row['metric'])}</td><td>{row['base']}</td>"
            f"<td>{row['fine_tuned']}</td>"
            f"<td class='{cls}'>{sign}{row['improvement_pct']}%</td></tr>"
        )
    return "\n".join(rows) if rows else "<tr><td colspan='4'>No comparable metrics yet.</td></tr>"


def _samples_html(samples: list[dict]) -> str:
    blocks = []
    for s in samples[:5]:
        blocks.append(
            "<pre>"
            f"PROMPT:\n{html.escape(s.get('prompt', ''))}\n\n"
            f"REFERENCE:\n{html.escape(s.get('reference', ''))}\n\n"
            f"PREDICTION:\n{html.escape(s.get('prediction', ''))}"
            "</pre>"
        )
    return "\n".join(blocks) if blocks else "<p>No samples available.</p>"


def generate_report(experiment: Experiment, comparison: list[dict], sample_predictions: list[dict]) -> str:
    report_dir = Path(settings.STORAGE_PATH, "experiments", str(experiment.id))
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "report.html"

    html_content = _TEMPLATE.format(
        name=html.escape(experiment.name),
        status=experiment.status,
        base_model=html.escape(experiment.base_model_id),
        method=experiment.fine_tuning_method or "N/A",
        duration=experiment.duration_seconds or "N/A",
        seed=experiment.seed or "N/A",
        created_at=experiment.created_at,
        hyperparameters=html.escape(json.dumps(experiment.hyperparameters, indent=2)),
        hardware=html.escape(json.dumps(experiment.hardware, indent=2)),
        comparison_rows=_comparison_rows_html(comparison),
        sample_predictions=_samples_html(sample_predictions),
    )
    report_path.write_text(html_content, encoding="utf-8")

    config_path = report_dir / "config.json"
    config_path.write_text(json.dumps(experiment.hyperparameters, indent=2), encoding="utf-8")

    return str(report_path)
