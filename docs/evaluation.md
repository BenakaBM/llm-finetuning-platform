# Evaluation

## Metrics

All metrics in `app/ml/evaluation/metrics.py` are computed from real model
generations against real reference outputs — nothing is hard-coded.

| Metric | Library | Notes |
|---|---|---|
| ROUGE-1/2/L | `rouge-score` | F-measure, stemmed |
| BLEU | `sacrebleu` | Corpus-level BLEU, scaled to 0-1 |
| Exact Match | custom | Case-insensitive, whitespace-trimmed string equality |
| BERTScore | `bert-score` | Optional; skipped gracefully if not installed |
| Accuracy/Precision/Recall/F1 | `scikit-learn` | Only computed when the reference set looks categorical (≤20 unique labels) |

New metrics can be added by writing a function `(predictions, references) -> dict`
and registering it in `METRIC_REGISTRY` — no other code needs to change.

## LLM-as-a-judge

`app/ml/evaluation/llm_judge.py` prompts a configurable judge model to
score a prediction on `correctness`, `relevance`, `helpfulness`,
`coherence`, `instruction_following`, and `faithfulness` on a 1-10 scale,
and parses the model's own JSON output. If the judge model's output can't
be parsed as valid JSON (which is expected with very small demo models
like `sshleifer/tiny-gpt2`, which are too small to reliably follow the
formatting instruction), the result is honestly reported as
`{"parsed": false, "raw_output": "..."}` rather than a fabricated score.
Use a capable instruction-tuned model (e.g. a 3B+ parameter chat model)
for meaningful judge scores in practice.

## Base vs Fine-Tuned comparison

Every training job automatically triggers two evaluation runs
(`EvaluationType.BASE` and `EvaluationType.FINE_TUNED`) against the exact
same held-out test split of the dataset, so the comparison is
apples-to-apples. `app/services/evaluation_service.py::compute_comparison`
computes the percentage improvement per metric, which is what's rendered
in the Experiment Detail page and the generated HTML report.

## Sample predictions

The first 5 test examples' prompt/reference/prediction triples are stored
on the `Evaluation` row (`sample_predictions` JSON column) and surfaced in
both the UI and the HTML report, so you can eyeball qualitative
differences, not just aggregate numbers.
