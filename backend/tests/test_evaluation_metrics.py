from app.ml.evaluation.metrics import (
    compute_bleu,
    compute_exact_match,
    compute_rouge,
    run_metrics,
)


def test_exact_match_perfect():
    preds = ["Paris", "Tokyo"]
    refs = ["Paris", "Tokyo"]
    assert compute_exact_match(preds, refs)["exact_match"] == 1.0


def test_exact_match_partial():
    preds = ["Paris", "Berlin"]
    refs = ["Paris", "Tokyo"]
    assert compute_exact_match(preds, refs)["exact_match"] == 0.5


def test_rouge_identical_text_is_high():
    preds = ["The cat sat on the mat"]
    refs = ["The cat sat on the mat"]
    scores = compute_rouge(preds, refs)
    assert scores["rouge1"] > 0.9
    assert scores["rougeL"] > 0.9


def test_rouge_unrelated_text_is_low():
    preds = ["completely different sentence about airplanes"]
    refs = ["The cat sat on the mat"]
    scores = compute_rouge(preds, refs)
    assert scores["rouge1"] < 0.5


def test_bleu_identical_text_is_high():
    preds = ["the quick brown fox jumps over the lazy dog"]
    refs = ["the quick brown fox jumps over the lazy dog"]
    scores = compute_bleu(preds, refs)
    assert scores["bleu"] > 0.9


def test_run_metrics_combines_registry():
    preds = ["Paris"]
    refs = ["Paris"]
    result = run_metrics(["exact_match", "rouge"], preds, refs)
    assert "exact_match" in result
    assert "rouge1" in result
