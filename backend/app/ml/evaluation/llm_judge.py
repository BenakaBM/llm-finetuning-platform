"""LLM-as-a-judge evaluation.

Uses a locally loaded HF causal LM (by default, the same small model used
elsewhere in the platform) to score a prediction against criteria and
returns structured integer scores parsed from the model's own output.
This is a real model call, not a hard-coded number - if the judge model's
output can't be parsed as JSON, the example is marked as unparsable rather
than assigned a fabricated score.
"""
from __future__ import annotations

import json
import re

from app.ml.inference.generate import generate, load_base_model

JUDGE_CRITERIA = [
    "correctness",
    "relevance",
    "helpfulness",
    "coherence",
    "instruction_following",
    "faithfulness",
]

JUDGE_PROMPT_TEMPLATE = """You are an evaluation judge. Score the RESPONSE to the INSTRUCTION on a
scale of 1-10 for each criterion below. Output ONLY a JSON object with integer
scores for each criterion, nothing else.

Criteria: {criteria}

INSTRUCTION:
{instruction}

REFERENCE ANSWER:
{reference}

RESPONSE TO SCORE:
{prediction}

JSON scores:
"""


def _extract_json(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def judge_single(
    instruction: str,
    reference: str,
    prediction: str,
    judge_model_id: str,
    criteria: list[str] | None = None,
) -> dict:
    criteria = criteria or JUDGE_CRITERIA
    model, tokenizer = load_base_model(judge_model_id)
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        criteria=", ".join(criteria),
        instruction=instruction,
        reference=reference,
        prediction=prediction,
    )
    result = generate(model, tokenizer, prompt, temperature=0.0, max_new_tokens=120)
    parsed = _extract_json(result["response"])

    if not parsed:
        return {"parsed": False, "raw_output": result["response"]}

    scores = {k: parsed[k] for k in criteria if k in parsed and isinstance(parsed[k], (int, float))}
    if scores:
        scores["overall"] = round(sum(scores.values()) / len(scores), 2)
    scores["parsed"] = True
    return scores
