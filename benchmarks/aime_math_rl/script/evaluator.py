"""Evaluate AIME 2024 + AIME 2025 final-answer predictions (60 problems)."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
from typing import Any

from farbench.evaluator import EvalResult, MetricEvaluatorBase
from farbench.schemas import TaskConfig


class AimeMathRLEvaluator(MetricEvaluatorBase):
    """Compare predicted AIME 24/25 answers against the held-out answer key."""

    def evaluate(
        self,
        predictions_path: str,
        test_data_dir: str,
        task_config: TaskConfig,
    ) -> EvalResult:
        with open(predictions_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        answers_path = os.path.join(test_data_dir, "test_answers.json")
        with open(answers_path, "r", encoding="utf-8") as f:
            gold = json.load(f)

        pred_map = _coerce_predictions(raw, gold)

        correct = 0
        for item_id, gold_answer in gold.items():
            pred_answer = pred_map[item_id]
            pred_norm = _normalize_answer(pred_answer)
            gold_norm = _normalize_answer(gold_answer)
            if pred_norm == gold_norm:
                correct += 1

        total = len(gold)
        exact_match = correct / total if total else 0.0

        # NOTE: do NOT include per-item details (gold answer or per-id
        # `correct`) in the returned metrics. EvalResult.metrics is fed back
        # into the agent prompt via obs.history, and AIME answers are integers
        # in [0, 999] over only 60 problems — leaking either the gold or a
        # per-id correctness bit would let the agent recover the test labels
        # across iterations.
        return EvalResult(
            metrics={"exact_match": round(exact_match, 4)},
            primary_metric_name="exact_match",
            primary_metric_value=exact_match,
        )


def _coerce_predictions(raw: Any, gold: Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(raw, Mapping) and "predictions" in raw:
        preds = raw["predictions"]
    else:
        preds = raw

    gold_ids = list(gold.keys())

    if isinstance(preds, Mapping):
        pred_map = {str(k): v for k, v in preds.items()}
    elif isinstance(preds, Sequence) and not isinstance(preds, (str, bytes)):
        if all(isinstance(item, Mapping) for item in preds):
            pred_map = {}
            for item in preds:
                item_id = str(item.get("id", "")).strip()
                if not item_id:
                    raise ValueError("Each prediction object must contain a non-empty 'id'")
                if item_id in pred_map:
                    raise ValueError(f"Duplicate prediction id: {item_id}")
                pred_map[item_id] = item.get("answer")
        else:
            if len(preds) != len(gold_ids):
                raise ValueError(
                    f"Prediction count mismatch: got {len(preds)}, expected {len(gold_ids)}"
                )
            pred_map = {item_id: answer for item_id, answer in zip(gold_ids, preds)}
    else:
        raise ValueError("Unsupported predictions format")

    missing = [item_id for item_id in gold_ids if item_id not in pred_map]
    if missing:
        raise ValueError(f"Missing predictions for ids: {missing[:5]}")
    return pred_map


def _normalize_answer(value: Any) -> str:
    if value is None:
        return ""

    text = str(value).strip()
    if not text:
        return ""

    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        text = boxed[-1]

    text = text.replace(",", " ")
    numbers = re.findall(r"-?\d+", text)
    if numbers:
        normalized = numbers[-1].lstrip("0")
        return normalized or "0"

    compact = re.sub(r"\s+", "", text)
    return compact
