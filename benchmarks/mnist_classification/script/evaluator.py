"""MNIST evaluator: compare predicted class indices against ground truth."""

from __future__ import annotations

import json
import os
from collections import Counter

import torch

from farbench.evaluator import EvalResult, MetricEvaluatorBase
from farbench.schemas import TaskConfig


class MNISTEvaluator(MetricEvaluatorBase):
    """Compare agent predictions against MNIST test labels."""

    def evaluate(
        self,
        predictions_path: str,
        test_data_dir: str,
        task_config: TaskConfig,
    ) -> EvalResult:
        # Load predictions
        with open(predictions_path) as f:
            data = json.load(f)
        preds = data["predictions"]

        # Load ground truth
        test_path = os.path.join(test_data_dir, "test.pt")
        raw = torch.load(test_path, map_location="cpu", weights_only=True)
        targets = raw["labels"].tolist()

        if len(preds) != len(targets):
            raise ValueError(
                f"Prediction count mismatch: got {len(preds)}, expected {len(targets)}"
            )

        # Accuracy
        correct = sum(p == t for p, t in zip(preds, targets))
        accuracy = correct / len(targets)

        # Macro F1
        f1 = _compute_macro_f1(targets, preds, num_classes=10)

        metrics = {
            "accuracy": round(accuracy, 4),
            "f1_score": round(f1, 4),
        }

        return EvalResult(
            metrics=metrics,
            primary_metric_name="accuracy",
            primary_metric_value=accuracy,
        )


def _compute_macro_f1(
    targets: list[int], preds: list[int], num_classes: int
) -> float:
    """Compute macro-averaged F1 score."""
    f1_scores = []
    for cls in range(num_classes):
        tp = sum(1 for p, t in zip(preds, targets) if p == cls and t == cls)
        fp = sum(1 for p, t in zip(preds, targets) if p == cls and t != cls)
        fn = sum(1 for p, t in zip(preds, targets) if p != cls and t == cls)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)

    return sum(f1_scores) / num_classes
