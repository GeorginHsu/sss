"""CIFAR-100N evaluator: accuracy on the clean CIFAR-100 test set.

Primary metric: accuracy (top-1, the universal metric in noisy label papers).
Secondary metric: balanced_accuracy (identical on the balanced test set).

Reference: Wei et al., "Learning with Noisy Labels Revisited" (ICLR 2022).
    All papers report test accuracy on the clean balanced CIFAR-100 test set
    (10K images, 100 per class).
"""

from __future__ import annotations

import json
import os

import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from farbench.evaluator import EvalResult, MetricEvaluatorBase
from farbench.schemas import TaskConfig


class CIFAR100NEvaluator(MetricEvaluatorBase):
    """Evaluate classification on CIFAR-100N (clean test set)."""

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

        # Load ground truth (clean labels)
        test_path = os.path.join(test_data_dir, "test.pt")
        raw = torch.load(test_path, map_location="cpu", weights_only=True)
        targets = raw["labels"].tolist()

        if len(preds) != len(targets):
            raise ValueError(
                f"Prediction count mismatch: got {len(preds)}, expected {len(targets)}"
            )

        # Accuracy (primary) — matches what all noisy label papers report
        acc = float(accuracy_score(targets, preds))

        # Balanced accuracy (= accuracy on this balanced test set, included for consistency)
        bal_acc = float(balanced_accuracy_score(targets, preds))

        metrics = {
            "accuracy": round(acc, 4),
            "balanced_accuracy": round(bal_acc, 4),
        }

        return EvalResult(
            metrics=metrics,
            primary_metric_name="accuracy",
            primary_metric_value=acc,
        )
