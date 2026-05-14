"""DomainNet Quickdraw evaluator: compare predicted class indices against ground truth.

Primary metric is plain top-1 accuracy, matching DomainBed / ERM++ / DPSPG / High-Rate Mixout.
Quickdraw's test set is class-balanced (~500 imgs per class × 345 classes), so
balanced_accuracy is reported as a secondary metric for reference but differs from
accuracy by less than 1 percentage point in practice.
"""

from __future__ import annotations

import json
import os

from sklearn.metrics import accuracy_score, balanced_accuracy_score

from farbench.evaluator import EvalResult, MetricEvaluatorBase
from farbench.schemas import TaskConfig


class DomainNetEvaluator(MetricEvaluatorBase):
    """Compare agent predictions against DomainNet Quickdraw test labels."""

    def evaluate(
        self,
        predictions_path: str,
        test_data_dir: str,
        task_config: TaskConfig,
    ) -> EvalResult:
        with open(predictions_path) as f:
            preds = json.load(f)["predictions"]

        labels_path = os.path.join(test_data_dir, "test_labels.txt")
        targets = []
        with open(labels_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    targets.append(int(line))

        if len(preds) != len(targets):
            raise ValueError(
                f"Prediction count mismatch: got {len(preds)}, expected {len(targets)}"
            )

        acc = accuracy_score(targets, preds)
        bal_acc = balanced_accuracy_score(targets, preds)

        metrics = {
            "accuracy": round(acc, 4),
            "balanced_accuracy": round(bal_acc, 4),
        }

        return EvalResult(
            metrics=metrics,
            primary_metric_name="accuracy",
            primary_metric_value=acc,
        )
