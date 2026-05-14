"""Split-CIFAR-100 evaluator: average accuracy over all classes after sequential training."""

from __future__ import annotations

import json
import os

import torch

from farbench.evaluator import EvalResult, MetricEvaluatorBase
from farbench.schemas import TaskConfig


class SplitCIFAR100Evaluator(MetricEvaluatorBase):
    """Evaluate class-incremental learning predictions on CIFAR-100 test set."""

    def evaluate(
        self,
        predictions_path: str,
        test_data_dir: str,
        task_config: TaskConfig,
    ) -> EvalResult:
        # Load predictions
        with open(predictions_path) as f:
            preds = json.load(f)["predictions"]

        # Load ground truth
        test_data = torch.load(
            os.path.join(test_data_dir, "test.pt"),
            map_location="cpu",
            weights_only=True,
        )
        targets = test_data["labels"].tolist()

        if len(preds) != len(targets):
            raise ValueError(
                f"Prediction count mismatch: got {len(preds)}, expected {len(targets)}"
            )

        # Load task metadata for per-task breakdown
        with open(os.path.join(test_data_dir, "meta.json")) as f:
            meta = json.load(f)
        classes_per_task = meta["classes_per_task"]

        # Overall accuracy = Average Accuracy (balanced since 100 images/class)
        correct = sum(p == t for p, t in zip(preds, targets))
        average_accuracy = correct / len(targets)

        # Per-task accuracy
        task_accuracies = []
        for task_idx, task_classes in enumerate(classes_per_task):
            task_class_set = set(task_classes)
            task_correct = 0
            task_total = 0
            for p, t in zip(preds, targets):
                if t in task_class_set:
                    task_total += 1
                    if p == t:
                        task_correct += 1
            task_acc = task_correct / task_total if task_total > 0 else 0.0
            task_accuracies.append(task_acc)

        worst_task_accuracy = min(task_accuracies)

        metrics = {
            "average_accuracy": round(average_accuracy, 4),
            "worst_task_accuracy": round(worst_task_accuracy, 4),
        }

        # Add per-task breakdown
        for i, acc in enumerate(task_accuracies):
            metrics[f"task_{i}_accuracy"] = round(acc, 4)

        return EvalResult(
            metrics=metrics,
            primary_metric_name="average_accuracy",
            primary_metric_value=average_accuracy,
        )
