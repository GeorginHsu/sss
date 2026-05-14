"""CIFAR-100-LT evaluator: balanced accuracy + per-group (Many/Medium/Few) accuracy.

Primary metric: balanced_accuracy (macro-average of per-class recall).
Secondary metrics: many_shot_acc, medium_shot_acc, few_shot_acc.

Group definitions (based on training set class counts):
    Many-shot:   classes with > 100 training samples
    Medium-shot: classes with 20-100 training samples
    Few-shot:    classes with < 20 training samples
"""

from __future__ import annotations

import json
import os

import torch
from sklearn.metrics import balanced_accuracy_score

from farbench.evaluator import EvalResult, MetricEvaluatorBase
from farbench.schemas import TaskConfig

# Long-tailed class counts for IR=100, C=100, n_max=490 (500-10 val)
# n_i = 490 * (1/100)^(i/99)
# Pre-computed group boundaries:
#   Many (>100): classes 0-33   (34 classes)
#   Medium (20-100): classes 34-68 (35 classes)
#   Few (<20): classes 69-99   (31 classes)


def _compute_class_counts(num_classes: int = 100, max_samples: int = 490,
                          imbalance_ratio: int = 100) -> list[int]:
    """Recompute per-class counts to determine group membership."""
    counts = []
    for i in range(num_classes):
        n_i = int(max_samples * (1.0 / imbalance_ratio) ** (i / (num_classes - 1)))
        n_i = max(n_i, 1)
        counts.append(n_i)
    return counts


def _get_group_classes(counts: list[int]):
    """Split classes into Many/Medium/Few groups based on training counts."""
    many, medium, few = [], [], []
    for cls_idx, count in enumerate(counts):
        if count > 100:
            many.append(cls_idx)
        elif count >= 20:
            medium.append(cls_idx)
        else:
            few.append(cls_idx)
    return many, medium, few


def _group_accuracy(targets: list[int], preds: list[int],
                    group_classes: list[int]) -> float:
    """Compute accuracy on test samples whose ground truth is in group_classes."""
    group_set = set(group_classes)
    correct = 0
    total = 0
    for t, p in zip(targets, preds):
        if t in group_set:
            total += 1
            if p == t:
                correct += 1
    return correct / total if total > 0 else 0.0


class CIFAR100LTEvaluator(MetricEvaluatorBase):
    """Evaluate long-tailed classification on CIFAR-100-LT."""

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

        # Balanced accuracy (primary) — macro-average of per-class recall
        bal_acc = float(balanced_accuracy_score(targets, preds))

        # Overall accuracy
        correct = sum(p == t for p, t in zip(preds, targets))
        accuracy = correct / len(targets)

        # Per-group accuracy
        counts = _compute_class_counts()
        many_cls, medium_cls, few_cls = _get_group_classes(counts)
        many_acc = _group_accuracy(targets, preds, many_cls)
        medium_acc = _group_accuracy(targets, preds, medium_cls)
        few_acc = _group_accuracy(targets, preds, few_cls)

        metrics = {
            "balanced_accuracy": round(bal_acc, 4),
            "accuracy": round(accuracy, 4),
            "many_shot_acc": round(many_acc, 4),
            "medium_shot_acc": round(medium_acc, 4),
            "few_shot_acc": round(few_acc, 4),
        }

        return EvalResult(
            metrics=metrics,
            primary_metric_name="balanced_accuracy",
            primary_metric_value=bal_acc,
        )
