"""TerraIncognita evaluator: balanced accuracy on the L100 test domain.

Primary metric: balanced_accuracy (macro-average of per-class recall).
L100 is heavily class-imbalanced (52% raccoon, 20% opossum), so plain
accuracy inflates scores. Balanced accuracy gives equal weight to each class,
better reflecting cross-domain generalization quality.

Secondary: plain accuracy, macro_f1, per-class accuracy breakdown.

Note: DomainBed convention uses plain accuracy, but FARBench adopts balanced
accuracy for TerraIncognita to avoid majority-class bias (consistent with
FARBench's domainnet_quickdraw which also uses balanced_accuracy).
"""

from __future__ import annotations

import json
import os
from collections import Counter

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from farbench.evaluator import EvalResult, MetricEvaluatorBase
from farbench.schemas import TaskConfig

CLASS_NAMES = [
    "bird", "bobcat", "cat", "coyote", "dog",
    "empty", "opossum", "rabbit", "raccoon", "squirrel",
]


class TerraIncognitaEvaluator(MetricEvaluatorBase):
    """Compare agent predictions against TerraIncognita test labels."""

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

        # Balanced accuracy (primary) — macro-average of per-class recall
        bal_acc = float(balanced_accuracy_score(targets, preds))

        # Plain accuracy (secondary, for reference vs DomainBed papers)
        acc = float(accuracy_score(targets, preds))

        macro_f1 = float(f1_score(targets, preds, average="macro"))

        # Per-class accuracy
        class_correct = Counter()
        class_total = Counter()
        for t, p in zip(targets, preds):
            class_total[t] += 1
            if t == p:
                class_correct[t] += 1

        metrics = {
            "balanced_accuracy": round(bal_acc, 4),
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
        }

        # Add per-class accuracy for diagnostics
        for cls_idx in sorted(class_total.keys()):
            name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else str(cls_idx)
            cls_acc = class_correct[cls_idx] / class_total[cls_idx] if class_total[cls_idx] > 0 else 0.0
            metrics[f"acc_{name}"] = round(cls_acc, 4)

        return EvalResult(
            metrics=metrics,
            primary_metric_name="balanced_accuracy",
            primary_metric_value=bal_acc,
        )
