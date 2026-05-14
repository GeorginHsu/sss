"""iWildCam-WILDS evaluator: compute macro F1 on OOD test set.

Macro F1 = unweighted mean of per-class F1 scores, restricted to the set of
classes that actually appear in the test-set ground truth. This matches the
official WILDS metric in wilds/common/metrics/all_metrics.py (F1 class, which
passes labels=torch.unique(y_true) to sklearn.metrics.f1_score). NOT passing
labels= would let sklearn include any extra class the model predicted, which
would drag the macro average down and make our numbers incomparable with
published baselines (ERM 30.8, FLYP 46.0, DRM 51.4, AutoFT 52.0).
"""

from __future__ import annotations

import json
import os

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score

from farbench.evaluator import EvalResult, MetricEvaluatorBase
from farbench.schemas import TaskConfig


class IWildCamWILDSEvaluator(MetricEvaluatorBase):
    """Compare agent predictions against iWildCam-WILDS OOD test labels."""

    def evaluate(
        self,
        predictions_path: str,
        test_data_dir: str,
        task_config: TaskConfig,
    ) -> EvalResult:
        # Load predictions
        with open(predictions_path) as f:
            preds = json.load(f)["predictions"]

        # Load ground truth labels
        labels_path = os.path.join(test_data_dir, "test_labels.txt")
        targets = []
        with open(labels_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    targets.append(int(line))

        if len(preds) != len(targets):
            raise ValueError(
                f"Prediction count mismatch: got {len(preds)}, expected {len(targets)}"
            )

        # Restrict macro-average to classes present in y_true — mirrors
        # wilds.common.metrics.all_metrics.F1 (labels=torch.unique(y_true)).
        y_true_classes = np.unique(targets)

        macro_f1 = float(f1_score(
            targets, preds,
            labels=y_true_classes, average="macro", zero_division=0,
        ))
        macro_recall = float(recall_score(
            targets, preds,
            labels=y_true_classes, average="macro", zero_division=0,
        ))
        accuracy = float(accuracy_score(targets, preds))

        metrics = {
            "macro_f1": round(macro_f1, 4),
            "accuracy": round(accuracy, 4),
            "macro_recall": round(macro_recall, 4),
            "n_eval_classes": int(len(y_true_classes)),
        }

        return EvalResult(
            metrics=metrics,
            primary_metric_name="macro_f1",
            primary_metric_value=macro_f1,
        )
