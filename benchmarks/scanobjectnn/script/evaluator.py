"""ScanObjectNN evaluator: overall accuracy and mean class accuracy on PB_T50_RS test set."""

from __future__ import annotations

import json
import os

import torch

from farbench.evaluator import EvalResult, MetricEvaluatorBase
from farbench.schemas import TaskConfig


class ScanObjectNNEvaluator(MetricEvaluatorBase):
    """Evaluate point cloud classification predictions on ScanObjectNN test set."""

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

        # Load metadata for class info
        with open(os.path.join(test_data_dir, "meta.json")) as f:
            meta = json.load(f)
        num_classes = meta["num_classes"]
        class_names = meta["class_names"]

        # Overall Accuracy
        correct = sum(p == t for p, t in zip(preds, targets))
        overall_accuracy = correct / len(targets)

        # Mean class Accuracy (mAcc)
        per_class_correct = [0] * num_classes
        per_class_total = [0] * num_classes
        for p, t in zip(preds, targets):
            per_class_total[t] += 1
            if p == t:
                per_class_correct[t] += 1

        per_class_acc = []
        for i in range(num_classes):
            if per_class_total[i] > 0:
                per_class_acc.append(per_class_correct[i] / per_class_total[i])
            else:
                per_class_acc.append(0.0)

        mean_class_accuracy = sum(per_class_acc) / len(per_class_acc)

        metrics = {
            "overall_accuracy": round(overall_accuracy, 4),
            "mean_class_accuracy": round(mean_class_accuracy, 4),
        }

        # Per-class breakdown
        for i, (name, acc) in enumerate(zip(class_names, per_class_acc)):
            metrics[f"class_{name}_accuracy"] = round(acc, 4)

        return EvalResult(
            metrics=metrics,
            primary_metric_name="overall_accuracy",
            primary_metric_value=overall_accuracy,
        )
