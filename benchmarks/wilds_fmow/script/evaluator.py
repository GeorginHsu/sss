"""WILDS-FMoW evaluator: compute worst-region accuracy on OOD test set.

Worst-region accuracy = min(per-region accuracy) across the 5 geographic
regions (Africa, Americas, Asia, Europe, Oceania). Samples with region=-1
("Other") are excluded from the worst-region computation but included in
overall accuracy.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

from sklearn.metrics import accuracy_score

from farbench.evaluator import EvalResult, MetricEvaluatorBase
from farbench.schemas import TaskConfig

REGION_NAMES = ["Africa", "Americas", "Asia", "Europe", "Oceania"]


class WildsFMoWEvaluator(MetricEvaluatorBase):
    """Compare agent predictions against WILDS-FMoW OOD test labels."""

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

        # Load region IDs
        regions_path = os.path.join(test_data_dir, "test_regions.txt")
        regions = []
        with open(regions_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    regions.append(int(line))

        if len(preds) != len(targets):
            raise ValueError(
                f"Prediction count mismatch: got {len(preds)}, expected {len(targets)}"
            )

        if len(regions) != len(targets):
            raise ValueError(
                f"Region count mismatch: got {len(regions)}, expected {len(targets)}"
            )

        # Overall accuracy
        overall_acc = accuracy_score(targets, preds)

        # Per-region accuracy (exclude "Other" region with id=-1)
        region_preds = defaultdict(list)
        region_targets = defaultdict(list)
        for p, t, r in zip(preds, targets, regions):
            if 0 <= r < len(REGION_NAMES):
                region_preds[r].append(p)
                region_targets[r].append(t)

        per_region_acc = {}
        for r_id in range(len(REGION_NAMES)):
            if r_id in region_targets and len(region_targets[r_id]) > 0:
                acc = accuracy_score(region_targets[r_id], region_preds[r_id])
                per_region_acc[REGION_NAMES[r_id]] = acc

        # Worst-region accuracy = min across all regions
        if per_region_acc:
            worst_region_acc = min(per_region_acc.values())
            worst_region_name = min(per_region_acc, key=per_region_acc.get)
        else:
            worst_region_acc = 0.0
            worst_region_name = "N/A"

        metrics = {
            "worst_region_accuracy": round(worst_region_acc, 4),
            "overall_accuracy": round(overall_acc, 4),
            "worst_region": worst_region_name,
        }
        # Add per-region breakdown
        for name, acc in per_region_acc.items():
            metrics[f"accuracy_{name.lower()}"] = round(acc, 4)

        return EvalResult(
            metrics=metrics,
            primary_metric_name="worst_region_accuracy",
            primary_metric_value=worst_region_acc,
        )
