"""QM9 evaluator: compare predicted HOMO-LUMO gap values against ground truth."""

from __future__ import annotations

import csv
import json
import math
import os

from farbench.evaluator import EvalResult, MetricEvaluatorBase
from farbench.schemas import TaskConfig


class QM9Evaluator(MetricEvaluatorBase):
    """Compare agent predictions against QM9 test targets.

    The agent's predict.py should output a list of predicted HOMO-LUMO gap
    values (floats, in eV), one per molecule in the same order as test.csv rows.
    The evaluator loads ground truth from test_labels.csv and computes
    MAE (primary) and RMSE (secondary).
    """

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

        # Load ground truth targets
        labels_path = os.path.join(test_data_dir, "test_labels.csv")
        targets = []
        with open(labels_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                targets.append(float(row["target"]))

        if len(preds) != len(targets):
            raise ValueError(
                f"Prediction count mismatch: got {len(preds)}, "
                f"expected {len(targets)} molecules"
            )

        # MAE (primary metric)
        abs_errors = [abs(p - t) for p, t in zip(preds, targets)]
        mae = sum(abs_errors) / len(abs_errors)

        # RMSE (secondary metric)
        sq_errors = [(p - t) ** 2 for p, t in zip(preds, targets)]
        rmse = math.sqrt(sum(sq_errors) / len(sq_errors))

        metrics = {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
        }

        return EvalResult(
            metrics=metrics,
            primary_metric_name="mae",
            primary_metric_value=mae,
        )
