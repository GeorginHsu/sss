"""FLIP AAV evaluator: Spearman rank correlation for protein fitness prediction."""

from __future__ import annotations

import csv
import json
import os

import numpy as np
from scipy.stats import spearmanr

from farbench.evaluator import EvalResult, MetricEvaluatorBase
from farbench.schemas import TaskConfig


class FLIPAAVEvaluator(MetricEvaluatorBase):
    """Compare agent fitness predictions against ground truth (Spearman ρ)."""

    def evaluate(
        self,
        predictions_path: str,
        test_data_dir: str,
        task_config: TaskConfig,
    ) -> EvalResult:
        # Load predictions
        with open(predictions_path) as f:
            preds = json.load(f)["predictions"]
        preds = np.array(preds, dtype=np.float64)

        # Load ground truth
        targets = []
        labels_path = os.path.join(test_data_dir, "test_labels.csv")
        with open(labels_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                targets.append(float(row["target"]))
        targets = np.array(targets, dtype=np.float64)

        if len(preds) != len(targets):
            raise ValueError(
                f"Prediction count mismatch: got {len(preds)}, expected {len(targets)}"
            )

        # Spearman rank correlation
        rho, p_value = spearmanr(preds, targets)

        # MSE and MAE as secondary metrics
        mse = float(np.mean(np.square(preds - targets)))
        mae = float(np.mean(np.abs(preds - targets)))

        # Pearson correlation
        pearson = float(np.corrcoef(preds, targets)[0, 1])

        metrics = {
            "spearman_rho": round(float(rho), 4),
            "pearson_r": round(pearson, 4),
            "mse": round(mse, 4),
            "mae": round(mae, 4),
        }

        return EvalResult(
            metrics=metrics,
            primary_metric_name="spearman_rho",
            primary_metric_value=float(rho),
        )
