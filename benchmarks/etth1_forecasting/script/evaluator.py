"""ETTh1 forecasting evaluator: compare predicted future values against ground truth.

The agent's predict.py outputs a nested list of shape [N, 720, 7] containing raw
(denormalized) predicted values for each test window. The evaluator normalizes
both predictions and ground truth using training-set statistics (StandardScaler),
then computes MSE (primary) and MAE (secondary) on the normalized values to
match the standard benchmark protocol.
"""

from __future__ import annotations

import json
import os

import numpy as np

from farbench.evaluator import EvalResult, MetricEvaluatorBase
from farbench.schemas import TaskConfig


class ETTh1Evaluator(MetricEvaluatorBase):
    """Evaluate multivariate time series forecasting on ETTh1."""

    def evaluate(
        self,
        predictions_path: str,
        test_data_dir: str,
        task_config: TaskConfig,
    ) -> EvalResult:
        # Load predictions
        with open(predictions_path) as f:
            data = json.load(f)
        preds = np.array(data["predictions"], dtype=np.float32)

        # Load ground truth
        labels_path = os.path.join(test_data_dir, "test_labels.npy")
        targets = np.load(labels_path)

        # Validate shape
        if preds.shape != targets.shape:
            raise ValueError(
                f"Prediction shape mismatch: got {preds.shape}, "
                f"expected {targets.shape}"
            )

        # Load normalization stats (training set mean/std)
        norm_path = os.path.join(test_data_dir, "norm_stats.json")
        with open(norm_path) as f:
            norm = json.load(f)
        mean = np.array(norm["mean"], dtype=np.float32)
        std = np.array(norm["std"], dtype=np.float32)

        # Normalize both predictions and ground truth
        std_safe = np.where(std == 0, 1.0, std)
        preds_norm = (preds - mean) / std_safe
        targets_norm = (targets - mean) / std_safe

        # MSE (primary metric) — mean over all windows, timesteps, features
        mse = float(np.mean((preds_norm - targets_norm) ** 2))

        # MAE (secondary metric)
        mae = float(np.mean(np.abs(preds_norm - targets_norm)))

        metrics = {
            "mse": round(mse, 4),
            "mae": round(mae, 4),
        }

        return EvalResult(
            metrics=metrics,
            primary_metric_name="mse",
            primary_metric_value=mse,
        )
