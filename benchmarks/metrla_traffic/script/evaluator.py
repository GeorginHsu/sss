"""METR-LA evaluator: MAE, RMSE, MAPE for traffic speed prediction."""

from __future__ import annotations

import json
import os

import numpy as np

from farbench.evaluator import EvalResult, MetricEvaluatorBase
from farbench.schemas import TaskConfig


class METRLAEvaluator(MetricEvaluatorBase):
    """Compare agent predictions against METR-LA test ground truth."""

    def evaluate(
        self,
        predictions_path: str,
        test_data_dir: str,
        task_config: TaskConfig,
    ) -> EvalResult:
        # Load predictions (file-based format)
        with open(predictions_path) as f:
            meta = json.load(f)

        preds_dir = meta["predictions_dir"]
        preds_file = os.path.join(preds_dir, "predictions.npy")
        if not os.path.exists(preds_file):
            raise FileNotFoundError(
                f"predictions.npy not found in {preds_dir}"
            )
        preds = np.load(preds_file).astype(np.float64)  # [K, 12, 207]

        # Load ground truth
        targets = np.load(
            os.path.join(test_data_dir, "test_y.npy")
        ).astype(np.float64)  # [K, 12, 207]

        if preds.shape != targets.shape:
            raise ValueError(
                f"Shape mismatch: predictions {preds.shape} vs targets {targets.shape}"
            )

        # Mask out zero/near-zero targets for MAPE (avoid division by zero)
        mask = targets > 1e-3

        # Overall metrics (across all windows, steps, sensors)
        mae = float(np.abs(preds - targets).mean())
        rmse = float(np.sqrt(np.square(preds - targets).mean()))
        mape = float(np.abs((preds[mask] - targets[mask]) / targets[mask]).mean() * 100)

        # Per-horizon metrics (average across windows and sensors for each step)
        horizon_mae = []
        for h in range(preds.shape[1]):
            h_mae = float(np.abs(preds[:, h, :] - targets[:, h, :]).mean())
            horizon_mae.append(round(h_mae, 4))

        # Report metrics at key horizons: 15min (step 3), 30min (step 6), 60min (step 12)
        mae_15 = horizon_mae[2] if len(horizon_mae) > 2 else None   # step 3
        mae_30 = horizon_mae[5] if len(horizon_mae) > 5 else None   # step 6
        mae_60 = horizon_mae[11] if len(horizon_mae) > 11 else None  # step 12

        metrics = {
            "mae_avg": round(mae, 4),
            "rmse": round(rmse, 4),
            "mape": round(mape, 4),
        }
        if mae_15 is not None:
            metrics["mae_15min"] = mae_15
        if mae_30 is not None:
            metrics["mae_30min"] = mae_30
        if mae_60 is not None:
            metrics["mae_60min"] = mae_60

        # Use horizon-12 (60min) MAE as primary metric to match
        # traffic forecasting literature convention (DCRNN, GWNet, STAEformer, etc.)
        primary_value = mae_60 if mae_60 is not None else mae

        return EvalResult(
            metrics=metrics,
            primary_metric_name="mae_60min",
            primary_metric_value=primary_value,
        )
