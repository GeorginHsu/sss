"""WeatherBench Z500+T850 evaluator: latitude-weighted RMSE.

Latitude-weighted RMSE accounts for the fact that grid cells near the equator
represent more surface area than cells near the poles on an equirectangular grid.

Formula (per variable, following Rasp et al. 2020):
    RMSE = (1/N) * sum_i sqrt( (1/(N_lat*N_lon)) * sum_j sum_k L(j) * (f - t)^2 )

where L(j) = cos(lat_j) / mean(cos(lat)) is the latitude weight.

Primary metric: rmse_z500 (latitude-weighted RMSE on Z500 in m^2/s^2)
Secondary metric: rmse_t850 (latitude-weighted RMSE on T850 in K)
"""

from __future__ import annotations

import json
import os

import numpy as np

from farbench.evaluator import EvalResult, MetricEvaluatorBase
from farbench.schemas import TaskConfig


def latitude_weighted_rmse(
    preds: np.ndarray,
    targets: np.ndarray,
    lat_weights: np.ndarray,
) -> float:
    """Compute latitude-weighted RMSE over spatial dimensions.

    Args:
        preds:   [N, 32, 64] predictions for one variable
        targets: [N, 32, 64] ground truth for one variable
        lat_weights: [32] normalized latitude weights (mean=1)

    Returns:
        Scalar RMSE averaged over all forecast samples.
    """
    # Squared error: [N, 32, 64]
    sq_err = (preds - targets) ** 2

    # Apply latitude weights: broadcast [32] -> [1, 32, 1]
    weighted_sq_err = sq_err * lat_weights[np.newaxis, :, np.newaxis]

    # Spatial mean for each sample: [N]
    spatial_mse = weighted_sq_err.mean(axis=(-2, -1))

    # RMSE per sample, then average
    rmse_per_sample = np.sqrt(spatial_mse)
    return float(rmse_per_sample.mean())


class WeatherBenchEvaluator(MetricEvaluatorBase):
    """Evaluate global weather field prediction on WeatherBench Z500+T850."""

    def evaluate(
        self,
        predictions_path: str,
        test_data_dir: str,
        task_config: TaskConfig,
    ) -> EvalResult:
        # Load predictions (file-based format: JSON points to .npy)
        with open(predictions_path) as f:
            meta = json.load(f)

        pred_file = meta["predictions_file"]
        preds = np.load(pred_file).astype(np.float32)

        # Load ground truth
        labels_path = os.path.join(test_data_dir, "test_labels.npy")
        targets = np.load(labels_path).astype(np.float32)

        # Validate shape
        if preds.shape != targets.shape:
            raise ValueError(
                f"Prediction shape mismatch: got {preds.shape}, "
                f"expected {targets.shape}"
            )

        # Load latitude values for weighting
        lat_path = os.path.join(test_data_dir, "lat.npy")
        lat = np.load(lat_path)

        # Compute latitude weights: L(j) = cos(lat_j) / mean(cos(lat))
        cos_lat = np.cos(np.deg2rad(lat))
        lat_weights = cos_lat / cos_lat.mean()

        # Compute latitude-weighted RMSE for each variable
        # Channel 0: Z500 (geopotential, m^2/s^2)
        rmse_z500 = latitude_weighted_rmse(preds[:, 0], targets[:, 0], lat_weights)

        # Channel 1: T850 (temperature, K)
        rmse_t850 = latitude_weighted_rmse(preds[:, 1], targets[:, 1], lat_weights)

        metrics = {
            "rmse_z500": round(rmse_z500, 4),
            "rmse_t850": round(rmse_t850, 4),
        }

        return EvalResult(
            metrics=metrics,
            primary_metric_name="rmse_z500",
            primary_metric_value=rmse_z500,
        )
