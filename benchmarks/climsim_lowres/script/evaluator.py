"""ClimSim low-res evaluator: compute per-target R² and mean R².

Compares agent predictions against scoring_target.npy (ground truth).
Both are in pre-normalized space, matching the NeurIPS 2023 paper protocol.

Output groups for secondary metrics (128 targets indexed 0-127):
  - ptend_t:       temperature tendency (indices 0-59, 60 vertical levels)
  - ptend_q0001:   humidity tendency (indices 60-119, 60 vertical levels)
  - radiation:     NETSW(120), FLWDS(121), SOLS(124), SOLL(125), SOLSD(126), SOLLD(127)
  - precipitation: PRECSC(122), PRECC(123)
"""

from __future__ import annotations

import json
import os

import numpy as np
from farbench.evaluator import EvalResult, MetricEvaluatorBase
from farbench.schemas import TaskConfig

PTEND_T_SLICE = slice(0, 60)
PTEND_Q_SLICE = slice(60, 120)
SCALAR_RADIATION_IDX = [120, 121, 124, 125, 126, 127]  # NETSW, FLWDS, SOLS, SOLL, SOLSD, SOLLD
SCALAR_PRECIP_IDX = [122, 123]  # PRECSC, PRECC


def _r2_per_target(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - y_true.mean(axis=0)) ** 2, axis=0)
    mask = ss_tot > 0
    r2 = np.full(y_true.shape[1], float("nan"))
    r2[mask] = 1.0 - ss_res[mask] / ss_tot[mask]
    return r2


class ClimSimEvaluator(MetricEvaluatorBase):
    def evaluate(
        self,
        predictions_path: str,
        test_data_dir: str,
        task_config: TaskConfig,
    ) -> EvalResult:
        with open(predictions_path) as f:
            meta = json.load(f)
        pred_file = meta["predictions_file"]
        preds = np.load(pred_file).astype(np.float64)

        targets_path = os.path.join(test_data_dir, "scoring_target.npy")
        targets = np.load(targets_path).astype(np.float64)

        if preds.shape != targets.shape:
            raise ValueError(
                f"Prediction shape mismatch: got {preds.shape}, expected {targets.shape}"
            )

        r2 = _r2_per_target(targets, preds)
        mean_r2 = float(np.nanmean(r2))

        r2_ptend_t = float(np.nanmean(r2[PTEND_T_SLICE]))
        r2_ptend_q = float(np.nanmean(r2[PTEND_Q_SLICE]))
        r2_radiation = float(np.nanmean(r2[SCALAR_RADIATION_IDX]))
        r2_precip = float(np.nanmean(r2[SCALAR_PRECIP_IDX]))

        metrics = {
            "mean_r2": round(mean_r2, 4),
            "r2_ptend_t": round(r2_ptend_t, 4),
            "r2_ptend_q": round(r2_ptend_q, 4),
            "r2_radiation": round(r2_radiation, 4),
            "r2_precipitation": round(r2_precip, 4),
        }

        return EvalResult(
            metrics=metrics,
            primary_metric_name="mean_r2",
            primary_metric_value=mean_r2,
        )
