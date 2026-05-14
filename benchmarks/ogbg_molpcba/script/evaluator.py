"""ogbg-molpcba evaluator: compute Average Precision across 128 binary tasks.

Uses sklearn's average_precision_score with NaN masking, matching the OGB
official evaluation protocol. AP is computed per-task then averaged across
all 128 tasks (only tasks with at least one positive and one negative label
contribute to the average).
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from farbench.evaluator import EvalResult, MetricEvaluatorBase
from farbench.schemas import TaskConfig


class OGBGMolpcbaEvaluator(MetricEvaluatorBase):
    """Evaluate multi-task molecular property predictions via Average Precision."""

    def evaluate(
        self,
        predictions_path: str,
        test_data_dir: str,
        task_config: TaskConfig,
    ) -> EvalResult:
        # 1. Load predictions
        with open(predictions_path) as f:
            data = json.load(f)
        preds_raw = data["predictions"]

        # 2. Load ground truth labels
        labels_path = os.path.join(test_data_dir, "test_labels.pt")
        labels_data = torch.load(labels_path, map_location="cpu", weights_only=True)
        targets_list = labels_data["labels"]  # list of [128] tensors
        targets = torch.stack(targets_list).numpy()  # [N, 128]

        # 3. Validate count
        if len(preds_raw) != len(targets):
            raise ValueError(
                f"Prediction count mismatch: got {len(preds_raw)}, "
                f"expected {len(targets)} molecules"
            )

        # 4. Convert predictions to numpy array (handle null -> NaN)
        n_samples = len(preds_raw)
        n_tasks = targets.shape[1]
        preds = np.full((n_samples, n_tasks), np.nan, dtype=np.float64)
        for i, row in enumerate(preds_raw):
            if len(row) != n_tasks:
                raise ValueError(
                    f"Molecule {i}: expected {n_tasks} predictions, got {len(row)}"
                )
            for j, val in enumerate(row):
                if val is not None:
                    preds[i, j] = float(val)

        # 5. Compute per-task AP (skip tasks with no labels or no predictions)
        ap_list = []
        for task_idx in range(n_tasks):
            y_true = targets[:, task_idx]
            y_pred = preds[:, task_idx]

            # Mask: both ground truth and prediction must be non-NaN
            valid = ~np.isnan(y_true) & ~np.isnan(y_pred)
            if valid.sum() == 0:
                continue

            y_true_valid = y_true[valid]
            y_pred_valid = y_pred[valid]

            # Need at least one positive and one negative for AP
            if y_true_valid.sum() == 0 or y_true_valid.sum() == len(y_true_valid):
                continue

            ap = average_precision_score(y_true_valid, y_pred_valid)
            ap_list.append(ap)

        if len(ap_list) == 0:
            raise ValueError("No valid tasks found for AP computation")

        avg_ap = float(np.mean(ap_list))

        # 6. Compute per-task ROC-AUC as secondary metric
        from sklearn.metrics import roc_auc_score

        auc_list = []
        for task_idx in range(n_tasks):
            y_true = targets[:, task_idx]
            y_pred = preds[:, task_idx]

            valid = ~np.isnan(y_true) & ~np.isnan(y_pred)
            if valid.sum() == 0:
                continue

            y_true_valid = y_true[valid]
            y_pred_valid = y_pred[valid]

            if y_true_valid.sum() == 0 or y_true_valid.sum() == len(y_true_valid):
                continue

            auc = roc_auc_score(y_true_valid, y_pred_valid)
            auc_list.append(auc)

        avg_auc = float(np.mean(auc_list)) if auc_list else 0.0

        metrics = {
            "avg_precision": round(avg_ap, 4),
            "avg_roc_auc": round(avg_auc, 4),
            "num_evaluated_tasks": len(ap_list),
        }

        return EvalResult(
            metrics=metrics,
            primary_metric_name="avg_precision",
            primary_metric_value=avg_ap,
        )
