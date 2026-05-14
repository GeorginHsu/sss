"""Knowledge Tracing evaluator: compare predicted probabilities against ground truth."""

from __future__ import annotations

import json
import os

import torch
from sklearn.metrics import roc_auc_score

from farbench.evaluator import EvalResult, MetricEvaluatorBase
from farbench.schemas import TaskConfig


class KTEvaluator(MetricEvaluatorBase):
    """Compare agent predictions against KT test labels.

    The agent's predict.py should output a flat list of probabilities,
    one per valid prediction step (after masking by sequence lengths).
    The evaluator independently computes the masked ground truth targets
    from test.pt and compares.
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

        # Load ground truth and compute masked targets
        test_path = os.path.join(test_data_dir, "test.pt")
        raw = torch.load(test_path, map_location="cpu", weights_only=True)
        corrects = raw["corrects"]   # (N, seq_len)
        lengths = raw["lengths"]     # (N,)

        # Target: corrects at step t+1 (shift by 1), masked by valid length
        # Same masking logic the agent should use in predict.py
        seq_len = corrects.shape[1] - 1  # predictions are for steps 1..T
        mask = torch.arange(seq_len).unsqueeze(0) < (lengths.unsqueeze(1) - 1)
        targets = corrects[:, 1:]  # ground truth for steps 1..T

        target_list = targets[mask].tolist()

        if len(preds) != len(target_list):
            raise ValueError(
                f"Prediction count mismatch: got {len(preds)}, "
                f"expected {len(target_list)} (masked valid steps)"
            )

        # AUC-ROC (sklearn handles edge cases: returns 0.5 for single-class)
        auc = roc_auc_score(target_list, preds)

        # Accuracy (threshold 0.5)
        binary_preds = [1 if p > 0.5 else 0 for p in preds]
        acc = sum(p == t for p, t in zip(binary_preds, target_list)) / len(target_list)

        metrics = {
            "auc_roc": round(auc, 4),
            "accuracy": round(acc, 4),
        }

        return EvalResult(
            metrics=metrics,
            primary_metric_name="auc_roc",
            primary_metric_value=auc,
        )
