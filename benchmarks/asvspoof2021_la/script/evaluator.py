"""ASVspoof 2021 LA evaluator: compute EER from bonafide/spoof scores."""

from __future__ import annotations

import json
import os

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve

from farbench.evaluator import EvalResult, MetricEvaluatorBase
from farbench.schemas import TaskConfig


def compute_eer(labels: list[int], scores: list[float]) -> float:
    """Compute Equal Error Rate.

    Args:
        labels: 1 for bonafide, 0 for spoof.
        scores: higher = more likely bonafide.

    Returns:
        EER as a fraction (0.0 to 1.0).
    """
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    return float(eer)


class ASVspoofEvaluator(MetricEvaluatorBase):
    """Evaluate audio deepfake detection using EER (Equal Error Rate)."""

    def evaluate(
        self,
        predictions_path: str,
        test_data_dir: str,
        task_config: TaskConfig,
    ) -> EvalResult:
        # Load predictions (list of float scores, one per test utterance)
        with open(predictions_path) as f:
            preds = json.load(f)["predictions"]

        # Load ground truth labels
        labels_path = os.path.join(test_data_dir, "test_labels.txt")
        utt_ids = []
        labels = []
        with open(labels_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                utt_ids.append(parts[0])
                # 1 = bonafide, 0 = spoof
                labels.append(1 if parts[1] == "bonafide" else 0)

        if len(preds) != len(labels):
            raise ValueError(
                f"Prediction count mismatch: got {len(preds)}, expected {len(labels)}"
            )

        scores = [float(s) for s in preds]
        eer = compute_eer(labels, scores)

        # Compute accuracy at EER threshold as secondary metric
        fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
        eer_threshold = float(
            interp1d(fpr, thresholds)(
                brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
            )
        )
        predictions_binary = [1 if s >= eer_threshold else 0 for s in scores]
        acc = sum(
            p == l for p, l in zip(predictions_binary, labels)
        ) / len(labels)

        metrics = {
            "eer": round(eer, 4),
            "accuracy_at_eer": round(acc, 4),
        }

        return EvalResult(
            metrics=metrics,
            primary_metric_name="eer",
            primary_metric_value=eer,
        )
