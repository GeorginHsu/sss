"""Qlib CSI300 stock prediction evaluator.

Reads the agent's predict.py output containing daily IC values
and computes aggregate metrics.

Expected predict.py output format:
    {
        "ic_mean": 0.045,
        "icir": 0.35,
        "rank_ic_mean": 0.052,
        "per_day": [
            {"date": "2022-01-04", "ic": 0.08, "rank_ic": 0.09},
            ...
        ],
        "total_days": 480
    }
"""

from __future__ import annotations

import json
import math

from farbench.evaluator import EvalResult, MetricEvaluatorBase
from farbench.schemas import TaskConfig


class QlibStockEvaluator(MetricEvaluatorBase):
    """Read stock prediction results produced by the agent's predict.py."""

    def evaluate(
        self,
        predictions_path: str,
        test_data_dir: str,
        task_config: TaskConfig,
    ) -> EvalResult:
        with open(predictions_path) as f:
            data = json.load(f)

        # Validate required fields
        for key in ["ic_mean", "icir", "rank_ic_mean"]:
            if key not in data:
                raise ValueError(
                    f"predict.py output missing '{key}' key. Got: {list(data.keys())}"
                )

        ic_mean = float(data["ic_mean"])
        icir = float(data["icir"])
        rank_ic_mean = float(data["rank_ic_mean"])
        total_days = int(data.get("total_days", 0))

        # Recompute from per_day if available (for verification)
        per_day = data.get("per_day", [])
        if per_day and len(per_day) > 0:
            ic_values = [d["ic"] for d in per_day if "ic" in d and d["ic"] is not None]
            rank_ic_values = [d["rank_ic"] for d in per_day if "rank_ic" in d and d["rank_ic"] is not None]

            if ic_values:
                recomputed_ic_mean = sum(ic_values) / len(ic_values)
                ic_std = math.sqrt(
                    sum((v - recomputed_ic_mean) ** 2 for v in ic_values) / len(ic_values)
                ) if len(ic_values) > 1 else 0.0
                recomputed_icir = recomputed_ic_mean / ic_std if ic_std > 0 else 0.0

                # Use recomputed values for consistency
                ic_mean = round(recomputed_ic_mean, 6)
                icir = round(recomputed_icir, 6)
                total_days = len(ic_values)

            if rank_ic_values:
                rank_ic_mean = round(sum(rank_ic_values) / len(rank_ic_values), 6)

        metrics: dict[str, float] = {
            "ic_mean": round(ic_mean, 6),
            "icir": round(icir, 4),
            "rank_ic_mean": round(rank_ic_mean, 6),
            "total_days": float(total_days),
        }

        return EvalResult(
            metrics=metrics,
            primary_metric_name="ic_mean",
            primary_metric_value=ic_mean,
        )
