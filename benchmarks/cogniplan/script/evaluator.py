"""CogniPlan evaluator.

Phase 1 (run by eval_harness): agent's predict.py runs the exploration planner
on evaluation maps and writes a JSON results file.

Phase 2 (this file): reads that JSON and returns an EvalResult.

Primary metric: exploration_score (higher is better) — combines success rate and
path efficiency:
    exploration_score = (1/N) * Σ_i [ S_i * (REF_DIST / d_i) ]
where S_i is binary success, d_i is travel distance, REF_DIST=222m(SOTA).
Failed episodes contribute 0. Successful episodes with shorter paths score higher.

Expected predict.py output format:
    {
        "explored_rate": 0.95,
        "travel_distance": 280.5,
        "success_rate": 0.72,
        "per_episode": [
            {"map_path": "room/map_001.png", "env_type": "room",
             "success": 1, "travel_distance": 245.3, "explored_rate": 0.9999},
            ...
        ],
        "per_type": {
            "room":    {"explored_rate": 0.97, "travel_distance": 220.1, "success_rate": 0.84},
            "tunnel":  {"explored_rate": 0.94, "travel_distance": 310.2, "success_rate": 0.68},
            "outdoor": {"explored_rate": 0.93, "travel_distance": 311.3, "success_rate": 0.64}
        },
        "total_episodes": 150
    }
"""

from __future__ import annotations

import json

from farbench.evaluator import EvalResult, MetricEvaluatorBase
from farbench.schemas import TaskConfig

ENV_TYPES = ["room", "tunnel", "outdoor"]
REF_DIST = 222.0


def _episode_score(success: float, travel_distance: float) -> float:
    if not success or travel_distance <= 0:
        return 0.0
    return REF_DIST / travel_distance


class CogniPlanEvaluator(MetricEvaluatorBase):
    """Read exploration results produced by the agent's predict.py."""

    def evaluate(
        self,
        predictions_path: str,
        test_data_dir: str,
        task_config: TaskConfig,
    ) -> EvalResult:
        with open(predictions_path) as f:
            data = json.load(f)

        for key in ["explored_rate", "travel_distance", "success_rate"]:
            if key not in data:
                raise ValueError(
                    f"predict.py output missing '{key}' key. Got: {list(data.keys())}"
                )

        per_episode = data.get("per_episode", [])
        per_type = data.get("per_type", {})
        total_episodes = int(data.get("total_episodes", 0))

        # ── Compute exploration_score ─────────────────────────────────────────
        if per_episode:
            scores = [
                _episode_score(
                    float(ep.get("success", 0)),
                    float(ep.get("travel_distance", 0.0)),
                )
                for ep in per_episode
            ]
            exploration_score = round(sum(scores) / len(scores), 4) if scores else 0.0
            total_episodes = total_episodes or len(per_episode)
        elif per_type:
            type_scores = []
            for env_type in ENV_TYPES:
                if env_type not in per_type:
                    raise ValueError(
                        f"predict.py output missing env type '{env_type}' in per_type. "
                        f"Got: {list(per_type.keys())}"
                    )
                td = per_type[env_type]
                sr = float(td.get("success_rate", 0.0))
                dist = float(td.get("travel_distance", 0.0))
                type_scores.append(sr * REF_DIST / dist if sr > 0 and dist > 0 else 0.0)
            exploration_score = round(sum(type_scores) / len(type_scores), 4)
        else:
            sr = float(data["success_rate"])
            dist = float(data["travel_distance"])
            exploration_score = round(
                sr * REF_DIST / dist if sr > 0 and dist > 0 else 0.0, 4
            )

        # ── Compute standard metrics from per_type or top-level ───────────────
        if per_type:
            explored_vals, distance_vals, success_vals = [], [], []
            for env_type in ENV_TYPES:
                type_data = per_type[env_type]
                explored_vals.append(float(type_data.get("explored_rate", 0.0)))
                distance_vals.append(float(type_data.get("travel_distance", 0.0)))
                success_vals.append(float(type_data.get("success_rate", 0.0)))
            explored_rate = round(sum(explored_vals) / len(explored_vals), 4)
            travel_distance = round(sum(distance_vals) / len(distance_vals), 4)
            success_rate = round(sum(success_vals) / len(success_vals), 4)
        else:
            explored_rate = round(float(data["explored_rate"]), 4)
            travel_distance = round(float(data["travel_distance"]), 4)
            success_rate = round(float(data["success_rate"]), 4)

        # ── Build metrics dict ────────────────────────────────────────────────
        metrics: dict[str, float] = {
            "exploration_score": exploration_score,
            "explored_rate": explored_rate,
            "travel_distance": travel_distance,
            "success_rate": success_rate,
            "total_episodes": float(total_episodes),
        }

        for env_type in ENV_TYPES:
            if env_type in per_type:
                type_data = per_type[env_type]
                metrics[f"explored_{env_type}"] = round(
                    float(type_data.get("explored_rate", 0.0)), 4
                )
                metrics[f"distance_{env_type}"] = round(
                    float(type_data.get("travel_distance", 0.0)), 4
                )
                metrics[f"success_{env_type}"] = round(
                    float(type_data.get("success_rate", 0.0)), 4
                )

        return EvalResult(
            metrics=metrics,
            primary_metric_name="exploration_score",
            primary_metric_value=exploration_score,
        )
