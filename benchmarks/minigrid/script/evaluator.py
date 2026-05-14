"""Minigrid evaluator.

Phase 1 (run by eval_harness): agent's predict.py creates Minigrid environments,
runs its RL policy, and writes a JSON results file.

Phase 2 (this file): reads that JSON and returns an EvalResult.

Expected predict.py output format:
    {
        "success_rate": 0.85,
        "mean_reward": 0.72,
        "per_env": {
            "MiniGrid-DoorKey-8x8-v0":            {"mean_reward": 0.45, "success_rate": 0.58},
            "MiniGrid-FourRooms-v0":              {"mean_reward": 0.40, "success_rate": 0.52},
            "MiniGrid-KeyCorridorS3R3-v0":        {"mean_reward": 0.42, "success_rate": 0.55},
            "MiniGrid-LockedRoom-v0":             {"mean_reward": 0.38, "success_rate": 0.48},
            "MiniGrid-Dynamic-Obstacles-8x8-v0":  {"mean_reward": 0.30, "success_rate": 0.40},
            "MiniGrid-LavaCrossingS9N2-v0":       {"mean_reward": 0.35, "success_rate": 0.45},
            "MiniGrid-UnlockPickup-v0":           {"mean_reward": 0.50, "success_rate": 0.60},
            "MiniGrid-MultiRoom-N4-S5-v0":        {"mean_reward": 0.48, "success_rate": 0.56},
            "MiniGrid-LavaGapS7-v0":              {"mean_reward": 0.32, "success_rate": 0.42},
            "MiniGrid-DistShift1-v0":             {"mean_reward": 0.44, "success_rate": 0.54}
        },
        "total_episodes": 500
    }
"""

from __future__ import annotations

import json

from farbench.evaluator import EvalResult, MetricEvaluatorBase
from farbench.schemas import TaskConfig

EVAL_ENVS = [
    "MiniGrid-DoorKey-8x8-v0",
    "MiniGrid-FourRooms-v0",
    "MiniGrid-KeyCorridorS3R3-v0",
    "MiniGrid-LockedRoom-v0",
    "MiniGrid-Dynamic-Obstacles-8x8-v0",
    "MiniGrid-LavaCrossingS9N2-v0",
    "MiniGrid-UnlockPickup-v0",
    "MiniGrid-MultiRoom-N4-S5-v0",
    "MiniGrid-LavaGapS7-v0",
    "MiniGrid-DistShift1-v0",
]

EPISODES_PER_ENV = 50


class MinigridEvaluator(MetricEvaluatorBase):
    """Read simulation results produced by the agent's predict.py."""

    def evaluate(
        self,
        predictions_path: str,
        test_data_dir: str,
        task_config: TaskConfig,
    ) -> EvalResult:
        with open(predictions_path) as f:
            data = json.load(f)

        # ── Validate required fields ──────────────────────────────────────────
        if "mean_reward" not in data:
            raise ValueError(
                f"predict.py output missing 'mean_reward' key. Got: {list(data.keys())}"
            )
        if "success_rate" not in data:
            raise ValueError(
                f"predict.py output missing 'success_rate' key. Got: {list(data.keys())}"
            )

        per_env = data.get("per_env", {})
        total_episodes = int(data.get("total_episodes", 0))

        # ── Re-compute overall metrics from per_env if available ──────────────
        # Protects against predict.py returning wrong aggregated values.
        if per_env:
            reward_vals = []
            success_vals = []
            for env_name in EVAL_ENVS:
                if env_name not in per_env:
                    raise ValueError(
                        f"predict.py output missing eval env '{env_name}' in per_env. "
                        f"Got: {list(per_env.keys())}"
                    )
                env_data = per_env[env_name]
                reward_vals.append(float(env_data.get("mean_reward", 0.0)))
                success_vals.append(float(env_data.get("success_rate", 0.0)))

            mean_reward = round(sum(reward_vals) / len(reward_vals), 4)
            success_rate = round(sum(success_vals) / len(success_vals), 4)
        else:
            mean_reward = round(float(data["mean_reward"]), 4)
            success_rate = round(float(data["success_rate"]), 4)

        # ── Build metrics dict ────────────────────────────────────────────────
        metrics: dict[str, float] = {
            "mean_reward": mean_reward,
            "success_rate": success_rate,
            "total_episodes": float(total_episodes),
        }
        for env_name in EVAL_ENVS:
            if env_name in per_env:
                env_data = per_env[env_name]
                short_name = env_name.replace("MiniGrid-", "").replace("-v0", "")
                metrics[f"reward_{short_name}"] = round(
                    float(env_data.get("mean_reward", 0.0)), 4
                )
                metrics[f"success_{short_name}"] = round(
                    float(env_data.get("success_rate", 0.0)), 4
                )

        return EvalResult(
            metrics=metrics,
            primary_metric_name="success_rate",
            primary_metric_value=success_rate,
        )
