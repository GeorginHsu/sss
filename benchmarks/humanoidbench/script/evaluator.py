"""HumanoidBench evaluator.

Phase 1 (run by eval_harness): the agent's predict.py rolls out its policy
in MuJoCo for every (task, episode) pair listed in `eval_config.json` and
writes a JSON results file.

Phase 2 (this file): reads that JSON and returns an `EvalResult`.

Expected predict.py output format (REQUIRED FIELDS):
    {
        "total_episodes": 200,
        "per_episode": [
            {"task_id": "h1hand-walk-v0", "episode_id": 0, "seed": 0,
             "reward": 355.2,           # CUMULATIVE episode reward
             "success": 0|1             # optional; recomputed server-side
            },
            ...
        ]
    }

Optional aggregate fields ("success_rate", "mean_reward", "per_task") are
ignored when `per_episode` is present — the evaluator recomputes them.

Success definition (HumanoidBench paper):
    episode_success = 1  iff  episode_total_reward >= task.success_bar
The success_bar per task lives in eval_config.json (one source of truth).
For h1hand-push-v0 / h1hand-package-v0 the env additionally exposes
`info["success"]` mid-episode; if predict.py reports a per-episode
`success` field for those tasks it is OR-merged with the threshold-based
recomputation (matching humanoid-bench's "success ever fired during the
episode" convention).
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any

from farbench.evaluator import EvalResult, MetricEvaluatorBase
from farbench.schemas import TaskConfig


def _coerce_int_success(v: Any) -> int:
    try:
        return 1 if float(v) > 0.5 else 0
    except Exception:
        return 0


def _coerce_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return 0.0


class HumanoidBenchEvaluator(MetricEvaluatorBase):
    """Aggregate per-episode rollout results across 10 HumanoidBench tasks.

    The evaluator recomputes `success` from `episode_total_reward >=
    success_bar` using the bars in test_data_dir/eval_config.json — this is
    necessary because `info["success"]` is ONLY emitted by 2 of the 10
    tasks (push, package). Trusting predict.py's own success accounting
    silently drops 8/10 of the leaderboard to zero, which is the most
    common HumanoidBench eval bug.
    """

    def evaluate(
        self,
        predictions_path: str,
        test_data_dir: str,
        task_config: TaskConfig,
    ) -> EvalResult:
        del task_config

        with open(predictions_path) as f:
            data = json.load(f)

        # ── Load eval protocol: per-task success_bar + info_has_success ────
        success_bars: dict[str, float] = {}
        info_has_success: dict[str, bool] = {}
        expected_tasks: list[str] = []
        expected_total = 0
        eval_cfg_path = os.path.join(test_data_dir, "eval_config.json")
        if os.path.exists(eval_cfg_path):
            with open(eval_cfg_path) as f:
                eval_cfg = json.load(f)
            for t in eval_cfg.get("tasks", []):
                tid = str(t.get("task_id", "")).strip()
                if not tid:
                    continue
                expected_tasks.append(tid)
                if "success_bar" in t:
                    success_bars[tid] = float(t["success_bar"])
                info_has_success[tid] = bool(t.get("info_has_success", False))
            expected_total = int(
                eval_cfg.get(
                    "total_episodes",
                    sum(int(t.get("episodes", 0)) for t in eval_cfg.get("tasks", [])),
                )
            )

        per_episode = data.get("per_episode") or []

        # ── Recompute everything from per_episode if available ─────────────
        if per_episode:
            total_success = 0
            total_reward = 0.0
            recomputed_count = 0
            per_task_success: dict[str, list[int]] = defaultdict(list)
            per_task_reward: dict[str, list[float]] = defaultdict(list)

            for ep in per_episode:
                task_id = str(ep.get("task_id", "unknown"))
                ep_reward = _coerce_float(ep.get("reward"))

                # Server-side recomputation: success = (reward >= success_bar)
                bar = success_bars.get(task_id)
                if bar is not None:
                    success = 1 if ep_reward >= bar else 0
                    recomputed_count += 1
                    # For push / package, OR-merge with predict.py's reported
                    # info-based success ("success ever fired during episode").
                    if info_has_success.get(task_id, False):
                        success = max(success, _coerce_int_success(ep.get("success")))
                else:
                    # Fallback: trust predict.py's reported success.
                    success = _coerce_int_success(ep.get("success"))

                per_task_success[task_id].append(success)
                per_task_reward[task_id].append(ep_reward)
                total_success += success
                total_reward += ep_reward

            total_episodes = len(per_episode)
            success_rate = round(total_success / max(total_episodes, 1), 4)
            mean_reward = round(total_reward / max(total_episodes, 1), 4)

            per_task_breakdown = {
                task: {
                    "success_rate": round(
                        sum(per_task_success[task]) / max(len(per_task_success[task]), 1),
                        4,
                    ),
                    "mean_reward": round(
                        sum(per_task_reward[task]) / max(len(per_task_reward[task]), 1),
                        4,
                    ),
                    "n_episodes": len(per_task_success[task]),
                    "success_bar": success_bars.get(task),
                }
                for task in per_task_success
            }
        else:
            # Fallback: trust reported aggregates (no per_episode = no
            # server-side success_bar recomputation possible).
            success_rate = round(_coerce_float(data.get("success_rate")), 4)
            mean_reward = round(_coerce_float(data.get("mean_reward")), 4)
            total_episodes = int(data.get("total_episodes", 0))
            per_task_breakdown = data.get("per_task", {}) or {}
            recomputed_count = 0

        # ── Task coverage audit ────────────────────────────────────────────
        reported_tasks = set(per_task_breakdown.keys())
        missing_tasks = [t for t in expected_tasks if t not in reported_tasks]
        extra_tasks = (
            [t for t in reported_tasks if t not in expected_tasks]
            if expected_tasks else []
        )

        metrics: dict[str, Any] = {
            "success_rate": success_rate,
            "mean_reward": mean_reward,
            "total_episodes": float(total_episodes),
            "num_tasks_evaluated": float(len(reported_tasks)),
            "num_tasks_expected": float(len(expected_tasks)),
            "num_tasks_missing": float(len(missing_tasks)),
            "num_tasks_extra": float(len(extra_tasks)),
            "total_episodes_expected": float(expected_total),
            "episodes_recomputed_via_success_bar": float(recomputed_count),
        }
        if per_task_breakdown:
            metrics["per_task"] = per_task_breakdown
        if missing_tasks:
            metrics["missing_tasks"] = missing_tasks
        if extra_tasks:
            metrics["extra_tasks"] = extra_tasks

        return EvalResult(
            metrics=metrics,
            primary_metric_name="success_rate",
            primary_metric_value=success_rate,
        )
