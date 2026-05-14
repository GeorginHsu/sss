"""VLABench evaluator.

Phase 1 (run by eval_harness): agent's predict.py runs VLABench simulation
and writes a JSON results file.

Phase 2 (this file): reads that JSON and returns an EvalResult.

Expected predict.py output format:
    {
        "episodes": [
            {"task": "add_condiment", "episode_index": 0, "success": true},
            ...
        ],
        "total_episodes": 60
    }
"""

from __future__ import annotations

import json
import os

from farbench.evaluator import EvalResult, MetricEvaluatorBase
from farbench.schemas import TaskConfig

BENCHMARK_TASKS = ["add_condiment", "select_fruit", "select_book"]


class VLABenchEvaluator(MetricEvaluatorBase):
    """Validate per-episode rollout results and compute success rate."""

    def evaluate(
        self,
        predictions_path: str,
        test_data_dir: str,
        task_config: TaskConfig,
    ) -> EvalResult:
        with open(predictions_path) as f:
            data = json.load(f)

        cfg_path = os.path.join(test_data_dir, "eval_config.json")
        with open(cfg_path) as f:
            cfg = json.load(f)

        tasks = cfg.get("benchmark_tasks")
        if (
            not isinstance(tasks, list)
            or not tasks
            or any(not isinstance(task_name, str) for task_name in tasks)
        ):
            raise ValueError(
                "eval_config.json must contain non-empty string list "
                f"'benchmark_tasks'; got {tasks!r}"
            )

        episodes_per_task = cfg.get("episodes_per_task")
        if (
            not isinstance(episodes_per_task, int)
            or isinstance(episodes_per_task, bool)
            or episodes_per_task <= 0
        ):
            raise ValueError(
                "eval_config.json must contain positive integer "
                f"'episodes_per_task'; got {episodes_per_task!r}"
            )

        protocol = cfg.get("evaluation_protocol")
        if protocol != "vlabench_official_track":
            raise ValueError(
                "eval_config.json must use evaluation_protocol="
                f"'vlabench_official_track'; got {protocol!r}"
            )
        expected_total = len(tasks) * episodes_per_task

        episodes = data.get("episodes")
        if not isinstance(episodes, list):
            raise ValueError(
                "predict.py output must contain an 'episodes' list with one "
                "record per evaluated rollout. Aggregated success_rate-only "
                f"outputs are not accepted. Got keys: {list(data.keys())}"
            )

        if len(episodes) != expected_total:
            raise ValueError(
                f"Expected {expected_total} episode records "
                f"({len(tasks)} tasks × {episodes_per_task}), got {len(episodes)}"
            )

        total_episodes = data.get("total_episodes")
        if (
            not isinstance(total_episodes, int)
            or isinstance(total_episodes, bool)
            or total_episodes != expected_total
        ):
            raise ValueError(
                "predict.py output must contain integer total_episodes "
                f"equal to {expected_total}; got {total_episodes!r}"
            )

        counts = {task_name: 0 for task_name in tasks}
        successes = {task_name: 0 for task_name in tasks}
        episode_indices = {task_name: [] for task_name in tasks}
        for i, episode in enumerate(episodes):
            if not isinstance(episode, dict):
                raise ValueError(f"Episode record {i} is not an object")
            task_name = episode.get("task")
            if task_name not in counts:
                raise ValueError(f"Episode record {i} has unexpected task {task_name!r}")
            episode_index = episode.get("episode_index")
            if (
                not isinstance(episode_index, int)
                or isinstance(episode_index, bool)
                or not 0 <= episode_index < episodes_per_task
            ):
                raise ValueError(
                    f"Episode record {i} has invalid episode_index "
                    f"{episode_index!r}; expected int in [0, {episodes_per_task})"
                )
            success = episode.get("success")
            if not isinstance(success, bool):
                raise ValueError(
                    f"Episode record {i} must contain boolean 'success', "
                    f"got {success!r}"
                )
            episode_indices[task_name].append(episode_index)
            counts[task_name] += 1
            successes[task_name] += int(success)

        for task_name in tasks:
            if counts[task_name] != episodes_per_task:
                raise ValueError(
                    f"Task {task_name!r}: expected {episodes_per_task} episodes, "
                    f"got {counts[task_name]}"
                )
            expected_indices = list(range(episodes_per_task))
            if sorted(episode_indices[task_name]) != expected_indices:
                raise ValueError(
                    f"Task {task_name!r}: episode_index values must be exactly "
                    f"{expected_indices}, got {sorted(episode_indices[task_name])}"
                )

        per_task = {
            task_name: successes[task_name] / float(counts[task_name])
            for task_name in tasks
        }
        success_rate = round(sum(per_task.values()) / len(per_task), 4)

        # ── Build metrics dict ────────────────────────────────────────────────
        metrics: dict[str, float] = {
            "success_rate":   success_rate,
            "total_episodes": float(len(episodes)),
        }
        for task_name, value in per_task.items():
            metrics[f"success_{task_name}"] = round(float(value), 4)

        return EvalResult(
            metrics=metrics,
            primary_metric_name="success_rate",
            primary_metric_value=success_rate,
        )
