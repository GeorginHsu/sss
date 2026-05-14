"""Habitat 3.0 Social Navigation evaluator.

Phase 1 (eval_harness): the agent's `predict.py` runs the H3 social_nav
multi-agent simulation (Spot + humanoid avatar in HSSD scenes) and writes
a JSON results file.

Phase 2 (this file): reads that JSON and returns an EvalResult with:
  • primary metric: nav_seek_success  (0–1, mean over episodes)
  • secondary:      social_nav_reward, found_human_rate_over_epi,
                    avg_robot_to_human_dis_over_epi, nav_to_pos_success.

The metric name `nav_seek_success` matches the actual key returned by
`env.get_metrics()` in habitat-lab 0.3.3 (despite the H3 paper using
"social_nav_seek_success" — that name does NOT exist as a top-level
metric in the simulator).

We re-aggregate from per_episode whenever it's available, defending
against a buggy predict.py that miscomputes its own means.

Expected predict.py output format (REQUIRED keys are starred)::

    {
        "primary_metric": "nav_seek_success",            # *
        "nav_seek_success": 0.42,                         # *
        "social_nav_reward": -3.1,
        "found_human_rate_over_epi": 0.55,
        "safe_follow_steps": 520,                      # optional trace-derived diagnostic
        "avg_robot_to_human_dis_over_epi": 1.85,
        "nav_to_pos_success": 0.30,
        "per_episode": [                                  # * (per-ep list)
            {
                "episode_id": "0",
                "nav_seek_success": 1,
                "social_nav_reward": 4.2,
                "found_human_rate_over_epi": 0.7,
                "safe_follow_steps": 520,
                "avg_robot_to_human_dis_over_epi": 1.5,
                "nav_to_pos_success": 1
            },
            ...
        ],
        "total_episodes": 50
    }
"""

from __future__ import annotations

import json
from typing import Iterable

from farbench.evaluator import EvalResult, MetricEvaluatorBase
from farbench.schemas import TaskConfig


# Order matters for human-readable logs; the first one is the primary.
_REPORTED_METRICS = (
    "nav_seek_success",
    "social_nav_reward",
    "found_human_rate_over_epi",
    "avg_robot_to_human_dis_over_epi",
    "nav_to_pos_success",
    "dist_to_goal",
    "safe_follow_steps",
)
_PRIMARY = "nav_seek_success"
_SAFE_FOLLOW_STEPS_THRESHOLD = 500


def _safe_mean(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        return 0.0
    return float(sum(vals)) / float(len(vals))


def _trace_safe_follow_steps(ep: dict) -> int | None:
    """Count steps at 1-2m while facing the human, if predict.py emitted a trace.

    Habitat's own `nav_seek_success` measurement already implements this
    condition. This fallback exists for agents that expose a step trace instead
    of trusting the final metric dict. Expected trace row keys are intentionally
    permissive: `dist`/`distance_to_human` and `angle`/`angle_to_human`, or a
    boolean `facing_human`.
    """
    trace = ep.get("step_trace")
    if not isinstance(trace, list):
        return None

    safe_steps = 0
    for row in trace:
        if not isinstance(row, dict):
            continue
        dist = row.get("dist", row.get("distance_to_human"))
        if dist is None:
            continue
        try:
            dist = float(dist)
        except (TypeError, ValueError):
            continue

        facing = row.get("facing_human")
        if facing is None:
            angle = row.get("angle", row.get("angle_to_human"))
            if angle is None:
                continue
            try:
                facing = abs(float(angle)) <= 0.5
            except (TypeError, ValueError):
                continue

        if isinstance(facing, str):
            facing = facing.strip().lower() in {"1", "true", "yes"}

        if 1.0 <= dist < 2.0 and bool(facing):
            safe_steps += 1
    return safe_steps


def _episode_nav_seek_success(ep: dict) -> float | None:
    """Return episode success under the Habitat3 social-nav definition.

    Success means maintaining the safe social distance band [1.0, 2.0) while
    facing the human for at least 500 accumulated steps. Prefer an explicit
    safe-step count or trace when present; otherwise fall back to Habitat's own
    `nav_seek_success` metric, which uses the same config.
    """
    safe_steps = ep.get("safe_follow_steps", ep.get("seek_steps_in_safe_band"))
    if safe_steps is None:
        safe_steps = _trace_safe_follow_steps(ep)
    if safe_steps is not None:
        try:
            return 1.0 if float(safe_steps) >= _SAFE_FOLLOW_STEPS_THRESHOLD else 0.0
        except (TypeError, ValueError):
            return None

    if "nav_seek_success" in ep and ep["nav_seek_success"] is not None:
        return float(ep["nav_seek_success"])
    return None


class HabitatSocialNavEvaluator(MetricEvaluatorBase):
    """Read H3 social-nav simulation results produced by the agent's predict.py."""

    def evaluate(
        self,
        predictions_path: str,
        test_data_dir: str,
        task_config: TaskConfig,
    ) -> EvalResult:
        with open(predictions_path) as f:
            data = json.load(f)

        per_episode = data.get("per_episode", [])
        total_episodes = int(data.get("total_episodes", len(per_episode) or 0))

        # ── Validate: at minimum we need either per_episode rows or a top-level
        # primary metric. Otherwise there's nothing to score.
        if not per_episode and _PRIMARY not in data:
            raise ValueError(
                f"predict.py output missing both 'per_episode' and '{_PRIMARY}'. "
                f"Got keys: {sorted(data.keys())}"
            )

        # ── Re-aggregate from per_episode if present (more reliable than
        # whatever predict.py wrote at the top level — agents have shipped
        # broken means in the past).
        metrics: dict[str, float] = {}
        if per_episode:
            primary_values = [
                v for ep in per_episode
                if isinstance(ep, dict)
                for v in [_episode_nav_seek_success(ep)]
                if v is not None
            ]
            if primary_values:
                metrics[_PRIMARY] = round(_safe_mean(primary_values), 4)

            for key in _REPORTED_METRICS:
                if key == _PRIMARY:
                    continue
                values = [
                    float(ep[key])
                    for ep in per_episode
                    if isinstance(ep, dict) and key in ep and ep[key] is not None
                ]
                if values:
                    metrics[key] = round(_safe_mean(values), 4)
            metrics["total_episodes"] = float(len(per_episode))
            primary_value = float(metrics.get(_PRIMARY, 0.0))
        else:
            for key in _REPORTED_METRICS:
                if key in data and data[key] is not None:
                    metrics[key] = round(float(data[key]), 4)
            metrics["total_episodes"] = float(total_episodes)
            primary_value = float(metrics.get(_PRIMARY, 0.0))

        # ── Always include the primary metric, even as 0.0, so downstream
        # consumers that index by name don't KeyError on a failed eval.
        metrics.setdefault(_PRIMARY, round(primary_value, 4))

        return EvalResult(
            metrics=metrics,
            primary_metric_name=_PRIMARY,
            primary_metric_value=round(primary_value, 4),
        )
