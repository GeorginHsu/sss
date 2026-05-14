"""Dashboard API: read-only endpoints for browsing experiment results.

Serves the dashboard HTML and provides JSON APIs for:
- Listing experiments
- Experiment detail and trajectory
- Per-iteration detail (action, output, eval)
- Workspace diffs between iterations
- SSE live stream for running experiments
"""

from __future__ import annotations

import asyncio
import csv
import difflib
import json
import math
import os
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, StreamingResponse

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BINARY_EXTENSIONS = frozenset({
    ".pt", ".pth", ".pkl", ".pickle", ".npy", ".npz", ".bin",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".svg",
    ".pdf", ".zip", ".tar", ".gz", ".bz2", ".xz",
    ".so", ".dylib", ".dll", ".o", ".a",
    ".pyc", ".pyo", ".whl", ".egg",
})

_MODEL_DISPLAY_NAMES = {
    "claude": "claude-4.6-sonnet",
    "opus47": "claude-4.7-opus",
    "claude47": "claude-4.7-opus",
    "claude47opus": "claude-4.7-opus",
    "claude-4.7-opus": "claude-4.7-opus",
    "opus46": "claude-4.6-opus",
    "claude46": "claude-4.6-sonnet",
    "claude46opus": "claude-4.6-opus",
    "claude-4.6-opus": "claude-4.6-opus",
    "sonnet46": "claude-4.6-sonnet",
    "claude46sonnet": "claude-4.6-sonnet",
    "claude-4.6-sonnet": "claude-4.6-sonnet",
    "gpt55": "gpt5.5",
    "gpt5.5": "gpt5.5",
    "gpt54": "gpt5.4",
    "gpt5.4": "gpt5.4",
    "gemini": "gemini-3.1-pro-preview",
    "gemini31": "gemini-3.1-pro-preview",
    "gemini-3.1-pro-preview": "gemini-3.1-pro-preview",
    "glm": "glm-5.1",
    "glm51": "glm-5.1",
    "glm-5.1": "glm-5.1",
    "kimi": "kimi-k2.6",
    "kimi26": "kimi-k2.6",
    "kimi-k2.6": "kimi-k2.6",
    "grok": "grok-4.20-0309-reasoning",
    "grok4": "grok-4.20-0309-reasoning",
    "grok420": "grok-4.20-0309-reasoning",
    "grok-4.20-0309-reasoning": "grok-4.20-0309-reasoning",
    "qwen": "qwen3-coder-plus-2025-09-23",
    "qwen3": "qwen3-coder-plus-2025-09-23",
    "qwen3coder": "qwen3-coder-plus-2025-09-23",
    "qwen3-coder-plus-2025-09-23": "qwen3-coder-plus-2025-09-23",
    "deepseek": "deepseek-v4-pro",
    "deepseekv4": "deepseek-v4-pro",
    "deepseek-v4-pro": "deepseek-v4-pro",
    "gpt4o": "gpt-4o",
    "gpt-4o": "gpt-4o",
}

_FAILED_TERMINAL_REASONS = {"error", "failed", "exception", "aborted", "cancelled", "canceled"}
_RUNNING_ACTIVITY_GRACE_SECONDS = 30 * 60

_PAPER_MODEL_LABELS = {
    "gpt5.5": "GPT-5.5",
    "gemini-3.1-pro-preview": "Gemini-3.1-Pro",
    "kimi-k2.6": "Kimi-K2.6",
    "claude-4.7-opus": "Claude-4.7-Opus",
    "grok-4.20-0309-reasoning": "Grok-4.20-0309",
    "claude-4.6-opus": "Claude-4.6-Opus",
    "glm-5.1": "GLM-5.1",
    "claude-4.6-sonnet": "Claude-4.6-Sonnet",
    "gpt5.4": "GPT-5.4",
    "deepseek-v4-pro": "DeepSeek-V4-Pro",
    "qwen3-coder-plus-2025-09-23": "Qwen3-Coder-Plus",
}

_PAPER_METRICS = [
    ("instruction_following", "Instruction Following", "IF"),
    ("code_execution_success", "Code Execution Success", "CES"),
    ("first_achieve", "First Achievement", "FA"),
    ("headroom_gain", "Headroom Gain", "HG"),
    ("progress_efficiency", "Progress Efficiency", "PE"),
    ("breadth_at_0_5", "Breadth at 0.5", "Pass@0.5"),
]

_PAPER_FAILURE_MODES = [
    ("failure_missing", "Missing trace", "#C7C7C7"),
    ("failure_eval_or_score", "Invalid/zero result", "#C9576B"),
    ("failure_command_fail_or_timeout", "Command fail", "#D45A4F"),
    ("failure_low_learning_gain", "Low learning gain", "#D8923B"),
    ("failure_low_achievement", "Low achievement", "#8F6AA3"),
    ("failure_productive", "Productive", "#3F8F3A"),
]

_PAPER_ITERATION_FAILURE_MODES = [
    ("productive", "Productive", "#3F8F3A"),
    ("low_learning_gain", "Low learning gain", "#D8923B"),
    ("low_achievement", "Low achievement", "#8F6AA3"),
    ("valid_zero", "Valid zero", "#C9576B"),
    ("command_fail_or_timeout", "Command fail", "#D45A4F"),
    ("no_scorable_metric", "No scorable metric", "#A8A8A8"),
]

_PAPER_DOMAIN_COLOURS = {
    "computer vision": "#2F6FBB",
    "AI for science": "#3F8F3A",
    "natural language processing": "#E0813B",
    "audio/speech understanding": "#8F6AA3",
    "robotics": "#C9576B",
}

_PAPER_DOMAIN_LABELS = {
    "computer vision": "Computer Vision",
    "AI for science": "AI for Science",
    "natural language processing": "NLP",
    "audio/speech understanding": "Audio / Speech",
    "robotics": "Robotics",
}

_ANALYSIS_CASE_STUDIES = [
    {
        "id": "productive_iteration",
        "index": "A",
        "title": "Productive iteration",
        "tagline": "Hypothesis-driven adaptation",
        "tone": "productive",
        "model": "DeepSeek-V4-Pro",
        "task": "flip_aav",
        "file": "DeepSeek-V4-Pro__flip_aav__trajectory.json",
        "thesis": "Feedback changes the research hypothesis, not just a hyperparameter.",
        "behavior": (
            "The agent moves from a weak frozen-embedding baseline to region-sliced modeling, "
            "then to end-to-end ESM2 fine-tuning after the validation/test gap exposes overfitting."
        ),
        "diagnosis": (
            "This is the positive control: evaluator feedback triggers a method-level pivot "
            "at the right abstraction layer."
        ),
        "evidence": [
            {
                "iteration": 3,
                "label": "Method-level hypothesis",
                "quote": "Frozen mean pooling over the full sequence dilutes mutation signals concentrated in positions 561-588.",
            },
            {
                "iteration": 5,
                "label": "Empirical check",
                "quote": "Val Spearman is promising; now need actual test metric to gauge progress and decide next steps.",
            },
            {
                "iteration": 9,
                "label": "Response to overfitting",
                "quote": "To improve generalization, we need to fine-tune ESM2-35M end-to-end.",
            },
        ],
    },
    {
        "id": "early_plateau",
        "index": "B",
        "title": "Early plateau",
        "tagline": "Polishing the first strong model",
        "tone": "plateau",
        "model": "Kimi-K2.6",
        "task": "weatherbench_z500t850",
        "file": "Kimi-K2.6__weatherbench_z500t850__trajectory.json",
        "thesis": "An early good result becomes an anchor for local search.",
        "behavior": (
            "The best score appears early. Later iterations retrain, ensemble, or bias-correct nearby variants "
            "but do not replace the modeling strategy."
        ),
        "diagnosis": (
            "The agent remains operationally competent, but feedback is interpreted as a reason to tune "
            "the same family rather than to reconsider the formulation."
        ),
        "evidence": [
            {
                "iteration": 5,
                "label": "Early best",
                "quote": "A deep fully-convolutional ResNet at constant 32x64 resolution should preserve spatial phase.",
            },
            {
                "iteration": 12,
                "label": "Retreat to known setup",
                "quote": "The iteration-5 config produced the best result. To beat it, we should train the same architecture longer.",
            },
            {
                "iteration": 28,
                "label": "Endgame",
                "quote": "Training is too risky now.",
            },
        ],
    },
    {
        "id": "stuck_low",
        "index": "C",
        "title": "Stuck-low",
        "tagline": "Many valid evaluations, little conceptual movement",
        "tone": "stuck",
        "model": "GPT-5.5",
        "task": "domainnet_quickdraw",
        "file": "GPT-5.5__domainnet_quickdraw__trajectory.json",
        "thesis": "The loop is active, but the search stays at the wrong abstraction level.",
        "behavior": (
            "The agent repeatedly tunes checkpoints, test-time augmentation, and ensemble weights, "
            "while the normalized achievement stays near its initial low value."
        ),
        "diagnosis": (
            "This is not an execution failure. It is a research-strategy failure: valid feedback produces "
            "local optimization rather than a new domain-adaptation idea."
        ),
        "evidence": [
            {
                "iteration": 8,
                "label": "Local optimum",
                "quote": "Testing the local training curve around the 2-epoch optimum is the highest-value next step.",
            },
            {
                "iteration": 18,
                "label": "Inference-side search",
                "quote": "The gain may come from TTA, checkpoint ensembling, or both.",
            },
            {
                "iteration": 28,
                "label": "Diminishing returns",
                "quote": "The gains are diminishing but still positive.",
            },
        ],
    },
    {
        "id": "scored_zero",
        "index": "D",
        "title": "Submitted but scored zero",
        "tagline": "Accepted artifacts, no task progress",
        "tone": "zero",
        "model": "GPT-5.5",
        "task": "crohme_hmer",
        "file": "GPT-5.5__crohme_hmer__trajectory.json",
        "thesis": "A zero score should prompt a method rethink; here it mostly extends the same pipeline.",
        "behavior": (
            "The agent submits multiple runnable recognizers, but every official exact-match score remains zero. "
            "Late iterations shift toward prediction-side fallbacks."
        ),
        "diagnosis": (
            "The evaluator is providing a clear negative signal. The missing capability is deciding that the "
            "current modeling layer is wrong."
        ),
        "evidence": [
            {
                "iteration": 2,
                "label": "Continuation after zero",
                "quote": "Hidden exprate was 0; within the remaining time the best chance is to continue training.",
            },
            {
                "iteration": 10,
                "label": "Fallback logic",
                "quote": "The best remaining chance is a prediction-side ensemble/fallback.",
            },
            {
                "iteration": 12,
                "label": "Final heuristic",
                "quote": "Further training is impossible; the only feasible chance is a prediction-side heuristic.",
            },
        ],
    },
    {
        "id": "never_enters_loop",
        "index": "E",
        "title": "Never enters the loop",
        "tagline": "Local success, no evaluator-accepted feedback",
        "tone": "contract",
        "model": "GPT-5.4",
        "task": "asvspoof2021_la",
        "file": "GPT-5.4__asvspoof2021_la__trajectory.json",
        "thesis": "Without an accepted evaluation, empirical learning never starts.",
        "behavior": (
            "The agent obtains strong local development results, but official submissions never yield a score. "
            "Most effort goes into rewriting the inference interface."
        ),
        "diagnosis": (
            "This regime separates local engineering progress from benchmark progress: the agent cannot convert "
            "a plausible model into a contract-satisfying artifact."
        ),
        "evidence": [
            {
                "iteration": 5,
                "label": "Contract focus",
                "quote": "The main blocker is path resolution in predict.py, not model quality.",
            },
            {
                "iteration": 10,
                "label": "Rewrite cycle",
                "quote": "Safest is to rewrite complete model.py/train.py/predict.py from scratch.",
            },
            {
                "iteration": 30,
                "label": "No official metric",
                "quote": "With exactly one iteration left and no official metric recorded, the only rational action is to submit again.",
            },
        ],
    },
]


def _model_key_from_agent_id(agent_id: str | None) -> str:
    if not agent_id:
        return ""
    return agent_id.split("_", 1)[0]


def _model_display_name(model_key: str | None) -> str:
    if not model_key:
        return ""
    return _MODEL_DISPLAY_NAMES.get(model_key, model_key)


def _paper_model_label(value: object) -> str:
    key = str(value or "")
    return _PAPER_MODEL_LABELS.get(key, key)


def _paper_number(value: object, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    try:
        number = float(text)
    except (TypeError, ValueError):
        return default
    if math.isnan(number) or math.isinf(number):
        return default
    return number


def _paper_int(value: object, default: int = 0) -> int:
    number = _paper_number(value)
    return int(round(number)) if number is not None else default


def _paper_median(values: list[float]) -> Optional[float]:
    nums = sorted(v for v in values if v is not None and math.isfinite(v))
    if not nums:
        return None
    mid = len(nums) // 2
    if len(nums) % 2:
        return nums[mid]
    return (nums[mid - 1] + nums[mid]) / 2.0


def _paper_rows_from_capabilities(capabilities: dict[str, Any]) -> dict[str, Any]:
    """Build the analysis chart schema from experiment diagnostics."""
    metric_aliases = {"first_achieve": "first_achievement"}
    model_scores = capabilities.get("model_scores") or {}
    capability_rows: list[dict[str, Any]] = []
    for model in capabilities.get("models") or []:
        stats = model_scores.get(model) or {}
        scores = stats.get("capability_scores") or {}
        capability_rows.append({
            "model": model,
            "agent": _paper_model_label(model),
            "final_score": None,
            "capability_score": _paper_number(stats.get("overall_capability_score")),
            "metrics": {
                key: _paper_number(scores.get(metric_aliases.get(key, key)))
                for key, _label, _short in _PAPER_METRICS
            },
        })
    capability_rows.sort(
        key=lambda row: _paper_number(row.get("capability_score"), -1.0) or -1.0,
        reverse=True,
    )

    failure_key_map = {
        "failure_missing": ("missing",),
        "failure_eval_or_score": ("no_valid_eval", "no_scorable_metric", "valid_zero"),
        "failure_command_fail_or_timeout": ("command_fail_or_timeout",),
        "failure_low_learning_gain": ("low_learning_gain",),
        "failure_low_achievement": ("low_achievement",),
        "failure_productive": ("productive",),
    }
    failure_rows: list[dict[str, Any]] = []
    for row in capability_rows:
        stats = model_scores.get(row["model"]) or {}
        raw_counts = stats.get("failure_modes") or {}
        counts = {
            key: sum(_paper_int(raw_counts.get(source), 0) for source in sources)
            for key, sources in failure_key_map.items()
        }
        active_tasks = _paper_int(stats.get("active_tasks"), 0) or sum(counts.values())
        failure_rows.append({
            "model": row["model"],
            "agent": row["agent"],
            "active_tasks": active_tasks,
            "counts": counts,
            "fractions": {
                key: (count / active_tasks if active_tasks else 0.0)
                for key, count in counts.items()
            },
        })

    iteration_modes = {key for key, _label, _color in _PAPER_ITERATION_FAILURE_MODES}
    iteration_rows: list[dict[str, Any]] = []
    for idx, episode in enumerate(capabilities.get("episodes") or []):
        if episode.get("status") != "valid":
            continue
        metrics = episode.get("metrics") or {}
        first = _paper_number(metrics.get("first_achievement"))
        best = _paper_number(episode.get("achievement"))
        headroom = _paper_number(metrics.get("headroom_gain"))
        n_valid_eval = _paper_number(episode.get("valid_eval_count"), _paper_number(episode.get("eval_count")))
        if first is None or best is None or headroom is None or n_valid_eval is None:
            continue
        domains = episode.get("domain") or []
        domain = str(domains[0] if isinstance(domains, list) and domains else domains).strip()
        failure_mode = str(episode.get("failure_mode") or "no_scorable_metric").strip()
        if failure_mode not in iteration_modes:
            failure_mode = "no_scorable_metric"
        iteration_rows.append({
            "id": idx,
            "agent": episode.get("model") or "",
            "agent_label": _paper_model_label(episode.get("model")),
            "task": str(episode.get("task", "")).strip(),
            "domain": domain,
            "domain_label": _PAPER_DOMAIN_LABELS.get(domain, domain),
            "status": "valid",
            "first_achievement": first,
            "best_achievement": best,
            "headroom_gain": headroom,
            "first_to_best_gain": best - first,
            "progress_efficiency": _paper_number(metrics.get("progress_efficiency")),
            "n_valid_eval_used": n_valid_eval,
            "total_iterations": _paper_number(episode.get("total_iterations")),
            "failure_mode": failure_mode,
            "terminal_reason": str(episode.get("terminal_reason", "")).strip(),
            "metric_name": str(episode.get("metric_name", "")).strip(),
        })

    return {
        "capability_rows": capability_rows,
        "failure_rows": failure_rows,
        "iteration_rows": iteration_rows,
    }


def _sanitize_floats(obj: Any) -> Any:
    """Replace NaN/Inf float values with None so JSON serialization succeeds."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize_floats(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_floats(v) for v in obj]
    return obj


def _is_within_path(path: str, root: str) -> bool:
    """Return True iff `path` resolves inside `root`."""
    try:
        real_path = os.path.realpath(path)
        real_root = os.path.realpath(root)
        return os.path.commonpath([real_path, real_root]) == real_root
    except ValueError:
        return False


def _safe_experiment_dir(exp_root: str, task: str, exp_id: str) -> str:
    """Resolve an experiment path and reject traversal outside its task dir."""
    if not exp_id or exp_id in {".", ".."} or os.path.isabs(exp_id):
        raise HTTPException(400, "Invalid experiment id")
    if "/" in exp_id or "\\" in exp_id:
        raise HTTPException(400, "Invalid experiment id")
    task_root = _safe_task_dir(exp_root, task)
    exp_dir = os.path.join(task_root, exp_id)
    if not _is_within_path(exp_dir, task_root):
        raise HTTPException(403, "Path traversal denied")
    return exp_dir


def _safe_task_dir(exp_root: str, task: str) -> str:
    """Resolve an experiment task directory and reject traversal."""
    if not task or task in {".", ".."} or os.path.isabs(task):
        raise HTTPException(400, "Invalid task")
    if "/" in task or "\\" in task:
        raise HTTPException(400, "Invalid task")
    task_dir = os.path.join(exp_root, task)
    if not _is_within_path(task_dir, exp_root):
        raise HTTPException(403, "Path traversal denied")
    return task_dir


def _read_json(path: str, max_bytes: int = 0) -> Optional[dict]:
    """Read a JSON file, return None if missing or malformed.

    If *max_bytes* > 0 and the file exceeds that size, return a slim
    placeholder instead of loading the entire file into memory.
    """
    try:
        if max_bytes > 0:
            size = os.path.getsize(path)
            if size > max_bytes:
                return _read_json_truncated(path, max_bytes)
        with open(path, "r") as f:
            return _sanitize_floats(json.load(f))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _as_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _safe_mtime(path: str) -> Optional[float]:
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


def _experiment_latest_activity_mtime(exp_dir: str) -> Optional[float]:
    """Return a cheap activity timestamp without walking large workspace files."""
    latest = _safe_mtime(exp_dir)
    for name in ("live_status.json", "experiment.log", "config.json"):
        mtime = _safe_mtime(os.path.join(exp_dir, name))
        if mtime is not None:
            latest = max(latest or mtime, mtime)

    try:
        entries = os.listdir(exp_dir)
    except OSError:
        return latest

    for entry in entries:
        if not entry.startswith("iter_"):
            continue
        iter_dir = os.path.join(exp_dir, entry)
        if not os.path.isdir(iter_dir):
            continue
        mtime = _safe_mtime(iter_dir)
        if mtime is not None:
            latest = max(latest or mtime, mtime)
        for name in ("obs.json", "action.json", "command_output.json", "eval_result.json"):
            mtime = _safe_mtime(os.path.join(iter_dir, name))
            if mtime is not None:
                latest = max(latest or mtime, mtime)
    return latest


def _is_live_experiment_current(exp_dir: str, live_status: Optional[dict]) -> bool:
    """Treat live_status as running only while its remaining budget is plausible."""
    now = time.time()
    live_path = os.path.join(exp_dir, "live_status.json")
    live_mtime = _safe_mtime(live_path)

    if live_status and live_mtime is not None:
        remaining_hours = _as_float(live_status.get("remaining_hours"))
        if remaining_hours is not None:
            remaining_seconds = max(0.0, remaining_hours) * 3600.0
            return now <= live_mtime + remaining_seconds + _RUNNING_ACTIVITY_GRACE_SECONDS
        return now <= live_mtime + _RUNNING_ACTIVITY_GRACE_SECONDS

    latest = _experiment_latest_activity_mtime(exp_dir)
    if latest is None:
        return False
    return now <= latest + _RUNNING_ACTIVITY_GRACE_SECONDS


# Max chars to keep from head / tail of large string fields
# Aligned with agent prompt limits (STDOUT_PROMPT_TAIL = 2000)
_TRUNC_HEAD = 2_000
_TRUNC_TAIL = 2_000


def _read_json_truncated(path: str, max_bytes: int) -> Optional[dict]:
    """Read a JSON file that is too large, truncating long string values.

    Strategy: read a generous head chunk so that json.loads can parse
    small fields that appear before the giant string, then splice in
    the tail of the file so the user sees both the beginning and end
    of long output.
    """
    try:
        size = os.path.getsize(path)

        # -- 1. Read head + tail raw bytes --------------------------------
        head_bytes = max_bytes  # e.g. 10 MB
        tail_bytes = _TRUNC_TAIL * 4  # generous for utf-8

        with open(path, "rb") as f:
            head_raw = f.read(head_bytes).decode("utf-8", errors="replace")

        with open(path, "rb") as f:
            f.seek(max(0, size - tail_bytes))
            tail_raw = f.read().decode("utf-8", errors="replace")

        # -- 2. Extract each top-level key + value start ------------------
        # For a structure like {"stdout": "...", "exit_code": 0, ...}
        # we find each key and grab what we can of its value.
        data: dict[str, Any] = {}
        data["_truncated"] = True
        data["_original_size"] = size

        # Find all top-level "key": patterns in the head
        for m in re.finditer(r'"(\w+)"\s*:\s*', head_raw):
            key = m.group(1)
            if key in data:
                continue
            val_start = m.end()
            ch = head_raw[val_start] if val_start < len(head_raw) else ""

            if ch == '"':
                # String value – extract head portion (up to _TRUNC_HEAD chars)
                # Find the content after the opening quote
                str_start = val_start + 1
                str_head = head_raw[str_start:str_start + _TRUNC_HEAD]
                # Un-escape basic JSON escapes for display
                try:
                    str_head = str_head.encode().decode("unicode_escape", errors="replace")
                except Exception:
                    pass

                # Try to grab the tail of this string value from file tail
                # The tail_raw ends with something like ...content", "next_key": ...}
                # Find the last substantial string-end pattern
                str_tail = ""
                # Look for closing pattern: ", "somekey" or "}\n etc
                # We grab the last _TRUNC_TAIL chars before a likely string terminator
                tail_match = re.search(r'^([\s\S]*?)"\s*[,}]', tail_raw)
                if tail_match:
                    raw_tail_content = tail_match.group(1)[-_TRUNC_TAIL:]
                    try:
                        str_tail = raw_tail_content.encode().decode("unicode_escape", errors="replace")
                    except Exception:
                        str_tail = raw_tail_content

                if str_tail:
                    omitted = max(0, size - len(str_head) - len(str_tail))
                    data[key] = (
                        str_head
                        + f"\n\n... [truncated ~{omitted:,} chars] ...\n\n"
                        + str_tail
                    )
                else:
                    data[key] = str_head + f"\n\n... [truncated, total file {size:,} bytes] ..."

                # After the first giant string, remaining keys are likely
                # short – try to extract them from tail_raw
                break
            else:
                # Non-string value (number, bool, null, object, array)
                # Try to parse just this value
                snippet = head_raw[val_start:val_start + 200]
                # Find end of value: , or } at top level
                end_m = re.search(r'[,}]', snippet)
                if end_m:
                    val_str = snippet[:end_m.start()].strip()
                    try:
                        data[key] = json.loads(val_str)
                    except (json.JSONDecodeError, ValueError):
                        data[key] = val_str

        # -- 3. Extract short fields from the tail (e.g. exit_code) -------
        for m in re.finditer(r'"(\w+)"\s*:\s*([^",{}[\]]+)', tail_raw):
            key = m.group(1)
            if key not in data:
                val_str = m.group(2).strip().rstrip(",}")
                try:
                    data[key] = json.loads(val_str)
                except (json.JSONDecodeError, ValueError):
                    data[key] = val_str

        return _sanitize_floats(data)
    except (FileNotFoundError, OSError):
        return None


def _truncate_strings(obj: Any, max_len: int = _TRUNC_HEAD + _TRUNC_TAIL) -> Any:
    """Recursively truncate long strings, keeping head + tail."""
    if isinstance(obj, str) and len(obj) > max_len:
        return (
            obj[:_TRUNC_HEAD]
            + f"\n\n... [truncated {len(obj) - _TRUNC_HEAD - _TRUNC_TAIL:,} chars] ...\n\n"
            + obj[-_TRUNC_TAIL:]
        )
    if isinstance(obj, dict):
        return {k: _truncate_strings(v, max_len) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_truncate_strings(v, max_len) for v in obj]
    return obj


def _is_text_file(filename: str) -> bool:
    return Path(filename).suffix.lower() not in _BINARY_EXTENSIONS


def _read_text_safe(path: str, max_bytes: int = 100_000) -> Optional[str]:
    """Read a text file, return None if binary or too large."""
    try:
        size = os.path.getsize(path)
        if size > max_bytes:
            return f"[File too large: {size:,} bytes]"
        with open(path, "r", errors="replace") as f:
            return f.read()
    except (OSError, UnicodeDecodeError):
        return None


def _to_float(value: Any) -> Optional[float]:
    """Return a finite float or None."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _median(values: list[float]) -> Optional[float]:
    if not values:
        return None
    vals = sorted(values)
    n = len(vals)
    mid = n // 2
    if n % 2:
        return float(vals[mid])
    return float((vals[mid - 1] + vals[mid]) / 2.0)


def _read_task_score_table(benchmarks_root: str) -> dict[str, dict[str, Any]]:
    """Read benchmarks/task.score.

    `low_score` and `high_score` are native-metric reference endpoints. For
    lower-is-better metrics, `low_score` is the target and `high_score` is the
    floor. The achievement calculation orients the metric before clipping.
    """
    path = os.path.join(benchmarks_root, "task.score")
    rows: dict[str, dict[str, Any]] = {}
    try:
        with open(path, newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                task = (row.get("name") or "").strip()
                metric = (row.get("metric") or "").strip()
                if not task or not metric:
                    continue
                low = _to_float(row.get("low_score"))
                high = _to_float(row.get("high_score"))
                if low is None or high is None or high == low:
                    continue
                hib_raw = str(row.get("higher_is_better", "")).strip().lower()
                rows[task] = {
                    "name": task,
                    "domain": [(row.get("dimension") or "").strip()] if row.get("dimension") else [],
                    "primary_metric": metric,
                    "higher_is_better": hib_raw in {"true", "1", "yes", "y"},
                    "low_score": low,
                    "high_score": high,
                }
    except OSError:
        return {}
    return rows


def _achievement(value: float, meta: dict[str, Any]) -> dict[str, float]:
    """Compute uncapped and clipped achievement for one native metric value."""
    low = float(meta["low_score"])
    high = float(meta["high_score"])
    span = high - low
    if span <= 0:
        raise ValueError("score table requires high_score > low_score")
    if meta["higher_is_better"]:
        uncapped = (value - low) / span
        oriented = value
        oriented_floor = low
        oriented_target = high
    else:
        uncapped = (high - value) / span
        oriented = -value
        oriented_floor = -high
        oriented_target = -low
    return {
        "oriented_score": oriented,
        "oriented_floor": oriented_floor,
        "oriented_target": oriented_target,
        "uncapped_score": uncapped,
        "achievement": max(0.0, min(1.0, uncapped)),
    }


def _extract_scored_metric(
    final: dict[str, Any],
    leaderboard: dict[str, Any],
    config: dict[str, Any],
    metric_name: str,
) -> tuple[Optional[float], str]:
    """Extract the task.score metric from an experiment summary.

    Prefer the named metric in `best_metrics`. Fall back to `best_primary_metric`
    only when the run's recorded primary metric matches the task.score metric,
    so old experiments whose primary metric changed do not get mis-scored.
    """
    for source_name, payload in (
        ("final.best_metrics", final.get("best_metrics")),
        ("leaderboard.best_metrics", leaderboard.get("best_metrics")),
    ):
        if isinstance(payload, dict) and metric_name in payload:
            val = _to_float(payload.get(metric_name))
            if val is not None:
                return val, source_name

    # Modern summaries always persist evaluator metrics. If that payload exists
    # but does not contain the requested task.score metric, do not reinterpret a
    # fallback zero `best_primary_metric` as a valid score; many failed/invalid
    # runs persist exactly that shape.
    if any(isinstance(payload, dict) for payload in (final.get("best_metrics"), leaderboard.get("best_metrics"))):
        return None, ""

    recorded_metrics = {
        final.get("primary_metric"),
        leaderboard.get("primary_metric"),
        config.get("primary_metric"),
    }
    if metric_name in recorded_metrics:
        val = _to_float(final.get("best_primary_metric"))
        if val is not None:
            return val, "final.best_primary_metric"
        val = _to_float(leaderboard.get("best_primary_metric"))
        if val is not None:
            return val, "leaderboard.best_primary_metric"

    return None, ""


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _tau_for_iteration(iteration: Any, max_iterations: Any) -> Optional[float]:
    """Normalize an iteration index to the paper budget fraction tau."""
    it = _to_float(iteration)
    max_it = _to_float(max_iterations)
    if it is None or max_it is None or max_it <= 0:
        return None
    return _clamp01(it / max_it)


def _extract_eval_metric_value(payload: Any, metric_name: str) -> Optional[float]:
    """Extract a named metric from evaluator payloads and trajectory rows."""
    if not isinstance(payload, dict):
        return None

    metrics = payload.get("metrics")
    if isinstance(metrics, dict):
        val = _to_float(metrics.get(metric_name))
        if val is not None:
            return val

    val = _to_float(payload.get(metric_name))
    if val is not None:
        return val

    if payload.get("primary_metric_name") == metric_name:
        val = _to_float(payload.get("primary_metric_value"))
        if val is not None:
            return val

    nested = payload.get("eval_result")
    if isinstance(nested, dict):
        return _extract_eval_metric_value(nested, metric_name)

    return None


def _iter_dirs(experiment_dir: str) -> list[tuple[int, str]]:
    """Return sorted (iteration_number, dir_path) pairs."""
    result = []
    for name in os.listdir(experiment_dir):
        m = re.match(r"iter_(\d+)$", name)
        if m:
            result.append((int(m.group(1)), os.path.join(experiment_dir, name)))
    result.sort(key=lambda x: x[0])
    return result


def _conversation_response_text(conversation: dict[str, Any]) -> str:
    response = conversation.get("response")
    if isinstance(response, dict):
        if isinstance(response.get("raw_text"), str):
            return response["raw_text"]
        content = response.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item, str):
                    parts.append(item)
            if parts:
                return "\n".join(parts)
        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0] if isinstance(choices[0], dict) else {}
            message = choice.get("message") if isinstance(choice, dict) else None
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return message["content"]
            if isinstance(choice.get("text"), str):
                return choice["text"]
    if isinstance(response, str):
        return response
    return ""


def _normalize_done_action(payload: Any) -> dict[str, Any]:
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except (json.JSONDecodeError, TypeError):
            return {"reasoning": payload, "done": True}
    if not isinstance(payload, dict):
        return {"done": True}

    action = payload.get("action") if isinstance(payload.get("action"), dict) else payload
    key_map = {
        "reasoning": "reasoning",
        "reason": "reasoning",
        "files_to_write": "files_to_write",
        "packages_to_install": "packages_to_install",
        "command": "command",
        "submit_eval": "submit_eval",
        "done": "done",
        "done_reason": "done_reason",
    }
    normalized: dict[str, Any] = {}
    for key, value in action.items():
        canonical = key_map.get(str(key).strip().lower())
        if canonical:
            normalized[canonical] = value
    normalized["done"] = bool(normalized.get("done", True))
    normalized.setdefault("command", "")
    normalized.setdefault("files_to_write", {})
    normalized.setdefault("packages_to_install", [])
    normalized.setdefault("submit_eval", None)
    return normalized


def _done_action(exp_dir: str, summary: Optional[dict[str, Any]] = None) -> Optional[dict[str, Any]]:
    done_path = os.path.join(exp_dir, "summary", "done_conversation.json")
    if not os.path.isfile(done_path):
        return None
    conversation = _read_json(done_path, max_bytes=2_500_000)
    if not isinstance(conversation, dict):
        return None
    action = _normalize_done_action(_conversation_response_text(conversation))
    if summary and not action.get("done_reason"):
        action["done_reason"] = summary.get("termination_detail") or ""
    action["done"] = True
    return action


def _append_done_iteration(
    exp_dir: str,
    summary: Optional[dict[str, Any]],
    iterations_data: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not summary or summary.get("terminal_reason") != "agent_done":
        return iterations_data
    action = _done_action(exp_dir, summary)
    if not action:
        return iterations_data
    rows = list(iterations_data)
    if any(row.get("agent_done") for row in rows):
        return rows
    last_iteration = max(
        [int(row.get("iteration") or 0) for row in rows if row.get("iteration") is not None] or [0]
    )
    done_reason = action.get("done_reason") or summary.get("termination_detail") or ""
    rows.append({
        "iteration": last_iteration + 1,
        "command": f"agent done: {done_reason}" if done_reason else "agent done",
        "files_written": [],
        "eval_submitted": False,
        "error_summary": "",
        "elapsed_seconds": 0,
        "agent_done": True,
        "done_reason": done_reason,
        "terminal_reason": "agent_done",
    })
    return rows


def _done_iteration_detail(exp_dir: str, summary: Optional[dict[str, Any]], n: int) -> Optional[dict[str, Any]]:
    trajectory = _read_json(os.path.join(exp_dir, "summary", "trajectory.json"))
    if trajectory:
        iterations_data = trajectory.get("iterations", [])
    else:
        iterations_data = [{"iteration": it_num} for it_num, _ in _iter_dirs(exp_dir)]
    rows = _append_done_iteration(exp_dir, summary, iterations_data)
    if not rows or not rows[-1].get("agent_done") or rows[-1].get("iteration") != n:
        return None
    action = _done_action(exp_dir, summary) or {"done": True}
    return {
        "iteration": n,
        "action": action,
        "command_output": None,
        "eval_result": None,
        "observation": None,
        "workspace_files": [],
        "agent_done": True,
        "done_reason": action.get("done_reason") or (summary or {}).get("termination_detail"),
    }


def _command_execution_stats(experiment_dir: str) -> dict[str, Any]:
    """Count executed commands and successful command completions."""
    executed = 0
    successes = 0
    timeouts = 0
    failures = 0
    try:
        iter_dirs = _iter_dirs(experiment_dir)
    except OSError:
        iter_dirs = []

    for _, it_dir in iter_dirs:
        payload = _read_json(os.path.join(it_dir, "command_output.json"), max_bytes=500_000)
        if not isinstance(payload, dict):
            continue
        executed += 1
        exit_code = _to_float(payload.get("exit_code"))
        exit_ok = exit_code is not None and exit_code == 0
        timed_out = bool(payload.get("timed_out"))
        if exit_ok and not timed_out:
            successes += 1
        else:
            failures += 1
            if timed_out:
                timeouts += 1

    return {
        "command_count": executed,
        "command_successes": successes,
        "command_failures": failures,
        "command_timeouts": timeouts,
        "code_execution_success": successes / executed if executed else 0.0,
    }


def _snapshot_files(iter_dir: str) -> dict[str, str]:
    """Read all files in workspace_snapshot/ of an iteration.

    Skips the same directories as the core framework (WORKSPACE_SKIP_DIRS)
    to avoid loading downloaded packages/models that leaked into snapshots.
    """
    from farbench.utils import WORKSPACE_SKIP_DIRS

    snap_dir = os.path.join(iter_dir, "workspace_snapshot")
    files: dict[str, str] = {}
    if not os.path.isdir(snap_dir):
        return files
    for root, dirnames, filenames in os.walk(snap_dir):
        dirnames[:] = [d for d in dirnames if d not in WORKSPACE_SKIP_DIRS]
        for fn in sorted(filenames):
            full = os.path.join(root, fn)
            if not _is_within_path(full, snap_dir):
                continue
            rel = os.path.relpath(full, snap_dir)
            if _is_text_file(fn):
                content = _read_text_safe(full)
                if content is not None:
                    files[rel] = content
    return files


def _compute_diff(old_files: dict[str, str], new_files: dict[str, str]) -> list[dict]:
    """Compute unified diffs between two file snapshots."""
    all_names = sorted(set(old_files) | set(new_files))
    diffs = []
    for name in all_names:
        old = old_files.get(name, "")
        new = new_files.get(name, "")
        if old == new:
            continue
        old_lines = old.splitlines(keepends=True)
        new_lines = new.splitlines(keepends=True)
        diff_lines = list(difflib.unified_diff(
            old_lines, new_lines,
            fromfile=f"a/{name}", tofile=f"b/{name}", lineterm="",
        ))
        if diff_lines:
            added = sum(1 for l in diff_lines if l.startswith("+") and not l.startswith("+++"))
            removed = sum(1 for l in diff_lines if l.startswith("-") and not l.startswith("---"))
            status = "added" if name not in old_files else ("deleted" if name not in new_files else "modified")
            diffs.append({
                "filename": name,
                "status": status,
                "added_lines": added,
                "removed_lines": removed,
                "diff": "\n".join(diff_lines),
            })
    return diffs


def _experiment_summary(task_name: str, exp_dirname: str, exp_dir: str) -> Optional[dict]:
    """Build a summary dict for one experiment."""
    config = _read_json(os.path.join(exp_dir, "config.json"))
    final = _read_json(os.path.join(exp_dir, "summary", "final_results.json"))

    # Determine status:
    #   "completed" = final_results reached a non-error terminal state
    #   "failed"    = final error, incomplete summary, or stale interrupted run
    #   "running"   = no summary and live_status is still within its plausible budget window
    has_summary = os.path.isdir(os.path.join(exp_dir, "summary"))

    # Extract agent_id from dirname: {agent_id}_{timestamp}_{...}
    # Try to get from leaderboard entry first
    leaderboard = _read_json(os.path.join(exp_dir, "summary", "leaderboard_entry.json"))
    agent_id = (leaderboard or {}).get("agent_id", exp_dirname.rsplit("_", 3)[0] if "_" in exp_dirname else exp_dirname)
    model = _model_key_from_agent_id(agent_id)

    # Count iterations
    iter_count = len(_iter_dirs(exp_dir))

    # Compute status
    if final:
        terminal_reason = str(final.get("terminal_reason") or "").strip().lower()
        status = "failed" if terminal_reason in _FAILED_TERMINAL_REASONS else "completed"
    elif has_summary:
        status = "failed"
    else:
        live = _read_json(os.path.join(exp_dir, "live_status.json"))
        status = "running" if _is_live_experiment_current(exp_dir, live) else "failed"

    summary: dict[str, Any] = {
        "id": exp_dirname,
        "task": task_name,
        "agent_id": agent_id,
        "model": model,
        "model_display_name": _model_display_name(model),
        "status": status,
        "iterations": iter_count,
    }

    if config:
        summary["time_budget_hours"] = config.get("total_time_budget_hours")
        summary["max_iterations"] = config.get("max_iterations")
        summary["compute_type"] = config.get("compute_type")
        summary["primary_metric"] = config.get("primary_metric")
        summary["higher_is_better"] = config.get("higher_is_better")

    if final:
        summary["best_metric"] = final.get("best_primary_metric")
        summary["best_metrics"] = final.get("best_metrics")
        summary["primary_metric"] = final.get("primary_metric", summary.get("primary_metric"))
        summary["higher_is_better"] = final.get("higher_is_better", summary.get("higher_is_better"))
        summary["total_elapsed_hours"] = final.get("total_elapsed_hours")
        summary["eval_submissions_used"] = final.get("eval_submissions_used")
        summary["total_input_tokens"] = final.get("total_input_tokens", 0)
        summary["total_output_tokens"] = final.get("total_output_tokens", 0)
        summary["total_thinking_tokens"] = final.get("total_thinking_tokens", 0)
        summary["total_tokens"] = final.get("total_tokens", 0)
        summary["terminal_reason"] = final.get("terminal_reason")
        summary["termination_detail"] = final.get("termination_detail")
    elif not has_summary:
        # Running experiment: read live_status.json (written by orchestrator each iteration)
        live_status = _read_json(os.path.join(exp_dir, "live_status.json"))
        if live_status:
            summary["best_metric"] = live_status.get("best_primary_metric")
            summary["best_metrics"] = live_status.get("best_metrics")
            summary["eval_submissions_used"] = live_status.get("eval_submissions_used", 0)
            summary["total_input_tokens"] = live_status.get("total_input_tokens", 0)
            summary["total_output_tokens"] = live_status.get("total_output_tokens", 0)
            summary["total_thinking_tokens"] = live_status.get("total_thinking_tokens", 0)
            summary["total_tokens"] = live_status.get("total_tokens", 0)
            summary["total_elapsed_hours"] = live_status.get("elapsed_hours")
            summary["remaining_hours"] = live_status.get("remaining_hours")
            summary["per_iteration_tokens"] = live_status.get("per_iteration_tokens", [])
        else:
            # Fallback: scan iteration dirs if live_status.json not yet written
            best_metric = None
            higher = summary.get("higher_is_better", True)
            total_input = 0
            total_output = 0
            total_thinking = 0
            eval_count = 0
            for _, it_dir in _iter_dirs(exp_dir):
                eval_res = _read_json(os.path.join(it_dir, "eval_result.json"))
                if eval_res:
                    eval_count += 1
                    val = eval_res.get("primary_metric_value")
                    if val is not None:
                        if best_metric is None:
                            best_metric = val
                        elif higher and val > best_metric:
                            best_metric = val
                        elif not higher and val < best_metric:
                            best_metric = val
                # Read tokens from history in obs.json
                obs_data = _read_json(os.path.join(it_dir, "obs.json"))
                if obs_data and "history" in obs_data:
                    for h in obs_data["history"]:
                        if h.get("iteration") == int(os.path.basename(it_dir).split("_")[1]):
                            total_input += h.get("input_tokens", 0)
                            total_output += h.get("output_tokens", 0)
                            total_thinking += h.get("thinking_tokens", 0)
                            break
            summary["best_metric"] = best_metric
            summary["eval_submissions_used"] = eval_count
            if total_input or total_output or total_thinking:
                summary["total_input_tokens"] = total_input
                summary["total_output_tokens"] = total_output
                summary["total_thinking_tokens"] = total_thinking
                # thinking_tokens is a breakdown of output_tokens (already
                # included); do not double-count.
                summary["total_tokens"] = total_input + total_output

    if leaderboard:
        summary["budget_utilization_pct"] = leaderboard.get("budget_utilization_pct")
        summary["timestamp"] = leaderboard.get("timestamp")

    return summary


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------

def create_dashboard_router(experiments_dir: str = "experiments") -> APIRouter:
    """Create the dashboard API router.

    The router is read-only: it never modifies experiment data.
    It only reads persisted JSON files from the experiments directory.
    """
    router = APIRouter()
    exp_root = os.path.abspath(experiments_dir)

    # ── HTML serving ──

    @router.get("/dashboard")
    def serve_dashboard():
        html_path = os.path.join(os.path.dirname(__file__), "dashboard.html")
        if not os.path.isfile(html_path):
            raise HTTPException(404, "dashboard.html not found")
        return FileResponse(html_path, media_type="text/html")

    @router.get("/favicon.png")
    def serve_favicon():
        # Look for image.png in project root (two levels up from gui/)
        gui_dir = os.path.dirname(__file__)
        favicon_path = os.path.join(gui_dir, "..", "image.png")
        favicon_path = os.path.abspath(favicon_path)
        if not os.path.isfile(favicon_path):
            raise HTTPException(404, "favicon not found")
        return FileResponse(favicon_path, media_type="image/png")

    @router.get("/api/agent-demo")
    def agent_demo():
        """Return a compact, paper-facing agent workflow replay for the overview."""
        demo_path = os.path.join(os.path.dirname(__file__), "agent_workflow_demo.json")
        data = _read_json(demo_path)
        if not data:
            raise HTTPException(404, "agent workflow demo not found")
        return data

    @router.get("/api/paper-analysis")
    def paper_analysis():
        """Return experiment-analysis data computed from experiment records."""
        analysis_rows = _paper_rows_from_capabilities(capabilities())
        capability_rows = analysis_rows["capability_rows"]
        failure_rows = analysis_rows["failure_rows"]
        iteration_rows = analysis_rows["iteration_rows"]

        deltas = [
            row["best_achievement"] - row["first_achievement"]
            for row in iteration_rows
            if row["best_achievement"] is not None and row["first_achievement"] is not None
        ]
        n_episodes = len(deltas)
        iteration_summary = {
            "valid_episodes": n_episodes,
            "median_delta": _paper_median(deltas),
            "pct_no_improvement": (
                100.0 * sum(1 for delta in deltas if delta <= 0.005) / n_episodes
                if n_episodes else None
            ),
            "pct_gain_under_0_10": (
                100.0 * sum(1 for delta in deltas if delta < 0.10) / n_episodes
                if n_episodes else None
            ),
        }

        bins = [(1, 1), (2, 3), (4, 6), (7, 10), (11, 30)]
        iteration_bins: list[dict[str, Any]] = []
        for lo, hi in bins:
            selected = [
                row for row in iteration_rows
                if lo <= row["n_valid_eval_used"] <= hi
                and row["headroom_gain"] is not None
            ]
            if not selected:
                continue
            iteration_bins.append({
                "label": f"{lo}" if lo == hi else f"{lo}-{hi}",
                "lo": lo,
                "hi": hi,
                "x": sum(row["n_valid_eval_used"] for row in selected) / len(selected),
                "mean_headroom_gain": sum(row["headroom_gain"] for row in selected) / len(selected),
                "count": len(selected),
            })

        model_display_names = {
            row["model"]: row["agent"]
            for row in capability_rows
            if row.get("model") and row.get("agent")
        }
        active_task_counts = [
            row.get("active_tasks")
            for row in failure_rows
            if isinstance(row.get("active_tasks"), int) and row.get("active_tasks") > 0
        ]

        return _sanitize_floats({
            "metrics": [
                {"key": key, "label": label, "short_label": short}
                for key, label, short in _PAPER_METRICS
            ],
            "capability_rows": capability_rows,
            "failure_modes": [
                {"key": key, "label": label, "color": color}
                for key, label, color in _PAPER_FAILURE_MODES
            ],
            "failure_rows": failure_rows,
            "iteration_failure_modes": [
                {"key": key, "label": label, "color": color}
                for key, label, color in _PAPER_ITERATION_FAILURE_MODES
            ],
            "domain_colors": _PAPER_DOMAIN_COLOURS,
            "domain_labels": _PAPER_DOMAIN_LABELS,
            "iteration_rows": iteration_rows,
            "iteration_bins": iteration_bins,
            "iteration_summary": iteration_summary,
            "model_display_names": model_display_names,
            "analysis_summary": {
                "data_source": "experiment_records",
                "agents": len(capability_rows),
                "active_tasks_per_agent": max(active_task_counts) if active_task_counts else None,
                "valid_episodes": n_episodes,
            },
            "data_sources": {
                "capability_metrics": "experiment records",
                "failure_modes": "experiment records",
                "iteration": "experiment records",
            },
        })

    benchmarks_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "benchmarks"))

    def _score_table() -> dict[str, dict[str, Any]]:
        return _read_task_score_table(benchmarks_root)

    def _case_number(value: Any) -> Optional[float]:
        val = _to_float(value)
        return val if val is not None and math.isfinite(val) else None

    def _format_case_number(value: Any) -> str:
        val = _case_number(value)
        if val is None:
            return "--"
        if abs(val) >= 100:
            return f"{val:.1f}"
        return f"{val:.3f}"

    def _case_eval_points(iterations: list[dict[str, Any]], score_meta: Optional[dict[str, Any]]) -> list[dict[str, Any]]:
        points: list[dict[str, Any]] = []
        for row in iterations:
            eval_result = row.get("eval_result") if isinstance(row, dict) else None
            if not isinstance(eval_result, dict):
                continue
            raw = _case_number(eval_result.get("primary_metric_value"))
            if raw is None:
                continue
            achievement = _achievement(raw, score_meta)["achievement"] if score_meta else None
            points.append({
                "iteration": row.get("iteration"),
                "metric_name": eval_result.get("primary_metric_name") or (score_meta or {}).get("primary_metric"),
                "raw_value": raw,
                "achievement": achievement,
            })
        return points

    @router.get("/api/analysis-cases")
    def analysis_cases():
        """Return curated trajectory-regime case studies from experiment logs."""
        example_dir = os.path.join(os.path.dirname(__file__), "paper_example")
        score_table = _score_table()
        cases: list[dict[str, Any]] = []

        for spec in _ANALYSIS_CASE_STUDIES:
            path = os.path.join(example_dir, spec["file"])
            trajectory = _read_json(path) or {}
            iterations = trajectory.get("iterations") if isinstance(trajectory, dict) else []
            if not isinstance(iterations, list):
                iterations = []

            task_name = spec["task"]
            score_meta = score_table.get(task_name)
            eval_points = _case_eval_points(iterations, score_meta)
            first_point = eval_points[0] if eval_points else None
            best_point = (
                max(eval_points, key=lambda p: p.get("achievement") if p.get("achievement") is not None else -math.inf)
                if eval_points and score_meta else None
            )
            metric_name = (
                (best_point or first_point or {}).get("metric_name")
                or (score_meta or {}).get("primary_metric")
                or ""
            )
            submitted_count = sum(1 for row in iterations if isinstance(row, dict) and row.get("eval_submitted"))
            accepted_count = len(eval_points)

            by_iter = {
                row.get("iteration"): row
                for row in iterations
                if isinstance(row, dict) and row.get("iteration") is not None
            }
            eval_by_iter = {point["iteration"]: point for point in eval_points}
            evidence = []
            for item in spec["evidence"]:
                point = eval_by_iter.get(item["iteration"])
                evidence.append({
                    **item,
                    "raw_value": point.get("raw_value") if point else None,
                    "achievement": point.get("achievement") if point else None,
                    "command": (by_iter.get(item["iteration"]) or {}).get("command", ""),
                })

            if first_point and best_point:
                achievement_summary = (
                    f"FA {_format_case_number(first_point.get('achievement'))} -> "
                    f"Best {_format_case_number(best_point.get('achievement'))}"
                )
                raw_summary = (
                    f"{metric_name}: {_format_case_number(first_point.get('raw_value'))} -> "
                    f"{_format_case_number(best_point.get('raw_value'))}"
                )
            elif accepted_count:
                achievement_summary = "Best 0.000"
                raw_summary = f"{metric_name}: 0.000"
            else:
                achievement_summary = "No accepted score"
                raw_summary = "official evaluation never returned a metric"

            cases.append({
                "id": spec["id"],
                "index": spec["index"],
                "title": spec["title"],
                "tagline": spec["tagline"],
                "tone": spec["tone"],
                "model": spec["model"],
                "task": task_name,
                "trajectory_file": spec["file"],
                "available": os.path.isfile(path),
                "total_iterations": len(iterations),
                "submitted_evaluations": submitted_count,
                "accepted_evaluations": accepted_count,
                "metric_name": metric_name,
                "achievement_summary": achievement_summary,
                "raw_summary": raw_summary,
                "best_iteration": best_point.get("iteration") if best_point else None,
                "thesis": spec["thesis"],
                "behavior": spec["behavior"],
                "diagnosis": spec["diagnosis"],
                "evidence": evidence,
            })

        return _sanitize_floats({
            "summary": {
                "case_count": len(cases),
                "source": "gui/paper_example trajectory logs",
                "framing": (
                    "Five recurring trajectory regimes that make the aggregate capability diagnostics concrete."
                ),
            },
            "cases": cases,
        })

    def _require_scored_task(task_name: str) -> None:
        if task_name not in _score_table():
            raise HTTPException(404, f"Task not found in benchmarks/task.score: {task_name}")

    # ── Benchmark task listing ──

    @router.get("/api/benchmark-tasks")
    def benchmark_tasks():
        """Return metadata for all packaged benchmark tasks.

        task.score is the source of truth for leaderboard scoring fields when
        present. task.yaml supplies descriptive UI fields for every task.
        """
        import yaml as _yaml

        score_table = _score_table()
        tasks = []
        if not os.path.isdir(benchmarks_root):
            return tasks

        entries = [
            entry for entry in os.listdir(benchmarks_root)
            if os.path.isfile(os.path.join(benchmarks_root, entry, "task.yaml"))
        ]
        for entry in sorted(entries):
            task_yaml = os.path.join(benchmarks_root, entry, "task.yaml")
            try:
                with open(task_yaml) as f:
                    data = _yaml.safe_load(f) or {}
                score_meta = score_table.get(entry, {})
                # Extract first paragraph of description as summary
                desc = (data.get("description") or "").strip()
                summary = desc.split("\n\n")[0].strip() if desc else ""
                domain = score_meta.get("domain", data.get("domain", []))
                if isinstance(domain, str):
                    domain = [domain] if domain else []
                tasks.append({
                    "name": score_meta.get("name") or data.get("name") or entry,
                    "domain": domain,
                    "subdomain": data.get("subdomain", ""),
                    "summary": summary,
                    "description": desc,
                    "primary_metric": score_meta.get("primary_metric") or data.get("primary_metric", ""),
                    "higher_is_better": score_meta.get("higher_is_better", data.get("higher_is_better", True)),
                    "compute_type": data.get("compute_type", "gpu"),
                    "total_time_budget_hours": data.get("total_time_budget_hours", 10.0),
                    "max_gpu_count": data.get("max_gpu_count", 4),
                    "network_access": data.get("network_access", False),
                    "scored": entry in score_table,
                })
            except Exception:
                pass
        return tasks

    # ── Experiment listing ──

    @router.get("/api/experiments")
    def list_experiments(task: Optional[str] = Query(None)):
        """List experiment directories, including unscored helper tasks."""
        results = []
        if not os.path.isdir(exp_root):
            return results

        score_table = _score_table()

        if task:
            task_dir = _safe_task_dir(exp_root, task)
            task_dirs = [task] if os.path.isdir(task_dir) else []
        else:
            task_dirs = [
                d for d in sorted(os.listdir(exp_root))
                if os.path.isdir(os.path.join(exp_root, d)) and not d.startswith(".")
            ]

        for task_name in task_dirs:
            task_dir = os.path.join(exp_root, task_name)
            for exp_dirname in sorted(os.listdir(task_dir), reverse=True):
                exp_dir = os.path.join(task_dir, exp_dirname)
                if not os.path.isdir(exp_dir) or exp_dirname.startswith("."):
                    continue
                s = _experiment_summary(task_name, exp_dirname, exp_dir)
                if s:
                    score_meta = score_table.get(task_name)
                    if score_meta:
                        s["primary_metric"] = score_meta["primary_metric"]
                        s["higher_is_better"] = score_meta["higher_is_better"]
                        s["scored"] = True
                    else:
                        s["scored"] = False
                    results.append(s)
        return results

    # ── Leaderboard ──

    @router.get("/api/leaderboard")
    def leaderboard():
        """Aggregate one scored run per model-task for the leaderboard view."""
        if not os.path.isdir(exp_root):
            return {"tasks": [], "models": [], "model_display_names": {}, "matrix": {}, "domains": [], "model_scores": {}}

        score_table = _score_table()
        if not score_table:
            return {
                "tasks": [],
                "models": [],
                "model_display_names": {},
                "matrix": {},
                "domains": [],
                "model_scores": {},
                "analysis": [],
                "scoring": {
                    "task_score_path": os.path.join(benchmarks_root, "task.score"),
                    "error": "missing or empty score table",
                },
            }

        def _best_iteration(lb: dict, trajectory: dict, pm: str, hib: bool) -> Optional[int]:
            rows = (lb or {}).get("trajectory_summary") or []
            if not rows:
                rows = [
                    {
                        "iteration": r.get("iteration"),
                        pm: (((r.get("eval_result") or {}).get("metrics") or {}).get(pm)),
                    }
                    for r in (trajectory or {}).get("iterations", [])
                ]
            best_iter = None
            best_val = None
            for row in rows:
                val = row.get(pm)
                if val is None:
                    continue
                try:
                    val = float(val)
                except (TypeError, ValueError):
                    continue
                if best_val is None or (hib and val > best_val) or (not hib and val < best_val):
                    best_val = val
                    best_iter = row.get("iteration")
            return best_iter

        def _select_seeded_run(model: str, task_name: str, runs: list[dict[str, Any]]) -> dict[str, Any]:
            ordered = sorted(runs, key=lambda r: r["best_exp_id"])
            seed = f"farbench-leaderboard-v2:seed=0:{task_name}:{model}"
            chosen = random.Random(seed).choice(ordered)
            return {**chosen, "selection_index": ordered.index(chosen), "selection_pool": len(ordered)}

        # candidates[(model, task)] contains scored runs, including legitimate
        # zero-achievement runs. invalid_candidates contains completed runs
        # without the task.score metric. If a scored run exists, no-score runs
        # are ignored for the representative cell instead of marking that cell
        # invalid.
        candidates: dict[tuple[str, str], list[dict[str, Any]]] = {}
        invalid_candidates: dict[tuple[str, str], list[dict[str, Any]]] = {}
        all_run_counts: dict[tuple[str, str], int] = {}
        invalid_run_counts: dict[tuple[str, str], int] = {}
        ignored_unscored_tasks: set[str] = set()

        for task_name in sorted(os.listdir(exp_root)):
            task_dir = os.path.join(exp_root, task_name)
            if not os.path.isdir(task_dir) or task_name.startswith("."):
                continue
            if task_name not in score_table:
                ignored_unscored_tasks.add(task_name)
                continue
            score_meta = score_table[task_name]
            pm = score_meta["primary_metric"]
            hib = bool(score_meta["higher_is_better"])
            for exp_dirname in os.listdir(task_dir):
                exp_dir = os.path.join(task_dir, exp_dirname)
                if not os.path.isdir(exp_dir) or exp_dirname.startswith("."):
                    continue
                config = _read_json(os.path.join(exp_dir, "config.json")) or {}
                final = _read_json(os.path.join(exp_dir, "summary", "final_results.json"))
                lb = _read_json(os.path.join(exp_dir, "summary", "leaderboard_entry.json"))
                if not final:
                    continue

                agent_id = (lb or {}).get("agent_id", exp_dirname.rsplit("_", 3)[0] if "_" in exp_dirname else exp_dirname)
                model = _model_key_from_agent_id(agent_id)
                key = (model, task_name)
                all_run_counts[key] = all_run_counts.get(key, 0) + 1

                best_val, metric_source = _extract_scored_metric(final, lb or {}, config, pm)
                if best_val is None:
                    invalid_run_counts[key] = invalid_run_counts.get(key, 0) + 1
                    invalid_candidates.setdefault(key, []).append({
                        "task": task_name,
                        "model": model,
                        "status": "invalid",
                        "best_metric": None,
                        "metric_name": pm,
                        "metric_source": "",
                        "oriented_score": None,
                        "oriented_floor": None,
                        "oriented_target": None,
                        "uncapped_score": 0.0,
                        "achievement": 0.0,
                        "target_reached": False,
                        "best_exp_id": exp_dirname,
                        "terminal_reason": final.get("terminal_reason") or (lb or {}).get("terminal_reason") or "invalid",
                        "termination_detail": final.get("termination_detail") or (lb or {}).get("termination_detail"),
                        "total_iterations": final.get("total_iterations") or (lb or {}).get("total_iterations") or 0,
                        "max_iterations": final.get("max_iterations") or (lb or {}).get("max_iterations") or 0,
                        "best_iteration": None,
                        "best_iteration_ratio": None,
                    })
                    continue
                score_parts = _achievement(best_val, score_meta)

                trajectory = {}
                if not (lb or {}).get("trajectory_summary"):
                    trajectory = _read_json(os.path.join(exp_dir, "summary", "trajectory.json")) or {}
                total_iters = final.get("total_iterations") or (lb or {}).get("total_iterations") or 0
                max_iters = final.get("max_iterations") or (lb or {}).get("max_iterations") or total_iters
                best_iter = _best_iteration(lb or {}, trajectory, pm, hib)
                candidates.setdefault(key, []).append({
                    "task": task_name,
                    "model": model,
                    "status": "valid",
                    "best_metric": best_val,
                    "metric_name": pm,
                    "metric_source": metric_source,
                    **score_parts,
                    "target_reached": score_parts["uncapped_score"] >= 1.0,
                    "best_exp_id": exp_dirname,
                    "terminal_reason": final.get("terminal_reason") or (lb or {}).get("terminal_reason") or "unknown",
                    "termination_detail": final.get("termination_detail") or (lb or {}).get("termination_detail"),
                    "total_iterations": total_iters,
                    "max_iterations": max_iters,
                    "best_iteration": best_iter,
                    "best_iteration_ratio": (
                        round(best_iter / total_iters, 4)
                        if best_iter is not None and total_iters else None
                    ),
                })

        matrix: dict[str, dict[str, Any]] = {}
        analysis: list[dict[str, Any]] = []
        selected_by_task: dict[str, list[dict[str, Any]]] = {}
        for (model, task_name), runs in candidates.items():
            chosen = _select_seeded_run(model, task_name, runs)
            key = (model, task_name)
            chosen = {
                **chosen,
                "runs": len(runs),
                "total_runs": all_run_counts.get(key, len(runs)),
                "invalid_runs": 0,
                "ignored_no_score_runs": invalid_run_counts.get(key, 0),
            }
            matrix.setdefault(model, {})[task_name] = {
                "status": "valid",
                "invalid": False,
                "best_metric": chosen["best_metric"],
                "achievement": round(chosen["achievement"], 6),
                "uncapped_score": round(chosen["uncapped_score"], 6),
                "target_reached": chosen["target_reached"],
                "runs": len(runs),
                "total_runs": chosen["total_runs"],
                "invalid_runs": 0,
                "ignored_no_score_runs": chosen["ignored_no_score_runs"],
                "best_exp_id": chosen["best_exp_id"],
                "selection_index": chosen["selection_index"],
            }
            analysis.append(chosen)
            selected_by_task.setdefault(task_name, []).append(chosen)

        # A task becomes active once at least one completed episode exists for it.
        # If every attempted episode has no score, it still remains active and
        # contributes zero achievement, matching the paper scoring rule.
        active_task_names = sorted({
            task_name
            for (_model, task_name) in all_run_counts
        })

        for (model, task_name), runs in invalid_candidates.items():
            if task_name not in active_task_names or task_name in matrix.get(model, {}):
                continue
            chosen = _select_seeded_run(model, task_name, runs)
            key = (model, task_name)
            invalid_count = invalid_run_counts.get(key, len(runs))
            total_count = all_run_counts.get(key, invalid_count)
            chosen = {
                **chosen,
                "runs": 0,
                "total_runs": total_count,
                "invalid_runs": invalid_count,
            }
            matrix.setdefault(model, {})[task_name] = {
                "status": "invalid",
                "invalid": True,
                "best_metric": None,
                "achievement": 0.0,
                "uncapped_score": 0.0,
                "target_reached": False,
                "runs": 0,
                "total_runs": total_count,
                "invalid_runs": invalid_count,
                "ignored_no_score_runs": 0,
                "best_exp_id": chosen["best_exp_id"],
                "selection_index": chosen["selection_index"],
            }
            analysis.append(chosen)

        tasks_list = [
            {
                "name": t,
                "primary_metric": score_table[t]["primary_metric"],
                "higher_is_better": score_table[t]["higher_is_better"],
                "domain": score_table[t]["domain"],
                "low_score": score_table[t]["low_score"],
                "high_score": score_table[t]["high_score"],
            }
            for t in active_task_names
        ]
        domains = sorted({d for t in tasks_list for d in t["domain"]})

        models_all = sorted({
            model
            for (model, task_name) in all_run_counts
            if task_name in active_task_names
        } | set(matrix.keys()))

        # Task ranks use the clipped achievement, which is monotonic with the
        # oriented native metric and comparable across lower/higher-is-better
        # tasks. Missing model-task cells are assigned a rank after all valid
        # scored cells, matching the paper-facing "no score gives zero
        # achievement" convention while still ignoring globally inactive tasks.
        task_ranks: dict[str, dict[str, float]] = {}
        for task_name, rows in selected_by_task.items():
            ordered = sorted(rows, key=lambda r: (-r["achievement"], r["model"]))
            ranks: dict[str, float] = {}
            i = 0
            while i < len(ordered):
                j = i + 1
                while j < len(ordered) and math.isclose(
                    ordered[j]["achievement"], ordered[i]["achievement"], rel_tol=1e-12, abs_tol=1e-12
                ):
                    j += 1
                # Average rank for ties, 1-indexed.
                rank = ((i + 1) + j) / 2.0
                for row in ordered[i:j]:
                    ranks[row["model"]] = rank
                i = j
            missing_rank = float(len(ordered) + 1)
            for model in models_all:
                ranks.setdefault(model, missing_rank)
            task_ranks[task_name] = ranks

        domain_task_names: dict[str, list[str]] = {d: [] for d in domains}
        for task in tasks_list:
            for domain in task["domain"]:
                domain_task_names.setdefault(domain, []).append(task["name"])
        domain_task_denominators = {
            domain: len(task_names)
            for domain, task_names in domain_task_names.items()
        }

        model_scores: dict[str, dict[str, Any]] = {}
        active_task_count = len(active_task_names)
        for model in models_all:
            domain_scores: dict[str, float] = {}
            domain_counts: dict[str, int] = {}
            for domain, task_names in domain_task_names.items():
                vals = [
                    matrix.get(model, {}).get(task_name, {}).get("achievement", 0.0)
                    for task_name in task_names
                ]
                valid_count = sum(
                    1
                    for task_name in task_names
                    if (cell := matrix.get(model, {}).get(task_name)) and cell.get("status") == "valid"
                )
                if vals:
                    domain_scores[domain] = round(100.0 * sum(vals) / len(vals), 4)
                domain_counts[domain] = valid_count

            scored_tasks = sum(
                1 for cell in matrix.get(model, {}).values()
                if cell.get("status") == "valid"
            )
            ranks = [
                task_ranks[task_name][model]
                for task_name in active_task_names
                if model in task_ranks.get(task_name, {})
            ]
            score = (
                sum(domain_scores.values()) / len(domain_scores)
                if domain_scores else 0.0
            )
            model_scores[model] = {
                "farbench_score": round(score, 4),
                "tasks_scored": scored_tasks,
                "active_tasks": active_task_count,
                "valid_score_rate": round(scored_tasks / active_task_count, 4) if active_task_count else 0.0,
                "targets_reached": sum(
                    1 for cell in matrix.get(model, {}).values()
                    if cell.get("target_reached")
                ),
                "median_rank": round(_median(ranks), 4) if ranks else None,
                "domain_scores": domain_scores,
                "domain_task_counts": domain_counts,
                "domain_task_denominators": domain_task_denominators,
            }

        domain_leaders: dict[str, dict[str, Any]] = {}
        for domain in domains:
            rows = [
                (model, stats["domain_scores"].get(domain))
                for model, stats in model_scores.items()
                if stats["domain_scores"].get(domain) is not None
            ]
            rows = [(model, score) for model, score in rows if score is not None]
            rows.sort(key=lambda x: (-x[1], x[0]))
            if rows:
                leader = rows[0]
                runner_up = rows[1] if len(rows) > 1 else None
                domain_leaders[domain] = {
                    "leader": leader[0],
                    "leader_score": round(leader[1], 4),
                    "runner_up": runner_up[0] if runner_up else None,
                    "runner_up_score": round(runner_up[1], 4) if runner_up else None,
                    "margin": round(leader[1] - runner_up[1], 4) if runner_up else None,
                }

        return {
            "tasks": tasks_list,
            "models": sorted(matrix.keys()),
            "model_display_names": {m: _model_display_name(m) for m in sorted(matrix.keys())},
            "matrix": matrix,
            "domains": domains,
            "domain_task_denominators": domain_task_denominators,
            "model_scores": model_scores,
            "domain_leaders": domain_leaders,
            "analysis": analysis,
            "scoring": {
                "task_score_path": os.path.join(benchmarks_root, "task.score"),
                "score_table_tasks": len(score_table),
                "active_scored_tasks": active_task_count,
                "ignored_unscored_tasks": sorted(ignored_unscored_tasks),
                "globally_inactive_tasks_ignored": True,
                "invalid_run_candidates_ignored": "no-score runs are ignored when a scored run exists for the same model-task; selected no-score-only cells are shown with zero achievement",
                "missing_active_model_task_cells": "zero achievement in FARBench Score; native metric left missing",
                "selection_policy": "fixed-seed random choice among valid runs per model-task after sorting by experiment id",
                "selection_seed": 0,
                "domain_denominator_policy": "fixed per active domain: denominator is the number of active scored tasks in that domain, independent of each model's valid/missing/zero cells",
                "aggregate": "domain-balanced mean of clipped task achievements, scaled to 100; globally inactive tasks ignored, completed no-score-only tasks and active missing cells count as zero achievement",
            },
        }

    # ── Capability diagnostics ──

    @router.get("/api/capabilities")
    def capabilities():
        """Paper-facing capability diagnostics for the leaderboard dashboard."""
        metric_defs = [
            {
                "key": "instruction_following",
                "label": "Instruction Following",
                "short_label": "IF",
                "name": "Instruction Following",
                "description": "1 - tau_first for the first valid scorable evaluation, where tau_first is max(iteration budget used, time budget used); zero if no valid evaluation exists.",
            },
            {
                "key": "code_execution_success",
                "label": "Code Execution Success",
                "short_label": "CES",
                "name": "Code Execution Success",
                "description": "Fraction of executed commands that exited with code 0 and did not time out; command-free episodes score zero.",
            },
            {
                "key": "first_achievement",
                "label": "First Achievement",
                "short_label": "FA",
                "name": "First Achievement",
                "description": "Clipped normalized achievement of the first valid scorable artifact.",
            },
            {
                "key": "headroom_gain",
                "label": "Headroom Gain",
                "short_label": "HG",
                "name": "Headroom Gain",
                "description": "(best - first) / max(1 - first, eps), clipped to [0, 1].",
            },
            {
                "key": "progress_efficiency",
                "label": "Progress Efficiency",
                "short_label": "PE",
                "name": "Progress Efficiency",
                "description": "Area under the best-so-far achievement curve M(tau) over tau in [0, 1].",
            },
            {
                "key": "breadth_at_0_5",
                "label": "Breadth at 0.5",
                "short_label": "B@0.5",
                "name": "Breadth at 0.5",
                "description": "Indicator that the selected run reaches clipped achievement >= 0.5.",
            },
        ]
        failure_defs = [
            {"key": "missing", "label": "Missing"},
            {"key": "no_valid_eval", "label": "No valid eval"},
            {"key": "no_scorable_metric", "label": "No scorable metric"},
            {"key": "command_fail_or_timeout", "label": "Command fail/timeout"},
            {"key": "valid_zero", "label": "Valid zero"},
            {"key": "low_learning_gain", "label": "Low learning gain"},
            {"key": "low_achievement", "label": "Low achievement"},
            {"key": "productive", "label": "Productive"},
        ]

        def _empty_response(error: str | None = None) -> dict[str, Any]:
            scoring = {
                "task_score_path": os.path.join(benchmarks_root, "task.score"),
                "tau_definition": "tau = max(iteration / max_iterations, cumulative_elapsed_seconds / total_time_budget_seconds)",
                "aggregate": "domain-balanced mean of per-task capability values in [0, 1]; active missing and no-score cells contribute zero",
                "selection_policy": "fixed-seed random choice among valid runs per model-task after sorting by experiment id; no-score-only cells use the same policy",
                "selection_seed": 0,
            }
            if error:
                scoring["error"] = error
            return {
                "metrics": metric_defs,
                "failure_modes": failure_defs,
                "tasks": [],
                "domains": [],
                "models": [],
                "model_display_names": {},
                "model_scores": {},
                "episodes": [],
                "scoring": scoring,
            }

        if not os.path.isdir(exp_root):
            return _empty_response("experiments directory not found")

        score_table = _score_table()
        if not score_table:
            return _empty_response("missing or empty score table")

        def _select_seeded_run(model: str, task_name: str, runs: list[dict[str, Any]]) -> dict[str, Any]:
            ordered = sorted(runs, key=lambda r: r["best_exp_id"])
            seed = f"farbench-leaderboard-v2:seed=0:{task_name}:{model}"
            chosen = random.Random(seed).choice(ordered)
            return {**chosen, "selection_index": ordered.index(chosen), "selection_pool": len(ordered)}

        def _iter_eval_rows(
            exp_dir: str,
            leaderboard: dict[str, Any],
            trajectory: dict[str, Any],
            metric_name: str,
        ):
            rows = (leaderboard or {}).get("trajectory_summary") or []
            if rows:
                for row in rows:
                    submitted = bool(row.get("eval_submitted"))
                    val = _extract_eval_metric_value(row, metric_name)
                    yield row.get("iteration"), submitted or val is not None, row
                return

            rows = (trajectory or {}).get("iterations") or []
            if rows:
                for row in rows:
                    payload = row.get("eval_result") if isinstance(row, dict) else None
                    submitted = bool(row.get("eval_submitted")) if isinstance(row, dict) else False
                    val = _extract_eval_metric_value(payload, metric_name)
                    yield row.get("iteration"), submitted or val is not None, payload or {}
                return

            try:
                iter_dirs = _iter_dirs(exp_dir)
            except OSError:
                iter_dirs = []
            for iter_num, it_dir in iter_dirs:
                eval_payload = _read_json(os.path.join(it_dir, "eval_result.json")) or {}
                action = _read_json(os.path.join(it_dir, "action.json")) or {}
                submitted = bool(action.get("submit_eval")) or bool(eval_payload)
                val = _extract_eval_metric_value(eval_payload, metric_name)
                yield iter_num, submitted or val is not None, eval_payload

        def _eval_points(
            exp_dir: str,
            leaderboard: dict[str, Any],
            trajectory: dict[str, Any],
            metric_name: str,
            score_meta: dict[str, Any],
            max_iterations: Any,
            time_budget_hours: Any,
        ) -> tuple[list[dict[str, Any]], int]:
            eval_count = 0
            points: list[dict[str, Any]] = []
            time_budget_seconds = _to_float(time_budget_hours)
            if time_budget_seconds is not None:
                time_budget_seconds *= 3600.0
            cumulative_elapsed = 0.0
            for iter_num, submitted, payload in _iter_eval_rows(exp_dir, leaderboard, trajectory, metric_name):
                elapsed = None
                if isinstance(payload, dict):
                    elapsed = _to_float(payload.get("elapsed_seconds"))
                if elapsed is None and iter_num is not None:
                    obs = _read_json(os.path.join(exp_dir, f"iter_{int(iter_num):03d}", "obs.json")) or {}
                    elapsed = _to_float(obs.get("elapsed_seconds")) if isinstance(obs, dict) else None
                if elapsed is not None and elapsed > 0:
                    cumulative_elapsed += elapsed
                val = _extract_eval_metric_value(payload, metric_name)
                if submitted or val is not None:
                    eval_count += 1
                if val is None:
                    continue
                tau_iter = _tau_for_iteration(iter_num, max_iterations)
                tau_time = (
                    _clamp01(cumulative_elapsed / time_budget_seconds)
                    if time_budget_seconds and time_budget_seconds > 0
                    else None
                )
                tau_candidates = [t for t in (tau_iter, tau_time) if t is not None]
                tau = max(tau_candidates) if tau_candidates else None
                if tau is None:
                    continue
                achievement = _achievement(val, score_meta)["achievement"]
                points.append({
                    "iteration": int(iter_num) if iter_num is not None else None,
                    "tau": tau,
                    "tau_iter": tau_iter,
                    "tau_time": tau_time,
                    "elapsed_seconds_until_eval": round(cumulative_elapsed, 6),
                    "metric": val,
                    "achievement": achievement,
                })
            points.sort(key=lambda p: (p["tau"], p["iteration"] or 0))
            return points, eval_count

        def _best_point_iteration(points: list[dict[str, Any]]) -> Optional[int]:
            best_iter = None
            best_achievement = None
            for point in points:
                achievement = point.get("achievement")
                if achievement is None:
                    continue
                if best_achievement is None or achievement > best_achievement:
                    best_achievement = achievement
                    best_iter = point.get("iteration")
            return best_iter

        def _curve_and_auc(
            points: list[dict[str, Any]],
            max_iterations: Any,
            fallback_achievement: Optional[float],
            fallback_iteration: Optional[int],
        ) -> tuple[list[dict[str, float]], float, Optional[dict[str, Any]]]:
            usable_points = list(points)
            if fallback_achievement is not None and fallback_iteration is not None:
                tau = _tau_for_iteration(fallback_iteration, max_iterations)
                if tau is not None:
                    usable_points.append({
                        "iteration": fallback_iteration,
                        "tau": tau,
                        "tau_iter": tau,
                        "tau_time": None,
                        "elapsed_seconds_until_eval": None,
                        "achievement": fallback_achievement,
                    })
            elif not usable_points and fallback_achievement is not None:
                usable_points.append({
                    "iteration": None,
                    "tau": 1.0,
                    "tau_iter": None,
                    "tau_time": None,
                    "elapsed_seconds_until_eval": None,
                    "achievement": fallback_achievement,
                })

            usable_points.sort(key=lambda p: (p["tau"], p.get("iteration") or 0))
            first = usable_points[0] if usable_points else None
            curve = [{"tau": 0.0, "achievement": 0.0}]
            auc = 0.0
            best = 0.0
            last_tau = 0.0
            for point in usable_points:
                tau = _clamp01(float(point.get("tau") or 0.0))
                if tau < last_tau:
                    tau = last_tau
                auc += (tau - last_tau) * best
                best = max(best, _clamp01(float(point.get("achievement") or 0.0)))
                last_tau = tau
                if math.isclose(curve[-1]["tau"], tau, rel_tol=1e-12, abs_tol=1e-12):
                    curve[-1]["achievement"] = max(curve[-1]["achievement"], best)
                else:
                    curve.append({"tau": round(tau, 6), "achievement": round(best, 6)})
            auc += (1.0 - last_tau) * best
            if not math.isclose(curve[-1]["tau"], 1.0, rel_tol=1e-12, abs_tol=1e-12):
                curve.append({"tau": 1.0, "achievement": round(best, 6)})
            else:
                curve[-1]["achievement"] = round(max(curve[-1]["achievement"], best), 6)
            return curve, _clamp01(auc), first

        def _classify_failure(
            status: str,
            best_achievement: float,
            headroom_gain: float,
            valid_eval_count: int,
            eval_count: int,
            command_stats: dict[str, Any],
        ) -> str:
            if status == "missing":
                return "missing"
            if valid_eval_count == 0:
                return "no_valid_eval" if eval_count == 0 else "no_scorable_metric"
            command_count = int(command_stats.get("command_count") or 0)
            ces = float(command_stats.get("code_execution_success") or 0.0)
            if command_count > 0 and ces < 0.5 and best_achievement < 0.5:
                return "command_fail_or_timeout"
            if best_achievement <= 1e-12:
                return "valid_zero"
            if best_achievement < 0.5 and headroom_gain < 0.1:
                return "low_learning_gain"
            if best_achievement < 0.5:
                return "low_achievement"
            return "productive"

        candidates: dict[tuple[str, str], list[dict[str, Any]]] = {}
        invalid_candidates: dict[tuple[str, str], list[dict[str, Any]]] = {}
        all_run_counts: dict[tuple[str, str], int] = {}
        invalid_run_counts: dict[tuple[str, str], int] = {}
        ignored_unscored_tasks: set[str] = set()

        for task_name in sorted(os.listdir(exp_root)):
            task_dir = os.path.join(exp_root, task_name)
            if not os.path.isdir(task_dir) or task_name.startswith("."):
                continue
            if task_name not in score_table:
                ignored_unscored_tasks.add(task_name)
                continue
            score_meta = score_table[task_name]
            pm = score_meta["primary_metric"]
            for exp_dirname in os.listdir(task_dir):
                exp_dir = os.path.join(task_dir, exp_dirname)
                if not os.path.isdir(exp_dir) or exp_dirname.startswith("."):
                    continue
                config = _read_json(os.path.join(exp_dir, "config.json")) or {}
                final = _read_json(os.path.join(exp_dir, "summary", "final_results.json"))
                lb = _read_json(os.path.join(exp_dir, "summary", "leaderboard_entry.json")) or {}
                if not final:
                    continue

                agent_id = lb.get("agent_id", exp_dirname.rsplit("_", 3)[0] if "_" in exp_dirname else exp_dirname)
                model = _model_key_from_agent_id(agent_id)
                key = (model, task_name)
                all_run_counts[key] = all_run_counts.get(key, 0) + 1

                total_iters = final.get("total_iterations") or lb.get("total_iterations") or len(_iter_dirs(exp_dir))
                max_iters = (
                    final.get("max_iterations")
                    or lb.get("max_iterations")
                    or config.get("max_iterations")
                    or total_iters
                    or 0
                )
                best_val, metric_source = _extract_scored_metric(final, lb, config, pm)
                base_row = {
                    "task": task_name,
                    "model": model,
                    "metric_name": pm,
                    "best_exp_id": exp_dirname,
                    "exp_dir": exp_dir,
                    "terminal_reason": final.get("terminal_reason") or lb.get("terminal_reason") or "unknown",
                    "termination_detail": final.get("termination_detail") or lb.get("termination_detail"),
                    "total_iterations": total_iters,
                    "max_iterations": max_iters,
                    "time_budget_hours": (
                        final.get("total_time_budget_hours")
                        or lb.get("total_time_budget_hours")
                        or config.get("total_time_budget_hours")
                    ),
                }
                if best_val is None:
                    invalid_run_counts[key] = invalid_run_counts.get(key, 0) + 1
                    invalid_candidates.setdefault(key, []).append({
                        **base_row,
                        "status": "invalid",
                        "best_metric": None,
                        "metric_source": "",
                        "achievement": 0.0,
                    })
                    continue

                score_parts = _achievement(best_val, score_meta)
                candidates.setdefault(key, []).append({
                    **base_row,
                    "status": "valid",
                    "best_metric": best_val,
                    "metric_source": metric_source,
                    "achievement": score_parts["achievement"],
                    "uncapped_score": score_parts["uncapped_score"],
                    "target_reached": score_parts["uncapped_score"] >= 1.0,
                })

        selected: dict[tuple[str, str], dict[str, Any]] = {}
        active_task_names = sorted({task_name for (_model, task_name) in all_run_counts})
        for (model, task_name), runs in candidates.items():
            chosen = _select_seeded_run(model, task_name, runs)
            key = (model, task_name)
            selected[key] = {
                **chosen,
                "runs": len(runs),
                "total_runs": all_run_counts.get(key, len(runs)),
                "invalid_runs": 0,
                "ignored_no_score_runs": invalid_run_counts.get(key, 0),
            }

        for (model, task_name), runs in invalid_candidates.items():
            key = (model, task_name)
            if task_name not in active_task_names or key in selected:
                continue
            chosen = _select_seeded_run(model, task_name, runs)
            invalid_count = invalid_run_counts.get(key, len(runs))
            selected[key] = {
                **chosen,
                "runs": 0,
                "total_runs": all_run_counts.get(key, invalid_count),
                "invalid_runs": invalid_count,
                "ignored_no_score_runs": 0,
            }

        tasks_list = [
            {
                "name": t,
                "primary_metric": score_table[t]["primary_metric"],
                "higher_is_better": score_table[t]["higher_is_better"],
                "domain": score_table[t]["domain"],
                "low_score": score_table[t]["low_score"],
                "high_score": score_table[t]["high_score"],
            }
            for t in active_task_names
        ]
        domains = sorted({d for t in tasks_list for d in t["domain"]})
        domain_task_names: dict[str, list[str]] = {d: [] for d in domains}
        for task in tasks_list:
            for domain in task["domain"]:
                domain_task_names.setdefault(domain, []).append(task["name"])

        models = sorted({model for (model, task_name) in all_run_counts if task_name in active_task_names})

        def _missing_episode(model: str, task: dict[str, Any]) -> dict[str, Any]:
            metrics = {m["key"]: 0.0 for m in metric_defs}
            return {
                "model": model,
                "task": task["name"],
                "domain": task["domain"],
                "status": "missing",
                "best_exp_id": None,
                "first_metric": None,
                "best_metric": None,
                "achievement": 0.0,
                "first_eval_iteration": None,
                "tau_first": None,
                "tau_first_iteration": None,
                "tau_first_time": None,
                "elapsed_seconds_until_first_eval": None,
                "eval_count": 0,
                "valid_eval_count": 0,
                "command_count": 0,
                "command_successes": 0,
                "command_failures": 0,
                "command_timeouts": 0,
                "terminal_reason": "missing",
                "time_budget_hours": None,
                "failure_mode": "missing",
                "metrics": metrics,
                "curve": [{"tau": 0.0, "achievement": 0.0}, {"tau": 1.0, "achievement": 0.0}],
            }

        def _build_episode(row: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
            score_meta = score_table[task["name"]]
            pm = score_meta["primary_metric"]
            exp_dir = row["exp_dir"]
            lb = _read_json(os.path.join(exp_dir, "summary", "leaderboard_entry.json")) or {}
            trajectory = {}
            if not lb.get("trajectory_summary"):
                trajectory = _read_json(os.path.join(exp_dir, "summary", "trajectory.json")) or {}

            points, eval_count = _eval_points(
                exp_dir,
                lb,
                trajectory,
                pm,
                score_meta,
                row.get("max_iterations"),
                row.get("time_budget_hours"),
            )
            best_achievement = _clamp01(float(row.get("achievement") or 0.0)) if row.get("status") == "valid" else 0.0
            fallback_iter = _best_point_iteration(points)
            if fallback_iter is None and row.get("status") == "valid":
                fallback_iter = row.get("total_iterations") or row.get("max_iterations")
            curve, progress_efficiency, first_point = _curve_and_auc(
                points,
                row.get("max_iterations"),
                best_achievement if row.get("status") == "valid" else None,
                fallback_iter,
            )

            first_achievement = _clamp01(float(first_point.get("achievement") or 0.0)) if first_point else 0.0
            first_metric = first_point.get("metric") if first_point else None
            first_eval_iteration = first_point.get("iteration") if first_point else None
            tau_first = first_point.get("tau") if first_point else None
            tau_first_iteration = first_point.get("tau_iter") if first_point else None
            tau_first_time = first_point.get("tau_time") if first_point else None
            elapsed_seconds_until_first_eval = (
                first_point.get("elapsed_seconds_until_eval") if first_point else None
            )
            valid_eval_count = len(points) or (1 if first_point and row.get("status") == "valid" else 0)
            effective_eval_count = eval_count or valid_eval_count
            instruction_following = 1.0 - float(tau_first) if tau_first is not None else 0.0
            headroom_gain = (
                (best_achievement - first_achievement) / max(1.0 - first_achievement, 1e-12)
                if first_point else 0.0
            )
            headroom_gain = _clamp01(headroom_gain)
            command_stats = _command_execution_stats(exp_dir)
            metrics = {
                "instruction_following": round(_clamp01(instruction_following), 6),
                "code_execution_success": round(_clamp01(float(command_stats["code_execution_success"])), 6),
                "first_achievement": round(first_achievement, 6),
                "headroom_gain": round(headroom_gain, 6),
                "progress_efficiency": round(progress_efficiency, 6),
                "breadth_at_0_5": 1.0 if best_achievement >= 0.5 else 0.0,
            }
            failure_mode = _classify_failure(
                row.get("status", "invalid"),
                best_achievement,
                headroom_gain,
                valid_eval_count,
                effective_eval_count,
                command_stats,
            )
            episode = {
                "model": row["model"],
                "task": task["name"],
                "domain": task["domain"],
                "status": row.get("status", "invalid"),
                "best_exp_id": row.get("best_exp_id"),
                "first_metric": first_metric,
                "best_metric": row.get("best_metric"),
                "metric_name": pm,
                "metric_source": row.get("metric_source", ""),
                "achievement": round(best_achievement, 6),
                "first_eval_iteration": first_eval_iteration,
                "tau_first": round(float(tau_first), 6) if tau_first is not None else None,
                "tau_first_iteration": (
                    round(float(tau_first_iteration), 6) if tau_first_iteration is not None else None
                ),
                "tau_first_time": round(float(tau_first_time), 6) if tau_first_time is not None else None,
                "elapsed_seconds_until_first_eval": elapsed_seconds_until_first_eval,
                "eval_count": effective_eval_count,
                "valid_eval_count": valid_eval_count,
                "terminal_reason": row.get("terminal_reason") or "unknown",
                "termination_detail": row.get("termination_detail"),
                "runs": row.get("runs", 0),
                "total_runs": row.get("total_runs", 0),
                "invalid_runs": row.get("invalid_runs", 0),
                "ignored_no_score_runs": row.get("ignored_no_score_runs", 0),
                "total_iterations": row.get("total_iterations"),
                "max_iterations": row.get("max_iterations"),
                "time_budget_hours": row.get("time_budget_hours"),
                "failure_mode": failure_mode,
                "metrics": metrics,
                "curve": curve,
            }
            episode.update({
                "command_count": command_stats["command_count"],
                "command_successes": command_stats["command_successes"],
                "command_failures": command_stats["command_failures"],
                "command_timeouts": command_stats["command_timeouts"],
            })
            return episode

        task_by_name = {task["name"]: task for task in tasks_list}
        episodes: list[dict[str, Any]] = []
        episode_map: dict[tuple[str, str], dict[str, Any]] = {}
        for model in models:
            for task_name in active_task_names:
                task = task_by_name[task_name]
                row = selected.get((model, task_name))
                episode = _build_episode(row, task) if row else _missing_episode(model, task)
                episodes.append(episode)
                episode_map[(model, task_name)] = episode

        def _count_failure_modes(rows: list[dict[str, Any]]) -> dict[str, int]:
            counts = {mode["key"]: 0 for mode in failure_defs}
            for row in rows:
                mode = row.get("failure_mode") or "missing"
                counts[mode] = counts.get(mode, 0) + 1
            return counts

        model_scores: dict[str, dict[str, Any]] = {}
        for model in models:
            domain_scores: dict[str, dict[str, float]] = {}
            domain_improvement_scores: dict[str, dict[str, float]] = {}
            domain_task_counts: dict[str, int] = {}
            domain_failure_modes: dict[str, dict[str, int]] = {}
            for domain, task_names in domain_task_names.items():
                rows = [episode_map[(model, task_name)] for task_name in task_names]
                domain_scores[domain] = {
                    metric["key"]: round(
                        sum(float(row["metrics"].get(metric["key"], 0.0)) for row in rows) / len(rows),
                        6,
                    )
                    for metric in metric_defs
                } if rows else {metric["key"]: 0.0 for metric in metric_defs}
                first_avg = (
                    sum(float(row["metrics"].get("first_achievement", 0.0)) for row in rows) / len(rows)
                    if rows else 0.0
                )
                best_avg = (
                    sum(float(row.get("achievement", 0.0)) for row in rows) / len(rows)
                    if rows else 0.0
                )
                best_avg = max(first_avg, best_avg)
                gain_to_best = max(0.0, best_avg - first_avg)
                remaining_headroom = max(0.0, 1.0 - best_avg)
                domain_improvement_scores[domain] = {
                    "first_achievement": round(first_avg, 6),
                    "gain_to_best": round(gain_to_best, 6),
                    "best_achievement": round(best_avg, 6),
                    "remaining_headroom": round(remaining_headroom, 6),
                }
                domain_task_counts[domain] = sum(1 for row in rows if row.get("status") == "valid")
                domain_failure_modes[domain] = _count_failure_modes(rows)

            capability_scores = {}
            for metric in metric_defs:
                vals = [domain_scores[domain][metric["key"]] for domain in domains if domain in domain_scores]
                capability_scores[metric["key"]] = round(sum(vals) / len(vals), 6) if vals else 0.0
            improvement_scores = {}
            for key in ("first_achievement", "gain_to_best", "best_achievement", "remaining_headroom"):
                vals = [
                    domain_improvement_scores[domain][key]
                    for domain in domains
                    if domain in domain_improvement_scores
                ]
                improvement_scores[key] = round(sum(vals) / len(vals), 6) if vals else 0.0

            model_rows = [episode_map[(model, task_name)] for task_name in active_task_names]
            overall = (
                sum(capability_scores.values()) / len(capability_scores)
                if capability_scores else 0.0
            )
            model_scores[model] = {
                "overall_capability_score": round(overall, 6),
                "capability_scores": capability_scores,
                "domain_scores": domain_scores,
                "improvement_scores": improvement_scores,
                "domain_improvement_scores": domain_improvement_scores,
                "domain_task_counts": domain_task_counts,
                "domain_failure_modes": domain_failure_modes,
                "failure_modes": _count_failure_modes(model_rows),
                "tasks_scored": sum(1 for row in model_rows if row.get("status") == "valid"),
                "active_tasks": len(active_task_names),
                "valid_score_rate": round(
                    sum(1 for row in model_rows if row.get("status") == "valid") / len(active_task_names),
                    6,
                ) if active_task_names else 0.0,
            }

        return {
            "metrics": metric_defs,
            "failure_modes": failure_defs,
            "tasks": tasks_list,
            "domains": domains,
            "models": models,
            "model_display_names": {m: _model_display_name(m) for m in models},
            "model_scores": model_scores,
            "episodes": episodes,
            "scoring": {
                "task_score_path": os.path.join(benchmarks_root, "task.score"),
                "score_table_tasks": len(score_table),
                "active_scored_tasks": len(active_task_names),
                "ignored_unscored_tasks": sorted(ignored_unscored_tasks),
                "tau_definition": "tau = max(iteration / max_iterations, cumulative_elapsed_seconds / total_time_budget_seconds)",
                "headroom_gain": "(b - f) / max(1 - f, eps), eps=1e-12, clipped to [0, 1]",
                "progress_efficiency": "left-continuous step-function AUC of best-so-far clipped achievement over tau in [0, 1]",
                "aggregate": "domain-balanced mean of per-task capability values in [0, 1]; active missing and no-score cells contribute zero",
                "selection_policy": "fixed-seed random choice among valid runs per model-task after sorting by experiment id; no-score-only cells use the same policy",
                "selection_seed": 0,
            },
        }

    # ── Research question analysis ──

    @router.get("/api/research-questions")
    def research_questions():
        """Question-driven aggregates for paper-style experiment analysis."""
        lb = leaderboard()
        cap = capabilities()
        models = lb.get("models") or cap.get("models") or []
        tasks = lb.get("tasks") or []
        domains = lb.get("domains") or cap.get("domains") or []
        lb_scores = lb.get("model_scores") or {}
        cap_scores = cap.get("model_scores") or {}
        episodes = cap.get("episodes") or []
        metrics = cap.get("metrics") or []
        failure_modes = cap.get("failure_modes") or []

        if not models:
            return {
                "summary": {"models": 0, "tasks": 0, "episodes": 0},
                "questions": [],
                "model_scores": [],
                "domain_scores": [],
                "capability_metrics": metrics,
                "capability_rows": [],
                "capability_leaders": [],
                "iteration_rows": [],
                "failure_modes": failure_modes,
                "failure_total": {},
                "failure_by_domain": [],
                "failure_by_model": [],
                "task_extremes": {"hardest": [], "easiest": []},
                "correlations": [],
            }

        def _score(model: str) -> float:
            return float((lb_scores.get(model) or {}).get("farbench_score") or 0.0)

        def _pct(value: Any) -> float:
            val = _to_float(value)
            return round(100.0 * val, 4) if val is not None else 0.0

        def _pearson(xs: list[float], ys: list[float]) -> float:
            n = len(xs)
            if n == 0 or n != len(ys):
                return 0.0
            mx = sum(xs) / n
            my = sum(ys) / n
            vx = sum((x - mx) ** 2 for x in xs)
            vy = sum((y - my) ** 2 for y in ys)
            if vx <= 0 or vy <= 0:
                return 0.0
            return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)

        model_rows = []
        for model in sorted(models, key=lambda m: (-_score(m), m)):
            stats = lb_scores.get(model) or {}
            model_rows.append({
                "model": model,
                "display_name": _model_display_name(model),
                "farbench_score": round(_score(model), 4),
                "remaining_headroom": round(max(0.0, 100.0 - _score(model)), 4),
                "valid_score_rate": _pct(stats.get("valid_score_rate")),
                "targets_reached": stats.get("targets_reached") or 0,
                "tasks_scored": stats.get("tasks_scored") or 0,
                "active_tasks": stats.get("active_tasks") or len(tasks),
            })

        top_model = model_rows[0]
        domain_rows = []
        for domain in domains:
            vals = [
                float((lb_scores.get(model) or {}).get("domain_scores", {}).get(domain) or 0.0)
                for model in models
            ]
            leader = (lb.get("domain_leaders") or {}).get(domain) or {}
            domain_rows.append({
                "domain": domain,
                "mean_score": round(sum(vals) / len(vals), 4) if vals else 0.0,
                "best_score": round(max(vals), 4) if vals else 0.0,
                "worst_score": round(min(vals), 4) if vals else 0.0,
                "leader": leader.get("leader"),
                "leader_score": leader.get("leader_score"),
                "runner_up": leader.get("runner_up"),
                "runner_up_score": leader.get("runner_up_score"),
                "margin": leader.get("margin"),
            })
        domain_rows.sort(key=lambda row: row["mean_score"])
        weakest_domain = domain_rows[0] if domain_rows else {}

        capability_rows = []
        for model in sorted(models, key=lambda m: (-_score(m), m)):
            stats = cap_scores.get(model) or {}
            capability_rows.append({
                "model": model,
                "display_name": _model_display_name(model),
                "farbench_score": round(_score(model), 4),
                "overall_capability_score": _pct(stats.get("overall_capability_score")),
                "metrics": {
                    metric["key"]: _pct((stats.get("capability_scores") or {}).get(metric["key"]))
                    for metric in metrics
                },
                "improvement": {
                    key: _pct((stats.get("improvement_scores") or {}).get(key))
                    for key in ("first_achievement", "gain_to_best", "best_achievement", "remaining_headroom")
                },
                "valid_score_rate": _pct(stats.get("valid_score_rate")),
            })

        capability_leaders = []
        for metric in metrics:
            rows = sorted(
                [
                    {
                        "model": model,
                        "display_name": _model_display_name(model),
                        "score": _pct((cap_scores.get(model) or {}).get("capability_scores", {}).get(metric["key"])),
                    }
                    for model in models
                ],
                key=lambda row: (-row["score"], row["model"]),
            )
            capability_leaders.append({
                "key": metric["key"],
                "label": metric["label"],
                "short_label": metric.get("short_label"),
                "leaders": rows[:3],
            })

        iteration_rows = []
        for model in sorted(models, key=lambda m: (-_score(m), m)):
            stats = cap_scores.get(model) or {}
            improvement = stats.get("improvement_scores") or {}
            valid_eps = [e for e in episodes if e.get("model") == model and e.get("status") == "valid"]
            gains = [
                max(0.0, float(e.get("achievement") or 0.0) - float((e.get("metrics") or {}).get("first_achievement") or 0.0))
                for e in valid_eps
            ]
            iteration_rows.append({
                "model": model,
                "display_name": _model_display_name(model),
                "first_achievement": _pct(improvement.get("first_achievement")),
                "gain_to_best": _pct(improvement.get("gain_to_best")),
                "best_achievement": _pct(improvement.get("best_achievement")),
                "remaining_headroom": _pct(improvement.get("remaining_headroom")),
                "valid_episodes": len(valid_eps),
                "episode_avg_gain": round((sum(gains) / len(gains)) * 100.0, 4) if gains else 0.0,
                "gain_over_1pt": sum(1 for g in gains if g > 0.01),
                "gain_over_5pt": sum(1 for g in gains if g > 0.05),
            })

        failure_total_counter = Counter(e.get("failure_mode") or "unknown" for e in episodes)
        failure_total = {mode.get("key"): failure_total_counter.get(mode.get("key"), 0) for mode in failure_modes}
        for mode, count in failure_total_counter.items():
            failure_total.setdefault(mode, count)

        by_domain: dict[str, Counter] = defaultdict(Counter)
        by_model: dict[str, Counter] = defaultdict(Counter)
        for episode in episodes:
            mode = episode.get("failure_mode") or "unknown"
            by_model[episode.get("model") or "unknown"][mode] += 1
            episode_domains = episode.get("domain") or ["unknown"]
            for domain in episode_domains:
                by_domain[domain][mode] += 1

        failure_by_domain = [
            {
                "domain": domain,
                "counts": {mode.get("key"): counter.get(mode.get("key"), 0) for mode in failure_modes},
                "total": sum(counter.values()),
            }
            for domain, counter in sorted(by_domain.items())
        ]
        failure_by_model = [
            {
                "model": model,
                "display_name": _model_display_name(model),
                "counts": {mode.get("key"): by_model[model].get(mode.get("key"), 0) for mode in failure_modes},
                "total": sum(by_model[model].values()),
            }
            for model in sorted(models, key=lambda m: (-_score(m), m))
        ]

        task_rows = []
        for task in tasks:
            task_name = task["name"]
            vals = [
                100.0 * float((lb.get("matrix") or {}).get(model, {}).get(task_name, {}).get("achievement") or 0.0)
                for model in models
            ]
            task_rows.append({
                "task": task_name,
                "domain": task.get("domain") or [],
                "mean_score": round(sum(vals) / len(vals), 4) if vals else 0.0,
                "best_score": round(max(vals), 4) if vals else 0.0,
                "worst_score": round(min(vals), 4) if vals else 0.0,
            })
        hardest = sorted(task_rows, key=lambda row: (row["best_score"], row["mean_score"], row["task"]))[:12]
        easiest = sorted(task_rows, key=lambda row: (-row["mean_score"], row["task"]))[:12]

        valid_episodes = [e for e in episodes if e.get("status") == "valid"]
        correlations = []
        for metric in metrics:
            xs = [float((e.get("metrics") or {}).get(metric["key"]) or 0.0) for e in valid_episodes]
            ys = [float(e.get("achievement") or 0.0) for e in valid_episodes]
            correlations.append({
                "key": metric["key"],
                "label": metric["label"],
                "correlation_with_best": round(_pearson(xs, ys), 4),
            })
        correlations.sort(key=lambda row: -abs(row["correlation_with_best"]))

        productive = failure_total.get("productive", 0)
        low_learning = failure_total.get("low_learning_gain", 0)
        command_fail = failure_total.get("command_fail_or_timeout", 0)
        questions = [
            {
                "key": "replace_researcher",
                "question": "Can current agents replace researchers?",
                "answer": (
                    f"No. The best selected agent reaches {top_model['farbench_score']:.1f}/100, "
                    f"leaving {top_model['remaining_headroom']:.1f} points of normalized headroom."
                ),
                "evidence": [
                    f"Top model: {_model_display_name(top_model['model'])}.",
                    f"Weakest domain by mean score: {weakest_domain.get('domain', 'n/a')} at {weakest_domain.get('mean_score', 0):.1f}.",
                    f"Active benchmark coverage: {len(tasks)} tasks across {len(domains)} domains.",
                ],
            },
            {
                "key": "weak_dimensions",
                "question": "Where do agents fall short?",
                "answer": "They are stronger at operating the protocol than at producing high-quality research artifacts.",
                "evidence": [
                    f"Best model Code Execution Success: {next((r['metrics'].get('code_execution_success', 0) for r in capability_rows if r['model'] == top_model['model']), 0):.1f}.",
                    f"Best model First Achievement: {next((r['metrics'].get('first_achievement', 0) for r in capability_rows if r['model'] == top_model['model']), 0):.1f}.",
                    "First Achievement and Progress Efficiency are the strongest correlates of final achievement.",
                ],
            },
            {
                "key": "capability_leaders",
                "question": "Which agent is strongest on each capability?",
                "answer": "Leadership differs by dimension; overall score should not be treated as a single capability.",
                "evidence": [
                    f"{row['label']}: {row['leaders'][0]['display_name']} ({row['leaders'][0]['score']:.1f})"
                    for row in capability_leaders[:6]
                    if row.get("leaders")
                ],
            },
            {
                "key": "self_iteration",
                "question": "Do agents improve through self-iteration?",
                "answer": "Yes, but the improvement is inconsistent and leaves large remaining headroom.",
                "evidence": [
                    f"{row['display_name']}: first {row['first_achievement']:.1f}, gain {row['gain_to_best']:.1f}, best {row['best_achievement']:.1f}."
                    for row in iteration_rows[:3]
                ],
            },
            {
                "key": "failure_modes",
                "question": "What are the dominant failure modes?",
                "answer": "Low learning gain and low achievement remain substantial even after excluding fully productive episodes.",
                "evidence": [
                    f"Productive episodes: {productive}.",
                    f"Low learning gain episodes: {low_learning}.",
                    f"Command fail or timeout episodes: {command_fail}.",
                ],
            },
        ]

        return {
            "summary": {
                "models": len(models),
                "tasks": len(tasks),
                "domains": len(domains),
                "episodes": len(episodes),
                "valid_episodes": len(valid_episodes),
                "top_model": top_model["model"],
                "top_model_display": top_model["display_name"],
                "top_score": top_model["farbench_score"],
                "top_remaining_headroom": top_model["remaining_headroom"],
                "weakest_domain": weakest_domain.get("domain"),
                "weakest_domain_mean_score": weakest_domain.get("mean_score"),
            },
            "questions": questions,
            "model_scores": model_rows,
            "domain_scores": domain_rows,
            "capability_metrics": metrics,
            "capability_rows": capability_rows,
            "capability_leaders": capability_leaders,
            "iteration_rows": iteration_rows,
            "failure_modes": failure_modes,
            "failure_total": failure_total,
            "failure_by_domain": failure_by_domain,
            "failure_by_model": failure_by_model,
            "task_extremes": {"hardest": hardest, "easiest": easiest},
            "correlations": correlations,
            "model_display_names": {m: _model_display_name(m) for m in models},
            "scoring": {
                "leaderboard": (lb.get("scoring") or {}),
                "capabilities": (cap.get("scoring") or {}),
            },
        }

    # ── Single experiment detail ──

    @router.get("/api/experiments/{task}/{exp_id}")
    def experiment_detail(task: str, exp_id: str):
        exp_dir = _safe_experiment_dir(exp_root, task, exp_id)
        if not os.path.isdir(exp_dir):
            raise HTTPException(404, f"Experiment not found: {task}/{exp_id}")

        summary = _experiment_summary(task, exp_id, exp_dir)
        config = _read_json(os.path.join(exp_dir, "config.json")) or {}
        leaderboard = _read_json(os.path.join(exp_dir, "summary", "leaderboard_entry.json")) or {}

        # Trajectory: use summary if available, otherwise build from iter dirs (live)
        trajectory = _read_json(os.path.join(exp_dir, "summary", "trajectory.json"))
        if trajectory:
            iterations_data = trajectory.get("iterations", [])
        else:
            iterations_data = []
            for it_num, it_dir in _iter_dirs(exp_dir):
                obs = _read_json(os.path.join(it_dir, "obs.json"))
                eval_res = _read_json(os.path.join(it_dir, "eval_result.json"))
                action = _read_json(os.path.join(it_dir, "action.json"))
                cmd_out = _read_json(os.path.join(it_dir, "command_output.json"))
                entry: dict[str, Any] = {"iteration": it_num}
                if eval_res:
                    entry["eval_result"] = eval_res.get("metrics")
                # error_summary: command failed (non-zero exit)
                if cmd_out and cmd_out.get("exit_code", 0) != 0:
                    stderr = cmd_out.get("stderr", "")
                    entry["error_summary"] = stderr[-300:] if stderr else "command failed"
                else:
                    entry["error_summary"] = ""
                # eval_submitted: action requested eval
                entry["eval_submitted"] = bool(
                    action and action.get("submit_eval")
                )
                if obs:
                    entry["elapsed_seconds"] = obs.get("elapsed_seconds", 0)
                    entry["command"] = (action or {}).get("command")
                    entry["files_written"] = list(
                        (action or {}).get("files_to_write", {}).keys()
                    )
                iterations_data.append(entry)
        iterations_data = _append_done_iteration(exp_dir, summary, iterations_data)

        return {
            "summary": summary,
            "config": config,
            "leaderboard": leaderboard,
            "trajectory": iterations_data,
        }

    # ── Trajectory data (optimized for charting) ──

    @router.get("/api/experiments/{task}/{exp_id}/trajectory")
    def experiment_trajectory(task: str, exp_id: str):
        exp_dir = _safe_experiment_dir(exp_root, task, exp_id)
        if not os.path.isdir(exp_dir):
            raise HTTPException(404, f"Experiment not found: {task}/{exp_id}")

        trajectory = _read_json(os.path.join(exp_dir, "summary", "trajectory.json"))
        summary = _experiment_summary(task, exp_id, exp_dir)
        if not trajectory:
            # Experiment may still be running — try live_status.json first
            live_status = _read_json(os.path.join(exp_dir, "live_status.json"))
            if live_status and "per_iteration_tokens" in live_status:
                iterations_data = live_status["per_iteration_tokens"]
                return {
                    "iterations": _append_done_iteration(exp_dir, summary, iterations_data),
                    "total_tokens": live_status.get("total_tokens", 0),
                    "source": "live",
                }

            # Fallback: build from iter dirs
            iterations_data = []
            for it_num, it_dir in _iter_dirs(exp_dir):
                obs = _read_json(os.path.join(it_dir, "obs.json"))
                eval_res = _read_json(os.path.join(it_dir, "eval_result.json"))
                action = _read_json(os.path.join(it_dir, "action.json"))
                cmd_out = _read_json(os.path.join(it_dir, "command_output.json"))
                entry: dict[str, Any] = {"iteration": it_num}
                if eval_res:
                    entry["eval_result"] = eval_res.get("metrics")
                if cmd_out and cmd_out.get("exit_code", 0) != 0:
                    stderr = cmd_out.get("stderr", "")
                    entry["error_summary"] = stderr[-300:] if stderr else "command failed"
                else:
                    entry["error_summary"] = ""
                entry["eval_submitted"] = bool(action and action.get("submit_eval"))
                if obs:
                    entry["elapsed_seconds"] = obs.get("elapsed_seconds", 0)
                iterations_data.append(entry)
            return {"iterations": _append_done_iteration(exp_dir, summary, iterations_data), "source": "live"}

        return {"iterations": _append_done_iteration(exp_dir, summary, trajectory.get("iterations", [])), "source": "summary"}

    # ── Per-iteration detail ──

    @router.get("/api/experiments/{task}/{exp_id}/iterations/{n}")
    def iteration_detail(task: str, exp_id: str, n: int):
        exp_dir = _safe_experiment_dir(exp_root, task, exp_id)
        iter_dir = os.path.join(exp_dir, f"iter_{n:03d}")
        if not os.path.isdir(iter_dir):
            done_detail = _done_iteration_detail(exp_dir, _experiment_summary(task, exp_id, exp_dir), n)
            if done_detail:
                return done_detail
            raise HTTPException(404, f"Iteration {n} not found")

        _MAX = 2_500_000  # 2.5 MB – truncate files larger than this
        return {
            "iteration": n,
            "action": _read_json(os.path.join(iter_dir, "action.json")),
            "command_output": _read_json(os.path.join(iter_dir, "command_output.json"), max_bytes=_MAX),
            "eval_result": _read_json(os.path.join(iter_dir, "eval_result.json")),
            "observation": _read_json(os.path.join(iter_dir, "obs.json"), max_bytes=_MAX),
            "workspace_files": list(_snapshot_files(iter_dir).keys()),
        }

    # ── Workspace diff between iterations ──

    @router.get("/api/experiments/{task}/{exp_id}/diff/{n}")
    def iteration_diff(task: str, exp_id: str, n: int):
        exp_dir = _safe_experiment_dir(exp_root, task, exp_id)
        curr_dir = os.path.join(exp_dir, f"iter_{n:03d}")
        prev_dir = os.path.join(exp_dir, f"iter_{n - 1:03d}") if n > 0 else None

        if not os.path.isdir(curr_dir):
            done_detail = _done_iteration_detail(exp_dir, _experiment_summary(task, exp_id, exp_dir), n)
            if done_detail:
                return {"iteration": n, "prev_iteration": n - 1 if n > 0 else None, "diffs": []}
            raise HTTPException(404, f"Iteration {n} not found")

        old_files = _snapshot_files(prev_dir) if prev_dir and os.path.isdir(prev_dir) else {}
        new_files = _snapshot_files(curr_dir)

        return {
            "iteration": n,
            "prev_iteration": n - 1 if n > 0 else None,
            "diffs": _compute_diff(old_files, new_files),
        }

    # ── Read a single workspace snapshot file ──

    @router.get("/api/experiments/{task}/{exp_id}/code/{n}/{path:path}")
    def read_code(task: str, exp_id: str, n: int, path: str):
        exp_dir = _safe_experiment_dir(exp_root, task, exp_id)
        snapshot_dir = os.path.join(exp_dir, f"iter_{n:03d}", "workspace_snapshot")
        file_path = os.path.join(snapshot_dir, path)

        # Path traversal protection
        if not _is_within_path(file_path, snapshot_dir):
            raise HTTPException(403, "Path traversal denied")

        content = _read_text_safe(file_path)
        if content is None:
            raise HTTPException(404, f"File not found: {path}")

        return {"path": path, "iteration": n, "content": content}

    # ── Training curves (metrics.jsonl from workspace/logs/) ──

    @router.get("/api/experiments/{task}/{exp_id}/curves/{n}")
    def training_curves(task: str, exp_id: str, n: int):
        """Read training curves for iteration n from workspace/logs/iter_{n}/metrics.jsonl."""
        exp_dir = _safe_experiment_dir(exp_root, task, exp_id)
        metrics_path = os.path.join(exp_dir, "workspace", "logs", f"iter_{n}", "metrics.jsonl")

        if not os.path.isfile(metrics_path):
            return {"iteration": n, "data": []}

        # Path traversal protection
        if not _is_within_path(metrics_path, exp_dir):
            raise HTTPException(403, "Path traversal denied")

        data = []
        try:
            with open(metrics_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
        except Exception:
            pass
        return {"iteration": n, "data": data}

    def _filesystem_live_status() -> dict[str, Any]:
        """Best-effort live status for standalone dashboard processes."""
        score_table = _score_table()
        if not os.path.isdir(exp_root):
            return {"status": "idle"}

        best: tuple[float, dict[str, Any]] | None = None
        task_names = set(score_table)
        try:
            task_names.update(
                entry for entry in os.listdir(exp_root)
                if os.path.isdir(os.path.join(exp_root, entry)) and not entry.startswith(".")
            )
        except OSError:
            return {"status": "idle"}

        for task_name in sorted(task_names):
            task_dir = os.path.join(exp_root, task_name)
            if not os.path.isdir(task_dir):
                continue
            try:
                exp_ids = os.listdir(task_dir)
            except OSError:
                continue
            for exp_id in exp_ids:
                if exp_id.startswith("."):
                    continue
                exp_dir = os.path.join(task_dir, exp_id)
                if not os.path.isdir(exp_dir):
                    continue
                live_path = os.path.join(exp_dir, "live_status.json")
                live = _read_json(live_path)
                if not live or not _is_live_experiment_current(exp_dir, live):
                    continue
                mtime = _safe_mtime(live_path) or _safe_mtime(exp_dir) or 0.0
                config = _read_json(os.path.join(exp_dir, "config.json")) or {}
                agent_id = config.get("agent_id") or (
                    exp_id.rsplit("_", 3)[0] if "_" in exp_id else exp_id
                )
                status = {
                    "status": "running",
                    "task_name": task_name,
                    "experiment_id": exp_id,
                    "agent_id": agent_id,
                    "current_iteration": live.get("total_iterations", 0),
                    "total_iterations": live.get("total_iterations", 0),
                    "total_tokens": live.get("total_tokens", 0),
                    "total_input_tokens": live.get("total_input_tokens", 0),
                    "total_output_tokens": live.get("total_output_tokens", 0),
                    "total_thinking_tokens": live.get("total_thinking_tokens", 0),
                    "best_primary_metric": live.get("best_primary_metric"),
                    "elapsed_hours": live.get("elapsed_hours"),
                    "remaining_hours": live.get("remaining_hours"),
                }
                if best is None or mtime > best[0]:
                    best = (mtime, status)
        return best[1] if best else {"status": "idle"}

    # ── SSE live stream ──

    @router.get("/api/live")
    async def live_stream(request: Request):
        """Server-Sent Events stream for live experiment monitoring.

        Polls the app state and experiment directory every 2 seconds.
        Emits: status_update, iteration_complete, experiment_done.
        """
        async def event_generator():
            last_iter_count = -1
            last_status = None
            last_exp_key = None

            while True:
                # Check client disconnect
                if await request.is_disconnected():
                    break

                state = getattr(request.app, "state", None)
                env = getattr(state, "env", None) if state else None

                if env is None:
                    status = _filesystem_live_status()
                    status_str = json.dumps(status, default=str)
                    if last_status != status_str:
                        yield f"event: status_update\ndata: {status_str}\n\n"
                        last_status = status_str

                    exp_key = (status.get("task_name"), status.get("experiment_id"))
                    iter_count = int(status.get("total_iterations") or 0)
                    if status.get("status") == "running" and exp_key[0] and exp_key[1]:
                        if exp_key != last_exp_key:
                            last_exp_key = exp_key
                            last_iter_count = iter_count
                        elif iter_count > last_iter_count:
                            yield f"event: iteration_complete\ndata: {json.dumps(status)}\n\n"
                            last_iter_count = iter_count
                    else:
                        last_exp_key = None
                        last_iter_count = -1
                    await asyncio.sleep(2)
                    continue

                try:
                    status = env.status()
                    status_str = json.dumps(status, default=str)

                    if last_status != status_str:
                        yield f"event: status_update\ndata: {status_str}\n\n"
                        last_status = status_str

                    # Check for new iterations and live token data
                    exp_dir = getattr(env, "_orchestrator", None)
                    if exp_dir and hasattr(exp_dir, "store"):
                        store = exp_dir.store
                        iter_count = len(_iter_dirs(store.experiment_dir))
                        if iter_count > last_iter_count and last_iter_count >= 0:
                            # Include live token data in iteration event
                            live = _read_json(os.path.join(store.experiment_dir, "live_status.json"))
                            event_data = {"iteration": iter_count}
                            if live:
                                event_data["total_tokens"] = live.get("total_tokens", 0)
                                event_data["total_input_tokens"] = live.get("total_input_tokens", 0)
                                event_data["total_output_tokens"] = live.get("total_output_tokens", 0)
                                event_data["total_thinking_tokens"] = live.get("total_thinking_tokens", 0)
                                event_data["best_primary_metric"] = live.get("best_primary_metric")
                                event_data["elapsed_hours"] = live.get("elapsed_hours")
                                event_data["remaining_hours"] = live.get("remaining_hours")
                            yield f"event: iteration_complete\ndata: {json.dumps(event_data)}\n\n"
                        last_iter_count = iter_count

                except Exception:
                    pass

                await asyncio.sleep(2)

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return router
