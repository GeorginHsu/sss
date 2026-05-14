"""Agent prompt builders for the single-agent flow.

The prompt is split into two parts for prefix-cache efficiency:
  - system_prompt: static per experiment (instructions + task description)
  - user_prompt:   dynamic per iteration (context JSON payload)

LLM APIs (Anthropic, OpenAI) cache the system prompt across calls,
so only the changing user_prompt is recomputed each iteration.
"""

from __future__ import annotations

import json
from pathlib import Path

from farbench.constants import (
    DETAIL_EVAL_ERROR_TAIL,
    DETAIL_OUTPUT_TAIL,
    HISTORY_COMPACT_MAX,
    HISTORY_RECENT_COUNT,
    MAX_WORKSPACE_CONTENT_SIZE,
    STDERR_PROMPT_TAIL,
    STDOUT_PROMPT_TAIL,
)
from farbench.schemas import Observation, IterationRecord, TaskConfig

# ── Workspace file inclusion strategy ──
# When truncating workspace files, prioritize in order:
# 1. Critical files (always keep if possible)
# 2. Secondary files (keep if space permits)
# 3. Fallback to critical only if still too large
WORKSPACE_CRITICAL_FILES = {
    "model.py", "train.py", "predict.py",
    "config.py", "config.yaml", "config.json", "settings.py",
    "requirements.txt"
}
WORKSPACE_SECONDARY_FILES = {
    "data_loader.py", "dataloaders.py", "dataset.py", "data.py",
    "utils.py", "helper.py", "helpers.py",
    "losses.py", "loss.py", "metrics.py", "metric.py",
    "eval.py", "evaluate.py", "test.py"
}

def _build_iteration_history(records: list[IterationRecord], *, higher_is_better: bool = True) -> list[dict]:
    """Unified history: all iters marked by detail level (recent=full, old=compact)."""
    # Trim if needed (keep all evals + latest non-evals)
    if len(records) <= HISTORY_COMPACT_MAX:
        compact = records
    else:
        evals = [r for r in records if r.eval_result]
        non_evals = [r for r in records if not r.eval_result]
        slots_left = max(0, HISTORY_COMPACT_MAX - len(evals))
        compact = evals + non_evals[-slots_left:] if slots_left > 0 else evals
        compact = sorted(compact, key=lambda r: r.iteration)

    # Find best eval for marking (respects metric direction)
    best_eval_iter = None
    evals_with_result = [r for r in compact if r.eval_result]
    if evals_with_result:
        best_fn = max if higher_is_better else min
        best_eval_iter = best_fn(
            evals_with_result,
            key=lambda r: r.eval_result.primary_metric_value
        ).iteration

    # Mark which iters are recent (should have detail)
    recent = {r.iteration for r in records[-HISTORY_RECENT_COUNT:]}

    history = []
    for rec in compact:
        is_recent = rec.iteration in recent
        entry = {
            "iter": rec.iteration,
            "time_sec": round(rec.elapsed_seconds, 1),
        }

        if rec.description:
            entry["reasoning"] = rec.description

        if rec.eval_result:
            entry["eval"] = rec.eval_result.metrics
            if rec.iteration == best_eval_iter:
                entry["is_best"] = True

        # Full detail only for recent iters
        if is_recent:
            if rec.command_output_summary:
                entry["stdout"] = rec.command_output_summary[-DETAIL_OUTPUT_TAIL:]
            if rec.error_summary:
                entry["stderr"] = rec.error_summary
            if rec.eval_error_log:
                entry["eval_error"] = rec.eval_error_log[-DETAIL_EVAL_ERROR_TAIL:]

        history.append(entry)

    if len(records) > len(compact):
        history.append({"omitted": len(records) - len(compact)})

    return history


# ── System prompt (static per experiment) ──

def _fmt_params_cap(cap_billion: float) -> str:
    """Render the max-model-params cap in a human-friendly way: '1' / '1.5' / '5'."""
    if cap_billion <= 0:
        return "1"
    if float(cap_billion).is_integer():
        return str(int(cap_billion))
    return f"{cap_billion:g}"


def _build_objectives_and_constraints(task_config: TaskConfig) -> str:
    """Consolidated: objective, time budget, environment, and evaluation protocol."""
    if task_config.higher_is_better:
        objective_text = f"Maximize {task_config.primary_metric} (higher is better)"
    else:
        objective_text = f"Minimize {task_config.primary_metric} (lower is better)"

    lines = [
        f"Objective: {objective_text}",
        "",
        "⏱️ PRIMARY CONSTRAINTS: remaining_time_budget_hours AND remaining_iterations",
        f"- Time budget: {task_config.total_time_budget_hours}h — commands exceeding this will be KILLED.",
        f"- Iteration budget: {task_config.max_iterations} iterations — experiment stops when exhausted.",
        "Plan carefully: measure iteration time early by submitting eval.",
        "",
        "Environment & Constraints:",
        f"- Compute: {task_config.compute_type.value}",
        f"- Data: $FARBENCH_DATA_DIR (/data in container, test data hidden)",
        "- Pre-installed (base): torch, torchvision, torchaudio, numpy, scipy, pandas, scikit-learn, pillow, matplotlib, tensorboard, transformers, datasets, accelerate, evaluate, tokenizers, sentencepiece, timm, torchmetrics, einops, albumentations, opencv-python-headless, librosa, soundfile, h5py, tqdm, pyyaml",
        f"- Network: {'enabled' if task_config.network_access else 'disabled'}",
        f"- Model size: ≤{_fmt_params_cap(task_config.max_model_params_billion)}B total parameters (pretrained + added). Exceeding this will incur score penalties.",
    ]

    gpu_count = task_config.max_gpu_count
    if gpu_count:
        gpu_mem = task_config.per_gpu_memory_gb
        if gpu_mem is None:
            from farbench.utils import collect_hardware_info
            hw = collect_hardware_info()
            if hw.get("gpus"):
                gpu_mem = hw["gpus"][0].get("memory_total_gb")
        gpu_line = f"- GPU: {gpu_count}x"
        if gpu_mem:
            gpu_line += f", {gpu_mem} GB VRAM per GPU"
        lines.append(gpu_line)
    lines.append(f"- CPU: {task_config.max_cpu_cores} cores")
    lines.append(
        f"- RAM: {task_config.max_memory_gb} GB (OOM kills your process — keep num_workers and batch_size in check)"
    )

    lines.extend([
        "",
        "TWO-PHASE EVALUATION — Understand this fully:",
        "",
        "Phase 1 (Training, this container):",
        "- You write train.py, predict.py, model.py",
        "- You can download models/data (if network enabled)",
        "- You MUST save everything to workspace",
        "",
        "Phase 2 (Evaluation, isolated + no network):",
        "- Your predict.py will be invoked according to the eval contract (see below for exact format)",
        "- Test data has NO labels — you only see predictions",
        "- predict.py MUST load all models from workspace (no remote APIs)",
        "",
        "KEY: Test your predict.py locally before submitting eval. Offline loading must work."
    ])

    return "\n".join(lines)


_ROLE_AND_ACTIONS = """\
You are an autonomous ML research agent.
Your goal: maximize the primary metric within the resource budget (time, iterations, gpu, model size).

You start with an empty workspace. You must write ALL code from scratch:
- Model architecture
- Training script
- Prediction script (required for evaluation)
- Any config files you need

Available actions per iteration (all optional, can combine):
1. files_to_write: dict of {relative_path: file_content} to create/update files
2. packages_to_install: list of pip packages to install before the command runs,
   e.g. ["open3d", "lerobot==2.1.0"]. Installations persist across all future
   iterations — you only need to install a package once.
3. command: bash shell command to execute (e.g. 'python train.py').
   ALWAYS write Python to .py files. Short inline Python (-c) is OK for simple operations.
   RULES:
   - Do NOT background processes with '&'. Commands must run to completion before the next step.
   - Do NOT use 2>&1 or pipe (| tail). stdout/stderr are captured separately and auto-truncated.
   If you need multiple steps, chain with && in one command, or use separate iterations.
4. submit_eval: dict of {checkpoint_path, predict_script} to request evaluation
   - checkpoint_path: relative path to your saved model checkpoint
   - predict_script: ALWAYS "predict.py" (mandatory filename)
   e.g. {"checkpoint_path": "checkpoints/best.pt", "predict_script": "predict.py"}
5. done: true to end the experiment
6. done_reason: string explaining why you chose to stop (only when done=true)"""

_WORKFLOW = """\
Workflow:
1. FIRST iteration: train with FEW epochs (2-10 max) + submit_eval immediately.
   This validates your pipeline AND measures iteration time. Do NOT skip this step.
2. Measure: Check elapsed time AND remaining_iterations. If iteration took 2min,
   you have ~N more time-wise — but never more than remaining_iterations.
   Whichever budget runs out first ends the experiment. Plan for the tighter one.
3. Iterate: Make incremental changes (edit existing code, don't rewrite).
   Scale epochs based on measured iteration time, not guesses.
4. Finalize: Submit best checkpoint before time runs out.

RULE: Never train for >20 epochs without first submitting eval. You need a baseline score
and iteration time measurement before committing to long training runs.

Reasoning: Each iteration, write Observation (what happened), Analysis (why), Decision (what next).
On errors, diagnose the root cause before retrying."""

_OBSERVATION_FORMAT = """\
Observation (what you receive each iteration):
Every iteration, the framework sends you a Context JSON with these fields:

- iteration: current iteration number
- total_time_budget_hours: total time budget for this experiment
- remaining_time_budget_hours: hours left in time budget
- max_iterations: maximum number of iterations allowed
- remaining_iterations: iterations left before experiment stops
- workspace_files: all files you've written
- workspace: file contents (may be truncated; see workspace_note)
- workspace_note (conditional): explains truncation
- history: past iterations (recent 3 include full stdout/stderr/eval_error)
- command_output (conditional): exit_code, timed_out, elapsed_sec, stdout, stderr
- eval (conditional): primary_metric, value, best_value, is_new_best, delta, other_metrics
- error (conditional): framework error message from last iteration
- alert (conditional): warning if time budget < 20% and no eval yet
- training_curves (conditional): training metrics as JSON (only when curve images are not attached)
  Note: when available, training curves are sent as attached images instead (preferred)

CRITICAL: Before editing any file, always check the current workspace contents to understand existing code structure."""

_TRAINING_LOGGING = """\
Structured logging (CRITICAL for efficiency):
- Write metrics to: logs/iter_{FARBENCH_ITERATION}/metrics.jsonl (NOT stdout)
  FARBENCH_ITERATION is set by framework; access via os.environ['FARBENCH_ITERATION']
- Each line: {"step": 100, "loss": 0.5, "acc": 0.85}
- System generates training curves from these logs (no truncation, complete data)

DO NOT print training loss/accuracy to stdout (wastes tokens, already sent to training_curves).
Keep stdout/stderr for errors, warnings, important events only.
Example train.py: use logging.basicConfig(level=logging.WARNING) or redirect print to file."""

_EXTERNAL_RESOURCES = """\
Network resources (if enabled):
STRATEGY: Spend 1-2 iterations researching the best pre-trained model/algorithm for this task,
then download and adapt. Reusing a strong pre-trained model almost always beats training from scratch.
Only train from scratch if your task is highly custom/domain-specific.

Available tools: huggingface-cli (download models), wget/curl (fetch files/papers).
Add any packages you need via packages_to_install

HARD LIMIT: Total model parameters ≤{max_params_b}B. Choose models wisely (e.g. ViT-B/16 ~86M ✓, ViT-L ~300M ✓, ViT-G ~1.8B ✗).
Models exceeding {max_params_b}B parameters will incur score penalties during evaluation.

Remember: Save all downloaded models/weights/tokenizers/configs to workspace during training.
During evaluation (Phase 2), predict.py will be network-isolated and must load everything from local workspace."""

_OUTPUT_FORMAT = """\
Strict JSON output (no markdown):
{
  "Reasoning": "Observation: accuracy 0.92→0.94 with dropout=0.3. Analysis: regularization works. Decision: increase epochs to 150 (loss still declining).",
  "files_to_write": {"model.py": "...", "train.py": "..."},
  "packages_to_install": [],
  "command": "python train.py --epochs 150",
  "submit_eval": null,
  "done": false,
  "done_reason": ""
}

Reasoning examples (for different scenarios):
- Early baseline: "Observation: baseline CNN trained (acc 0.72). Analysis: architecture works but underfitting. Decision: add regularization, data augmentation, train longer."
- Optimization: "Observation: acc 0.85→0.88 with batch_size=32. Analysis: smaller batches help generalization. Decision: try learning_rate=1e-3 (currently loss plateauing)."
- Error recovery: "Observation: OOM error with batch_size=128. Analysis: model+data too large. Decision: reduce hidden_dim to 256, retry with batch_size=64."
- Late exploration: "Observation: stuck at 0.89 for 3 iters. Analysis: local optimum reached. Decision: try different architecture (residual blocks) + reset learning rate to explore new region."
"""


def _build_system_prompt(task_config: TaskConfig) -> str:
    """Minimal, focused system prompt. Static per experiment (cacheable)."""
    sections = [
        _ROLE_AND_ACTIONS,
        _build_objectives_and_constraints(task_config),
        _WORKFLOW,
        _OBSERVATION_FORMAT,
        _TRAINING_LOGGING,
        _OUTPUT_FORMAT,
    ]

    # Only add if task has eval contract
    if task_config.eval_contract:
        sections.append(
            "Eval contract:\n"
            f"{json.dumps(task_config.eval_contract, ensure_ascii=False, indent=2)}"
        )

    # Only add if network access enabled
    if task_config.network_access:
        sections.append(
            _EXTERNAL_RESOURCES.format(
                max_params_b=_fmt_params_cap(task_config.max_model_params_billion)
            )
        )

    # Task-specific guidance (if provided)
    if getattr(task_config, "agent_hints", ""):
        sections.append("Task-specific guidance:\n" + task_config.agent_hints)

    # Finally: task description
    sections.append("Task description:\n" + task_config.description)

    return "\n\n".join(sections)


# ── User prompt (dynamic per iteration) ──

def _build_context_payload(
    obs: Observation,
    task_config: TaskConfig,
    *,
    has_curve_images: bool = False,
    curve_image_labels: list[str] | None = None,
) -> dict:
    """Lean context: only essential per-iteration state."""
    payload = {
        "iteration": obs.iteration,
        "total_time_budget_hours": task_config.total_time_budget_hours,
        "remaining_time_budget_hours": obs.remaining_time_budget_hours,
        "max_iterations": task_config.max_iterations,
        "remaining_iterations": obs.remaining_iterations,
        "workspace_files": obs.workspace_files,
    }

    if obs.error:
        payload["error"] = obs.error

    # History: recent iters get full detail (stdout, stderr), old iters are compact
    if obs.history:
        payload["history"] = _build_iteration_history(
            obs.history, higher_is_better=task_config.higher_is_better
        )

    # Workspace files: prioritize critical + secondary; keep only critical files if too large.
    # Match by basename so files in subdirectories (e.g. src/model.py) are still recognized
    if obs.workspace_file_contents:
        total = sum(len(v) for v in obs.workspace_file_contents.values())
        if total > MAX_WORKSPACE_CONTENT_SIZE:
            # Try to keep critical + secondary files
            included = WORKSPACE_CRITICAL_FILES | WORKSPACE_SECONDARY_FILES
            workspace = {k: v for k, v in obs.workspace_file_contents.items()
                         if Path(k).name in included}

            # If still too large, keep critical files only.
            if sum(len(v) for v in workspace.values()) > MAX_WORKSPACE_CONTENT_SIZE:
                workspace = {k: v for k, v in obs.workspace_file_contents.items()
                             if Path(k).name in WORKSPACE_CRITICAL_FILES}
                payload["workspace_note"] = "Workspace truncated to essential files (model, train, predict, config). Use 'cat' to inspect others."
            else:
                payload["workspace_note"] = "Workspace includes utility files. Use 'cat' to inspect other files in workspace_files list."

            payload["workspace"] = workspace
        else:
            payload["workspace"] = obs.workspace_file_contents

    # Command output
    if obs.command_output:
        co = obs.command_output
        payload["command_output"] = {
            "exit_code": co.exit_code,
            "timed_out": co.timed_out,
            "elapsed_sec": co.elapsed_seconds,
            "stdout": co.stdout[-STDOUT_PROMPT_TAIL:],
            "stderr": co.stderr[-STDERR_PROMPT_TAIL:] if co.stderr else None,
        }

    # Eval results: consolidated into single field for clarity
    if obs.eval_result:
        # Determine if this is a new personal best
        # Note: best_eval_result was last updated in a prior iteration,
        # so we can compare to see if current result would beat it
        is_new_best = obs.best_eval_result is None
        delta = 0.0
        best_value = obs.eval_result.primary_metric_value

        if obs.best_eval_result is not None:
            best_value = obs.best_eval_result.primary_metric_value
            current = obs.eval_result.primary_metric_value
            # Determine improvement based on metric direction
            if task_config.higher_is_better:
                is_new_best = current > best_value
                delta = current - best_value
            else:
                is_new_best = current < best_value
                delta = best_value - current

        # Build other_metrics: all metrics except primary_metric
        other_metrics = {
            k: v for k, v in obs.eval_result.metrics.items()
            if k != obs.eval_result.primary_metric_name
        }

        payload["eval"] = {
            "primary_metric": obs.eval_result.primary_metric_name,
            "value": obs.eval_result.primary_metric_value,
            "best_value": best_value,
            "is_new_best": is_new_best,
            "delta": round(delta, 6),
        }
        if other_metrics:
            payload["eval"]["other_metrics"] = other_metrics

    # Alert if either budget running low + no eval
    remaining_time = obs.remaining_time_budget_hours or 0
    remaining_iters = obs.remaining_iterations if obs.remaining_iterations is not None else task_config.max_iterations
    time_low = remaining_time > 0 and remaining_time < task_config.total_time_budget_hours * 0.2
    iters_low = remaining_iters <= max(1, int(task_config.max_iterations * 0.2))
    if (time_low or iters_low) and not obs.best_eval_result:
        payload["alert"] = (
            f"Only {remaining_time:.1f}h and {remaining_iters} iterations left, "
            f"NO eval yet. Score will be ZERO."
        )

    # Training curves: prefer images (sent as multimodal); fall back to JSON
    if has_curve_images:
        payload["training_curve_images"] = {
            "attached": True,
            "order": curve_image_labels or [],
        }
    elif obs.training_curves:
        payload["training_curves"] = obs.training_curves

    return payload


# ── Single-agent prompt ──

def build_agent_prompt(
    obs: Observation,
    task_config: TaskConfig,
    *,
    has_curve_images: bool = False,
    curve_image_labels: list[str] | None = None,
) -> tuple[str, str]:
    """Build a prompt for autonomous ML research agents.

    Args:
        has_curve_images: True if curve image files were successfully loaded.
            When True, payload includes image ordering metadata; when False,
            payload includes raw numeric training curve data instead.

    Returns:
        (system_prompt, user_prompt) — split for prefix-cache efficiency.
        system_prompt is static per experiment; user_prompt changes each iteration.
    """
    system_prompt = _build_system_prompt(task_config)
    payload = _build_context_payload(
        obs,
        task_config,
        has_curve_images=has_curve_images,
        curve_image_labels=curve_image_labels,
    )
    user_prompt = f"Context:\n{json.dumps(payload, ensure_ascii=False)}"
    return system_prompt, user_prompt
