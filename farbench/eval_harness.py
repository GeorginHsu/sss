"""Eval harness: container-side entry point for Docker-based evaluation.

Two-phase evaluation:
  Phase 1: Run agent's predict.py to produce predictions JSON.
  Phase 2: Run task evaluator to compare predictions against ground truth.

Usage inside container:
    python -m farbench.eval_harness

Environment variables:
    FARBENCH_EVAL_CHECKPOINT     — path to model checkpoint
    FARBENCH_EVAL_WORKSPACE      — path to agent workspace (contains predict.py)
    FARBENCH_EVAL_OUTPUT         — path to write final EvalResult JSON
    FARBENCH_PREDICT_SCRIPT      — path to agent's predict.py
    FARBENCH_TASK_CONFIG_JSON    — serialized TaskConfig
    FARBENCH_EVAL_SCRIPT_DIR     — directory containing evaluator.py
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict

from farbench.evaluator import MetricEvaluatorBase
from farbench.schemas import EvalResult, TaskConfig


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Required environment variable {name} is not set")
    return value


def main():
    checkpoint = _require_env("FARBENCH_EVAL_CHECKPOINT")
    workspace = _require_env("FARBENCH_EVAL_WORKSPACE")
    output_path = _require_env("FARBENCH_EVAL_OUTPUT")
    predict_script = _require_env("FARBENCH_PREDICT_SCRIPT")
    task_config_json = _require_env("FARBENCH_TASK_CONFIG_JSON")

    task_config = TaskConfig.from_dict(json.loads(task_config_json))
    test_data_dir = task_config.test_data_dir

    # ── Phase 1: Run agent's predict.py ──
    predictions_path = output_path.replace(".json", "_predictions.json")
    if predictions_path == output_path:
        predictions_path = output_path + ".predictions.json"

    # Pick which Python interpreter runs predict.py.
    #   - Default: sys.executable (the same Python that's running eval_harness,
    #     i.e. /usr/bin/python in the eval image). Works for CPU-bound / HF
    #     transformers-only predict scripts.
    #   - Override: task.yaml `eval_contract.interpreter` lets a task point at
    #     a sibling venv inside the image, e.g. /usr/local/bin/vllm-python for
    #     images that keep vLLM in an isolated /opt/vllm venv (to sidestep the
    #     cu118/cu128 torch split and transformers version clash). Without
    #     honoring this field, predict.py runs under /usr/bin/python where
    #     `import vllm` is ModuleNotFoundError even though the image does ship
    #     vLLM — just in a different venv.
    interpreter = task_config.eval_contract.get("interpreter") or sys.executable
    if not os.path.exists(interpreter):
        # Fail loudly now rather than in subprocess with a cryptic exec error.
        raise RuntimeError(
            f"eval_contract.interpreter={interpreter!r} does not exist in the "
            f"eval container. Check the path or remove the field to fall back "
            f"to {sys.executable}."
        )

    print(f"[eval_harness] Phase 1: Running predict script: {predict_script}")
    print(f"  interpreter: {interpreter}")
    print(f"  checkpoint:  {checkpoint}")
    print(f"  data_path:   {test_data_dir}")
    print(f"  output:      {predictions_path}")

    start_time = time.time()
    result = subprocess.run(
        [
            interpreter, predict_script,
            "--data_path", test_data_dir,
            "--checkpoint", checkpoint,
            "--output", predictions_path,
        ],
        cwd=workspace,
        capture_output=True,
        text=True,
    )

    predict_elapsed_ms = (time.time() - start_time) * 1000

    if result.stdout:
        print(f"[predict.py stdout] {result.stdout[:2000]}")
    if result.stderr:
        print(f"[predict.py stderr] {result.stderr[:2000]}", file=sys.stderr)

    if result.returncode != 0:
        raise RuntimeError(
            f"predict.py failed (exit {result.returncode}):\n"
            f"stderr: {result.stderr[:1000]}"
        )

    if not os.path.exists(predictions_path):
        raise RuntimeError(
            f"predict.py did not produce output file: {predictions_path}"
        )

    # ── Phase 2: Run evaluator ──
    print(f"[eval_harness] Phase 2: Running evaluator")

    eval_script = os.path.join(
        os.environ.get("FARBENCH_EVAL_SCRIPT_DIR", "/eval_script"),
        "evaluator.py",
    )
    spec = importlib.util.spec_from_file_location("_farbench_task_evaluator", eval_script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Find the MetricEvaluatorBase subclass
    evaluator_cls = None
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, MetricEvaluatorBase) and obj is not MetricEvaluatorBase:
            evaluator_cls = obj
            break

    if evaluator_cls is None:
        raise RuntimeError(f"No MetricEvaluatorBase subclass found in {eval_script}")

    evaluator = evaluator_cls()
    eval_result = evaluator.evaluate(predictions_path, test_data_dir, task_config)

    # Override inference time with actual measured time
    eval_result.inference_time_ms = predict_elapsed_ms

    # Attach predict.py logs so the agent can diagnose issues
    log_parts = []
    if result.stderr:
        log_parts.append(f"predict.py stderr: {result.stderr[:500]}")
    if result.stdout:
        log_parts.append(f"predict.py stdout: {result.stdout[:500]}")
    if log_parts:
        eval_result.eval_log = "\n".join(log_parts)

    print(f"[eval_harness] Result: {eval_result.metrics}")

    with open(output_path, "w") as f:
        json.dump(asdict(eval_result), f)


if __name__ == "__main__":
    main()
