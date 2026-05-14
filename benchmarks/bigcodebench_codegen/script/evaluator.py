from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from farbench.evaluator import EvalResult, MetricEvaluatorBase
from farbench.schemas import TaskConfig


# Defensive: if the agent forgets to strip markdown code fences in predict.py,
# we strip them here so the subprocess doesn't immediately SyntaxError on
# ``` tokens and report 0 pass@1 for a trivially recoverable issue.
_FENCE_RE = re.compile(r"```(?:[A-Za-z0-9_+-]*)?\s*\n(.*?)```", flags=re.DOTALL)


def _sanitize_completion(text: str) -> str:
    """Strip a surrounding ```python ... ``` fence, if present.

    Multiple fences collapse into their concatenated contents, matching the
    typical ``triple-backtick -> code -> triple-backtick`` pattern emitted by
    chat-tuned / completion-tuned LLMs. If no fence is found the text is
    returned unchanged so clean completions keep round-tripping.
    """
    if "```" not in text:
        return text
    blocks = _FENCE_RE.findall(text)
    if not blocks:
        # Lone/unterminated backticks — remove them so Python can still parse.
        return text.replace("```", "")
    return "\n".join(blocks)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _coerce_prediction_map(raw: Any, ordered_ids: list[str]) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}

    payload = raw.get("predictions")
    if payload is None:
        return {}

    if isinstance(payload, dict):
        return {str(k): str(v) for k, v in payload.items()}

    if isinstance(payload, list):
        if not payload:
            return {}
        if isinstance(payload[0], str):
            mapping: dict[str, str] = {}
            for idx, completion in enumerate(payload):
                if idx >= len(ordered_ids):
                    break
                mapping[ordered_ids[idx]] = str(completion)
            return mapping
        mapping = {}
        for item in payload:
            if not isinstance(item, dict):
                continue
            sample_id = item.get("id")
            completion = item.get("completion")
            if sample_id is None or completion is None:
                continue
            mapping[str(sample_id)] = str(completion)
        return mapping

    return {}


def _build_script(starter_code: str, completion: str, test_code: str) -> str:
    parts = [
        "from __future__ import annotations",
        "",
    ]
    if starter_code.strip():
        parts.extend(["# Starter code", starter_code.strip(), ""])
    parts.extend(["# Model completion", completion.strip(), ""])
    parts.extend(["# Unit tests", test_code.strip(), ""])
    return "\n".join(parts) + "\n"


def _run_one_case(
    completion: str,
    test_case: dict[str, Any],
    timeout_seconds: int,
) -> tuple[bool, str]:
    script_text = _build_script(
        starter_code=str(test_case.get("starter_code", "")),
        completion=_sanitize_completion(completion),
        test_code=str(test_case.get("test_code", "")),
    )
    with tempfile.TemporaryDirectory(prefix="bigcodebench_eval_") as tmp_dir:
        script_path = Path(tmp_dir) / "case.py"
        script_path.write_text(script_text, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, str(script_path)],
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return False, "timeout"

    if proc.returncode == 0:
        return True, ""
    stderr = (proc.stderr or "").strip()
    stdout = (proc.stdout or "").strip()
    msg = stderr or stdout or f"non-zero exit code: {proc.returncode}"
    return False, msg


class BigCodeBenchCodegenEvaluator(MetricEvaluatorBase):
    def evaluate(
        self,
        predictions_path: str,
        test_data_dir: str,
        task_config: TaskConfig,
    ) -> EvalResult:
        del task_config
        pred_path = Path(predictions_path)
        if not pred_path.exists():
            return EvalResult(
                metrics={
                    "pass_at_1": 0.0,
                    "num_passed": 0,
                    "total_problems": 0,
                    "error": "predictions.json not found",
                },
                primary_metric_name="pass_at_1",
                primary_metric_value=0.0,
            )

        test_cases_path = Path(test_data_dir) / "test_cases.jsonl"
        if not test_cases_path.exists():
            return EvalResult(
                metrics={
                    "pass_at_1": 0.0,
                    "num_passed": 0,
                    "total_problems": 0,
                    "error": "test_cases.jsonl not found",
                },
                primary_metric_name="pass_at_1",
                primary_metric_value=0.0,
            )

        test_cases = _read_jsonl(test_cases_path)
        ordered_ids = [str(row.get("id", "")) for row in test_cases]
        raw_pred = _read_json(pred_path)
        pred_map = _coerce_prediction_map(raw_pred, ordered_ids)

        timeout_seconds = int(os.getenv("BIGCODEBENCH_EVAL_TIMEOUT_SECONDS", "240"))
        # Parallelism knob. Threads are fine here because subprocess.run
        # releases the GIL while waiting; we just need one thread per concurrent
        # subprocess. Default 16 matches a typical 32-64 core host and keeps
        # peak RSS manageable (~16 x 300 MB ~= 5 GB).
        max_workers = max(1, int(os.getenv("BIGCODEBENCH_EVAL_WORKERS", "16")))
        # Optional global wall-clock deadline for the entire eval exec phase.
        # Unfinished tasks after the deadline are recorded as "deadline
        # exceeded". 0 or negative disables it.
        deadline_seconds = int(os.getenv("BIGCODEBENCH_EVAL_DEADLINE_SECONDS", "0"))

        passed = 0
        missing = 0
        failed_ids: list[str] = []
        errors: dict[str, str] = {}

        # First pass: pull predictions out and separate missing from pending.
        pending: list[tuple[str, dict[str, Any], str]] = []
        for row in test_cases:
            sample_id = str(row.get("id", ""))
            completion = pred_map.get(sample_id)
            if completion is None:
                missing += 1
                failed_ids.append(sample_id)
                errors[sample_id] = "missing prediction"
                continue
            pending.append((sample_id, row, completion))

        # Second pass: run the test cases in parallel.
        effective_workers = min(max_workers, max(1, len(pending)))
        start_time = time.monotonic()

        def _task(sample_id: str, row: dict[str, Any], completion: str) -> tuple[str, bool, str]:
            ok, err = _run_one_case(completion, row, timeout_seconds)
            return sample_id, ok, err

        if effective_workers <= 1:
            # Sequential fallback (keeps logic simple for debugging).
            for sample_id, row, completion in pending:
                if deadline_seconds > 0 and time.monotonic() - start_time > deadline_seconds:
                    failed_ids.append(sample_id)
                    errors[sample_id] = "deadline exceeded"
                    continue
                _, ok, err = _task(sample_id, row, completion)
                if ok:
                    passed += 1
                else:
                    failed_ids.append(sample_id)
                    errors[sample_id] = err
        else:
            with ThreadPoolExecutor(max_workers=effective_workers) as ex:
                futures = {
                    ex.submit(_task, sid, row, comp): sid
                    for sid, row, comp in pending
                }
                try:
                    if deadline_seconds > 0:
                        elapsed = time.monotonic() - start_time
                        remaining = max(0.0, deadline_seconds - elapsed)
                        iterator = as_completed(futures, timeout=remaining)
                    else:
                        iterator = as_completed(futures)
                    for fut in iterator:
                        sample_id, ok, err = fut.result()
                        if ok:
                            passed += 1
                        else:
                            failed_ids.append(sample_id)
                            errors[sample_id] = err
                except TimeoutError:
                    # Wall-clock deadline hit — record everything still
                    # outstanding as "deadline exceeded" and bail out.
                    for fut, sid in futures.items():
                        if fut.done():
                            try:
                                sample_id, ok, err = fut.result(timeout=0)
                            except Exception as e:  # noqa: BLE001
                                sample_id = sid
                                ok = False
                                err = f"worker error: {e}"
                            if sid in errors:
                                continue
                            if ok:
                                passed += 1
                            else:
                                failed_ids.append(sample_id)
                                errors[sample_id] = err
                        else:
                            fut.cancel()
                            if sid not in errors:
                                failed_ids.append(sid)
                                errors[sid] = "deadline exceeded"

        exec_wall_seconds = round(time.monotonic() - start_time, 2)
        total = len(test_cases)
        pass_at_1 = (passed / total) if total > 0 else 0.0

        metrics = {
            "pass_at_1": round(pass_at_1, 4),
            "num_passed": passed,
            "num_failed": total - passed,
            "num_missing_predictions": missing,
            "total_problems": total,
            "timeout_seconds_per_problem": timeout_seconds,
            "eval_workers": effective_workers,
            "eval_deadline_seconds": deadline_seconds,
            "exec_wall_seconds": exec_wall_seconds,
            "failed_ids_preview": failed_ids[:20],
            "error_preview": {k: errors[k] for k in list(errors)[:5]},
        }

        return EvalResult(
            metrics=metrics,
            primary_metric_name="pass_at_1",
            primary_metric_value=pass_at_1,
        )
