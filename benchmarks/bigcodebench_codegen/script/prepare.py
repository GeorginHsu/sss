#!/usr/bin/env python3
"""BigCodeBench code-generation data preparation.

Downloads the training set (nvidia/OpenCodeInstruct) and the test set
(bigcode/bigcodebench) from the Hugging Face Hub and writes normalized
JSONL files to ``$FARBENCH_DATA_DIR`` / ``$FARBENCH_TEST_DATA_DIR``.

prepare.py only handles DATA. Model downloading is done in docker/Dockerfile
(pre-cached into HF cache, loaded via from_pretrained at runtime).

Split strategy: OpenCodeInstruct has a single ``train`` split. We stream it
  and take the first MAX_VAL_SAMPLES rows as val, the next MAX_TRAIN_SAMPLES
  rows as train. BigCodeBench ``v0.1.4`` split is taken as-is.

Output layout:
    FARBENCH_DATA_DIR/
        train.jsonl        — ~195K training samples
        val.jsonl          — ~5K validation samples
        dataset_info.json  — metadata
    FARBENCH_TEST_DATA_DIR/
        test_prompts.jsonl — 1,140 BigCodeBench-Full problems (visible to agent)
        test_cases.jsonl   — ground-truth test cases (evaluator only)

Training record schema (one per JSONL line)::

    {
      "id": "...",
      "prompt": "<input>",
      "completion": "<output, possibly wrapped in ```python ... ``` fence>",
      "source": "nvidia/OpenCodeInstruct",
      "domain": "generic|algorithmic|...",        # may be ""
      "average_test_score": 0.0-1.0,              # -1.0 if unavailable
      "tests_execution_status": "pass|fail|..."   # may be "unknown"
    }

The completion field intentionally KEEPS the upstream markdown fence. The
agent's predict.py is responsible for stripping it before producing
predictions.json (see task.yaml agent_hints).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from datasets import load_dataset

# Defaults are set via ENV in docker/Dockerfile (single source of truth).
TRAIN_REPO = os.environ.get("BIGCODEBENCH_TRAIN_REPO", "nvidia/OpenCodeInstruct")
TEST_REPO = os.environ.get("BIGCODEBENCH_TEST_REPO", "bigcode/bigcodebench")

TRAIN_SPLIT = os.environ.get("BIGCODEBENCH_TRAIN_SPLIT", "train")
TEST_SPLIT = os.environ.get("BIGCODEBENCH_TEST_SPLIT", "v0.1.4")

MAX_TRAIN_SAMPLES = int(os.environ.get("BIGCODEBENCH_MAX_TRAIN_SAMPLES", "200000"))
MAX_VAL_SAMPLES = int(os.environ.get("BIGCODEBENCH_MAX_VAL_SAMPLES", "5000"))
MAX_TEST_SAMPLES = int(os.environ.get("BIGCODEBENCH_MAX_TEST_SAMPLES", "0"))


def _first_non_empty(record: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        value = record.get(key)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _extract_from_messages(messages: Any) -> tuple[str, str] | None:
    if not isinstance(messages, list):
        return None
    prompt_parts: list[str] = []
    answer_parts: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "")).lower()
        content = msg.get("content")
        if content is None:
            continue
        text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
        text = text.strip()
        if not text:
            continue
        if role in {"assistant", "model"}:
            answer_parts.append(text)
        else:
            prompt_parts.append(f"[{role or 'user'}]\n{text}")
    if not prompt_parts or not answer_parts:
        return None
    return "\n\n".join(prompt_parts), "\n\n".join(answer_parts)


def _coerce_float(value: Any) -> float:
    """Best-effort numeric coercion; returns -1.0 for None / invalid values."""
    if value is None:
        return -1.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return -1.0


def _normalize_train_record(record: dict[str, Any], fallback_id: int) -> dict[str, Any] | None:
    prompt = _first_non_empty(
        record,
        ["prompt", "instruction", "question", "problem", "input", "task", "query"],
    )
    completion = _first_non_empty(
        record,
        [
            "completion",
            "response",
            "output",
            "answer",
            "solution",
            "code",
            "canonical_solution",
        ],
    )

    if prompt is None or completion is None:
        extracted = _extract_from_messages(record.get("messages"))
        if extracted is not None:
            prompt, completion = extracted

    if prompt is None or completion is None:
        return None

    sample_id = _first_non_empty(record, ["id", "task_id", "problem_id", "uuid"])
    if sample_id is None:
        repo_slug = TRAIN_REPO.replace("/", "_")
        sample_id = f"{repo_slug}-{fallback_id:08d}"

    # Pass through OpenCodeInstruct-specific quality metadata so the agent can
    # choose to filter on it (the dataset is NOT pre-filtered).
    domain = _first_non_empty(record, ["domain", "category", "topic"])
    test_status = _first_non_empty(
        record,
        ["tests_execution_status", "test_status", "execution_status"],
    )
    avg_score = _coerce_float(
        record.get("average_test_score")
        if "average_test_score" in record
        else record.get("avg_test_score")
    )

    return {
        "id": str(sample_id),
        "prompt": str(prompt).strip(),
        # IMPORTANT: do not strip markdown fences from completion — the schema
        # contract in task.yaml promises they are preserved and the agent is
        # expected to handle them.
        "completion": str(completion).strip(),
        "source": TRAIN_REPO,
        "domain": "" if domain is None else str(domain),
        "tests_execution_status": "unknown" if test_status is None else str(test_status),
        "average_test_score": avg_score,
    }


def _normalize_test_record(record: dict[str, Any], fallback_id: int) -> dict[str, str] | None:
    sample_id = _first_non_empty(record, ["id", "task_id", "problem_id", "uuid"])
    if sample_id is None:
        sample_id = f"bigcodebench-{fallback_id:08d}"

    prompt = _first_non_empty(
        record,
        ["complete_prompt", "prompt", "question", "problem", "instruction", "description"],
    )
    starter_code = _first_non_empty(
        record,
        ["code_prompt", "starter_code", "declaration", "signature", "function_signature"],
    )
    test_code = _first_non_empty(
        record,
        ["test", "tests", "test_code", "unit_tests", "checker"],
    )
    entry_point = _first_non_empty(
        record,
        ["entry_point", "function_name", "fn_name", "target_function"],
    )

    if prompt is None or test_code is None:
        return None

    return {
        "id": str(sample_id),
        "prompt": str(prompt).strip(),
        "starter_code": "" if starter_code is None else str(starter_code).strip(),
        "test_code": str(test_code).strip(),
        "entry_point": "" if entry_point is None else str(entry_point).strip(),
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _prepare_train_data(train_dir: Path) -> dict[str, int]:
    dataset = load_dataset(TRAIN_REPO, split=TRAIN_SPLIT, streaming=True)

    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    accepted = 0
    skipped = 0

    for idx, raw in enumerate(dataset):
        row = _normalize_train_record(raw, idx)
        if row is None:
            skipped += 1
            continue
        accepted += 1
        if len(val_rows) < MAX_VAL_SAMPLES:
            val_rows.append(row)
        elif len(train_rows) < MAX_TRAIN_SAMPLES:
            train_rows.append(row)
        else:
            break

    if not train_rows or not val_rows:
        raise RuntimeError(
            f"Failed to build train/val data from {TRAIN_REPO}. "
            "Please check BIGCODEBENCH_* env configuration."
        )

    _write_jsonl(train_dir / "train.jsonl", train_rows)
    _write_jsonl(train_dir / "val.jsonl", val_rows)

    return {
        "accepted": accepted,
        "skipped": skipped,
        "train_count": len(train_rows),
        "val_count": len(val_rows),
    }


def _prepare_test_data(test_dir: Path) -> dict[str, int]:
    dataset = load_dataset(TEST_REPO, split=TEST_SPLIT, streaming=True)

    prompts: list[dict[str, str]] = []
    cases: list[dict[str, str]] = []
    skipped = 0

    for idx, raw in enumerate(dataset):
        row = _normalize_test_record(raw, idx)
        if row is None:
            skipped += 1
            continue
        prompt_entry = {
            "id": row["id"],
            "prompt": row["prompt"],
            "starter_code": row["starter_code"],
            "entry_point": row["entry_point"],
        }
        case_entry = {
            "id": row["id"],
            "test_code": row["test_code"],
            "starter_code": row["starter_code"],
            "entry_point": row["entry_point"],
        }
        prompts.append(prompt_entry)
        cases.append(case_entry)

        if MAX_TEST_SAMPLES > 0 and len(cases) >= MAX_TEST_SAMPLES:
            break

    if not cases:
        raise RuntimeError(
            "No valid BigCodeBench test cases were built. "
            "Please verify BIGCODEBENCH_TEST_REPO and split configuration."
        )

    _write_jsonl(test_dir / "test_prompts.jsonl", prompts)
    _write_jsonl(test_dir / "test_cases.jsonl", cases)

    return {
        "test_count": len(cases),
        "skipped": skipped,
    }


def _is_already_prepared(train_dir: Path, test_dir: Path) -> bool:
    """Check if data already exists and is non-empty (idempotency)."""
    required = [
        train_dir / "train.jsonl",
        train_dir / "val.jsonl",
        test_dir / "test_prompts.jsonl",
        test_dir / "test_cases.jsonl",
    ]
    min_size = 100  # bytes — guard against corrupt empty files
    return all(p.exists() and p.stat().st_size > min_size for p in required)


def main() -> None:
    train_dir = Path(os.environ["FARBENCH_DATA_DIR"])
    test_dir = Path(os.environ["FARBENCH_TEST_DATA_DIR"])

    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    if _is_already_prepared(train_dir, test_dir):
        print("[prepare] Data already prepared, skipping.")
        return

    train_stats = _prepare_train_data(train_dir)
    test_stats = _prepare_test_data(test_dir)

    info = {
        "task": "bigcodebench_codegen",
        "train_repo": TRAIN_REPO,
        "train_split": TRAIN_SPLIT,
        "test_repo": TEST_REPO,
        "test_split": TEST_SPLIT,
        "limits": {
            "max_train_samples": MAX_TRAIN_SAMPLES,
            "max_val_samples": MAX_VAL_SAMPLES,
            "max_test_samples": MAX_TEST_SAMPLES,
        },
        "train_stats": train_stats,
        "test_stats": test_stats,
    }
    info_path = train_dir / "dataset_info.json"
    info_path.write_text(
        json.dumps(info, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
