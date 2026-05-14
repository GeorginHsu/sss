"""AIME Math RL data preparation.

Output layout:
    FARBENCH_DATA_DIR/
        train.jsonl          — normalized Skywork-OR1 math problems (~100K)
        val.jsonl            — 5% deterministic holdout (~5K)
        sample_train.jsonl   — 256-row debug subset
        dataset_info.json    — sources + schema

    FARBENCH_TEST_DATA_DIR/
        test.jsonl           — AIME 2024 + 2025 problems (60 rows total)
        test_answers.json    — {id: answer}, evaluator-only
        dataset_info.json

Data sources:
    Training: Skywork/Skywork-OR1-RL-Data (split='math', 105K rows)
              De-contaminated against AIME 2024 / 2025 by the authors.
    Test:     Maxwell-Jia/AIME_2024 (30 problems)
              math-ai/aime25        (30 problems)

NOTE on the base model:
    Qwen/Qwen3-4B is NOT staged here. It is pre-downloaded into the system
    HF cache at Docker build time (see benchmarks/aime_math_rl/docker/
    Dockerfile, the trailing "Pre-download Qwen/Qwen3-4B" RUN). Agents access
    it via the canonical HF id `Qwen/Qwen3-4B` from both the training env
    (`python`) and the inference venv (`vllm-python`); no task-data path is
    involved. This keeps prepare.py focused on data only and avoids the
    "FileNotFoundError: Could not locate local Qwen3-4B under /data" class
    of mistakes that arises when agents have to remember a magic absolute
    snapshot path.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from typing import Any

from datasets import get_dataset_split_names, load_dataset


TRAIN_REPO = "Skywork/Skywork-OR1-RL-Data"
TRAIN_SPLIT = "math"
VAL_RATIO = 0.05
SAMPLE_N = 256

AIME24_REPO = "Maxwell-Jia/AIME_2024"
AIME25_REPO = "math-ai/aime25"


# ── Helpers ──────────────────────────────────────────────────────────────────

def _first(d: dict[str, Any], *keys: str) -> Any:
    """Return the first value present (and truthy) for any of `keys` in `d`."""
    for k in keys:
        v = d.get(k)
        if v not in (None, ""):
            return v
    return None


def _parse_skywork_ground_truth(raw: Any) -> str:
    """Skywork packs the gold answer as a JSON-stringified list, e.g. '["15625"]'."""
    if raw is None:
        return ""
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
        if isinstance(parsed, list) and parsed:
            return str(parsed[0]).strip()
    except (ValueError, TypeError):
        pass
    return str(raw).strip()


def _parse_skywork_prompt(prompt: Any) -> str:
    """Skywork stores the question in prompt[0].content (chat format)."""
    if isinstance(prompt, list) and prompt:
        first = prompt[0]
        if isinstance(first, dict):
            return str(first.get("content", "")).strip()
    if isinstance(prompt, str):
        return prompt.strip()
    return ""


def _skywork_difficulty_for_qwen3_4b(row: dict) -> int | None:
    """Return the Skywork 7B-proxy difficulty (closest to our 4B base model).

    Skywork keys look like 'DeepSeek-R1-Distill-Qwen-{1.5B,7B,32B}'; we match
    '-7B' (with the hyphen to avoid grabbing '17B' or '57B' if upstream grows).
    Values are ints in [0, 16]; None when unavailable.
    """
    extra = row.get("extra_info") or {}
    md = extra.get("model_difficulty") if isinstance(extra, dict) else None
    if not isinstance(md, dict):
        return None
    for k, v in md.items():
        if "-7B" in k:
            try:
                return int(v)
            except (TypeError, ValueError):
                return None
    return None


def _load_first_available_split(repo: str):
    """Load a dataset, auto-picking the first available split name."""
    splits = get_dataset_split_names(repo)
    if not splits:
        raise RuntimeError(f"{repo}: no splits available")
    return load_dataset(repo, split=splits[0])


# ── Training set ─────────────────────────────────────────────────────────────

def _prepare_train(data_dir: str) -> dict[str, Any]:
    print(f"[prepare] train <- {TRAIN_REPO} (split={TRAIN_SPLIT})", flush=True)
    ds = load_dataset(TRAIN_REPO, split=TRAIN_SPLIT)
    print(f"[prepare]   rows={len(ds)}", flush=True)

    train_path = os.path.join(data_dir, "train.jsonl")
    val_path = os.path.join(data_dir, "val.jsonl")
    sample_path = os.path.join(data_dir, "sample_train.jsonl")

    source_counter: dict[str, int] = {}
    n_train = n_val = n_skip = n_sample = 0

    with open(train_path, "w", encoding="utf-8") as train_f, \
         open(val_path, "w", encoding="utf-8") as val_f, \
         open(sample_path, "w", encoding="utf-8") as sample_f:

        for i, row in enumerate(ds):
            problem = _parse_skywork_prompt(row.get("prompt"))
            reward_model = row.get("reward_model") or {}
            raw_gt = reward_model.get("ground_truth") if isinstance(reward_model, dict) else None
            answer = _parse_skywork_ground_truth(raw_gt)

            if not problem or not answer:
                n_skip += 1
                continue

            data_source = str(row.get("data_source", "") or "")
            source_counter[data_source] = source_counter.get(data_source, 0) + 1

            out = {
                "id": f"skywork-math-{i:09d}",
                "problem": problem,
                "expected_answer": answer,
                "data_source": data_source,
                "messages": row.get("prompt") or [],
                "difficulty": _skywork_difficulty_for_qwen3_4b(row),
                "raw_ground_truth": "" if raw_gt is None else str(raw_gt),
            }
            line = json.dumps(out, ensure_ascii=False) + "\n"

            h = hashlib.sha1(problem.encode("utf-8", "ignore")).hexdigest()
            bucket = int(h[:8], 16) / 0xFFFFFFFF
            if bucket < VAL_RATIO:
                val_f.write(line)
                n_val += 1
            else:
                train_f.write(line)
                n_train += 1
                if n_sample < SAMPLE_N:
                    sample_f.write(line)
                    n_sample += 1

    info = {
        "train_dataset_repo": TRAIN_REPO,
        "train_split": TRAIN_SPLIT,
        "train_rows": n_train,
        "val_rows": n_val,
        "sample_rows": n_sample,
        "skipped_rows": n_skip,
        "val_ratio": VAL_RATIO,
        "split_policy": "sha1(problem) bucketed",
        "data_source_counts": source_counter,
        "schema": [
            "id", "problem", "expected_answer", "data_source", "messages",
            "difficulty", "raw_ground_truth",
        ],
        "difficulty_source": "Skywork model_difficulty @ DeepSeek-R1-Distill-Qwen-7B (proxy for Qwen3-4B)",
    }
    print(f"[prepare]   train={n_train}  val={n_val}  sample={n_sample}  skipped={n_skip}")
    return info


# ── Test set (AIME 2024 + 2025) ──────────────────────────────────────────────

def _prepare_test(test_dir: str) -> dict[str, Any]:
    test_path = os.path.join(test_dir, "test.jsonl")
    answers_path = os.path.join(test_dir, "test_answers.json")

    answers: dict[str, str] = {}
    per_year: dict[str, int] = {"2024": 0, "2025": 0}

    with open(test_path, "w", encoding="utf-8") as test_f:
        # AIME 2024
        print(f"[prepare] test <- {AIME24_REPO}", flush=True)
        ds24 = _load_first_available_split(AIME24_REPO)
        print(f"[prepare]   rows={len(ds24)}  cols={ds24.column_names}", flush=True)
        for i, row in enumerate(ds24):
            raw_id = str(_first(row, "ID", "id", "problem_id") or f"{i+1:02d}").strip()
            slug = raw_id.replace(" ", "_")
            sid = slug if slug.lower().startswith("aime24") else f"aime24-{slug}"
            problem = str(_first(row, "Problem", "problem", "question") or "").strip()
            answer = str(_first(row, "Answer", "answer", "ground_truth") or "").strip()
            if not problem or not answer:
                continue
            test_f.write(json.dumps({"id": sid, "problem": problem, "year": 2024},
                                    ensure_ascii=False) + "\n")
            answers[sid] = answer
            per_year["2024"] += 1

        # AIME 2025
        print(f"[prepare] test <- {AIME25_REPO}", flush=True)
        ds25 = _load_first_available_split(AIME25_REPO)
        print(f"[prepare]   rows={len(ds25)}  cols={ds25.column_names}", flush=True)
        for i, row in enumerate(ds25):
            raw_id = str(_first(row, "id", "ID", "problem_id") or f"{i+1:02d}").strip()
            sid = raw_id if raw_id.lower().startswith("aime25") else f"aime25-{raw_id}"
            problem = str(_first(row, "problem", "Problem", "question") or "").strip()
            answer = str(_first(row, "answer", "Answer", "ground_truth") or "").strip()
            if not problem or not answer:
                continue
            test_f.write(json.dumps({"id": sid, "problem": problem, "year": 2025},
                                    ensure_ascii=False) + "\n")
            answers[sid] = answer
            per_year["2025"] += 1

    with open(answers_path, "w", encoding="utf-8") as f:
        json.dump(answers, f, indent=2, ensure_ascii=False)

    total = sum(per_year.values())
    if total < 50:
        raise RuntimeError(
            f"[prepare] expected ~60 AIME problems, got {total} "
            f"(aime24={per_year['2024']}, aime25={per_year['2025']})"
        )

    info = {
        "aime24_repo": AIME24_REPO,
        "aime25_repo": AIME25_REPO,
        "num_per_year": per_year,
        "total": total,
        "schema_test": ["id", "problem", "year"],
        "schema_answers": "dict[id -> answer]",
    }
    print(f"[prepare]   aime24={per_year['2024']}  aime25={per_year['2025']}  total={total}")
    return info


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    data_dir = os.environ["FARBENCH_DATA_DIR"]
    test_dir = os.environ["FARBENCH_TEST_DATA_DIR"]
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Each phase is independently idempotent (checks its own outputs), so we
    # don't short-circuit here — it's safe to re-run prepare after partial
    # failure. The base model (Qwen/Qwen3-4B) is intentionally not handled
    # here; it is pre-cached at Docker build time and resolved by the agent
    # via the canonical HF id (see module docstring + Dockerfile).

    train_path = os.path.join(data_dir, "train.jsonl")
    if os.path.exists(train_path) and os.path.getsize(train_path) > 1_000_000:
        print(f"[prepare] {train_path} already exists, skipping train build")
        with open(os.path.join(data_dir, "dataset_info.json"), "r", encoding="utf-8") as f:
            train_info = json.load(f)
    else:
        train_info = _prepare_train(data_dir)

    test_path = os.path.join(test_dir, "test.jsonl")
    if os.path.exists(test_path) and os.path.getsize(test_path) > 1000:
        print(f"[prepare] {test_path} already exists, skipping test build")
        with open(os.path.join(test_dir, "dataset_info.json"), "r", encoding="utf-8") as f:
            test_info = json.load(f)
    else:
        test_info = _prepare_test(test_dir)

    # Record the base model as an HF id for downstream tooling / debugging,
    # but make explicit it does not live in FARBENCH_DATA_DIR.
    train_info["base_model"] = {
        "source": "hf://Qwen/Qwen3-4B",
        "location": "image HF cache (/root/.cache/huggingface/hub/)",
        "loader": "AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-4B')",
    }
    with open(os.path.join(data_dir, "dataset_info.json"), "w", encoding="utf-8") as f:
        json.dump(train_info, f, indent=2)
    with open(os.path.join(test_dir, "dataset_info.json"), "w", encoding="utf-8") as f:
        json.dump(test_info, f, indent=2)

    print(f"[prepare] data_dir       = {data_dir}")
    print(f"[prepare] test_data_dir  = {test_dir}")
    print("[prepare] AIME Math RL ready.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[prepare] ERROR: {exc}", file=sys.stderr)
        raise
