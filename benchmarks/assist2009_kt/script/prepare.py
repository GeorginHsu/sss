"""Knowledge Tracing data preparation: download ASSISTments 2009, split, serialize.

Split strategy: 70/15/15 student-level random split with seed=0.
  Literature standard (pyKT, AKT, simpleKT, UKT AAAI 2025) uses 5-fold student-level
  cross-validation. FARBench uses a single fixed split for deterministic single-run
  evaluation, so reported AUC numbers are NOT directly comparable to literature 5-fold
  CV values. This deviation is documented in tasks.md.
  Preprocessing follows theophilee/learner-performance-prediction (order_id ordering,
  skill remapping) — different from pyKT standard preprocessing.

Data source: ASSISTments 2009-2010 Skill-Builder dataset.
Reference: https://sites.google.com/site/assistmentsdata/datasets/2009-2010-assistment-data

Output layout:
    FARBENCH_DATA_DIR/
        train.pt   — student interaction sequences for training  (70% of students)
        val.pt     — student interaction sequences for validation (15% of students)
    FARBENCH_TEST_DATA_DIR/
        test.pt    — student interaction sequences for evaluation (15% of students)

Agent can only access FARBENCH_DATA_DIR.  Test data is isolated in FARBENCH_TEST_DATA_DIR
and never exposed to the agent — it is only used by the system-side evaluator.
"""

import csv
import os
import shutil
import urllib.error
import urllib.request
from collections import defaultdict

import numpy as np

# ── Download sources ──

DATA_URLS = [
    # GitHub mirrors of the ASSISTments 2009 processed data (tab-separated)
    "https://raw.githubusercontent.com/theophilee/learner-performance-prediction/master/data/assistments09/preprocessed_data.csv",
]

VAL_RATIO = 0.15
TEST_RATIO = 0.15
MIN_SEQ_LEN = 3      # skip students with fewer interactions
MAX_SEQ_LEN = 200     # truncate longer sequences
SPLIT_SEED = 0        # deterministic split


def download_file(url: str, dest: str) -> None:
    print(f"  Downloading {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "FARBench/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        with open(dest, "wb") as f:
            shutil.copyfileobj(resp, f)


def _detect_columns(header: list[str]) -> tuple[str, str, str, str | None]:
    """Detect user, skill, correct, and optional order columns from CSV header."""
    lower = [h.strip().lower() for h in header]

    user_col = None
    for candidate in ["user_id", "student_id"]:
        if candidate in lower:
            user_col = header[lower.index(candidate)]
            break

    skill_col = None
    for candidate in ["skill_id", "skill", "skill_name", "problem_id"]:
        if candidate in lower:
            skill_col = header[lower.index(candidate)]
            break

    correct_col = None
    for candidate in ["correct", "is_correct"]:
        if candidate in lower:
            correct_col = header[lower.index(candidate)]
            break

    order_col = None
    for candidate in ["order_id", "row_number", "timestamp", "log_id"]:
        if candidate in lower:
            order_col = header[lower.index(candidate)]
            break

    if not user_col or not skill_col or not correct_col:
        raise ValueError(
            f"Cannot detect required columns (user, skill, correct) from header: {header}\n"
            f"Detected: user={user_col}, skill={skill_col}, correct={correct_col}"
        )
    return user_col, skill_col, correct_col, order_col


def _parse_csv(csv_path: str) -> tuple[dict, int]:
    """Parse CSV into per-student interaction sequences.

    Returns:
        sequences: dict mapping user_id -> list of (skill_idx, correct)
        num_skills: number of unique skills
    """
    # Read CSV (auto-detect delimiter: tab or comma)
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        sample = f.read(4096)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters="\t,")
        reader = csv.DictReader(f, dialect=dialect)
        header = reader.fieldnames
        if not header:
            raise ValueError("CSV has no header row")

        user_col, skill_col, correct_col, order_col = _detect_columns(header)
        print(f"  Columns: user={user_col}, skill={skill_col}, correct={correct_col}, order={order_col}")

        raw_interactions: dict[str, list] = defaultdict(list)
        skill_set: set[str] = set()

        for row in reader:
            user = row.get(user_col, "").strip()
            skill = row.get(skill_col, "").strip()
            correct_str = row.get(correct_col, "").strip()
            order_str = row.get(order_col, "0").strip() if order_col else "0"

            # Skip rows with missing values
            if not user or not skill or not correct_str:
                continue

            # Parse correct as int (handle "1.0" etc.)
            try:
                correct = int(float(correct_str))
            except (ValueError, TypeError):
                continue
            if correct not in (0, 1):
                continue

            try:
                order = int(float(order_str)) if order_str else 0
            except (ValueError, TypeError):
                order = 0

            skill_set.add(skill)
            raw_interactions[user].append((order, skill, correct))

    # Map skills to contiguous integers
    skill_to_idx = {s: i for i, s in enumerate(sorted(skill_set))}
    num_skills = len(skill_to_idx)
    print(f"  Found {len(raw_interactions)} students, {num_skills} skills")

    # Build sequences: sort by order within each student
    sequences = {}
    for user, interactions in raw_interactions.items():
        interactions.sort(key=lambda x: x[0])  # sort by order
        seq = [(skill_to_idx[skill], correct) for _, skill, correct in interactions]

        if len(seq) < MIN_SEQ_LEN:
            continue

        # Truncate to last MAX_SEQ_LEN interactions
        if len(seq) > MAX_SEQ_LEN:
            seq = seq[-MAX_SEQ_LEN:]

        sequences[user] = seq

    print(f"  After filtering (min_len={MIN_SEQ_LEN}): {len(sequences)} students")
    return sequences, num_skills


def _split(sequences: dict) -> tuple[list, list, list]:
    """Split students into train/val/test sets (deterministic)."""
    users = sorted(sequences.keys())
    rng = np.random.RandomState(SPLIT_SEED)
    rng.shuffle(users)

    n = len(users)
    n_test = int(n * TEST_RATIO)
    n_val = int(n * VAL_RATIO)

    test_users = users[:n_test]
    val_users = users[n_test:n_test + n_val]
    train_users = users[n_test + n_val:]

    train_seqs = [sequences[u] for u in train_users]
    val_seqs = [sequences[u] for u in val_users]
    test_seqs = [sequences[u] for u in test_users]

    print(f"  Split: train={len(train_seqs)}, val={len(val_seqs)}, test={len(test_seqs)}")
    return train_seqs, val_seqs, test_seqs


def _save(sequences: list, num_skills: int, path: str, max_seq_len: int) -> None:
    """Pad sequences and save as a .pt file.

    All splits are padded to max_seq_len so that tensor shapes are consistent
    across train/val/test. This avoids shape mismatches when the agent trains
    on train.pt and runs predict.py on test.pt.
    """
    import torch

    n = len(sequences)
    max_len = max_seq_len

    skill_ids = torch.zeros(n, max_len, dtype=torch.long)
    corrects = torch.zeros(n, max_len, dtype=torch.long)
    lengths = torch.zeros(n, dtype=torch.long)

    for i, seq in enumerate(sequences):
        seq_len = min(len(seq), max_len)
        lengths[i] = seq_len
        for j in range(seq_len):
            skill_ids[i, j] = seq[j][0]
            corrects[i, j] = seq[j][1]

    torch.save({
        "skill_ids": skill_ids,
        "corrects": corrects,
        "lengths": lengths,
        "num_skills": num_skills,
    }, path)
    print(f"  Saved {path}  ({n} students, max_len={max_len})")


def main():
    data_dir = os.environ["FARBENCH_DATA_DIR"]
    test_data_dir = os.environ["FARBENCH_TEST_DATA_DIR"]
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(test_data_dir, exist_ok=True)

    # Skip if already prepared (check existence AND minimum file size to
    # guard against leftover empty/corrupt files from a failed prior run)
    MIN_PT_BYTES = 1024
    required = [
        os.path.join(data_dir, "train.pt"),
        os.path.join(data_dir, "val.pt"),
        os.path.join(test_data_dir, "test.pt"),
    ]
    if all(
        os.path.exists(p) and os.path.getsize(p) > MIN_PT_BYTES
        for p in required
    ):
        print("Knowledge Tracing data already prepared, skipping.")
        return

    # Download raw data
    raw_dir = os.path.join(data_dir, "_raw")
    os.makedirs(raw_dir, exist_ok=True)
    csv_path = os.path.join(raw_dir, "assistments2009.csv")

    if not os.path.exists(csv_path):
        downloaded = False
        for url in DATA_URLS:
            try:
                download_file(url, csv_path)
                downloaded = True
                break
            except Exception as e:
                print(f"  Failed: {e}")
                continue

        if not downloaded:
            raise RuntimeError(
                "Failed to download ASSISTments 2009 data.\n"
                "Please download manually:\n"
                "  1. Visit: https://sites.google.com/site/assistmentsdata/datasets/2009-2010-assistment-data\n"
                "  2. Download 'skill_builder_data_corrected_collapsed.csv'\n"
                f"  3. Place it at: {csv_path}\n"
                "  4. Re-run: farbench tasks prepare assist2009_kt"
            )

    # Parse CSV
    print("Parsing ASSISTments 2009 data...")
    sequences, num_skills = _parse_csv(csv_path)

    # Split students into train / val / test
    train_seqs, val_seqs, test_seqs = _split(sequences)

    # Serialize
    _save(train_seqs, num_skills, os.path.join(data_dir, "train.pt"), MAX_SEQ_LEN)
    _save(val_seqs, num_skills, os.path.join(data_dir, "val.pt"), MAX_SEQ_LEN)
    _save(test_seqs, num_skills, os.path.join(test_data_dir, "test.pt"), MAX_SEQ_LEN)

    # Clean up raw files
    shutil.rmtree(raw_dir)

    print(f"\nKnowledge Tracing data ready:")
    print(f"  Train/val: {data_dir}")
    print(f"  Test:      {test_data_dir}")


if __name__ == "__main__":
    main()
