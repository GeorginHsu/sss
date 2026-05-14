"""FLIP AAV protein fitness prediction data preparation: download, split, serialize.

Split strategy: Official FLIP 'two_vs_many' split (Dallago et al., NeurIPS 2021).
  Train: sequences with ≤2 mutations from wild type (~24,835 after val holdout)
  Val: 20% of training set, stratified by mutation count (~6,209)
  Test: sequences with >2 mutations (~44,195)
  This tests extrapolation from low-order to higher-order combinatorial mutants.

Output layout:
    FARBENCH_DATA_DIR/
        train.csv          — sequence, target columns (~24,835 rows)
        val.csv            — sequence, target columns (~6,209 rows)
    FARBENCH_TEST_DATA_DIR/
        test.csv           — sequence column only (~44,195 rows)
        test_labels.csv    — target column (evaluator only)
"""

import csv
import os
import shutil
import urllib.request
import zipfile

import numpy as np

# FLIP GitHub repo — splits.zip contains all split CSVs for AAV
SPLITS_ZIP_URL = (
    "https://github.com/J-SNACKKB/FLIP/raw/main/splits/aav/splits.zip"
)
SPLIT_NAME = "two_vs_many"

VAL_RATIO = 0.2
SPLIT_SEED = 42
MIN_FILE_BYTES = 100


def download_file(url: str, dest: str) -> None:
    print(f"  Downloading {url[:80]}...")
    req = urllib.request.Request(url, headers={"User-Agent": "FARBench/1.0"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        with open(dest, "wb") as f:
            shutil.copyfileobj(resp, f)


def main():
    data_dir = os.environ["FARBENCH_DATA_DIR"]
    test_data_dir = os.environ["FARBENCH_TEST_DATA_DIR"]
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(test_data_dir, exist_ok=True)

    # Check if already prepared
    required = [
        os.path.join(data_dir, "train.csv"),
        os.path.join(data_dir, "val.csv"),
        os.path.join(test_data_dir, "test.csv"),
        os.path.join(test_data_dir, "test_labels.csv"),
    ]
    if all(os.path.exists(p) and os.path.getsize(p) > MIN_FILE_BYTES for p in required):
        print("FLIP AAV data already prepared, skipping.")
        return

    # ── Download ──
    raw_dir = os.path.join(data_dir, "_raw")
    os.makedirs(raw_dir, exist_ok=True)
    zip_path = os.path.join(raw_dir, "splits.zip")
    csv_path = os.path.join(raw_dir, f"{SPLIT_NAME}.csv")

    if not os.path.exists(csv_path):
        if not os.path.exists(zip_path) or os.path.getsize(zip_path) < MIN_FILE_BYTES:
            try:
                download_file(SPLITS_ZIP_URL, zip_path)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download FLIP AAV splits: {e}\n"
                    "Please download manually from:\n"
                    f"  {SPLITS_ZIP_URL}\n"
                    f"  Place splits.zip at: {zip_path}"
                )

        # Extract the specific split CSV from the zip
        print(f"  Extracting {SPLIT_NAME}.csv from splits.zip...")
        with zipfile.ZipFile(zip_path, "r") as z:
            # List contents to find the right file
            names = z.namelist()
            target_name = None
            for name in names:
                if SPLIT_NAME in name and name.endswith(".csv"):
                    target_name = name
                    break
            if target_name is None:
                raise RuntimeError(
                    f"Could not find {SPLIT_NAME}.csv in splits.zip. "
                    f"Available files: {names}"
                )
            with z.open(target_name) as src, open(csv_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
        print(f"  Extracted: {target_name}")

    # ── Parse CSV ──
    print("  Parsing FLIP AAV two_vs_many split...")
    train_seqs = []
    train_targets = []
    train_validations = []
    test_seqs = []
    test_targets = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seq = row["sequence"]
            target = float(row["target"])
            split = row["set"]

            if split == "train":
                train_seqs.append(seq)
                train_targets.append(target)
                is_val = row.get("validation", "False").strip().lower() == "true"
                train_validations.append(is_val)
            elif split == "test":
                test_seqs.append(seq)
                test_targets.append(target)

    total_train = len(train_seqs)
    total_test = len(test_seqs)
    print(f"  Raw counts: train={total_train}, test={total_test}")

    # ── Create train/val split ──
    # Use FLIP's own validation flag if available, otherwise random split
    val_from_flip = sum(train_validations)
    if val_from_flip > 0:
        print(f"  Using FLIP validation flags: {val_from_flip} val samples")
        final_train_seqs = []
        final_train_targets = []
        val_seqs = []
        val_targets = []
        for i in range(total_train):
            if train_validations[i]:
                val_seqs.append(train_seqs[i])
                val_targets.append(train_targets[i])
            else:
                final_train_seqs.append(train_seqs[i])
                final_train_targets.append(train_targets[i])
    else:
        # Random split with fixed seed
        print(f"  Creating random val split: {VAL_RATIO*100:.0f}% of train")
        rng = np.random.RandomState(SPLIT_SEED)
        indices = rng.permutation(total_train)
        val_size = int(total_train * VAL_RATIO)
        val_idx = set(indices[:val_size].tolist())

        final_train_seqs = []
        final_train_targets = []
        val_seqs = []
        val_targets = []
        for i in range(total_train):
            if i in val_idx:
                val_seqs.append(train_seqs[i])
                val_targets.append(train_targets[i])
            else:
                final_train_seqs.append(train_seqs[i])
                final_train_targets.append(train_targets[i])

    print(f"  Final: train={len(final_train_seqs)}, val={len(val_seqs)}, test={total_test}")

    # ── Save CSVs ──
    def write_csv(path, seqs, targets):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["sequence", "target"])
            for s, t in zip(seqs, targets):
                writer.writerow([s, f"{t:.6f}"])
        print(f"  Saved {path} ({len(seqs)} rows)")

    write_csv(os.path.join(data_dir, "train.csv"), final_train_seqs, final_train_targets)
    write_csv(os.path.join(data_dir, "val.csv"), val_seqs, val_targets)

    # Test: sequences only (no targets visible to agent)
    test_csv_path = os.path.join(test_data_dir, "test.csv")
    with open(test_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sequence"])
        for s in test_seqs:
            writer.writerow([s])
    print(f"  Saved {test_csv_path} ({len(test_seqs)} rows, sequences only)")

    # Test labels (evaluator only)
    labels_path = os.path.join(test_data_dir, "test_labels.csv")
    with open(labels_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["target"])
        for t in test_targets:
            writer.writerow([f"{t:.6f}"])
    print(f"  Saved {labels_path} ({len(test_targets)} rows)")

    # Clean up
    shutil.rmtree(raw_dir)

    # Summary statistics
    all_targets = final_train_targets + val_targets + test_targets
    print(f"\nFLIP AAV data ready:")
    print(f"  Train: {len(final_train_seqs)}, Val: {len(val_seqs)}, Test: {total_test}")
    print(f"  Sequence length: ~{len(final_train_seqs[0])} AA")
    print(f"  Target range: [{min(all_targets):.3f}, {max(all_targets):.3f}]")
    print(f"  Target mean: {np.mean(all_targets):.3f}, std: {np.std(all_targets):.3f}")


if __name__ == "__main__":
    main()
