"""ETTh1 forecasting data preparation: download, split, build test windows.

Output layout:
    FARBENCH_DATA_DIR/
        train.csv       — 8640 rows x 7 features (12 months hourly)
        val.csv         — 2880 rows x 7 features (4 months hourly)
    FARBENCH_TEST_DATA_DIR/
        test_windows.npy  — float32 [91, 96, 7] input windows (agent reads)
        test_labels.npy   — float32 [91, 720, 7] ground truth (evaluator only)
        norm_stats.json   — {"mean": [...], "std": [...]} from training set

Data source: ETTh1 from ETDataset (Zhou et al., AAAI 2021 — Informer).
Standard split: 12/4/4 months → borders [0, 8640, 11520, 14400].
Test windows: seq_len=96, pred_len=720, stride=1 → 2161 windows.
  Matches Time-Series-Library standard evaluation protocol.
"""

import csv
import json
import os
import shutil
import urllib.request

import numpy as np


DATA_URLS = [
    "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
    "https://huggingface.co/datasets/ETDataset/ett/resolve/main/ETT-small/ETTh1.csv",
]

FEATURE_COLS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]

# Standard borders (12 / 4 / 4 months, hourly)
BORDER_TRAIN = 12 * 30 * 24        # 8640
BORDER_VAL = BORDER_TRAIN + 4 * 30 * 24  # 11520
BORDER_TEST = BORDER_VAL + 4 * 30 * 24   # 14400

SEQ_LEN = 96
PRED_LEN = 720
STRIDE = 1  # stride=1 matches Time-Series-Library standard evaluation protocol


def download_file(url: str, dest: str) -> None:
    print(f"  Downloading {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "FARBench/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        with open(dest, "wb") as f:
            shutil.copyfileobj(resp, f)


def main():
    data_dir = os.environ["FARBENCH_DATA_DIR"]
    test_data_dir = os.environ["FARBENCH_TEST_DATA_DIR"]
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(test_data_dir, exist_ok=True)

    required = [
        os.path.join(data_dir, "train.csv"),
        os.path.join(data_dir, "val.csv"),
        os.path.join(test_data_dir, "test_windows.npy"),
        os.path.join(test_data_dir, "test_labels.npy"),
        os.path.join(test_data_dir, "norm_stats.json"),
    ]
    if all(os.path.exists(p) and os.path.getsize(p) > 100 for p in required):
        print("ETTh1 data already prepared, skipping.")
        return

    # Download
    raw_dir = os.path.join(data_dir, "_raw")
    os.makedirs(raw_dir, exist_ok=True)
    csv_path = os.path.join(raw_dir, "ETTh1.csv")

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
                "Failed to download ETTh1 dataset.\n"
                "Please download manually from:\n"
                "  https://github.com/zhouhaoyi/ETDataset\n"
                f"  Place ETTh1.csv at: {csv_path}"
            )

    # Parse CSV
    print("Parsing ETTh1 dataset...")
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vals = [float(row[c]) for c in FEATURE_COLS]
            rows.append(vals)
    data = np.array(rows, dtype=np.float32)
    print(f"  Loaded {data.shape[0]} rows x {data.shape[1]} features")

    if data.shape[0] < BORDER_TEST:
        raise ValueError(
            f"Expected at least {BORDER_TEST} rows, got {data.shape[0]}"
        )

    # Split
    train_data = data[:BORDER_TRAIN]           # [0, 8640)
    val_data = data[BORDER_TRAIN:BORDER_VAL]   # [8640, 11520)
    test_region = data[BORDER_VAL - SEQ_LEN:BORDER_TEST]  # [11424, 14400) = 2976 rows

    print(f"  Train: {train_data.shape[0]} rows")
    print(f"  Val:   {val_data.shape[0]} rows")
    print(f"  Test region (with lookback): {test_region.shape[0]} rows")

    # Save train.csv, val.csv
    def write_csv(path, arr):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(FEATURE_COLS)
            for row in arr:
                writer.writerow([f"{v:.6f}" for v in row])
        print(f"  Saved {path}  ({arr.shape[0]} rows)")

    write_csv(os.path.join(data_dir, "train.csv"), train_data)
    write_csv(os.path.join(data_dir, "val.csv"), val_data)

    # Compute normalization stats from training set
    train_mean = train_data.mean(axis=0).tolist()
    train_std = train_data.std(axis=0).tolist()

    norm_stats = {"mean": train_mean, "std": train_std}
    norm_path = os.path.join(test_data_dir, "norm_stats.json")
    with open(norm_path, "w") as f:
        json.dump(norm_stats, f, indent=2)
    print(f"  Saved {norm_path}")

    # Build test windows: seq_len=96, pred_len=720, stride=24
    windows_in = []
    windows_out = []
    total_len = test_region.shape[0]  # 2976

    for start in range(0, total_len - SEQ_LEN - PRED_LEN + 1, STRIDE):
        inp = test_region[start:start + SEQ_LEN]            # [96, 7]
        tgt = test_region[start + SEQ_LEN:start + SEQ_LEN + PRED_LEN]  # [720, 7]
        windows_in.append(inp)
        windows_out.append(tgt)

    windows_in = np.array(windows_in, dtype=np.float32)   # [N, 96, 7]
    windows_out = np.array(windows_out, dtype=np.float32)  # [N, 720, 7]

    print(f"  Test windows: {windows_in.shape[0]} windows")
    print(f"    Input shape:  {windows_in.shape}")
    print(f"    Target shape: {windows_out.shape}")

    np.save(os.path.join(test_data_dir, "test_windows.npy"), windows_in)
    np.save(os.path.join(test_data_dir, "test_labels.npy"), windows_out)
    print(f"  Saved test_windows.npy and test_labels.npy")

    # Clean up
    shutil.rmtree(raw_dir)

    print(f"\nETTh1 data ready:")
    print(f"  Train/val: {data_dir}")
    print(f"  Test:      {test_data_dir}")


if __name__ == "__main__":
    main()
