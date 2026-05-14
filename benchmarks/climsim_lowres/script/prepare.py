"""ClimSim low-res data preparation: download pre-processed quickstart npy files.

Data source: LEAP/subsampled_low_res on HuggingFace
  Pre-normalized, subsampled (every 7th timestep) low-resolution (384-column)
  E3SM-MMF output. This is the exact dataset used for training and evaluation
  in the NeurIPS 2023 ClimSim paper (Yu et al., 2023).

Split strategy: Official ClimSim splits (no re-splitting needed):
  - train: years 1-7 + Jan year 8, stride=7  (~10.1M samples)
  - val: Feb year 8 - Jan year 9, stride=7   (~1.44M samples)
  - scoring (test): same period as val, stride=6  (~1.68M samples)

Output layout:
    FARBENCH_DATA_DIR/
        train_input.npy   — float32 [~10.1M, 124] pre-normalized inputs
        train_target.npy  — float32 [~10.1M, 128] pre-normalized targets
        val_input.npy     — float32 [~1.44M, 124]
        val_target.npy    — float32 [~1.44M, 128]
    FARBENCH_TEST_DATA_DIR/
        scoring_input.npy  — float32 [~1.68M, 124] (agent sees this)
        scoring_target.npy — float32 [~1.68M, 128] (evaluator only)
"""

from __future__ import annotations

import os
import shutil

from huggingface_hub import hf_hub_download

REPO_ID = "LEAP/subsampled_low_res"
REPO_TYPE = "dataset"

TRAIN_FILES = ["train_input.npy", "train_target.npy"]
VAL_FILES = ["val_input.npy", "val_target.npy"]
TEST_FILES = ["scoring_input.npy", "scoring_target.npy"]

MIN_SIZES = {
    "train_input.npy": 4_000_000_000,
    "train_target.npy": 4_000_000_000,
    "val_input.npy": 500_000_000,
    "val_target.npy": 500_000_000,
    "scoring_input.npy": 500_000_000,
    "scoring_target.npy": 500_000_000,
}


def download_file(filename: str, dest_dir: str) -> str:
    dest_path = os.path.join(dest_dir, filename)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > MIN_SIZES.get(filename, 0):
        print(f"  {filename} already exists ({os.path.getsize(dest_path) / 1e9:.2f} GB), skipping.")
        return dest_path

    print(f"  Downloading {filename} from {REPO_ID} ...")
    cached_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        repo_type=REPO_TYPE,
    )
    shutil.copy2(cached_path, dest_path)
    size_gb = os.path.getsize(dest_path) / 1e9
    print(f"  → {dest_path} ({size_gb:.2f} GB)")
    return dest_path


def main() -> None:
    data_dir = os.environ["FARBENCH_DATA_DIR"]
    test_data_dir = os.environ["FARBENCH_TEST_DATA_DIR"]
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(test_data_dir, exist_ok=True)

    all_files = [
        (f, data_dir) for f in TRAIN_FILES + VAL_FILES
    ] + [
        (f, test_data_dir) for f in TEST_FILES
    ]

    all_present = all(
        os.path.exists(os.path.join(d, f)) and os.path.getsize(os.path.join(d, f)) > MIN_SIZES.get(f, 0)
        for f, d in all_files
    )
    if all_present:
        print("ClimSim low-res data already prepared, skipping.")
        return

    print("Downloading ClimSim low-res quickstart data from HuggingFace ...")
    for filename, dest_dir in all_files:
        download_file(filename, dest_dir)

    import numpy as np
    train_in = np.load(os.path.join(data_dir, "train_input.npy"), mmap_mode="r")
    val_in = np.load(os.path.join(data_dir, "val_input.npy"), mmap_mode="r")
    test_in = np.load(os.path.join(test_data_dir, "scoring_input.npy"), mmap_mode="r")

    print(f"\nClimSim low-res data ready:")
    print(f"  Train: {train_in.shape[0]:,} samples, {train_in.shape[1]} features")
    print(f"  Val:   {val_in.shape[0]:,} samples")
    print(f"  Test:  {test_in.shape[0]:,} samples")
    print(f"  Data dir:      {data_dir}")
    print(f"  Test data dir: {test_data_dir}")


if __name__ == "__main__":
    main()
