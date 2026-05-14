"""Qlib CSI300 stock prediction data preparation.

Downloads CSI300 daily stock data (qlib binary format) and writes
the evaluation config for the held-out test period.

Data source: https://github.com/chenditc/investment_data/releases
(pre-processed qlib-compatible CN stock data)

Data periods:
  Train:  2008-01-01 to 2020-12-31
  Valid:  2021-01-01 to 2021-12-31
  Test:   2022-01-01 to 2023-12-31

Output layout:
    FARBENCH_DATA_DIR/
        meta.json
        qlib_data/       — Qlib binary data directory
    FARBENCH_TEST_DATA_DIR/
        eval_config.json — test period specification
"""

from __future__ import annotations

import json
import os
import shutil
import tarfile
import urllib.request


# Qlib CN stock data (binary format) — pinned release from chenditc/investment_data
QLIB_DATA_URL = "https://github.com/chenditc/investment_data/releases/download/2026-04-18/qlib_bin.tar.gz"

UNIVERSE = "csi300"
FEATURES = "Alpha158"
TRAIN_START = "2008-01-01"
TRAIN_END = "2020-12-31"
VALID_START = "2021-01-01"
VALID_END = "2021-12-31"
TEST_START = "2022-01-01"
TEST_END = "2023-12-31"
LABEL = "Ref($close, -1) / $close - 1"

EVAL_CONFIG = {
    "universe": UNIVERSE,
    "features": FEATURES,
    "test_period": [TEST_START, TEST_END],
    "train_period": [TRAIN_START, TRAIN_END],
    "valid_period": [VALID_START, VALID_END],
    "label": LABEL,
}


def download_file(url: str, dest: str, timeout: int = 600) -> None:
    """Download with progress."""
    print(f"  Downloading {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "FARBench/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(dest, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        while True:
            chunk = resp.read(65536)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = downloaded * 100 // total
                print(
                    f"\r  Progress: {pct}% ({downloaded // (1024*1024)}MB / {total // (1024*1024)}MB)",
                    end="", flush=True,
                )
        print()


def is_nonempty_dir(path: str) -> bool:
    return os.path.isdir(path) and len(os.listdir(path)) >= 1


def main() -> None:
    data_dir = os.environ["FARBENCH_DATA_DIR"]
    test_data_dir = os.environ["FARBENCH_TEST_DATA_DIR"]
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(test_data_dir, exist_ok=True)

    qlib_data_dir = os.path.join(data_dir, "qlib_data")
    test_qlib_data = os.path.join(test_data_dir, "qlib_data")

    # Idempotency check
    required = [
        os.path.join(data_dir, "meta.json"),
        qlib_data_dir,
        os.path.join(test_data_dir, "eval_config.json"),
    ]
    if all(os.path.exists(p) for p in required):
        if is_nonempty_dir(qlib_data_dir):
            if not is_nonempty_dir(test_qlib_data):
                print("Qlib train data already exists; copying qlib_data into test_data_dir ...")
                if os.path.exists(test_qlib_data):
                    shutil.rmtree(test_qlib_data)
                shutil.copytree(qlib_data_dir, test_qlib_data)
            print("Qlib data already prepared. Skipping.")
            return

    # Download qlib binary data
    print("Step 1: Downloading Qlib CN stock data ...")
    raw_dir = os.path.join(data_dir, "_raw")
    os.makedirs(raw_dir, exist_ok=True)

    tar_path = os.path.join(raw_dir, "qlib_bin.tar.gz")
    if not os.path.exists(tar_path) or os.path.getsize(tar_path) < 1_000_000:
        download_file(QLIB_DATA_URL, tar_path, timeout=1800)

    # Extract
    print("  Extracting qlib_bin.tar.gz ...")
    os.makedirs(qlib_data_dir, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(raw_dir)

    # Find the extracted data directory
    # It might extract as qlib_bin/ or cn_data/ or directly
    extracted = None
    for candidate in ["qlib_bin", "cn_data", "qlib_data"]:
        p = os.path.join(raw_dir, candidate)
        if os.path.isdir(p) and len(os.listdir(p)) >= 1:
            extracted = p
            break

    if extracted is None:
        # Check if files were extracted directly into raw_dir
        bin_files = [f for f in os.listdir(raw_dir) if f.endswith(".bin") or f.endswith(".d")]
        if bin_files:
            extracted = raw_dir
        else:
            # List what's in raw_dir for debugging
            print(f"  Contents of {raw_dir}:")
            for item in os.listdir(raw_dir):
                item_path = os.path.join(raw_dir, item)
                if os.path.isdir(item_path):
                    n = len(os.listdir(item_path))
                    print(f"    {item}/ ({n} items)")
                else:
                    size = os.path.getsize(item_path)
                    print(f"    {item} ({size} bytes)")
            raise RuntimeError("Could not find extracted qlib data")

    # Move to final location
    if extracted != qlib_data_dir:
        if os.path.exists(qlib_data_dir):
            shutil.rmtree(qlib_data_dir)
        shutil.move(extracted, qlib_data_dir)

    n_files = sum(1 for _ in os.scandir(qlib_data_dir))
    print(f"  Qlib data ready: {n_files} files/dirs in {qlib_data_dir}")

    # Verify with qlib
    print("Step 2: Verifying data with qlib ...")
    try:
        import qlib
        qlib.init(provider_uri=qlib_data_dir, region="cn")
        from qlib.data import D
        instruments = D.instruments(market="csi300")
        print(f"  CSI300 universe loaded successfully")
    except Exception as e:
        print(f"  Warning: qlib verification failed: {e}")
        print("  Data files exist but qlib init failed. Agent may need to adjust provider_uri.")

    # Write meta.json
    print("Step 3: Writing metadata ...")
    meta = {
        "universe": UNIVERSE,
        "features": FEATURES,
        "train_period": [TRAIN_START, TRAIN_END],
        "valid_period": [VALID_START, VALID_END],
        "test_period": [TEST_START, TEST_END],
        "label": LABEL,
        "qlib_data_dir": "qlib_data",
    }
    with open(os.path.join(data_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Write eval_config.json
    with open(os.path.join(test_data_dir, "eval_config.json"), "w") as f:
        json.dump(EVAL_CONFIG, f, indent=2)

    # Copy qlib_data into test_data_dir so predict.py can access it during evaluation.
    # The eval container mounts test_data_dir as /data, so /data/qlib_data must exist.
    # (Symlinks won't work across Docker bind-mount boundaries.)
    if not is_nonempty_dir(test_qlib_data):
        print("  Copying qlib_data into test_data_dir for evaluation ...")
        shutil.copytree(qlib_data_dir, test_qlib_data, dirs_exist_ok=True)
        print(f"  Done: {test_qlib_data}")

    # Clean up
    shutil.rmtree(raw_dir, ignore_errors=True)

    print(f"\nQlib CSI300 data ready:")
    print(f"  Data dir:    {qlib_data_dir}")
    print(f"  Train:       {TRAIN_START} to {TRAIN_END}")
    print(f"  Valid:       {VALID_START} to {VALID_END}")
    print(f"  Test:        {TEST_START} to {TEST_END}")


if __name__ == "__main__":
    main()
