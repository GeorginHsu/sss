"""ScanObjectNN data preparation: download PB_T50_RS variant, split, serialize.

Split strategy:
  Official ScanObjectNN 80/20 train/test split (provided in h5 files).
  Training data further split 90/10 into train/val (stratified by class, seed=42).

The PB_T50_RS variant is the hardest: perturbed bounding boxes with 50% translation,
random rotation, and random scaling applied to objects from real indoor 3D scans.

Output layout:
    FARBENCH_DATA_DIR/
        meta.json     — class names, num_points, variant info
        train.pt      — ~10,274 point clouds for training
        val.pt        — ~1,142 point clouds for validation
    FARBENCH_TEST_DATA_DIR/
        test.pt       — ~2,882 point clouds for evaluation
        meta.json     — same metadata
"""

import json
import os
import zipfile
import urllib.request

import numpy as np


# PB_T50_RS h5 file names within the zip
TRAIN_H5 = "h5_files/main_split/training_objectdataset_augmentedrot_scale75.h5"
TEST_H5 = "h5_files/main_split/test_objectdataset_augmentedrot_scale75.h5"

# Download URLs (try in order)
DOWNLOAD_URLS = [
    "https://hkust-vgd.ust.hk/scanobjectnn/h5_files.zip",
]

SPLIT_SEED = 42
VAL_RATIO = 0.1
MIN_PT_BYTES = 1024

CLASS_NAMES = [
    "bag", "bed", "bin", "box", "cabinet", "chair", "desk", "display",
    "door", "pillow", "shelf", "sink", "sofa", "table", "toilet",
]


def download_file(url, dest):
    """Download with User-Agent header."""
    print(f"  Downloading {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "FARBench/1.0"})
    with urllib.request.urlopen(req, timeout=600) as resp, open(dest, "wb") as f:
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
                print(f"\r  Progress: {pct}% ({downloaded // (1024*1024)}MB / {total // (1024*1024)}MB)", end="", flush=True)
        print()


def download_with_gdown(file_id, dest):
    """Fallback: download from Google Drive using gdown."""
    import gdown
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"  Downloading from Google Drive ({file_id}) ...")
    gdown.download(url, dest, quiet=False)


def load_h5(path):
    """Load ScanObjectNN h5 file, return points [N, 2048, 3] and labels [N]."""
    import h5py
    with h5py.File(path, "r") as f:
        points = np.array(f["data"], dtype=np.float32)   # [N, 2048, 3]
        labels = np.array(f["label"], dtype=np.int64)     # [N]
    return points, labels


def main():
    import torch

    data_dir = os.environ["FARBENCH_DATA_DIR"]
    test_data_dir = os.environ["FARBENCH_TEST_DATA_DIR"]
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(test_data_dir, exist_ok=True)

    # Idempotency check
    required = [
        os.path.join(data_dir, "train.pt"),
        os.path.join(data_dir, "val.pt"),
        os.path.join(data_dir, "meta.json"),
        os.path.join(test_data_dir, "test.pt"),
    ]
    if all(os.path.exists(p) and os.path.getsize(p) > MIN_PT_BYTES for p in required):
        print("ScanObjectNN data already prepared, skipping.")
        return

    # Download h5_files.zip
    raw_dir = os.path.join(data_dir, "_raw")
    os.makedirs(raw_dir, exist_ok=True)
    zip_path = os.path.join(raw_dir, "h5_files.zip")

    if not os.path.exists(zip_path) or os.path.getsize(zip_path) < 1_000_000:
        downloaded = False
        for url in DOWNLOAD_URLS:
            try:
                download_file(url, zip_path)
                downloaded = True
                break
            except Exception as e:
                print(f"  Failed: {e}")

        if not downloaded:
            # Fallback: try gdown with known Google Drive folder
            try:
                download_with_gdown("1Z1x3WHnhPkLY8n6dJ9E1kQGsfqbHbLfF", zip_path)
                downloaded = True
            except Exception as e:
                print(f"  gdown failed: {e}")

        if not downloaded:
            raise RuntimeError(
                "Failed to download ScanObjectNN. Please manually download h5_files.zip from:\n"
                "  https://hkust-vgd.github.io/scanobjectnn/\n"
                f"and place it at: {zip_path}"
            )

    # Extract PB_T50_RS h5 files
    print("  Extracting h5 files...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        for h5_name in [TRAIN_H5, TEST_H5]:
            if h5_name in zf.namelist():
                zf.extract(h5_name, raw_dir)
            else:
                raise FileNotFoundError(
                    f"{h5_name} not found in zip. Available: {zf.namelist()[:10]}..."
                )

    # Load h5 data
    train_points, train_labels = load_h5(os.path.join(raw_dir, TRAIN_H5))
    test_points, test_labels = load_h5(os.path.join(raw_dir, TEST_H5))

    print(f"  Loaded: {len(train_labels)} train, {len(test_labels)} test samples")
    print(f"  Point cloud shape: {train_points.shape[1:]} (num_points, xyz)")
    print(f"  Classes: {len(set(train_labels.tolist()))}")

    # Stratified train/val split
    rng = np.random.RandomState(SPLIT_SEED)
    train_idx, val_idx = [], []

    for cls in range(len(CLASS_NAMES)):
        cls_indices = np.where(train_labels == cls)[0]
        perm = rng.permutation(len(cls_indices))
        cls_indices = cls_indices[perm]
        val_size = max(1, int(len(cls_indices) * VAL_RATIO))
        val_idx.extend(cls_indices[:val_size].tolist())
        train_idx.extend(cls_indices[val_size:].tolist())

    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)

    # Save train.pt
    torch.save({
        "points": torch.from_numpy(train_points[train_idx]),
        "labels": torch.from_numpy(train_labels[train_idx]),
    }, os.path.join(data_dir, "train.pt"))
    print(f"  Train: {len(train_idx)} samples")

    # Save val.pt
    torch.save({
        "points": torch.from_numpy(train_points[val_idx]),
        "labels": torch.from_numpy(train_labels[val_idx]),
    }, os.path.join(data_dir, "val.pt"))
    print(f"  Val: {len(val_idx)} samples")

    # Save test.pt
    torch.save({
        "points": torch.from_numpy(test_points),
        "labels": torch.from_numpy(test_labels),
    }, os.path.join(test_data_dir, "test.pt"))
    print(f"  Test: {len(test_labels)} samples")

    # Save meta.json
    meta = {
        "num_classes": len(CLASS_NAMES),
        "num_points": int(train_points.shape[1]),
        "variant": "PB_T50_RS",
        "class_names": CLASS_NAMES,
        "split_seed": SPLIT_SEED,
    }
    for d in [data_dir, test_data_dir]:
        with open(os.path.join(d, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    # Clean up
    import shutil
    shutil.rmtree(raw_dir)

    print(f"\nScanObjectNN (PB_T50_RS) data ready:")
    print(f"  Train/val: {data_dir}")
    print(f"  Test:      {test_data_dir}")
    print(f"  Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_labels)}")


if __name__ == "__main__":
    main()
