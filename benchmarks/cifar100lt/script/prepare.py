"""CIFAR-100-LT data preparation: download CIFAR-100, apply long-tailed sampling.

Output layout:
    FARBENCH_DATA_DIR/
        train.pt          — long-tailed training set (~10,626 samples)
        val.pt            — balanced validation set (~1,000 samples, 10 per class)
        class_counts.json — {class_idx: count} for the training set
    FARBENCH_TEST_DATA_DIR/
        test.pt           — balanced test set (10,000 samples, 100 per class)

Long-tailed sampling follows Cui et al. (CVPR 2019):
    n_i = n_max * (1/IR)^(i/(C-1))
    n_max=500, IR=100, C=100 → exponential decay from 500 to 5 per class.

The original CIFAR-100 train set (50,000) is first split: 1,000 images are
reserved for a balanced validation set (10 per class from the TAIL of each class,
to preserve head class mass). The remaining ~49,000 are subsampled per-class
to create the long-tailed training set.
"""

import json
import os
import pickle
import shutil
import tarfile
import urllib.request

import numpy as np

DATA_URLS = [
    "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
]

IMBALANCE_RATIO = 100
NUM_CLASSES = 100
MAX_SAMPLES_PER_CLASS = 500  # original CIFAR-100 has 500 per class
VAL_PER_CLASS = 10
SPLIT_SEED = 42

MIN_PT_BYTES = 1024


def download_file(url: str, dest: str) -> None:
    print(f"  Downloading {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "FARBench/1.0"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        with open(dest, "wb") as f:
            shutil.copyfileobj(resp, f)


def compute_class_counts(num_classes: int, max_samples: int, imbalance_ratio: int):
    """Compute per-class sample counts for exponential long-tailed distribution."""
    counts = []
    for i in range(num_classes):
        exponent = i / (num_classes - 1)
        n_i = int(max_samples * (1.0 / imbalance_ratio) ** exponent)
        n_i = max(n_i, 1)  # at least 1 sample
        counts.append(n_i)
    return counts


def main():
    import torch

    data_dir = os.environ["FARBENCH_DATA_DIR"]
    test_data_dir = os.environ["FARBENCH_TEST_DATA_DIR"]
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(test_data_dir, exist_ok=True)

    required = [
        os.path.join(data_dir, "train.pt"),
        os.path.join(data_dir, "val.pt"),
        os.path.join(data_dir, "class_counts.json"),
        os.path.join(test_data_dir, "test.pt"),
    ]
    if all(os.path.exists(p) and os.path.getsize(p) > MIN_PT_BYTES for p in required):
        print("CIFAR-100-LT data already prepared, skipping.")
        return

    # Download
    raw_dir = os.path.join(data_dir, "_raw")
    os.makedirs(raw_dir, exist_ok=True)
    tar_path = os.path.join(raw_dir, "cifar-100-python.tar.gz")

    if not os.path.exists(tar_path):
        downloaded = False
        for url in DATA_URLS:
            try:
                download_file(url, tar_path)
                downloaded = True
                break
            except Exception as e:
                print(f"  Failed: {e}")
        if not downloaded:
            raise RuntimeError("Failed to download CIFAR-100.")

    # Extract
    print("Extracting CIFAR-100...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(raw_dir)

    # Load train batch
    train_batch_path = os.path.join(raw_dir, "cifar-100-python", "train")
    with open(train_batch_path, "rb") as f:
        train_batch = pickle.load(f, encoding="bytes")

    train_images_raw = train_batch[b"data"]  # (50000, 3072)
    train_labels_raw = np.array(train_batch[b"fine_labels"], dtype=np.int64)

    # Reshape to CHW: (N, 3, 32, 32)
    train_images_raw = train_images_raw.reshape(-1, 3, 32, 32).astype(np.uint8)

    # Load test batch
    test_batch_path = os.path.join(raw_dir, "cifar-100-python", "test")
    with open(test_batch_path, "rb") as f:
        test_batch = pickle.load(f, encoding="bytes")

    test_images = test_batch[b"data"].reshape(-1, 3, 32, 32).astype(np.uint8)
    test_labels = np.array(test_batch[b"fine_labels"], dtype=np.int64)

    print(f"  Loaded train: {train_images_raw.shape}, test: {test_images.shape}")

    # Group training indices by class
    rng = np.random.RandomState(SPLIT_SEED)
    class_indices = {c: [] for c in range(NUM_CLASSES)}
    for idx, label in enumerate(train_labels_raw):
        class_indices[label].append(idx)

    # Shuffle within each class
    for c in range(NUM_CLASSES):
        rng.shuffle(class_indices[c])

    # Reserve balanced validation set: take VAL_PER_CLASS from tail of each class
    val_indices = []
    remaining_indices = {}
    for c in range(NUM_CLASSES):
        indices = class_indices[c]
        val_indices.extend(indices[-VAL_PER_CLASS:])
        remaining_indices[c] = indices[:-VAL_PER_CLASS]

    # Compute long-tailed class counts
    lt_counts = compute_class_counts(NUM_CLASSES, MAX_SAMPLES_PER_CLASS - VAL_PER_CLASS, IMBALANCE_RATIO)

    # Sample long-tailed training set
    train_indices = []
    actual_counts = {}
    for c in range(NUM_CLASSES):
        available = remaining_indices[c]
        n_take = min(lt_counts[c], len(available))
        train_indices.extend(available[:n_take])
        actual_counts[str(c)] = n_take

    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)

    # Shuffle
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    total_train = len(train_indices)
    print(f"  Long-tailed train: {total_train} samples")
    print(f"    Head class (0): {actual_counts['0']} samples")
    print(f"    Tail class (99): {actual_counts['99']} samples")
    print(f"    Imbalance ratio: {actual_counts['0'] / max(actual_counts['99'], 1):.1f}")
    print(f"  Balanced val: {len(val_indices)} samples ({VAL_PER_CLASS} per class)")

    # Save as .pt
    def _save(images, labels, path):
        t_images = torch.from_numpy(images.copy())
        t_labels = torch.from_numpy(labels.copy())
        torch.save({"images": t_images, "labels": t_labels}, path)
        print(f"  Saved {path}  ({len(t_labels)} samples)")

    _save(train_images_raw[train_indices], train_labels_raw[train_indices],
          os.path.join(data_dir, "train.pt"))
    _save(train_images_raw[val_indices], train_labels_raw[val_indices],
          os.path.join(data_dir, "val.pt"))
    _save(test_images, test_labels,
          os.path.join(test_data_dir, "test.pt"))

    # Save class counts
    counts_path = os.path.join(data_dir, "class_counts.json")
    with open(counts_path, "w") as f:
        json.dump(actual_counts, f, indent=2)
    print(f"  Saved {counts_path}")

    # Clean up
    shutil.rmtree(raw_dir)

    print(f"\nCIFAR-100-LT data ready:")
    print(f"  Train/val: {data_dir}")
    print(f"  Test:      {test_data_dir}")


if __name__ == "__main__":
    main()
