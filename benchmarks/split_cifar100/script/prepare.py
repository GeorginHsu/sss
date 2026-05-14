"""Split-CIFAR-100 data preparation for continual learning (Class-IL).

Split strategy:
  CIFAR-100 official 50K train / 10K test split.
  100 classes randomly permuted (seed=42), split into 10 tasks x 10 classes.
  Training data further split 90/10 into train/val (stratified by class, seed=42).
  Following GEM (Lopez-Paz & Ranzato, NeurIPS 2017) 10-task protocol.

Output layout:
    FARBENCH_DATA_DIR/
        meta.json              — task ordering and metadata
        task_0/train.pt        — ~4,500 images for task 0's 10 classes
        task_1/train.pt        — ~4,500 images for task 1's 10 classes
        ...
        task_9/train.pt        — ~4,500 images for task 9's 10 classes
        val.pt                 — ~5,000 images across all 100 classes
    FARBENCH_TEST_DATA_DIR/
        test.pt                — 10,000 test images (all 100 classes)
        meta.json              — same task ordering (for evaluator)
"""

import json
import os
import pickle
import tarfile
import urllib.request

import numpy as np


CIFAR100_URL = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
CIFAR100_MIRROR = "https://ossci-datasets.s3.amazonaws.com/cifar-100-python.tar.gz"

SPLIT_SEED = 42
NUM_TASKS = 10
CLASSES_PER_TASK = 10
VAL_RATIO = 0.1
MIN_PT_BYTES = 1024


def download_file(url, dest):
    """Download with User-Agent header to avoid 403."""
    print(f"  Downloading {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "FARBench/1.0"})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as f:
        while True:
            chunk = resp.read(8192)
            if not chunk:
                break
            f.write(chunk)


def load_cifar100_batch(path):
    """Load a CIFAR-100 pickle batch, return images [N,3,32,32] uint8 and labels [N] int64."""
    with open(path, "rb") as f:
        batch = pickle.load(f, encoding="bytes")
    images = batch[b"data"].reshape(-1, 3, 32, 32)  # uint8 [N, 3, 32, 32]
    labels = np.array(batch[b"fine_labels"], dtype=np.int64)
    return images, labels


def main():
    import torch

    data_dir = os.environ["FARBENCH_DATA_DIR"]
    test_data_dir = os.environ["FARBENCH_TEST_DATA_DIR"]
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(test_data_dir, exist_ok=True)

    # Idempotency check
    required = [
        os.path.join(data_dir, "meta.json"),
        os.path.join(data_dir, "val.pt"),
        os.path.join(data_dir, "task_0", "train.pt"),
        os.path.join(data_dir, "task_9", "train.pt"),
        os.path.join(test_data_dir, "test.pt"),
    ]
    if all(os.path.exists(p) and os.path.getsize(p) > MIN_PT_BYTES for p in required):
        print("Split-CIFAR-100 data already prepared, skipping.")
        return

    # Download CIFAR-100
    raw_dir = os.path.join(data_dir, "_raw")
    os.makedirs(raw_dir, exist_ok=True)
    tar_path = os.path.join(raw_dir, "cifar-100-python.tar.gz")

    if not os.path.exists(tar_path) or os.path.getsize(tar_path) < 100_000_000:
        try:
            download_file(CIFAR100_URL, tar_path)
        except Exception:
            print("  Primary URL failed, trying mirror...")
            download_file(CIFAR100_MIRROR, tar_path)

    # Extract
    print("  Extracting CIFAR-100...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(raw_dir)

    # Load raw data
    train_images, train_labels = load_cifar100_batch(
        os.path.join(raw_dir, "cifar-100-python", "train")
    )
    test_images, test_labels = load_cifar100_batch(
        os.path.join(raw_dir, "cifar-100-python", "test")
    )

    print(f"  Loaded: {len(train_labels)} train, {len(test_labels)} test images")

    # Generate fixed class permutation
    rng = np.random.RandomState(SPLIT_SEED)
    class_order = rng.permutation(100).tolist()
    tasks = [class_order[i * CLASSES_PER_TASK:(i + 1) * CLASSES_PER_TASK]
             for i in range(NUM_TASKS)]

    meta = {
        "num_tasks": NUM_TASKS,
        "classes_per_task": tasks,
        "total_classes": 100,
        "backbone": "ResNet-18 (from scratch, no pretrained weights)",
        "scenario": "Class-Incremental Learning (Class-IL)",
        "split_seed": SPLIT_SEED,
    }

    # Stratified train/val split per class
    val_images_list, val_labels_list = [], []

    for task_idx, task_classes in enumerate(tasks):
        task_dir = os.path.join(data_dir, f"task_{task_idx}")
        os.makedirs(task_dir, exist_ok=True)

        task_train_imgs, task_train_lbls = [], []

        for cls in task_classes:
            cls_mask = train_labels == cls
            cls_images = train_images[cls_mask]
            cls_labels = train_labels[cls_mask]

            # Shuffle within class
            n = len(cls_labels)
            perm = rng.permutation(n)
            cls_images = cls_images[perm]
            cls_labels = cls_labels[perm]

            # 90/10 split
            val_size = max(1, int(n * VAL_RATIO))
            val_images_list.append(cls_images[:val_size])
            val_labels_list.append(cls_labels[:val_size])
            task_train_imgs.append(cls_images[val_size:])
            task_train_lbls.append(cls_labels[val_size:])

        # Save per-task train data
        t_images = torch.from_numpy(np.concatenate(task_train_imgs))
        t_labels = torch.from_numpy(np.concatenate(task_train_lbls))
        torch.save({"images": t_images, "labels": t_labels},
                    os.path.join(task_dir, "train.pt"))
        print(f"  Task {task_idx}: {len(t_labels)} train images, classes {task_classes}")

    # Save global val set
    all_val_images = torch.from_numpy(np.concatenate(val_images_list))
    all_val_labels = torch.from_numpy(np.concatenate(val_labels_list))
    torch.save({"images": all_val_images, "labels": all_val_labels},
               os.path.join(data_dir, "val.pt"))
    print(f"  Val: {len(all_val_labels)} images across all 100 classes")

    # Save test data
    t_test_images = torch.from_numpy(test_images.copy())
    t_test_labels = torch.from_numpy(test_labels.copy())
    torch.save({"images": t_test_images, "labels": t_test_labels},
               os.path.join(test_data_dir, "test.pt"))
    print(f"  Test: {len(t_test_labels)} images")

    # Save meta.json in both dirs
    for d in [data_dir, test_data_dir]:
        with open(os.path.join(d, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    # Clean up raw files
    import shutil
    shutil.rmtree(raw_dir)

    print(f"\nSplit-CIFAR-100 data ready:")
    print(f"  Train/val: {data_dir}")
    print(f"  Test:      {test_data_dir}")
    print(f"  Tasks: {NUM_TASKS} x {CLASSES_PER_TASK} classes")
    print(f"  Train: ~{len(all_val_labels) * 9} | Val: {len(all_val_labels)} | Test: {len(t_test_labels)}")


if __name__ == "__main__":
    main()
