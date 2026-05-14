"""CIFAR-100N data preparation: download CIFAR-100 + human noisy labels.

Split strategy: 50K CIFAR-100 training images split into 45K train / 5K val
    (stratified by noisy label, seed=42). Training and validation labels are
    real human annotation noise (~40% noise rate) from Wei et al. (ICLR 2022).
    Test set is the standard CIFAR-100 test set (10K, balanced) with clean labels.

Output layout:
    FARBENCH_DATA_DIR/
        train.pt          — ~45,000 training samples (noisy labels)
        val.pt            — ~5,000 validation samples (noisy labels)
    FARBENCH_TEST_DATA_DIR/
        test.pt           — 10,000 test samples (clean labels)

Noisy labels source: Wei et al., "Learning with Noisy Labels Revisited:
    A Study Using Real-World Human Annotations" (ICLR 2022).
    https://github.com/UCSC-REAL/cifar-10-100n
"""

import os
import pickle
import shutil
import tarfile
import urllib.request

import numpy as np

CIFAR100_URL = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
NOISY_LABEL_URL = (
    "https://raw.githubusercontent.com/UCSC-REAL/cifar-10-100n/main/data/CIFAR-100_human.pt"
)

SPLIT_SEED = 42
VAL_RATIO = 0.1
MIN_PT_BYTES = 1024


def download_file(url: str, dest: str) -> None:
    print(f"  Downloading {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "FARBench/1.0"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        with open(dest, "wb") as f:
            shutil.copyfileobj(resp, f)


def main():
    import torch
    from sklearn.model_selection import train_test_split

    data_dir = os.environ["FARBENCH_DATA_DIR"]
    test_data_dir = os.environ["FARBENCH_TEST_DATA_DIR"]
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(test_data_dir, exist_ok=True)

    required = [
        os.path.join(data_dir, "train.pt"),
        os.path.join(data_dir, "val.pt"),
        os.path.join(test_data_dir, "test.pt"),
    ]
    if all(os.path.exists(p) and os.path.getsize(p) > MIN_PT_BYTES for p in required):
        print("CIFAR-100N data already prepared, skipping.")
        return

    # Download CIFAR-100
    raw_dir = os.path.join(data_dir, "_raw")
    os.makedirs(raw_dir, exist_ok=True)

    tar_path = os.path.join(raw_dir, "cifar-100-python.tar.gz")
    if not os.path.exists(tar_path):
        download_file(CIFAR100_URL, tar_path)

    # Download CIFAR-100N noisy labels
    noisy_path = os.path.join(raw_dir, "CIFAR-100_human.pt")
    if not os.path.exists(noisy_path):
        download_file(NOISY_LABEL_URL, noisy_path)

    # Extract CIFAR-100
    print("Extracting CIFAR-100...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(raw_dir)

    # Load CIFAR-100 training data
    train_batch_path = os.path.join(raw_dir, "cifar-100-python", "train")
    with open(train_batch_path, "rb") as f:
        train_batch = pickle.load(f, encoding="bytes")

    images_all = train_batch[b"data"].reshape(-1, 3, 32, 32).astype(np.uint8)
    clean_labels_all = np.array(train_batch[b"fine_labels"], dtype=np.int64)
    print(f"  Loaded CIFAR-100 train: {images_all.shape}")

    # Load noisy labels
    noise_file = torch.load(noisy_path, map_location="cpu", weights_only=False)
    noisy_labels_all = np.array(noise_file["noisy_label"], dtype=np.int64)
    clean_labels_check = np.array(noise_file["clean_label"], dtype=np.int64)

    # Verify alignment: clean labels from CIFAR-100 must match CIFAR-100N clean labels
    assert np.array_equal(clean_labels_all, clean_labels_check), \
        "Clean label mismatch between CIFAR-100 and CIFAR-100N!"

    noise_rate = np.mean(noisy_labels_all != clean_labels_all)
    print(f"  Noise rate: {noise_rate:.2%}")

    # Stratified train/val split by noisy label
    indices = np.arange(len(images_all))
    train_idx, val_idx = train_test_split(
        indices, test_size=VAL_RATIO, random_state=SPLIT_SEED,
        stratify=noisy_labels_all,
    )
    print(f"  Train split: {len(train_idx)} samples (noisy labels)")
    print(f"  Val split: {len(val_idx)} samples (noisy labels)")

    # Load test data (clean labels)
    test_batch_path = os.path.join(raw_dir, "cifar-100-python", "test")
    with open(test_batch_path, "rb") as f:
        test_batch = pickle.load(f, encoding="bytes")

    test_images = test_batch[b"data"].reshape(-1, 3, 32, 32).astype(np.uint8)
    test_labels = np.array(test_batch[b"fine_labels"], dtype=np.int64)
    print(f"  Test: {len(test_labels)} samples (clean labels)")

    # Save as .pt — training/val get NOISY labels only, test gets clean labels
    def _save(images, labels, path):
        t_images = torch.from_numpy(images.copy())
        t_labels = torch.from_numpy(labels.copy())
        torch.save({"images": t_images, "labels": t_labels}, path)
        print(f"  Saved {path}  ({len(t_labels)} samples)")

    _save(images_all[train_idx], noisy_labels_all[train_idx],
          os.path.join(data_dir, "train.pt"))
    _save(images_all[val_idx], noisy_labels_all[val_idx],
          os.path.join(data_dir, "val.pt"))
    _save(test_images, test_labels,
          os.path.join(test_data_dir, "test.pt"))

    # Clean up
    shutil.rmtree(raw_dir)

    print(f"\nCIFAR-100N data ready:")
    print(f"  Train/val: {data_dir}")
    print(f"  Test:      {test_data_dir}")


if __name__ == "__main__":
    main()
