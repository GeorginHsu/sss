"""MNIST data preparation: download, split, and serialize as .pt files.

Output layout:
    FARBENCH_DATA_DIR/
        train.pt   — (images, labels) for training (85% of MNIST train split)
        val.pt     — (images, labels) for validation (15% of MNIST train split)
    FARBENCH_TEST_DATA_DIR/
        test.pt    — (images, labels) for final evaluation (MNIST test split)

Agent can only access FARBENCH_DATA_DIR.  Test data is isolated in FARBENCH_TEST_DATA_DIR
and never exposed to the agent — it is only used by the system-side evaluator.
"""

import gzip
import os
import shutil
import struct
import urllib.request

import numpy as np

# ── Download sources ──

FILES = [
    ("train-images-idx3-ubyte.gz", "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", False),
    ("train-labels-idx1-ubyte.gz", "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", False),
    ("t10k-images-idx3-ubyte.gz",  "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",  True),
    ("t10k-labels-idx1-ubyte.gz",  "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",  True),
]

MIRROR_URLS = {
    "train-images-idx3-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz":  "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz":  "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
}

VAL_RATIO = 0.15
SPLIT_SEED = 0  # must match data_loader.py to guarantee identical split


def download_file(url: str, dest: str) -> None:
    print(f"  Downloading {url} ...")
    urllib.request.urlretrieve(url, dest)


# ── IDX file parsing (no torch/torchvision dependency) ──

def _read_idx_images(path: str) -> np.ndarray:
    """Read IDX image file → uint8 array of shape (N, 28, 28)."""
    with open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, rows, cols)


def _read_idx_labels(path: str) -> np.ndarray:
    """Read IDX label file → int64 array of shape (N,)."""
    with open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 2049
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.astype(np.int64)


def main():
    import torch

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
        print("MNIST data already prepared, skipping.")
        return

    # Download and decompress raw IDX files into a temp directory
    raw_dir = os.path.join(data_dir, "_raw")
    os.makedirs(raw_dir, exist_ok=True)

    for filename, url, _is_test in FILES:
        gz_path = os.path.join(raw_dir, filename)
        raw_path = gz_path.replace(".gz", "")

        if os.path.exists(raw_path):
            print(f"  Already exists: {raw_path}")
            continue

        if not os.path.exists(gz_path):
            try:
                download_file(url, gz_path)
            except Exception:
                mirror_url = MIRROR_URLS.get(filename, url)
                print(f"  Primary URL failed, trying mirror...")
                download_file(mirror_url, gz_path)

        print(f"  Decompressing {filename} ...")
        with gzip.open(gz_path, "rb") as f_in:
            with open(raw_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

    # ── Parse raw files ──
    train_images = _read_idx_images(os.path.join(raw_dir, "train-images-idx3-ubyte"))
    train_labels = _read_idx_labels(os.path.join(raw_dir, "train-labels-idx1-ubyte"))
    test_images  = _read_idx_images(os.path.join(raw_dir, "t10k-images-idx3-ubyte"))
    test_labels  = _read_idx_labels(os.path.join(raw_dir, "t10k-labels-idx1-ubyte"))

    # ── Train / val split (deterministic, matches data_loader.py) ──
    n = len(train_images)
    val_size = int(n * VAL_RATIO)
    rng = np.random.RandomState(SPLIT_SEED)
    indices = rng.permutation(n)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    # Convert to torch tensors (uint8 images, int64 labels)
    def _save(images: np.ndarray, labels: np.ndarray, path: str) -> None:
        t_images = torch.from_numpy(images.copy())     # uint8 (N, 28, 28)
        t_labels = torch.from_numpy(labels.copy())     # int64 (N,)
        torch.save({"images": t_images, "labels": t_labels}, path)
        print(f"  Saved {path}  ({len(t_labels)} samples)")

    _save(train_images[train_indices], train_labels[train_indices],
          os.path.join(data_dir, "train.pt"))
    _save(train_images[val_indices], train_labels[val_indices],
          os.path.join(data_dir, "val.pt"))
    _save(test_images, test_labels,
          os.path.join(test_data_dir, "test.pt"))

    # Clean up raw files
    shutil.rmtree(raw_dir)

    print(f"\nMNIST data ready:")
    print(f"  Train/val: {data_dir}")
    print(f"  Test:      {test_data_dir}")


if __name__ == "__main__":
    main()
