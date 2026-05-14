"""DIV2K 4x super-resolution data preparation: download HR and LR bicubic x4.

Split strategy:
  Official DIV2K split: 800 train / 100 val / 100 test (test GT hidden).
  We use the 100 val images as our test set (community standard practice).
  Agent gets both train and val (with HR) for training and monitoring.

Output layout:
    FARBENCH_DATA_DIR/
        meta.json
        train/hr/0001.png ... 0800.png         — 800 HR training images
        train/lr_x4/0001x4.png ... 0800x4.png   — 800 LR training images (bicubic x4)
        val/hr/0801.png ... 0900.png             — 100 HR val images (for agent monitoring)
        val/lr_x4/0801x4.png ... 0900x4.png      — 100 LR val images
    FARBENCH_TEST_DATA_DIR/
        meta.json
        hr/0801.png ... 0900.png                 — ground truth HR (for evaluator)
        lr_x4/0801x4.png ... 0900x4.png          — LR test images (input to predict.py)
"""

import json
import os
import shutil
import zipfile
import urllib.request


DOWNLOAD_URLS = {
    "train_hr": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
    "train_lr": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X4.zip",
    "val_hr": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
    "val_lr": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X4.zip",
}

MIN_FILE_BYTES = 1024


def download_file(url, dest):
    """Download with User-Agent header and progress."""
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
            if total > 0 and downloaded % (50 * 65536) == 0:
                pct = downloaded * 100 // total
                print(f"\r  {pct}% ({downloaded // (1024*1024)}MB / {total // (1024*1024)}MB)",
                      end="", flush=True)
        print()


def move_images(src_dir, dst_dir):
    """Move all PNG images from src_dir to dst_dir."""
    os.makedirs(dst_dir, exist_ok=True)
    count = 0
    for f in sorted(os.listdir(src_dir)):
        if f.lower().endswith(".png"):
            shutil.move(os.path.join(src_dir, f), os.path.join(dst_dir, f))
            count += 1
    return count


def main():
    data_dir = os.environ["FARBENCH_DATA_DIR"]
    test_data_dir = os.environ["FARBENCH_TEST_DATA_DIR"]
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(test_data_dir, exist_ok=True)

    # Idempotency check
    required = [
        os.path.join(data_dir, "meta.json"),
        os.path.join(data_dir, "train", "hr"),
        os.path.join(data_dir, "train", "lr_x4"),
        os.path.join(test_data_dir, "hr"),
        os.path.join(test_data_dir, "lr_x4"),
    ]
    train_hr_dir = os.path.join(data_dir, "train", "hr")
    if all(os.path.exists(p) for p in required) and \
       os.path.isdir(train_hr_dir) and len(os.listdir(train_hr_dir)) >= 800:
        print("DIV2K SR x4 data already prepared, skipping.")
        return

    # Download all zips
    raw_dir = os.path.join(data_dir, "_raw")
    os.makedirs(raw_dir, exist_ok=True)

    for key, url in DOWNLOAD_URLS.items():
        zip_path = os.path.join(raw_dir, f"{key}.zip")
        if not os.path.exists(zip_path) or os.path.getsize(zip_path) < MIN_FILE_BYTES:
            download_file(url, zip_path)

        print(f"  Extracting {key}...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(raw_dir)

    # Organize files
    # Train HR: DIV2K_train_HR/ → train/hr/
    n = move_images(os.path.join(raw_dir, "DIV2K_train_HR"),
                    os.path.join(data_dir, "train", "hr"))
    print(f"  Train HR: {n} images")

    # Train LR x4: DIV2K_train_LR_bicubic/X4/ → train/lr_x4/
    n = move_images(os.path.join(raw_dir, "DIV2K_train_LR_bicubic", "X4"),
                    os.path.join(data_dir, "train", "lr_x4"))
    print(f"  Train LR x4: {n} images")

    # Val HR: DIV2K_valid_HR/ → val/hr/ (agent) + test hr/ (evaluator)
    val_hr_src = os.path.join(raw_dir, "DIV2K_valid_HR")
    val_hr_agent = os.path.join(data_dir, "val", "hr")
    val_hr_test = os.path.join(test_data_dir, "hr")
    os.makedirs(val_hr_agent, exist_ok=True)
    os.makedirs(val_hr_test, exist_ok=True)

    for f in sorted(os.listdir(val_hr_src)):
        if f.lower().endswith(".png"):
            src = os.path.join(val_hr_src, f)
            shutil.copy2(src, os.path.join(val_hr_agent, f))
            shutil.move(src, os.path.join(val_hr_test, f))
    n_val = len(os.listdir(val_hr_agent))
    print(f"  Val HR: {n_val} images")

    # Val LR x4: DIV2K_valid_LR_bicubic/X4/ → val/lr_x4/ (agent) + test lr_x4/ (evaluator)
    val_lr_src = os.path.join(raw_dir, "DIV2K_valid_LR_bicubic", "X4")
    val_lr_agent = os.path.join(data_dir, "val", "lr_x4")
    val_lr_test = os.path.join(test_data_dir, "lr_x4")
    os.makedirs(val_lr_agent, exist_ok=True)
    os.makedirs(val_lr_test, exist_ok=True)

    for f in sorted(os.listdir(val_lr_src)):
        if f.lower().endswith(".png"):
            src = os.path.join(val_lr_src, f)
            shutil.copy2(src, os.path.join(val_lr_agent, f))
            shutil.move(src, os.path.join(val_lr_test, f))
    print(f"  Val LR x4: {len(os.listdir(val_lr_agent))} images")

    # Save meta.json
    meta = {
        "scale": 4,
        "n_train": 800,
        "n_val": 100,
        "degradation": "bicubic",
        "mean": [0.4488, 0.4371, 0.4040],
        "std": [1.0, 1.0, 1.0],
        "eval_border_crop": 4,
        "eval_channel": "Y (YCbCr)",
    }
    for d in [data_dir, test_data_dir]:
        with open(os.path.join(d, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    # Clean up
    shutil.rmtree(raw_dir)

    print(f"\nDIV2K SR x4 data ready:")
    print(f"  Train: {data_dir}/train/ (800 HR + 800 LR)")
    print(f"  Val:   {data_dir}/val/ (100 HR + 100 LR)")
    print(f"  Test:  {test_data_dir}/ (100 HR + 100 LR)")


if __name__ == "__main__":
    main()
