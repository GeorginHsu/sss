"""ADE20K data preparation: download and organize for semantic segmentation.

Split strategy:
  Official ADE20K train/val split (MIT Scene Parsing Benchmark).
  Training: 20,210 images. Validation: 2,000 images (used as test set, labels public).
  No further splitting — the agent uses training images for training and can hold out
  a subset for validation if desired.

The test server's hidden test set (~3,000 images) is not used; we evaluate on val set
which is the community standard (Papers with Code, MMSegmentation, etc.).

Output layout:
    FARBENCH_DATA_DIR/
        meta.json
        images/training/ADE_train_*.jpg       — 20,210 training images
        annotations/training/ADE_train_*.png  — 20,210 training masks
        images/validation/ADE_val_*.jpg       — 2,000 validation images
        annotations/validation/ADE_val_*.png  — 2,000 validation masks
    FARBENCH_TEST_DATA_DIR/
        meta.json
        images/validation/ADE_val_*.jpg                   — 2,000 test images
        labels/annotations/validation/ADE_val_*.png       — evaluator ground truth
"""

import json
import os
import shutil
import zipfile
import urllib.request


# ADEChallengeData2016.zip from MIT — the standard 150-class benchmark
DOWNLOAD_URLS = [
    "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip",
]

MIN_FILE_BYTES = 1024

# 150 class names (index 0-149) from ADE20K
CLASS_NAMES = [
    "wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed",
    "windowpane", "grass", "cabinet", "sidewalk", "person", "earth",
    "door", "table", "mountain", "plant", "curtain", "chair", "car",
    "water", "painting", "sofa", "shelf", "house", "sea", "mirror",
    "rug", "field", "armchair", "seat", "fence", "desk", "rock",
    "wardrobe", "lamp", "bathtub", "railing", "cushion", "base",
    "box", "column", "signboard", "chest of drawers", "counter",
    "sand", "sink", "skyscraper", "fireplace", "refrigerator",
    "grandstand", "path", "stairs", "runway", "case", "pool table",
    "pillow", "screen door", "stairway", "river", "bridge", "bookcase",
    "blind", "coffee table", "toilet", "flower", "book", "hill",
    "bench", "countertop", "stove", "palm", "kitchen island",
    "computer", "swivel chair", "boat", "bar", "arcade machine",
    "hovel", "bus", "towel", "light", "truck", "tower", "chandelier",
    "awning", "streetlight", "booth", "television", "airplane",
    "dirt track", "apparel", "pole", "land", "bannister", "escalator",
    "ottoman", "bottle", "buffet", "poster", "stage", "van", "ship",
    "fountain", "conveyer belt", "canopy", "washer", "plaything",
    "swimming pool", "stool", "barrel", "basket", "waterfall", "tent",
    "bag", "minibike", "cradle", "oven", "ball", "food", "step",
    "tank", "trade name", "microwave", "pot", "animal", "bicycle",
    "lake", "dishwasher", "screen", "blanket", "sculpture", "hood",
    "sconce", "vase", "traffic light", "tray", "ashcan", "fan",
    "pier", "crt screen", "plate", "monitor", "bulletin board",
    "shower", "radiator", "glass", "clock", "flag",
]


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
            if total > 0 and downloaded % (10 * 65536) == 0:
                pct = downloaded * 100 // total
                print(f"\r  Progress: {pct}% ({downloaded // (1024*1024)}MB / {total // (1024*1024)}MB)",
                      end="", flush=True)
        print()


def _copy_existing_meta(src_dir, *dst_dirs):
    src = os.path.join(src_dir, "meta.json")
    if not os.path.exists(src):
        return
    for dst_dir in dst_dirs:
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy2(src, os.path.join(dst_dir, "meta.json"))


def _migrate_legacy_test_layout(test_data_dir):
    """Move old or input-split test layouts to images/ + labels/."""
    legacy_images = os.path.join(test_data_dir, "images")
    legacy_annotations = os.path.join(test_data_dir, "annotations")
    input_dir = os.path.join(test_data_dir, "input")
    input_images = os.path.join(input_dir, "images")
    label_annotations = os.path.join(test_data_dir, "labels", "annotations")

    if os.path.isdir(input_images) and not os.path.exists(legacy_images):
        shutil.move(input_images, legacy_images)
    if os.path.isdir(input_dir):
        shutil.rmtree(input_dir)
    if os.path.isdir(legacy_annotations) and not os.path.exists(label_annotations):
        os.makedirs(os.path.dirname(label_annotations), exist_ok=True)
        shutil.move(legacy_annotations, label_annotations)

    _copy_existing_meta(
        test_data_dir,
        os.path.join(test_data_dir, "labels"),
    )


def main():
    data_dir = os.environ["FARBENCH_DATA_DIR"]
    test_data_dir = os.environ["FARBENCH_TEST_DATA_DIR"]
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(test_data_dir, exist_ok=True)
    _migrate_legacy_test_layout(test_data_dir)

    # Idempotency check
    required = [
        os.path.join(data_dir, "meta.json"),
        os.path.join(data_dir, "images", "training"),
        os.path.join(data_dir, "annotations", "training"),
        os.path.join(test_data_dir, "meta.json"),
        os.path.join(test_data_dir, "images", "validation"),
        os.path.join(test_data_dir, "labels", "annotations", "validation"),
    ]
    if all(os.path.exists(p) for p in required):
        # Quick count check
        train_imgs = os.path.join(data_dir, "images", "training")
        test_imgs = os.path.join(test_data_dir, "images", "validation")
        test_labels = os.path.join(test_data_dir, "labels", "annotations", "validation")
        if (
            os.path.isdir(train_imgs)
            and os.path.isdir(test_imgs)
            and os.path.isdir(test_labels)
            and len(os.listdir(train_imgs)) >= 20000
            and len(os.listdir(test_imgs)) == 2000
            and len(os.listdir(test_labels)) == 2000
        ):
            print("ADE20K data already prepared, skipping.")
            return

    # Download
    raw_dir = os.path.join(data_dir, "_raw")
    os.makedirs(raw_dir, exist_ok=True)
    zip_path = os.path.join(raw_dir, "ADEChallengeData2016.zip")

    if not os.path.exists(zip_path) or os.path.getsize(zip_path) < 100_000_000:
        downloaded = False
        for url in DOWNLOAD_URLS:
            try:
                download_file(url, zip_path)
                downloaded = True
                break
            except Exception as e:
                print(f"  Failed: {e}")
        if not downloaded:
            raise RuntimeError(
                "Failed to download ADE20K. Please manually download from:\n"
                "  http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip\n"
                f"and place it at: {zip_path}"
            )

    # Extract
    print("  Extracting ADE20K...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(raw_dir)

    # The zip extracts to ADEChallengeData2016/ with images/ and annotations/ subdirs
    src_root = os.path.join(raw_dir, "ADEChallengeData2016")

    # Move training data to data_dir
    for subdir in ["images", "annotations"]:
        src = os.path.join(src_root, subdir, "training")
        dst = os.path.join(data_dir, subdir, "training")
        if os.path.exists(dst):
            shutil.rmtree(dst)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)
        n_files = len(os.listdir(dst))
        print(f"  {subdir}/training: {n_files} files")

    # Copy validation images+masks to data_dir for public-val monitoring.
    # Evaluation test_data_dir keeps images in the normal input location and
    # stores labels separately under labels/ for the evaluator.
    for subdir in ["images", "annotations"]:
        src = os.path.join(src_root, subdir, "validation")

        # Agent gets public val images/masks for monitoring during training.
        dst_data = os.path.join(data_dir, subdir, "validation")
        if os.path.exists(dst_data):
            shutil.rmtree(dst_data)
        os.makedirs(os.path.dirname(dst_data), exist_ok=True)
        shutil.copytree(src, dst_data)

        if subdir == "images":
            dst_test = os.path.join(test_data_dir, "images", "validation")
        else:
            dst_test = os.path.join(test_data_dir, "labels", "annotations", "validation")
        if os.path.exists(dst_test):
            shutil.rmtree(dst_test)
        os.makedirs(os.path.dirname(dst_test), exist_ok=True)
        shutil.copytree(src, dst_test)

        n_files = len(os.listdir(dst_data))
        print(f"  {subdir}/validation: {n_files} files")

    # Save meta.json
    meta = {
        "num_classes": 150,
        "ignore_index": 255,
        "class_names": CLASS_NAMES,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "crop_size": 512,
    }
    for d in [
        data_dir,
        test_data_dir,
        os.path.join(test_data_dir, "labels"),
    ]:
        with open(os.path.join(d, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    # Clean up
    shutil.rmtree(raw_dir)

    # Count
    n_train = len(os.listdir(os.path.join(data_dir, "images", "training")))
    n_val = len(os.listdir(os.path.join(data_dir, "images", "validation")))
    n_test = len(os.listdir(os.path.join(test_data_dir, "images", "validation")))
    print(f"\nADE20K data ready:")
    print(f"  Train: {data_dir} ({n_train} images)")
    print(f"  Public val: {data_dir} ({n_val} images)")
    print(f"  Eval images: {test_data_dir}/images ({n_test} images)")
    print(f"  Eval labels: {test_data_dir}/labels ({n_test} masks)")
    print(f"  Classes: 150")


if __name__ == "__main__":
    main()
