"""TerraIncognita data preparation: download, filter, split, organize.

Follows the DomainBed processing pipeline:
  1. Download ECCV 2018 Caltech Camera Traps images (small/resized) and annotations.
  2. Filter to 4 locations (L38, L43, L46, L100) and 10 categories.
  3. Use L100 as the held-out test domain; L38, L43, L46 as training domains.
  4. Stratified 80/20 train/val split within the 3 training domains.

Output layout:
    FARBENCH_DATA_DIR/
        train.txt         — "relative_path label" per line
        val.txt           — "relative_path label" per line
        class_names.txt   — 10 class names (line index = class index)
        domain_labels.txt — per-sample domain ID matching train.txt (0=L38, 1=L43, 2=L46)
        images/           — image files organized by location/class
    FARBENCH_TEST_DATA_DIR/
        test.txt          — "relative_path" only (no labels)
        test_labels.txt   — ground truth labels (evaluator only)
        images/           — test image files
"""

import collections
import json
import os
import shutil
import tarfile
import urllib.request

import numpy as np


IMAGES_URL = "https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/eccv_18_all_images_sm.tar.gz"
ANNOTATIONS_URL = "https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/eccv_18_annotations.tar.gz"

# Fallback URLs (Azure blob mirror)
IMAGES_URL_FALLBACK = "https://lilablobssc.blob.core.windows.net/caltechcameratraps/eccv_18_all_images_sm.tar.gz"
ANNOTATIONS_URL_FALLBACK = "https://lilablobssc.blob.core.windows.net/caltechcameratraps/eccv_18_annotations.tar.gz"

INCLUDE_LOCATIONS = ["38", "46", "100", "43"]
INCLUDE_CATEGORIES = [
    "bird", "bobcat", "cat", "coyote", "dog",
    "empty", "opossum", "rabbit", "raccoon", "squirrel",
]

# L100 is the held-out test domain; the rest are training domains
TEST_LOCATION = "100"
TRAIN_LOCATIONS = ["38", "43", "46"]
DOMAIN_ID_MAP = {"38": 0, "43": 1, "46": 2}

TRAIN_VAL_RATIO = 0.8
SPLIT_SEED = 42


def download_file(url: str, dest: str, timeout: int = 1200) -> None:
    print(f"  Downloading {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "FARBench/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        with open(dest, "wb") as f:
            shutil.copyfileobj(resp, f)


def download_with_fallback(primary_url: str, fallback_url: str, dest: str, name: str) -> None:
    """Try primary URL first, then fallback."""
    for url in [primary_url, fallback_url]:
        try:
            download_file(url, dest)
            return
        except Exception as e:
            print(f"  Failed ({url}): {e}")
    raise RuntimeError(
        f"Failed to download {name} from all URLs.\n"
        f"  Please download manually and place at: {dest}\n"
        f"  Then re-run: farbench tasks prepare terra_incognita"
    )


def extract_tarball(tar_path: str, dest_dir: str) -> None:
    print(f"  Extracting {tar_path} ...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(dest_dir)


def load_annotations(annotations_dir: str) -> dict:
    """Load and merge all 5 annotation JSON files from the ECCV 2018 release."""
    annotation_files = [
        "cis_test_annotations.json",
        "cis_val_annotations.json",
        "train_annotations.json",
        "trans_test_annotations.json",
        "trans_val_annotations.json",
    ]
    merged = collections.defaultdict(list)
    for fname in annotation_files:
        fpath = os.path.join(annotations_dir, fname)
        if not os.path.exists(fpath):
            print(f"  Warning: annotation file not found: {fpath}")
            continue
        with open(fpath, "r") as f:
            data = json.load(f)
            for key, val in data.items():
                if isinstance(val, list):
                    merged[key].extend(val)
                else:
                    merged[key] = val
    return dict(merged)


def build_image_annotation_map(data: dict):
    """Build mapping from image_id -> (file_name, location, category_name)."""
    # Category ID -> name
    cat_map = {item["id"]: item["name"] for item in data["categories"]}

    # Image ID -> (file_name, location)
    img_info = {}
    for img in data["images"]:
        img_info[img["id"]] = (img["file_name"], str(img["location"]))

    # Annotation: image_id -> category
    ann_map = {}
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        cat_name = cat_map.get(ann["category_id"], "")
        if img_id in img_info and cat_name in INCLUDE_CATEGORIES:
            fname, loc = img_info[img_id]
            if loc in INCLUDE_LOCATIONS:
                ann_map[img_id] = (fname, loc, cat_name)

    return ann_map


def main():
    data_dir = os.environ["FARBENCH_DATA_DIR"]
    test_data_dir = os.environ["FARBENCH_TEST_DATA_DIR"]
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(test_data_dir, exist_ok=True)

    # Idempotency check
    required = [
        os.path.join(data_dir, "train.txt"),
        os.path.join(data_dir, "val.txt"),
        os.path.join(data_dir, "class_names.txt"),
        os.path.join(test_data_dir, "test.txt"),
        os.path.join(test_data_dir, "test_labels.txt"),
    ]
    if all(os.path.exists(p) and os.path.getsize(p) > 10 for p in required):
        print("TerraIncognita data already prepared, skipping.")
        return

    raw_dir = os.path.join(data_dir, "_raw")
    os.makedirs(raw_dir, exist_ok=True)

    # ---------- Download ----------
    images_tar = os.path.join(raw_dir, "eccv_18_all_images_sm.tar.gz")
    if not os.path.exists(images_tar) or os.path.getsize(images_tar) < 1_000_000:
        download_with_fallback(IMAGES_URL, IMAGES_URL_FALLBACK, images_tar, "images")

    annot_tar = os.path.join(raw_dir, "eccv_18_annotations.tar.gz")
    if not os.path.exists(annot_tar) or os.path.getsize(annot_tar) < 1_000:
        download_with_fallback(ANNOTATIONS_URL, ANNOTATIONS_URL_FALLBACK, annot_tar, "annotations")

    # ---------- Extract ----------
    images_folder = os.path.join(raw_dir, "eccv_18_all_images_sm")
    if not os.path.isdir(images_folder):
        extract_tarball(images_tar, raw_dir)

    annot_folder = os.path.join(raw_dir, "eccv_18_annotation_files")
    if not os.path.isdir(annot_folder):
        extract_tarball(annot_tar, raw_dir)

    # ---------- Parse annotations ----------
    print("Parsing annotations ...")
    data = load_annotations(annot_folder)
    ann_map = build_image_annotation_map(data)
    print(f"  Found {len(ann_map)} images matching location/category filters")

    # ---------- Organize by location/category ----------
    # class_name -> index (sorted alphabetically for consistency)
    class_names = sorted(INCLUDE_CATEGORIES)
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    # Separate train-domain and test-domain samples
    train_domain_samples = []  # (relative_path, label, domain_id)
    test_domain_samples = []   # (relative_path, label)

    train_img_dir = os.path.join(data_dir, "images")
    test_img_dir = os.path.join(test_data_dir, "images")

    for img_id, (fname, loc, cat_name) in ann_map.items():
        src = os.path.join(images_folder, fname)
        if not os.path.exists(src):
            continue

        label = class_to_idx[cat_name]
        rel_path = os.path.join(f"location_{loc}", cat_name, fname)

        if loc == TEST_LOCATION:
            dst = os.path.join(test_img_dir, rel_path)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
            test_domain_samples.append((rel_path, label))
        elif loc in TRAIN_LOCATIONS:
            dst = os.path.join(train_img_dir, rel_path)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
            domain_id = DOMAIN_ID_MAP[loc]
            train_domain_samples.append((rel_path, label, domain_id))

    print(f"  Train-domain samples: {len(train_domain_samples)}")
    print(f"  Test-domain samples (L100): {len(test_domain_samples)}")

    # ---------- Stratified train/val split (within training domains) ----------
    rng = np.random.RandomState(SPLIT_SEED)

    # Group by (domain_id, label) for stratified split
    groups = collections.defaultdict(list)
    for sample in train_domain_samples:
        key = (sample[2], sample[1])  # (domain_id, label)
        groups[key].append(sample)

    train_split, val_split = [], []
    for key in sorted(groups.keys()):
        samples = groups[key]
        indices = list(range(len(samples)))
        rng.shuffle(indices)
        n_train = max(1, int(len(indices) * TRAIN_VAL_RATIO))
        train_split.extend(samples[indices[i]] for i in range(n_train))
        val_split.extend(samples[indices[i]] for i in range(n_train, len(indices)))

    rng.shuffle(train_split)
    rng.shuffle(val_split)
    rng.shuffle(test_domain_samples)

    print(f"  Split: train={len(train_split)}, val={len(val_split)}, test={len(test_domain_samples)}")

    # ---------- Save files ----------
    # train.txt
    with open(os.path.join(data_dir, "train.txt"), "w") as f:
        for rel_path, label, _ in train_split:
            f.write(f"{rel_path} {label}\n")

    # domain_labels.txt (matching train.txt order)
    with open(os.path.join(data_dir, "domain_labels.txt"), "w") as f:
        for _, _, domain_id in train_split:
            f.write(f"{domain_id}\n")

    # val.txt
    with open(os.path.join(data_dir, "val.txt"), "w") as f:
        for rel_path, label, _ in val_split:
            f.write(f"{rel_path} {label}\n")

    # class_names.txt
    with open(os.path.join(data_dir, "class_names.txt"), "w") as f:
        for name in class_names:
            f.write(f"{name}\n")

    # test.txt (no labels)
    with open(os.path.join(test_data_dir, "test.txt"), "w") as f:
        for rel_path, _ in test_domain_samples:
            f.write(f"{rel_path}\n")

    # test_labels.txt (evaluator only)
    with open(os.path.join(test_data_dir, "test_labels.txt"), "w") as f:
        for _, label in test_domain_samples:
            f.write(f"{label}\n")

    print(f"  Saved train.txt ({len(train_split)} lines)")
    print(f"  Saved val.txt ({len(val_split)} lines)")
    print(f"  Saved class_names.txt ({len(class_names)} classes)")
    print(f"  Saved domain_labels.txt ({len(train_split)} lines)")
    print(f"  Saved test.txt ({len(test_domain_samples)} lines)")
    print(f"  Saved test_labels.txt ({len(test_domain_samples)} labels)")

    # ---------- Clean up ----------
    shutil.rmtree(raw_dir)

    print(f"\nTerraIncognita data ready:")
    print(f"  Train/val: {data_dir}")
    print(f"  Test:      {test_data_dir}")


if __name__ == "__main__":
    main()
