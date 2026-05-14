"""Objaverse 3D Generation data preparation.

Data source:
  Arb-Objaverse (lizb6626/Arb-Objaverse) — pre-rendered multi-view images
  of curated Objaverse objects. 12 views per object, 512x512, transparent BG,
  with explicit camera parameters (azimuth, elevation, radius) in camera.json.

  We download a single zip shard (~10GB), extract ~4000 objects,
  select 1000, resize to 256x256, and split into train/val/test.

Split strategy:
  800 train / 100 val / 100 test from the same Arb-Objaverse curated set.
  View 0 = input view; views 1-11 = target views.

Output layout:
    FARBENCH_DATA_DIR/
        meta.json
        train/<object_uid>/
            input.png                         — view 0, 256x256 RGBA
            target_views/view_01.png ... view_11.png
            cameras.json
        val/<object_uid>/   — same structure
    FARBENCH_TEST_DATA_DIR/
        meta.json
        test/<object_uid>/
            input.png
            cameras.json
        test_gt/<object_uid>/
            view_01.png ... view_11.png
"""

from __future__ import annotations

import json
import os
import shutil
import urllib.request
import zipfile

import numpy as np

SPLIT_SEED = 42
N_TRAIN = 800
N_VAL = 100
N_TEST = 100
N_TOTAL = N_TRAIN + N_VAL + N_TEST  # 1000
N_VIEWS = 12
IMAGE_SIZE = 256  # resize from 512 to 256
N_TARGET_VIEWS = N_VIEWS - 1
EVAL_CONFIG = {
    "split": "test",
    "n_objects": N_TEST,
    "n_views": N_VIEWS,
    "n_target_views": N_TARGET_VIEWS,
    "image_size": IMAGE_SIZE,
    "input_layout": "<test_data_dir>/<object_uid>/input.png",
    "target_views": [f"view_{vi:02d}.png" for vi in range(1, N_VIEWS)],
    "ground_truth_dir": "test_gt",
}

# Arb-Objaverse: download the smallest shard first
# Each shard has ~400-500 objects; we need 1000, so download 3 shards
ARB_BASE_URL = (
    "https://huggingface.co/datasets/lizb6626/Arb-Objaverse/resolve/main/data"
)
# Pick shards to download (smallest first to minimize bandwidth)
ARB_SHARDS = ["000-110.zip", "000-000.zip", "000-001.zip"]


def _object_dirs(path: str) -> list[str]:
    if not os.path.isdir(path):
        return []
    return [
        name for name in os.listdir(path)
        if os.path.isdir(os.path.join(path, name))
    ]


def _has_target_views(path: str) -> bool:
    target_dir = os.path.join(path, "target_views")
    return all(
        os.path.exists(os.path.join(target_dir, f"view_{vi:02d}.png"))
        for vi in range(1, N_VIEWS)
    )


def _has_test_gt(path: str) -> bool:
    return all(
        os.path.exists(os.path.join(path, f"view_{vi:02d}.png"))
        for vi in range(1, N_VIEWS)
    )


def _has_test_input(path: str) -> bool:
    return (
        os.path.exists(os.path.join(path, "input.png"))
        and os.path.exists(os.path.join(path, "cameras.json"))
    )


def count_complete_split(path: str, kind: str) -> int:
    """Count complete objects in a prepared split."""
    n = 0
    for uid in _object_dirs(path):
        obj_dir = os.path.join(path, uid)
        if kind in {"train", "val"}:
            ok = _has_test_input(obj_dir) and _has_target_views(obj_dir)
        elif kind == "test":
            ok = _has_test_input(obj_dir)
        elif kind == "test_gt":
            ok = _has_test_gt(obj_dir)
        else:
            raise ValueError(f"unknown split kind: {kind}")
        n += int(ok)
    return n


def write_eval_config(test_data_dir: str) -> None:
    with open(os.path.join(test_data_dir, "eval_config.json"), "w") as f:
        json.dump(EVAL_CONFIG, f, indent=2)


def migrate_legacy_test_layout(test_data_dir: str) -> None:
    """Move old test/<uid>/ inputs to <uid>/ so --data_path matches task.yaml."""
    legacy_dir = os.path.join(test_data_dir, "test")
    if not os.path.isdir(legacy_dir):
        return

    moved = 0
    for uid in _object_dirs(legacy_dir):
        src = os.path.join(legacy_dir, uid)
        if not _has_test_input(src):
            continue
        dst = os.path.join(test_data_dir, uid)
        if os.path.exists(dst):
            if _has_test_input(dst):
                continue
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        moved += 1

    if moved:
        print(f"  Copied {moved} test input objects from legacy test/ to test_data_dir root.")

    if count_complete_split(test_data_dir, "test") >= N_TEST:
        try:
            shutil.rmtree(legacy_dir)
            print("  Removed legacy test/ directory.")
        except OSError as e:
            print(f"  Warning: could not remove legacy test/ directory: {e}")


def download_file(url: str, dest: str, timeout: int = 600) -> None:
    """Download a file with progress."""
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
            if total > 0 and downloaded % (1024 * 1024 * 10) < 65536:
                pct = downloaded * 100 // total
                mb = downloaded // (1024 * 1024)
                total_mb = total // (1024 * 1024)
                print(f"\r  Progress: {pct}% ({mb}MB / {total_mb}MB)",
                      end="", flush=True)
        print()


def extract_objects_from_shard(
    zip_path: str, extract_dir: str,
) -> dict[str, str]:
    """Extract objects from a shard zip.

    Arb-Objaverse structure inside zip:
        <shard_name>/<object_uid>/albedo_00.png ... albedo_11.png
        <shard_name>/<object_uid>/camera.json

    Returns dict mapping uid -> absolute path to the object directory.
    """
    print(f"  Extracting {zip_path} ...")
    uid_to_path: dict[str, str] = {}

    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            parts = name.strip("/").split("/")
            # Structure: shard_name/uid/file.png -> parts[1] is the UID
            if len(parts) >= 3:
                uid = parts[1]
                if uid not in uid_to_path:
                    shard_name = parts[0]
                    uid_to_path[uid] = os.path.join(extract_dir, shard_name, uid)

        zf.extractall(extract_dir)

    print(f"  Extracted {len(uid_to_path)} objects from {os.path.basename(zip_path)}")
    return uid_to_path


def validate_object(obj_dir: str) -> bool:
    """Check if an extracted object has the expected files."""
    camera_file = os.path.join(obj_dir, "camera.json")
    if not os.path.exists(camera_file):
        return False

    # Check we have at least 12 color images (albedo or color_L)
    n_images = 0
    for fname in os.listdir(obj_dir):
        if fname.startswith("albedo_") and fname.endswith(".png"):
            n_images += 1
    if n_images < N_VIEWS:
        # Try color_L images instead
        n_images = 0
        for fname in os.listdir(obj_dir):
            if fname.startswith("color_L_") and fname.endswith(".png"):
                n_images += 1

    return n_images >= N_VIEWS


def convert_camera_json(camera_path: str) -> list[dict]:
    """Convert Arb-Objaverse camera.json to our standardized format.

    Arb-Objaverse format:
        {"azimuths": [...], "elevations": [...], "radius": [...],
         "transform_matrix": [...]}

    Our format:
        [{"view_id": 0, "azimuth": ..., "elevation": ..., "distance": ...}, ...]
    """
    with open(camera_path) as f:
        cam_data = json.load(f)

    azimuths = cam_data["azimuths"]
    elevations = cam_data["elevations"]
    # Note: Arb-Objaverse has a typo — key is "raidus" not "radius"
    radii = cam_data.get("radius") or cam_data.get("raidus", [])

    cameras = []
    for i in range(min(N_VIEWS, len(azimuths))):
        cameras.append({
            "view_id": i,
            "azimuth": float(azimuths[i]),
            "elevation": float(elevations[i]),
            "distance": float(radii[i]),
        })
    return cameras


def save_split(
    uids: list[str],
    uid_to_path: dict[str, str],
    split_dir: str,
    is_test: bool = False,
    gt_dir: str | None = None,
):
    """Process and save a data split.

    Reads pre-rendered images from uid_to_path, resizes to IMAGE_SIZE,
    and saves in the standardized layout.
    """
    from PIL import Image

    os.makedirs(split_dir, exist_ok=True)
    if gt_dir:
        os.makedirs(gt_dir, exist_ok=True)

    saved = 0
    for uid in uids:
        src_obj = uid_to_path.get(uid)
        if not src_obj or not os.path.isdir(src_obj):
            print(f"    Warning: {uid} not found in extracted data, skipping")
            continue

        obj_dir = os.path.join(split_dir, uid)
        os.makedirs(obj_dir, exist_ok=True)

        # Determine image prefix (prefer albedo for cleaner appearance)
        prefix = "albedo_"
        sample = os.path.join(src_obj, "albedo_00.png")
        if not os.path.exists(sample):
            prefix = "color_L_"
            sample = os.path.join(src_obj, "color_L_00.png")
            if not os.path.exists(sample):
                print(f"    Warning: no images found for {uid}, skipping")
                continue

        # Convert camera.json
        camera_src = os.path.join(src_obj, "camera.json")
        cameras = convert_camera_json(camera_src)
        with open(os.path.join(obj_dir, "cameras.json"), "w") as f:
            json.dump(cameras, f, indent=2)

        # Save input view (view 0)
        img0_path = os.path.join(src_obj, f"{prefix}00.png")
        img0 = Image.open(img0_path).convert("RGBA")
        img0 = img0.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
        img0.save(os.path.join(obj_dir, "input.png"))

        # Save target views (views 1-11)
        if is_test and gt_dir:
            gt_obj_dir = os.path.join(gt_dir, uid)
            os.makedirs(gt_obj_dir, exist_ok=True)
            for vi in range(1, N_VIEWS):
                src_img = os.path.join(src_obj, f"{prefix}{vi:02d}.png")
                if os.path.exists(src_img):
                    img = Image.open(src_img).convert("RGBA")
                    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
                    img.save(os.path.join(gt_obj_dir, f"view_{vi:02d}.png"))
        else:
            tv_dir = os.path.join(obj_dir, "target_views")
            os.makedirs(tv_dir, exist_ok=True)
            for vi in range(1, N_VIEWS):
                src_img = os.path.join(src_obj, f"{prefix}{vi:02d}.png")
                if os.path.exists(src_img):
                    img = Image.open(src_img).convert("RGBA")
                    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
                    img.save(os.path.join(tv_dir, f"view_{vi:02d}.png"))

        saved += 1
        if saved % 50 == 0:
            print(f"    Processed {saved}/{len(uids)} objects")

    print(f"  Saved {saved} objects to {split_dir}")
    return saved


def download_triposr(models_dir: str) -> None:
    """Ensure TripoSR model is available (pre-cached in /models or download to models_dir)."""
    # Check if pre-cached in /models (from Dockerfile)
    precached_triposr = "/models/triposr"
    if os.path.exists(os.path.join(precached_triposr, "model.ckpt")):
        print(f"TripoSR found at {precached_triposr} (pre-cached), skipping download.")
        return

    # Otherwise try to download to models_dir
    triposr_dir = os.path.join(models_dir, "triposr")
    marker = os.path.join(triposr_dir, "model.ckpt")
    if os.path.exists(marker):
        print("TripoSR weights already downloaded, skipping.")
        return

    print("Downloading TripoSR weights (~520MB)...")
    os.makedirs(triposr_dir, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="stabilityai/TripoSR",
            local_dir=triposr_dir,
            ignore_patterns=["examples*", "*.md", "*.txt", ".gitattributes"],
        )
        print(f"TripoSR downloaded to {triposr_dir}")
    except Exception as e:
        print(f"Warning: Failed to download TripoSR: {e}")


def main():
    data_dir = os.environ["FARBENCH_DATA_DIR"]
    test_data_dir = os.environ["FARBENCH_TEST_DATA_DIR"]
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(test_data_dir, exist_ok=True)

    # Idempotency check
    migrate_legacy_test_layout(test_data_dir)
    required_dirs = [
        os.path.join(data_dir, "train"),
        os.path.join(data_dir, "val"),
        os.path.join(test_data_dir, "test_gt"),
    ]
    required_files = [
        os.path.join(data_dir, "meta.json"),
        os.path.join(test_data_dir, "meta.json"),
        os.path.join(test_data_dir, "eval_config.json"),
    ]
    if all(os.path.isdir(d) for d in required_dirs):
        try:
            n_train = count_complete_split(os.path.join(data_dir, "train"), "train")
            n_val = count_complete_split(os.path.join(data_dir, "val"), "val")
            n_test = count_complete_split(test_data_dir, "test")
            n_test_gt = count_complete_split(os.path.join(test_data_dir, "test_gt"), "test_gt")
            if (
                n_train >= N_TRAIN
                and n_val >= N_VAL
                and n_test >= N_TEST
                and n_test_gt >= N_TEST
            ):
                if not all(os.path.exists(f) for f in required_files):
                    write_eval_config(test_data_dir)
                print("Objaverse 3D generation data already prepared, skipping.")
                return
        except OSError:
            pass

    # Download pretrained model weights only when we really need to build data.
    # The task Dockerfile pre-caches TripoSR at /models/triposr for normal
    # FARBench image archive build runs, so complete prepared data should not trigger
    # any network access here.
    models_dir = os.path.join(data_dir, "models")
    download_triposr(models_dir)

    raw_dir = os.path.join(data_dir, "_raw")
    os.makedirs(raw_dir, exist_ok=True)

    # ── Step 1: Download and extract Arb-Objaverse shards ─────────
    print("Step 1: Downloading Arb-Objaverse pre-rendered images...")
    extract_dir = os.path.join(raw_dir, "extracted")
    os.makedirs(extract_dir, exist_ok=True)

    # uid -> absolute path to object directory
    uid_to_path: dict[str, str] = {}

    # Check if previous extraction already has enough objects
    if os.path.isdir(extract_dir):
        for shard_dir_name in os.listdir(extract_dir):
            shard_dir = os.path.join(extract_dir, shard_dir_name)
            if not os.path.isdir(shard_dir):
                continue
            for uid in os.listdir(shard_dir):
                obj_path = os.path.join(shard_dir, uid)
                if os.path.isdir(obj_path) and validate_object(obj_path):
                    uid_to_path[uid] = obj_path
        if uid_to_path:
            print(f"  Found {len(uid_to_path)} valid objects from previous extraction")

    for shard_name in ARB_SHARDS:
        if len(uid_to_path) >= N_TOTAL:
            break

        zip_path = os.path.join(raw_dir, shard_name)
        shard_url = f"{ARB_BASE_URL}/{shard_name}"

        # Download if needed
        if not os.path.exists(zip_path) or os.path.getsize(zip_path) < 1_000_000:
            download_file(shard_url, zip_path, timeout=1800)

        # Extract — returns {uid: abs_path}
        shard_uid_paths = extract_objects_from_shard(zip_path, extract_dir)

        # Validate each object
        for uid, obj_path in shard_uid_paths.items():
            if len(uid_to_path) >= N_TOTAL * 2:  # oversample for validation failures
                break
            if validate_object(obj_path):
                uid_to_path[uid] = obj_path

        print(f"  Valid objects so far: {len(uid_to_path)}")

        # Remove zip to save disk space
        os.remove(zip_path)
        print(f"  Removed {shard_name} to save space")

    if len(uid_to_path) < N_TOTAL:
        raise RuntimeError(
            f"Only found {len(uid_to_path)} valid objects, need {N_TOTAL}. "
            "May need to download more shards."
        )

    # ── Step 2: Split into train/val/test ────────────────────────
    print("Step 2: Splitting into train/val/test...")
    all_uids = sorted(uid_to_path.keys())
    rng = np.random.RandomState(SPLIT_SEED)
    perm = rng.permutation(len(all_uids))
    selected = [all_uids[i] for i in perm[:N_TOTAL]]

    train_uids = selected[:N_TRAIN]
    val_uids = selected[N_TRAIN:N_TRAIN + N_VAL]
    test_uids = selected[N_TRAIN + N_VAL:N_TOTAL]

    # ── Step 3: Save splits ──────────────────────────────────────
    print(f"Step 3: Saving train split ({N_TRAIN} objects)...")
    save_split(train_uids, uid_to_path, os.path.join(data_dir, "train"))

    print(f"Step 4: Saving val split ({N_VAL} objects)...")
    save_split(val_uids, uid_to_path, os.path.join(data_dir, "val"))

    print(f"Step 5: Saving test split ({N_TEST} objects)...")
    save_split(
        test_uids, uid_to_path,
        test_data_dir,
        is_test=True,
        gt_dir=os.path.join(test_data_dir, "test_gt"),
    )

    # ── Save metadata ────────────────────────────────────────────
    meta = {
        "n_objects_train": N_TRAIN,
        "n_objects_val": N_VAL,
        "n_objects_test": N_TEST,
        "n_views": N_VIEWS,
        "image_size": IMAGE_SIZE,
        "source": "arb_objaverse",
        "source_resolution": 512,
        "split_seed": SPLIT_SEED,
    }
    for d in [data_dir, test_data_dir]:
        with open(os.path.join(d, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
    write_eval_config(test_data_dir)

    # Save uid lists for reproducibility
    for name, uids in [("train", train_uids), ("val", val_uids), ("test", test_uids)]:
        dest = data_dir if name != "test" else test_data_dir
        with open(os.path.join(dest, f"{name}_uids.json"), "w") as f:
            json.dump(uids, f, indent=2)

    # Clean up extracted raw data
    shutil.rmtree(raw_dir)

    print(f"\nObjaverse 3D generation data ready:")
    print(f"  Train/val: {data_dir}")
    print(f"  Test:      {test_data_dir}")
    print(f"  Train: {N_TRAIN} | Val: {N_VAL} | Test: {N_TEST}")
    print(f"  Views per object: {N_VIEWS} (1 input + {N_VIEWS - 1} targets)")
    print(f"  Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")


if __name__ == "__main__":
    main()
