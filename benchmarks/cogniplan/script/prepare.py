"""CogniPlan data preparation: download exploration maps from GitHub Releases.

The CogniPlan dataset contains 2D occupancy grid maps in three categories:
  - room:    indoor room-like environments
  - tunnel:  narrow corridor environments
  - outdoor: open outdoor-like environments

Download source: GitHub Releases of marmotlab/CogniPlan (tag: paper_model_exploration).

Assets used:
  - maps_train.tar.gz  (1.44 MB) — training maps
  - maps_eval.tar.gz   (37.3 KB) — evaluation maps

Output layout:
    FARBENCH_DATA_DIR/
        meta.json
        maps_train/       — training maps (PNG, organized by env type)
    FARBENCH_TEST_DATA_DIR/
        eval_config.json
        maps_eval/        — evaluation maps (PNG, organized by env type)
"""

from __future__ import annotations

import json
import hashlib
import os
import pathlib
import shutil
import tarfile
import urllib.request

GITHUB_RELEASE_TAG = "paper_model_exploration"
GITHUB_REPO = "marmotlab/CogniPlan"
RELEASE_BASE_URL = f"https://github.com/{GITHUB_REPO}/releases/download/{GITHUB_RELEASE_TAG}"

# Assets to download (destination name -> (filename in release, sha256))
ASSETS = {
    "maps_train": (
        "maps_train.tar.gz",
        "28b7227320cfd7794bfe29d9be75a022e39f33ba98283aaac2f877fc5c1b49c5",
    ),
    "maps_eval": (
        "maps_eval.tar.gz",
        "b5f5bfb8fe6ec3aedd857b518c162eb63d4ecc12917af643cb95f3cde580f7e5",
    ),
}

ENV_TYPES = ["room", "tunnel", "outdoor"]
SENSOR_RANGE = 16.0
MAX_STEPS = 128
SUCCESS_THRESHOLD = 0.9999

EVAL_CONFIG = {
    "env_types": ENV_TYPES,
    "sensor_range": SENSOR_RANGE,
    "max_steps": MAX_STEPS,
    "success_threshold": SUCCESS_THRESHOLD,
    "episodes_per_map": 1,  # deterministic start per map
}


def download_file(url: str, dest: str, timeout: int = 600) -> None:
    """Download with progress and User-Agent header."""
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


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_sha256(path: str, expected: str) -> bool:
    actual = sha256_file(path)
    if actual == expected:
        return True
    print(f"  SHA256 mismatch for {os.path.basename(path)}:")
    print(f"    expected: {expected}")
    print(f"    actual:   {actual}")
    return False


def safe_extractall(tf: tarfile.TarFile, dest: str) -> None:
    dest_root = pathlib.Path(dest).resolve()
    for member in tf.getmembers():
        target = (dest_root / member.name).resolve()
        if not target.is_relative_to(dest_root):
            raise RuntimeError(f"Unsafe path in archive: {member.name}")
    tf.extractall(dest)


def count_maps(map_dir: str) -> int:
    """Count PNG files recursively under map_dir."""
    count = 0
    for root, _, files in os.walk(map_dir):
        count += sum(1 for f in files if f.lower().endswith(".png"))
    return count


def find_maps_root(extract_dir: str, name_hint: str) -> str | None:
    """Find the actual maps directory after extraction.

    tar.gz might extract as:
      - maps_train/room/...  (direct)
      - room/...             (no wrapper)
      - some_dir/maps_train/room/...  (nested)
    """
    # Check if extract_dir itself contains env type subdirs
    for env_type in ENV_TYPES:
        if os.path.isdir(os.path.join(extract_dir, env_type)):
            return extract_dir

    # Check if there's a subdirectory matching the name hint
    for d in os.listdir(extract_dir):
        candidate = os.path.join(extract_dir, d)
        if not os.path.isdir(candidate):
            continue
        # Direct match
        if name_hint in d.lower():
            # Check if it contains env type subdirs or PNGs
            for env_type in ENV_TYPES:
                if os.path.isdir(os.path.join(candidate, env_type)):
                    return candidate
            if count_maps(candidate) > 0:
                return candidate
        # Recurse one level
        for sub in os.listdir(candidate):
            sub_path = os.path.join(candidate, sub)
            if os.path.isdir(sub_path):
                for env_type in ENV_TYPES:
                    if os.path.isdir(os.path.join(sub_path, env_type)):
                        return sub_path

    # Fallback: any directory with PNGs
    for d in os.listdir(extract_dir):
        candidate = os.path.join(extract_dir, d)
        if os.path.isdir(candidate) and count_maps(candidate) > 0:
            return candidate

    return None


def main() -> None:
    data_dir = os.environ["FARBENCH_DATA_DIR"]
    test_data_dir = os.environ["FARBENCH_TEST_DATA_DIR"]
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(test_data_dir, exist_ok=True)

    # Idempotency check
    dst_train = os.path.join(data_dir, "maps_train")
    dst_eval = os.path.join(test_data_dir, "maps_eval")
    required = [
        os.path.join(data_dir, "meta.json"),
        dst_train,
        os.path.join(test_data_dir, "eval_config.json"),
        dst_eval,
    ]
    if all(os.path.exists(p) for p in required):
        n_train = count_maps(dst_train)
        n_eval = count_maps(dst_eval)
        if n_train > 0 and n_eval > 0:
            print(f"CogniPlan data already prepared ({n_train} train, {n_eval} eval maps). Skipping.")
            return

    raw_dir = os.path.join(data_dir, "_raw")
    os.makedirs(raw_dir, exist_ok=True)

    # Download and extract each asset
    for asset_name, (filename, expected_sha256) in ASSETS.items():
        tar_path = os.path.join(raw_dir, filename)
        url = f"{RELEASE_BASE_URL}/{filename}"

        # Download
        if (
            not os.path.exists(tar_path)
            or os.path.getsize(tar_path) < 100
            or not verify_sha256(tar_path, expected_sha256)
        ):
            download_file(url, tar_path)
        if not verify_sha256(tar_path, expected_sha256):
            raise RuntimeError(f"Downloaded file failed SHA256 check: {tar_path}")

        # Extract
        extract_subdir = os.path.join(raw_dir, asset_name)
        os.makedirs(extract_subdir, exist_ok=True)
        print(f"  Extracting {filename} ...")
        with tarfile.open(tar_path, "r:gz") as tf:
            safe_extractall(tf, extract_subdir)

        # List what was extracted
        for root, dirs, files in os.walk(extract_subdir):
            depth = root.replace(extract_subdir, "").count(os.sep)
            if depth <= 2:
                indent = "    " + "  " * depth
                print(f"{indent}{os.path.basename(root)}/  ({len(files)} files)")

    # Organize: maps_train -> data_dir/maps_train/
    raw_train_dir = os.path.join(raw_dir, "maps_train")
    maps_root = find_maps_root(raw_train_dir, "train")
    if maps_root:
        if os.path.exists(dst_train):
            shutil.rmtree(dst_train)
        shutil.copytree(maps_root, dst_train)
        print(f"  Train maps copied to {dst_train}")
    else:
        # If no structured dirs found, just copy everything
        if os.path.exists(dst_train):
            shutil.rmtree(dst_train)
        shutil.copytree(raw_train_dir, dst_train)
        print(f"  Train maps copied (flat) to {dst_train}")

    # Organize: maps_eval -> test_data_dir/maps_eval/
    raw_eval_dir = os.path.join(raw_dir, "maps_eval")
    maps_root = find_maps_root(raw_eval_dir, "eval")
    if maps_root:
        if os.path.exists(dst_eval):
            shutil.rmtree(dst_eval)
        shutil.copytree(maps_root, dst_eval)
        print(f"  Eval maps copied to {dst_eval}")
    else:
        if os.path.exists(dst_eval):
            shutil.rmtree(dst_eval)
        shutil.copytree(raw_eval_dir, dst_eval)
        print(f"  Eval maps copied (flat) to {dst_eval}")

    # Count maps
    n_train = count_maps(dst_train)
    n_eval = count_maps(dst_eval)

    if n_train == 0:
        # List what's actually in dst_train for debugging
        print(f"  WARNING: No PNGs found in {dst_train}")
        for root, dirs, files in os.walk(dst_train):
            print(f"    {root}: {files[:5]}")
        raise RuntimeError(f"No training maps found in {dst_train}")
    if n_eval == 0:
        print(f"  WARNING: No PNGs found in {dst_eval}")
        for root, dirs, files in os.walk(dst_eval):
            print(f"    {root}: {files[:5]}")
        raise RuntimeError(f"No evaluation maps found in {dst_eval}")

    # Write meta.json
    meta = {
        "n_maps_train": n_train,
        "n_maps_eval": n_eval,
        "env_types": ENV_TYPES,
        "sensor_range": SENSOR_RANGE,
        "max_steps": MAX_STEPS,
        "success_threshold": SUCCESS_THRESHOLD,
    }
    with open(os.path.join(data_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Write eval_config.json
    EVAL_CONFIG["n_maps_eval"] = n_eval
    with open(os.path.join(test_data_dir, "eval_config.json"), "w") as f:
        json.dump(EVAL_CONFIG, f, indent=2)

    # Clean up raw files
    shutil.rmtree(raw_dir, ignore_errors=True)

    print(f"\nCogniPlan data ready:")
    print(f"  Train maps: {n_train} ({dst_train})")
    print(f"  Eval maps:  {n_eval} ({dst_eval})")


if __name__ == "__main__":
    main()
