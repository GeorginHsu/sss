"""iWildCam-WILDS data preparation: download and organize splits.

Downloads from HuggingFace mirror (primary) or CodaLab (fallback).

Split construction matches the official WILDS Python loader
(wilds/datasets/iwildcam_dataset.py, v2.0.0) bit-for-bit. The raw
metadata.csv's "split" column is a STRING with 5 values; WILDS maps them
to integer IDs via {"train":0, "val":1, "test":2, "id_val":3, "id_test":4}
and applies NO additional filtering (no timestamp / year / location mask,
unlike FMoW). FARBench keeps only split IDs 0/1/2 per leaderboard convention:

  - split "train"   (0, 243 locations): 129,809 images  -> FARBench train
  - split "val"     (1,  32 locations):  14,961 images  -> FARBench val   (OOD)
  - split "test"    (2,  48 locations):  42,791 images  -> FARBench test  (OOD)
  - split "id_val"  (3):  excluded (same locations as train, different days)
  - split "id_test" (4):  excluded (same locations as train, different days)

Train/val/test use completely disjoint sets of camera trap locations for
OOD evaluation. Literature baselines (ERM 30.8, FLYP 46.0, DRM 51.4,
AutoFT 52.0 — all Macro F1) train on exactly this train split and report
on this OOD test split.

Output layout (final):
    FARBENCH_DATA_DIR/
        train.txt                 -- "<filename>.jpg {label}" per line (129,809 lines)
        val.txt                   -- same format (14,961 lines)
        location_labels_train.txt -- per-sample location ID matching train.txt
        location_labels_val.txt   -- per-sample location ID matching val.txt
        class_names.txt           -- 182 species names, one per line
        images/                   -- 144,770 REAL .jpg files (train + val combined)
        _raw/                     -- kept for idempotent re-runs:
            metadata.csv                    (source of truth for splits)
            categories.csv                  (class names)
            train/                          DELETED at end of prepare.py to reclaim disk
            iwildcam_v2.0.tar.gz            DELETED after extraction
    FARBENCH_TEST_DATA_DIR/
        test.txt                  -- "<filename>.jpg" per line (42,791 lines, no labels)
        test_labels.txt           -- ground truth labels (evaluator only)
        images/                   -- 42,791 REAL .jpg files

The images/ dirs contain real files (not symlinks), so each prepared directory
is fully self-contained and portable: tar / move / mount on any host without
re-running prepare.py.
"""

import os
import shutil
import subprocess
import tarfile

import numpy as np
import pandas as pd


def _stage_image(dst_dir: str, fname: str, src_lookup_dirs: list) -> bool:
    """Ensure ``<dst_dir>/<fname>`` is a real file (not a symlink).

    Strategy: if the destination already exists as a real file, do nothing.
    If it exists as a stale symlink (from an older buggy prepare.py), remove
    it. Then look through ``src_lookup_dirs`` in order and ``shutil.move``
    the first matching source into place. Returns True iff the destination
    ends up as a real file.

    Implementation note: we use ``shutil.move`` rather than ``os.rename``
    because FARBench bind-mounts ``data_dir`` and ``test_data_dir`` separately
    into the prepare container, so what looks like one filesystem on the
    host appears as two devices inside the container. ``os.rename`` raises
    ``OSError(EXDEV, "Invalid cross-device link")`` in that case;
    ``shutil.move`` falls back to copy+remove transparently and stays fast
    on the same-device path (train/val both live under data_dir).

    Why move instead of symlink: the prepared train/val and test directories
    become FULLY SELF-CONTAINED — no cross-directory dependency, no reliance
    on ``_raw/`` staying intact, and the trees can be tar'd / shipped /
    re-mounted on any host without re-running prepare.py.
    """
    dst = os.path.join(dst_dir, fname)

    if os.path.isfile(dst) and not os.path.islink(dst):
        return True

    if os.path.lexists(dst):
        try:
            os.remove(dst)
        except OSError:
            return False

    for src_dir in src_lookup_dirs:
        src = os.path.join(src_dir, fname)
        if os.path.isfile(src) and not os.path.islink(src):
            shutil.move(src, dst)
            return True

    return False


def _purge_unexpected_entries(image_dir: str, expected_filenames: set) -> int:
    """Remove any entry in ``image_dir`` whose basename is not in
    ``expected_filenames``. Returns the number removed."""
    if not os.path.isdir(image_dir):
        return 0
    removed = 0
    for name in os.listdir(image_dir):
        if name not in expected_filenames:
            try:
                os.remove(os.path.join(image_dir, name))
                removed += 1
            except OSError:
                pass
    return removed


# HuggingFace mirror (primary) — CDLA-Permissive-1.0 license
HF_REPO_ID = "FARBench/iwildcam-wilds-v2"
HF_FILENAME = "iwildcam_v2.0.tar.gz"

# CodaLab bundle (fallback — may return 500 or have Content-Length: 0 bug)
CODALAB_URL = "https://worksheets.codalab.org/rest/bundles/0x6313da2b204647e79a14b468131fcd64/contents/blob/"
ARCHIVE_NAME = "iwildcam_v2.0.tar.gz"

# Expected archive size ~11.3 GB
EXPECTED_SIZE_MIN = 10_000_000_000


def download_from_huggingface(dest: str) -> bool:
    """Download archive from HuggingFace Hub. Returns True on success."""
    try:
        from huggingface_hub import hf_hub_download

        print(f"  Downloading from HuggingFace: {HF_REPO_ID}/{HF_FILENAME} ...")
        path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_FILENAME,
            repo_type="dataset",
            local_dir=os.path.dirname(dest),
            local_dir_use_symlinks=False,
        )
        # hf_hub_download may place file in a cache dir; ensure it's at dest
        if os.path.abspath(path) != os.path.abspath(dest):
            os.rename(path, dest)
        if os.path.exists(dest) and os.path.getsize(dest) > EXPECTED_SIZE_MIN:
            print(f"  Downloaded {os.path.getsize(dest) / 1e9:.1f} GB from HuggingFace")
            return True
        print("  HuggingFace download too small, file may be corrupted")
        return False
    except Exception as e:
        print(f"  HuggingFace download failed: {e}")
        return False


def download_from_codalab(dest: str, timeout: int = 7200) -> bool:
    """Download archive from CodaLab. Returns True on success."""
    print(f"  Downloading from CodaLab (fallback) ...")
    print(f"  (This is ~11 GB, please be patient)")

    # curl -sL handles CodaLab's Content-Length: 0 bug correctly
    try:
        result = subprocess.run(
            ["curl", "-sL", "--connect-timeout", "60",
             "--max-time", str(timeout), "-o", dest, CODALAB_URL],
            capture_output=False,
            timeout=timeout + 120,
        )
        if result.returncode == 0 and os.path.exists(dest) and os.path.getsize(dest) > EXPECTED_SIZE_MIN:
            print(f"  Downloaded {os.path.getsize(dest) / 1e9:.1f} GB from CodaLab")
            return True
        print(f"  CodaLab curl failed (exit {result.returncode})")
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"  CodaLab download error: {e}")

    # Fallback: urllib (handles some cases curl doesn't)
    try:
        import ssl
        import urllib.request

        print("  Trying urllib fallback ...")
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        req = urllib.request.Request(CODALAB_URL, headers={"User-Agent": "FARBench/1.0"})
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            downloaded = 0
            with open(dest, "wb") as f:
                while True:
                    chunk = resp.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if downloaded % (500 * 1024 * 1024) < 8 * 1024 * 1024:
                        print(f"\r  Progress: {downloaded / 1e9:.1f} GB", end="", flush=True)
            print()
        if os.path.exists(dest) and os.path.getsize(dest) > EXPECTED_SIZE_MIN:
            print(f"  Downloaded {os.path.getsize(dest) / 1e9:.1f} GB via urllib")
            return True
    except Exception as e:
        print(f"  urllib fallback failed: {e}")

    return False


def download_archive(raw_dir: str) -> str:
    """Download and extract iwildcam_v2.0.tar.gz. Returns path to extracted root."""
    # The archive may extract into a subdirectory or flat into raw_dir
    candidates = [os.path.join(raw_dir, "iwildcam_v2.0"), raw_dir]

    # Already extracted?
    for candidate in candidates:
        if os.path.exists(os.path.join(candidate, "metadata.csv")):
            print(f"  Data already extracted at {candidate}")
            return candidate

    archive_path = os.path.join(raw_dir, ARCHIVE_NAME)

    # Download if archive doesn't exist or is too small
    if not os.path.exists(archive_path) or os.path.getsize(archive_path) < EXPECTED_SIZE_MIN:
        print("Downloading iWildCam-WILDS v2.0 (~11 GB) ...")

        # Try CodaLab first (official source), then HuggingFace mirror
        if not download_from_codalab(archive_path):
            if not download_from_huggingface(archive_path):
                raise RuntimeError(
                    "Failed to download iWildCam-WILDS v2.0 from both CodaLab and HuggingFace.\n"
                    f"You can manually download and place the archive at:\n  {archive_path}"
                )

    # Extract
    print(f"  Extracting {archive_path} ...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(raw_dir)
    print("  Extraction complete.")

    # Find where metadata.csv ended up
    for candidate in candidates:
        if os.path.exists(os.path.join(candidate, "metadata.csv")):
            # Clean up archive to save space
            if os.path.exists(archive_path):
                os.remove(archive_path)
                print(f"  Removed archive: {archive_path}")
            return candidate

    raise RuntimeError(
        f"Could not find metadata.csv after extraction.\n"
        f"Contents of {raw_dir}: {os.listdir(raw_dir)}"
    )


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
        print("iWildCam-WILDS data already prepared, skipping.")
        return

    # ---------- Download / locate data ----------
    raw_dir = os.path.join(data_dir, "_raw")
    os.makedirs(raw_dir, exist_ok=True)

    iwc_root = download_archive(raw_dir)
    print(f"  Data root: {iwc_root}")

    # ---------- Load metadata ----------
    metadata_path = os.path.join(iwc_root, "metadata.csv")
    meta = pd.read_csv(metadata_path)
    print(f"  Total images in metadata: {len(meta)}")
    print(f"  Columns: {list(meta.columns)}")

    # ---------- Create image directories ----------
    train_img_dir = os.path.join(data_dir, "images")
    test_img_dir = os.path.join(test_data_dir, "images")
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(test_img_dir, exist_ok=True)

    # ---------- Process splits ----------
    # WILDS splits: 0=train, 1=val(OOD), 2=test(OOD), 3=id_val, 4=id_test
    train_entries = []  # (img_fname, label, location)
    val_entries = []
    test_entries = []

    filenames = meta["filename"].values
    labels = meta["y"].values.astype(int)
    locations = meta["location_remapped"].values.astype(int)
    raw_splits = meta["split"].values

    # split column may be int (from wilds package) or string (from raw CodaLab archive)
    SPLIT_MAP = {"train": 0, "val": 1, "test": 2, "id_val": 3, "id_test": 4}

    # Per WILDS iwildcam_dataset.py:147-156 ("All images are in the train folder"),
    # every image — train, val, and test alike — lives at <iwc_root>/train/<filename>.
    # The "train" prefix here is a directory name, NOT the train split.
    raw_train_dir = os.path.join(iwc_root, "train")
    # Source lookup chain: prefer the original location in _raw/train/, but
    # on re-runs files may have already been moved to the destination dir.
    train_lookup = [raw_train_dir, train_img_dir]
    test_lookup = [raw_train_dir, test_img_dir]

    skipped = 0
    for i in range(len(meta)):
        s = raw_splits[i]
        split_id = SPLIT_MAP[s] if isinstance(s, str) else int(s)
        label = int(labels[i])
        location = int(locations[i])
        filename = str(filenames[i])
        img_fname = os.path.basename(filename)

        if split_id == 0:  # train
            if not _stage_image(train_img_dir, img_fname, train_lookup):
                skipped += 1
                continue
            train_entries.append((img_fname, label, location))
        elif split_id == 1:  # OOD val
            # Val images go to the same images/ dir as train (agent sees both)
            if not _stage_image(train_img_dir, img_fname, train_lookup):
                skipped += 1
                continue
            val_entries.append((img_fname, label, location))
        elif split_id == 2:  # OOD test
            if not _stage_image(test_img_dir, img_fname, test_lookup):
                skipped += 1
                continue
            test_entries.append((img_fname, label, location))
        # Skip id_val (3) and id_test (4)

    print(f"  Train: {len(train_entries)}, Val (OOD): {len(val_entries)}, "
          f"Test (OOD): {len(test_entries)}")
    if skipped > 0:
        print(f"  Skipped {skipped} images (file not found)")

    # ---------- Determine class names ----------
    n_classes = int(labels.max()) + 1
    print(f"  Number of classes: {n_classes}")

    # Try reading categories from the dataset or a categories file
    class_names = None
    categories_path = os.path.join(iwc_root, "categories.csv")
    if os.path.exists(categories_path):
        try:
            cat_df = pd.read_csv(categories_path)
            class_names = [f"species_{i}" for i in range(n_classes)]
            id_col = cat_df.columns[0]
            name_col = cat_df.columns[1] if len(cat_df.columns) > 1 else cat_df.columns[0]
            for _, row in cat_df.iterrows():
                idx = int(row[id_col])
                name = str(row[name_col]).strip()
                if 0 <= idx < n_classes:
                    class_names[idx] = name
            print(f"  Loaded {sum(1 for n in class_names if not n.startswith('species_'))} "
                  f"category names from categories.csv")
        except Exception as e:
            print(f"  Warning: could not parse categories.csv ({e}), using generic names")
            class_names = None

    if class_names is None:
        class_names = [f"species_{i}" for i in range(n_classes)]

    # ---------- Save files ----------
    with open(os.path.join(data_dir, "train.txt"), "w") as f:
        for fname, label, _ in train_entries:
            f.write(f"{fname} {label}\n")

    with open(os.path.join(data_dir, "location_labels_train.txt"), "w") as f:
        for _, _, loc in train_entries:
            f.write(f"{loc}\n")

    with open(os.path.join(data_dir, "val.txt"), "w") as f:
        for fname, label, _ in val_entries:
            f.write(f"{fname} {label}\n")

    with open(os.path.join(data_dir, "location_labels_val.txt"), "w") as f:
        for _, _, loc in val_entries:
            f.write(f"{loc}\n")

    with open(os.path.join(data_dir, "class_names.txt"), "w") as f:
        for name in class_names:
            f.write(f"{name}\n")

    with open(os.path.join(test_data_dir, "test.txt"), "w") as f:
        for fname, _, _ in test_entries:
            f.write(f"{fname}\n")

    with open(os.path.join(test_data_dir, "test_labels.txt"), "w") as f:
        for _, label, _ in test_entries:
            f.write(f"{label}\n")

    # ---------- Sweep unexpected entries left over from prior runs ----------
    train_val_expected = (
        {f for f, _, _ in train_entries} | {f for f, _, _ in val_entries}
    )
    test_expected = {f for f, _, _ in test_entries}
    n_train_val_purged = _purge_unexpected_entries(train_img_dir, train_val_expected)
    n_test_purged = _purge_unexpected_entries(test_img_dir, test_expected)
    if n_train_val_purged or n_test_purged:
        print(
            f"  Cleaned {n_train_val_purged} unexpected entries from train/val images/, "
            f"{n_test_purged} from test images/"
        )

    # ---------- Reclaim disk: whitelist cleanup of _raw/ ----------
    # Keep only metadata.csv and categories.csv so prepare.py stays cheap to
    # re-run if the user accidentally deletes train/val/test.txt (no need to
    # re-download the 11 GB tar — we rebuild splits from metadata). Everything
    # else in _raw/ is redundant:
    #   - train/ (~30 GB): the 217K source images, of which 187K were moved
    #     to dst dirs and ~30K id_val/id_test are dropped per leaderboard
    #   - iwildcam2021_train_annotations_final.json (~85 MB): upstream raw
    #     annotations, already distilled into metadata.csv
    #   - RELEASE_v2.0.txt: extraction artifact
    #   - iwildcam_v2.0/ empty stub dir from interrupted extractions
    #   - iwildcam_v2.0.tar.gz: the archive itself (if still around)
    IWC_KEEP_RAW_FILES = {"metadata.csv", "categories.csv"}

    purged_count = 0
    purged_bytes = 0
    if os.path.isdir(iwc_root):
        for name in sorted(os.listdir(iwc_root)):
            if name in IWC_KEEP_RAW_FILES:
                continue
            path = os.path.join(iwc_root, name)
            try:
                if os.path.isdir(path):
                    dir_bytes = 0
                    for dp, _, files in os.walk(path):
                        for f in files:
                            try:
                                dir_bytes += os.path.getsize(os.path.join(dp, f))
                            except OSError:
                                pass
                    shutil.rmtree(path, ignore_errors=True)
                    if dir_bytes >= 1e6:
                        print(f"  Removed {path}/ (~{dir_bytes / 1e9:.2f} GB)")
                    else:
                        print(f"  Removed {path}/")
                    purged_count += 1
                    purged_bytes += dir_bytes
                elif os.path.isfile(path):
                    sz = os.path.getsize(path)
                    os.remove(path)
                    if sz >= 1e6:
                        print(f"  Removed {path} ({sz / 1e6:.1f} MB)")
                    else:
                        print(f"  Removed {path}")
                    purged_count += 1
                    purged_bytes += sz
            except OSError as e:
                print(f"  Warning: could not remove {path}: {e}")

    # Also clean any tar copies historical bugs may have dropped at data_dir level
    stray_tar = os.path.join(data_dir, ARCHIVE_NAME)
    if os.path.exists(stray_tar):
        try:
            sz = os.path.getsize(stray_tar)
            os.remove(stray_tar)
            purged_count += 1
            purged_bytes += sz
            print(f"  Removed stray archive: {stray_tar} (~{sz / 1e9:.1f} GB)")
        except OSError as e:
            print(f"  Warning: could not remove {stray_tar}: {e}")

    if purged_count > 0:
        print(
            f"  Total reclaimed: ~{purged_bytes / 1e9:.2f} GB ({purged_count} entries)"
        )

    # ---------- Summary ----------
    print(f"\niWildCam-WILDS data ready:")
    print(f"  Train/val: {data_dir}      ({len(train_entries) + len(val_entries)} real image files in images/)")
    print(f"  Test:      {test_data_dir} ({len(test_entries)} real image files in images/)")
    print(f"  Both directories are FULLY SELF-CONTAINED (no symlinks, no cross-")
    print(f"  directory dependency). Each can be tar'd / shipped / mounted on")
    print(f"  another host without re-running prepare.py.")
    print(f"Done. Train: {len(train_entries)}, Val: {len(val_entries)}, "
          f"Test: {len(test_entries)}, Classes: {n_classes}")


if __name__ == "__main__":
    main()
