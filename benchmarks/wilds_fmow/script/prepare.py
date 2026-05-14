"""WILDS-FMoW data preparation: download, derive splits, organize.

Split construction matches the official WILDS Python loader
(wilds/datasets/fmow_dataset.py) bit-for-bit so that our train/val/test sets
are identical to the subsets used by WILDS leaderboard submissions (ERM,
FLYP, AutoFT, DRM, VDPG, ...).

The raw CSV "split" column has only 4 literal values {train, val, test, seq},
but each value spans the FULL year range 2002-2017 — the partition between
in-distribution (year<=2012) and OOD (year>=2013) is enforced by the WILDS
loader's `~ood_mask & split_mask` logic, NOT by the CSV authors. So we must
apply a year filter to ALL THREE splits, not just val/test:

  - CSV "train" AND year <= 2012        → WILDS id_train   (~76,863, our train)
  - CSV "train" AND year >= 2013        → DROPPED (would leak into eval window)
  - CSV "val"   AND 2013 <= year <= 2015→ WILDS val        (~19,915, our val, OOD)
  - CSV "val"   AND year <  2013        → WILDS id_val     (dropped, leaderboard does not use)
  - CSV "test"  AND year >= 2016        → WILDS test       (~22,108, our test, OOD)
  - CSV "test"  AND year <  2013        → WILDS id_test    (dropped, leaderboard does not use)
  - CSV "seq"                           → dropped

FARBench exposes only the three splits the leaderboard uses:
  FARBench train = WILDS id_train
  FARBench val   = WILDS OOD val
  FARBench test  = WILDS OOD test

Region is derived by joining country_code with country_code_mapping.csv
(5 regions: Africa / Americas / Asia / Europe / Oceania; unknown → "Other" = -1).

Output layout (final):
    FARBENCH_DATA_DIR/
        train.txt               — "rgb_img_{idx}.png {label}" per line (76,863 lines)
        val.txt                 — same format (19,915 lines)
        region_labels_train.txt — per-sample region ID (0-4 or -1) matching train.txt
        region_labels_val.txt   — per-sample region ID matching val.txt
        class_names.txt         — 62 class names, one per line
        images/                 — 96,778 REAL .png files (train + val combined)
        _raw/                   — kept for idempotent re-runs:
            rgb_metadata.csv             (~30 MB, source of truth for splits)
            country_code_mapping.csv     (small)
            images/                      DELETED at end of prepare.py to reclaim ~140 GB
            fmow_v1.1.tar.gz             DELETED after extraction
    FARBENCH_TEST_DATA_DIR/
        test.txt                — "rgb_img_{idx}.png" per line (22,108 lines, no labels)
        test_labels.txt         — ground truth labels (evaluator only)
        test_regions.txt        — ground truth region IDs (evaluator only)
        images/                 — 22,108 REAL .png files

The images/ dirs contain real files (not symlinks), so each prepared directory
is fully self-contained and portable: tar / move / mount on any host without
re-running prepare.py.
"""

import os
import shutil
import subprocess

import numpy as np
import pandas as pd


# WILDS FMoW download URL (CodaLab bundle, official source from wilds==2.0.0)
DOWNLOAD_URL = "https://worksheets.codalab.org/rest/bundles/0xaec91eb7c9d548ebb15e1b5e60f966ab/contents/blob/"
ARCHIVE_NAME = "fmow_v1.1.tar.gz"

# Region name -> ID mapping (alphabetical order)
REGION_NAMES = ["Africa", "Americas", "Asia", "Europe", "Oceania"]
REGION_TO_ID = {name: idx for idx, name in enumerate(REGION_NAMES)}

# The 62 FMoW category names (official WILDS order, index 0-61)
CATEGORY_NAMES = [
    "airport", "airport_hangar", "airport_terminal", "amusement_park",
    "aquaculture", "archaeological_site", "barn", "border_checkpoint",
    "burial_site", "car_dealership", "construction_site", "crop_field",
    "dam", "debris_or_rubble", "educational_institution", "electric_substation",
    "factory_or_powerplant", "fire_station", "flooded_road", "fountain",
    "gas_station", "golf_course", "ground_transportation_station", "helipad",
    "hospital", "impoverished_settlement", "interchange", "lake_or_pond",
    "lighthouse", "military_facility", "multi-unit_residential",
    "nuclear_powerplant", "office_building", "oil_or_gas_facility", "park",
    "parking_lot_or_garage", "place_of_worship", "police_station", "port",
    "prison", "race_track", "railway_bridge", "recreational_facility",
    "road_bridge", "runway", "shipyard", "shopping_mall",
    "single-unit_residential", "smokestack", "solar_farm", "space_facility",
    "stadium", "storage_tank", "surface_mine", "swimming_pool", "toll_booth",
    "tower", "tunnel_opening", "waste_disposal", "water_treatment_facility",
    "wind_farm", "zoo",
]
CATEGORY_TO_IDX = {name: idx for idx, name in enumerate(CATEGORY_NAMES)}


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
        return True  # already a real file, nothing to do

    if os.path.lexists(dst):
        # Stale symlink (or broken entry) from a prior run — remove
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
    ``expected_filenames``. Returns the number of entries removed.

    Handles re-runs cleanly: if an earlier prepare.py (e.g. one with the
    pre-fix train-split bug) left extra files / symlinks we no longer
    reference, this sweep deletes them so the prepared tree is exactly
    self-consistent with the txt files.
    """
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


def download_file(url: str, dest: str, timeout: int = 7200) -> None:
    """Download a file using curl (preferred) or urllib fallback."""
    print(f"  Downloading {url} ...")
    print(f"  (This is a large file ~50GB, please be patient)")

    # Try curl first — handles SSL and redirects more robustly
    try:
        result = subprocess.run(
            ["curl", "-fSL", "--retry", "3", "--connect-timeout", "60",
             "--max-time", str(timeout), "-o", dest, url],
            capture_output=False,
            timeout=timeout + 60,
        )
        if result.returncode == 0 and os.path.exists(dest) and os.path.getsize(dest) > 0:
            print(f"  Downloaded {os.path.getsize(dest) / 1e9:.1f} GB via curl")
            return
        print(f"  curl failed (exit {result.returncode}), trying urllib fallback ...")
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"  curl not available or timed out ({e}), trying urllib fallback ...")

    # Fallback: urllib with SSL workaround
    import ssl
    import urllib.request

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    req = urllib.request.Request(url, headers={"User-Agent": "FARBench/1.0"})
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            while True:
                chunk = resp.read(8 * 1024 * 1024)  # 8MB chunks
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded / total * 100
                    print(f"\r  Progress: {downloaded / 1e9:.1f} / {total / 1e9:.1f} GB ({pct:.1f}%)",
                          end="", flush=True)
        print()


def extract_tarball(tar_path: str, dest_dir: str) -> None:
    """Extract a tar.gz archive."""
    import tarfile

    print(f"  Extracting {tar_path} (this may take a while) ...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(dest_dir)
    print("  Extraction complete.")


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
        os.path.join(test_data_dir, "test_regions.txt"),
    ]
    if all(os.path.exists(p) and os.path.getsize(p) > 10 for p in required):
        print("WILDS-FMoW data already prepared, skipping.")
        return

    raw_dir = os.path.join(data_dir, "_raw")
    os.makedirs(raw_dir, exist_ok=True)

    # ---------- Find or download+extract data ----------
    # Check if already extracted (skip download entirely if so)
    fmow_root = None
    for candidate in [raw_dir, os.path.join(raw_dir, "fmow_v1.1")]:
        if os.path.exists(os.path.join(candidate, "rgb_metadata.csv")):
            fmow_root = candidate
            break

    if fmow_root is None:
        # Need to download and extract
        archive_path = os.path.join(raw_dir, ARCHIVE_NAME)
        if not os.path.exists(archive_path) or os.path.getsize(archive_path) < 1_000_000_000:
            download_file(DOWNLOAD_URL, archive_path)
        extract_tarball(archive_path, raw_dir)

        # Find where metadata ended up
        for candidate in [raw_dir, os.path.join(raw_dir, "fmow_v1.1")]:
            if os.path.exists(os.path.join(candidate, "rgb_metadata.csv")):
                fmow_root = candidate
                break
        if fmow_root is None:
            # Search recursively
            for root, dirs, files in os.walk(raw_dir):
                if "rgb_metadata.csv" in files:
                    fmow_root = root
                    break
        if fmow_root is None:
            raise RuntimeError(
                f"Could not find rgb_metadata.csv after extraction.\n"
                f"Contents of {raw_dir}: {os.listdir(raw_dir)}"
            )

    metadata_path = os.path.join(fmow_root, "rgb_metadata.csv")

    print(f"  FMoW data root: {fmow_root}")

    # ---------- Load metadata ----------
    print("Loading metadata ...")
    meta = pd.read_csv(metadata_path)
    print(f"  Total images in metadata: {len(meta)}")
    print(f"  Columns: {list(meta.columns)}")

    # ---------- Derive region from country_code ----------
    cc_mapping_path = os.path.join(fmow_root, "country_code_mapping.csv")
    if os.path.exists(cc_mapping_path):
        cc_map = pd.read_csv(cc_mapping_path)
        # Map alpha-3 -> region
        code_to_region = dict(zip(cc_map["alpha-3"], cc_map["region"]))
        meta["region"] = meta["country_code"].map(code_to_region).fillna("Other")
    else:
        print("  WARNING: country_code_mapping.csv not found, region info unavailable")
        meta["region"] = "Other"

    # ---------- Derive year from timestamp ----------
    # Absolute calendar year (used for WILDS OOD split filter, see below).
    meta["timestamp_dt"] = pd.to_datetime(meta["timestamp"], format="ISO8601")
    meta["year_abs"] = meta["timestamp_dt"].dt.year

    # ---------- Map category strings to integer indices ----------
    # The category column may be string names or already integers
    if meta["category"].dtype == object:
        meta["label"] = meta["category"].map(CATEGORY_TO_IDX)
        unmapped = meta["label"].isna().sum()
        if unmapped > 0:
            print(f"  WARNING: {unmapped} rows have unknown category, dropping them")
            meta = meta.dropna(subset=["label"])
        meta["label"] = meta["label"].astype(int)
    else:
        meta["label"] = meta["category"].astype(int)

    # ---------- Map CSV splits to FARBench splits (WILDS OOD protocol) ----------
    # IMPORTANT: the raw CSV "split" column has 4 values {train, val, test, seq}, BUT
    # CSV `split=="train"` rows span the full year range 2002-2017 (363K rows in v1.1),
    # NOT just 2002-2012. The official WILDS loader (wilds/datasets/fmow_dataset.py)
    # applies `~ood_mask & split_mask` to ALL three of train/val/test, where
    # `ood_mask = (year >= 2013)` (= val OOD window 2013-2015 ∪ test OOD window 2016+).
    # We must replicate that year filter on every split — otherwise our train set
    # contains 287K extra year-2013-2017 rows that leak into the OOD evaluation
    # window and break literature comparability.
    #
    #   CSV "train" AND year <= 2012             → FARBench train  ≡ WILDS id_train  (~76,863, years 2002-2012)
    #   CSV "val"   AND 2013 <= year <= 2015     → FARBench val    ≡ WILDS val       (~19,915, OOD 2013-2015)
    #   CSV "test"  AND year >= 2016             → FARBench test   ≡ WILDS test      (~22,108, OOD 2016-2017)
    #   CSV "train" AND year >= 2013             → DROPPED (would leak into eval years)
    #   CSV "val"/"test" AND year < 2013         → id_val / id_test (DROPPED, not on leaderboard)
    #   CSV "seq"                                → DROPPED

    images_root = os.path.join(fmow_root, "images")
    if not os.path.isdir(images_root):
        raise RuntimeError(f"Images directory not found: {images_root}")

    train_img_dir = os.path.join(data_dir, "images")
    test_img_dir = os.path.join(test_data_dir, "images")
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(test_img_dir, exist_ok=True)

    train_entries = []  # (rel_path, label, region_id)
    val_entries = []
    test_entries = []

    print("Processing images ...")
    skipped = 0
    dropped_train_ood_year = 0  # CSV train rows with year>=2013 (would leak into eval window)
    dropped_id_val_test = 0     # CSV val/test rows with year<2013 (id_val / id_test)
    dropped_seq = 0
    # Source lookup chain: prefer the original location in _raw/images/, but on
    # re-runs (after a prior successful prepare moved files out of _raw/) the
    # file may already be staged in the destination dir. _stage_image checks
    # both sites; this keeps the script idempotent across re-runs.
    train_lookup = [images_root, train_img_dir]
    test_lookup = [images_root, test_img_dir]

    for idx, row in meta.iterrows():
        csv_split = row["split"]
        year_abs = int(row["year_abs"])
        label = int(row["label"])
        region_name = row["region"]
        region_id = REGION_TO_ID.get(region_name, -1)

        # WILDS FMoW stores images as rgb_img_{row_index}.png flat in images/
        img_fname = f"rgb_img_{idx}.png"

        if csv_split == "train" and year_abs <= 2012:
            # WILDS id_train (2002-2012). Literature's standard training set.
            if not _stage_image(train_img_dir, img_fname, train_lookup):
                skipped += 1
                continue
            train_entries.append((img_fname, label, region_id))
        elif csv_split == "val" and 2013 <= year_abs <= 2015:
            # WILDS OOD validation (2013-2015). Excludes id_val (year<2013).
            if not _stage_image(train_img_dir, img_fname, train_lookup):
                skipped += 1
                continue
            val_entries.append((img_fname, label, region_id))
        elif csv_split == "test" and year_abs >= 2016:
            # WILDS OOD test (2016-2017). Excludes id_test (year<2013).
            if not _stage_image(test_img_dir, img_fname, test_lookup):
                skipped += 1
                continue
            test_entries.append((img_fname, label, region_id))
        elif csv_split == "train":
            # CSV train rows with year>=2013 would overlap the OOD eval years —
            # WILDS' loader excludes them via ~ood_mask. We do the same.
            dropped_train_ood_year += 1
        elif csv_split in ("val", "test"):
            # id_val / id_test — not part of WILDS OOD protocol, drop to stay
            # apples-to-apples with published leaderboard numbers.
            dropped_id_val_test += 1
        elif csv_split == "seq":
            dropped_seq += 1

    print(f"  Train (id_train, year<=2012): {len(train_entries)}")
    print(f"  Val   (OOD, 2013-2015):       {len(val_entries)}")
    print(f"  Test  (OOD, 2016-2017):       {len(test_entries)}")
    print(f"  Dropped CSV-train year>=2013 (would leak into OOD years): {dropped_train_ood_year}")
    print(f"  Dropped id_val+id_test (year<2013):                       {dropped_id_val_test}")
    print(f"  Dropped seq split:                                        {dropped_seq}")
    if skipped > 0:
        print(f"  Skipped {skipped} images (file not found)")

    # ---------- Save files ----------
    with open(os.path.join(data_dir, "train.txt"), "w") as f:
        for rel_path, label, _ in train_entries:
            f.write(f"{rel_path} {label}\n")

    with open(os.path.join(data_dir, "region_labels_train.txt"), "w") as f:
        for _, _, region_id in train_entries:
            f.write(f"{region_id}\n")

    with open(os.path.join(data_dir, "val.txt"), "w") as f:
        for rel_path, label, _ in val_entries:
            f.write(f"{rel_path} {label}\n")

    with open(os.path.join(data_dir, "region_labels_val.txt"), "w") as f:
        for _, _, region_id in val_entries:
            f.write(f"{region_id}\n")

    with open(os.path.join(data_dir, "class_names.txt"), "w") as f:
        for name in CATEGORY_NAMES:
            f.write(f"{name}\n")

    with open(os.path.join(test_data_dir, "test.txt"), "w") as f:
        for rel_path, _, _ in test_entries:
            f.write(f"{rel_path}\n")

    with open(os.path.join(test_data_dir, "test_labels.txt"), "w") as f:
        for _, label, _ in test_entries:
            f.write(f"{label}\n")

    with open(os.path.join(test_data_dir, "test_regions.txt"), "w") as f:
        for _, _, region_id in test_entries:
            f.write(f"{region_id}\n")

    print(f"  Saved train.txt ({len(train_entries)} lines)")
    print(f"  Saved val.txt ({len(val_entries)} lines)")
    print(f"  Saved class_names.txt ({len(CATEGORY_NAMES)} classes)")
    print(f"  Saved test.txt ({len(test_entries)} lines)")
    print(f"  Saved test_labels.txt ({len(test_entries)} labels)")
    print(f"  Saved test_regions.txt ({len(test_entries)} region IDs)")

    # ---------- Sweep unexpected entries left over from prior runs ----------
    # Re-running an older buggy prepare.py (e.g. one without the year filter,
    # or one that used symlinks) may have left files / symlinks in images/
    # that are no longer referenced by train/val/test.txt. Remove them so the
    # prepared tree is exactly self-consistent.
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
    # Keep only rgb_metadata.csv and country_code_mapping.csv so prepare.py
    # stays cheap to re-run if the user accidentally deletes train/val/test.txt
    # (no need to re-download the 50 GB tar — we rebuild splits from metadata).
    # Everything else in _raw/ is redundant:
    #   - images/ (~140 GB): the 482K source images, of which 119K were moved
    #     to dst dirs and ~363K (CSV-train year>=2013, id_val, id_test, seq)
    #     are dropped per WILDS leaderboard protocol
    #   - fmow_v1.1.tar.gz: the archive itself (if still around)
    #   - any other extraction artifacts shipped by the WILDS bundle
    FMOW_KEEP_RAW_FILES = {"rgb_metadata.csv", "country_code_mapping.csv"}

    purged_count = 0
    purged_bytes = 0
    # Purge inside fmow_root (the dir that holds rgb_metadata.csv)
    if os.path.isdir(fmow_root):
        for name in sorted(os.listdir(fmow_root)):
            if name in FMOW_KEEP_RAW_FILES:
                continue
            path = os.path.join(fmow_root, name)
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

    # If fmow_root is a subdir of raw_dir (e.g., _raw/fmow_v1.1/), also sweep
    # any sibling files/dirs at raw_dir that aren't the fmow_root directory.
    if os.path.realpath(fmow_root) != os.path.realpath(raw_dir):
        fmow_base = os.path.basename(os.path.realpath(fmow_root))
        for name in sorted(os.listdir(raw_dir)):
            if name == fmow_base:
                continue
            path = os.path.join(raw_dir, name)
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    os.remove(path)
                print(f"  Removed stray {path}")
                purged_count += 1
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

    print(f"\nWILDS-FMoW data ready:")
    print(f"  Train/val: {data_dir}            ({len(train_entries) + len(val_entries)} real image files in images/)")
    print(f"  Test:      {test_data_dir}       ({len(test_entries)} real image files in images/)")
    print(f"  Both directories are FULLY SELF-CONTAINED (no symlinks, no cross-")
    print(f"  directory dependency). Each can be tar'd / shipped / mounted on")
    print(f"  another host without re-running prepare.py.")


if __name__ == "__main__":
    main()
