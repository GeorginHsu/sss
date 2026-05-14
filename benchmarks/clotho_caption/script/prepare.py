"""Clotho v2.1 audio captioning data preparation.

Downloads Clotho v2.1 from Zenodo, extracts .7z audio archives,
organizes into train/val/test splits following the official split.

Split strategy: Official Clotho v2.1 splits (development / validation / evaluation).
  - Train: development split (3,839 clips, 19,195 captions)
  - Val: validation split (1,045 clips, 5,225 captions)
  - Test: evaluation split (1,045 clips, 5,225 captions)

Output layout:
    FARBENCH_DATA_DIR/
        audio/           — 3,839 training WAV files (44,100 Hz mono)
        captions.csv     — file_name,caption_1,...,caption_5 for training
        val_audio/       — 1,045 validation WAV files
        val_captions.csv — same format for validation
    FARBENCH_TEST_DATA_DIR/
        audio/               — 1,045 evaluation WAV files (agent-visible)
        test_files.txt       — one filename per line (defines evaluation order)
        reference_captions.csv — 5 reference captions per file (evaluator only)
"""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
import sys

ZENODO_BASE = "https://zenodo.org/api/records/4783391/files"

FILES = {
    "audio_dev": f"{ZENODO_BASE}/clotho_audio_development.7z/content",
    "audio_val": f"{ZENODO_BASE}/clotho_audio_validation.7z/content",
    "audio_eval": f"{ZENODO_BASE}/clotho_audio_evaluation.7z/content",
    "captions_dev": f"{ZENODO_BASE}/clotho_captions_development.csv/content",
    "captions_val": f"{ZENODO_BASE}/clotho_captions_validation.csv/content",
    "captions_eval": f"{ZENODO_BASE}/clotho_captions_evaluation.csv/content",
}

# Minimum file sizes (bytes) for integrity check
MIN_SIZES = {
    "audio_dev": 100_000_000,   # ~4.5 GB
    "audio_val": 100_000_000,   # ~1.2 GB
    "audio_eval": 100_000_000,  # ~1.2 GB
    "captions_dev": 100_000,    # ~1.3 MB
    "captions_val": 10_000,     # ~368 KB
    "captions_eval": 10_000,    # ~362 KB
}


def download_file(url: str, dest: str) -> None:
    """Download a file with curl (fallback to wget)."""
    print(f"  Downloading {os.path.basename(dest)} ...")
    if shutil.which("curl"):
        subprocess.check_call(
            ["curl", "-fSL", "-o", dest, url],
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
    else:
        subprocess.check_call(
            ["wget", "-q", "--show-progress", "-O", dest, url],
            stdout=sys.stdout,
            stderr=sys.stderr,
        )


def extract_7z(archive_path: str, out_dir: str) -> None:
    """Extract .7z archive using py7zr."""
    import py7zr
    import stat

    print(f"  Extracting {os.path.basename(archive_path)} ...")
    os.makedirs(out_dir, exist_ok=True)
    with py7zr.SevenZipFile(archive_path, mode="r") as z:
        z.extractall(path=out_dir)
    # py7zr preserves archive permissions which may be too restrictive;
    # ensure all extracted dirs/files are world-readable
    for root, dirs, files in os.walk(out_dir):
        for d in dirs:
            p = os.path.join(root, d)
            os.chmod(p, os.stat(p).st_mode | stat.S_IROTH | stat.S_IXOTH | stat.S_IRGRP | stat.S_IXGRP)
        for f in files:
            p = os.path.join(root, f)
            os.chmod(p, os.stat(p).st_mode | stat.S_IROTH | stat.S_IRGRP)


def count_wavs(directory: str) -> int:
    """Count WAV files in a directory."""
    if not os.path.isdir(directory):
        return 0
    return sum(1 for f in os.listdir(directory) if f.lower().endswith(".wav"))


def main() -> None:
    data_dir = os.environ["FARBENCH_DATA_DIR"]
    test_data_dir = os.environ["FARBENCH_TEST_DATA_DIR"]
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(test_data_dir, exist_ok=True)

    # Target directories
    train_audio = os.path.join(data_dir, "audio")
    val_audio = os.path.join(data_dir, "val_audio")
    test_audio = os.path.join(test_data_dir, "audio")
    train_captions = os.path.join(data_dir, "captions.csv")
    val_captions = os.path.join(data_dir, "val_captions.csv")
    test_files_txt = os.path.join(test_data_dir, "test_files.txt")
    ref_captions = os.path.join(test_data_dir, "reference_captions.csv")

    # Check if already prepared
    required_dirs = [train_audio, val_audio, test_audio]
    required_files = [train_captions, val_captions, test_files_txt, ref_captions]
    if (all(os.path.isdir(d) and count_wavs(d) > 0 for d in required_dirs)
            and all(os.path.isfile(f) and os.path.getsize(f) > 0 for f in required_files)):
        print("Data already prepared, skipping.")
        return

    tmp_dir = os.path.join(data_dir, "_tmp_download")
    os.makedirs(tmp_dir, exist_ok=True)

    # Download all files
    local_paths = {}
    for key, url in FILES.items():
        ext = ".7z" if "audio" in key else ".csv"
        dest = os.path.join(tmp_dir, f"{key}{ext}")
        min_size = MIN_SIZES.get(key, 1000)
        if not os.path.exists(dest) or os.path.getsize(dest) < min_size:
            download_file(url, dest)
        local_paths[key] = dest

    # Extract audio archives
    print("Extracting development audio (this may take several minutes) ...")
    extract_7z(local_paths["audio_dev"], train_audio)
    # 7z may extract into a subdirectory — flatten if needed
    _flatten_subdir(train_audio, "development", "clotho_audio_development")

    print("Extracting validation audio ...")
    extract_7z(local_paths["audio_val"], val_audio)
    _flatten_subdir(val_audio, "validation", "clotho_audio_validation")

    print("Extracting evaluation audio ...")
    extract_7z(local_paths["audio_eval"], test_audio)
    _flatten_subdir(test_audio, "evaluation", "clotho_audio_evaluation")

    # Copy caption CSVs
    shutil.copy2(local_paths["captions_dev"], train_captions)
    shutil.copy2(local_paths["captions_val"], val_captions)
    shutil.copy2(local_paths["captions_eval"], ref_captions)

    # Verify extraction produced expected file counts
    for name, directory, expected_min in [
        ("train", train_audio, 3800),
        ("val", val_audio, 1000),
        ("test", test_audio, 1000),
    ]:
        n = count_wavs(directory)
        if n < expected_min:
            raise RuntimeError(
                f"Expected ≥{expected_min} WAV files in {name} ({directory}), "
                f"found {n}. Extraction may have failed."
            )

    # Write test_files.txt (evaluation order)
    test_wavs = sorted(f for f in os.listdir(test_audio) if f.lower().endswith(".wav"))
    with open(test_files_txt, "w") as f:
        for fname in test_wavs:
            f.write(fname + "\n")

    # Clean up downloads
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # Print summary
    n_train = count_wavs(train_audio)
    n_val = count_wavs(val_audio)
    n_test = count_wavs(test_audio)
    print(f"\nDone. Train: {n_train} clips, Val: {n_val} clips, Test: {n_test} clips")
    print("All audio at 44,100 Hz mono (original Clotho format).")


def _flatten_subdir(parent: str, *candidate_names: str) -> None:
    """If extraction created a subdirectory, move its contents up.

    Tries each candidate name so we handle varying archive layouts.
    """
    for subdir_name in candidate_names:
        subdir = os.path.join(parent, subdir_name)
        if os.path.isdir(subdir):
            for fname in os.listdir(subdir):
                src = os.path.join(subdir, fname)
                dst = os.path.join(parent, fname)
                if not os.path.exists(dst):
                    shutil.move(src, dst)
            shutil.rmtree(subdir, ignore_errors=True)
            return


if __name__ == "__main__":
    main()
