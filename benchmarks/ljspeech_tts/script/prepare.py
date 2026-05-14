"""LJSpeech-1.1 TTS data preparation: download, split, organize.

Split strategy: Deterministic random shuffle (seed=42) of all 13,100 utterances.
  - Test: first 500 (after shuffle)
  - Val: next 100
  - Train: remaining 12,500
This ensures phonetic/prosodic diversity across all splits.

Output layout:
    FARBENCH_DATA_DIR/
        wavs/            — 12,500 training WAV files (22,050 Hz mono)
        metadata.csv     — id|normalized_text for training
        val_wavs/        — 100 validation WAV files
        val_metadata.csv — id|normalized_text for validation
    FARBENCH_TEST_DATA_DIR/
        test_texts.txt   — id|normalized_text for 500 test utterances (agent-visible)
        reference_wavs/  — 500 ground truth WAV files (evaluator only, for MCD)
"""

from __future__ import annotations

import csv
import os
import random
import shutil
import subprocess
import sys
import tarfile

LJSPEECH_URL = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
SPLIT_SEED = 42
N_TEST = 500
N_VAL = 100
MIN_FILE_SIZE = 1_000_000  # 1 MB — tar.bz2 is ~2.6 GB


def download_file(url: str, dest: str) -> None:
    """Download a file with curl (fallback to wget)."""
    print(f"  Downloading {url} ...")
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


def parse_metadata(metadata_path: str) -> list[tuple[str, str]]:
    """Parse LJSpeech metadata.csv → list of (id, normalized_text).

    Format: id|raw_text|normalized_text (pipe-delimited, no header).
    Some entries may have only 2 columns (raw == normalized).
    """
    entries = []
    with open(metadata_path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|", quoting=csv.QUOTE_NONE)
        for row in reader:
            if len(row) >= 3:
                entries.append((row[0].strip(), row[2].strip()))
            elif len(row) == 2:
                entries.append((row[0].strip(), row[1].strip()))
    return entries


def write_metadata(entries: list[tuple[str, str]], path: str) -> None:
    """Write id|text metadata file."""
    with open(path, "w", encoding="utf-8") as f:
        for uid, text in entries:
            f.write(f"{uid}|{text}\n")


def main() -> None:
    data_dir = os.environ["FARBENCH_DATA_DIR"]
    test_data_dir = os.environ["FARBENCH_TEST_DATA_DIR"]
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(test_data_dir, exist_ok=True)

    # Target directories
    train_wav_dir = os.path.join(data_dir, "wavs")
    val_wav_dir = os.path.join(data_dir, "val_wavs")
    ref_wav_dir = os.path.join(test_data_dir, "reference_wavs")
    train_meta = os.path.join(data_dir, "metadata.csv")
    val_meta = os.path.join(data_dir, "val_metadata.csv")
    test_texts = os.path.join(test_data_dir, "test_texts.txt")

    # Check if already prepared
    required = [train_wav_dir, val_wav_dir, ref_wav_dir]
    required_files = [train_meta, val_meta, test_texts]
    if (all(os.path.isdir(d) and len(os.listdir(d)) > 0 for d in required)
            and all(os.path.isfile(f) and os.path.getsize(f) > 0 for f in required_files)):
        print("Data already prepared, skipping.")
        return

    # Download LJSpeech
    tmp_dir = os.path.join(data_dir, "_tmp_download")
    os.makedirs(tmp_dir, exist_ok=True)
    tar_path = os.path.join(tmp_dir, "LJSpeech-1.1.tar.bz2")

    if not os.path.exists(tar_path) or os.path.getsize(tar_path) < MIN_FILE_SIZE:
        download_file(LJSPEECH_URL, tar_path)

    # Extract
    extract_dir = os.path.join(tmp_dir, "extracted")
    lj_dir = os.path.join(extract_dir, "LJSpeech-1.1")
    if not os.path.isdir(lj_dir):
        print("  Extracting tar.bz2 (this may take a few minutes) ...")
        os.makedirs(extract_dir, exist_ok=True)
        with tarfile.open(tar_path, "r:bz2") as tar:
            tar.extractall(extract_dir)

    # Parse metadata
    orig_meta = os.path.join(lj_dir, "metadata.csv")
    orig_wav_dir = os.path.join(lj_dir, "wavs")
    entries = parse_metadata(orig_meta)
    print(f"  Parsed {len(entries)} utterances from metadata.csv")

    if len(entries) != 13100:
        print(f"  WARNING: Expected 13,100 entries, got {len(entries)}")

    # Deterministic random split
    rng = random.Random(SPLIT_SEED)
    indices = list(range(len(entries)))
    rng.shuffle(indices)

    test_entries = [entries[i] for i in indices[:N_TEST]]
    val_entries = [entries[i] for i in indices[N_TEST:N_TEST + N_VAL]]
    train_entries = [entries[i] for i in indices[N_TEST + N_VAL:]]

    # Create output directories
    os.makedirs(train_wav_dir, exist_ok=True)
    os.makedirs(val_wav_dir, exist_ok=True)
    os.makedirs(ref_wav_dir, exist_ok=True)

    # Copy WAV files to splits
    print("  Copying training WAVs ...")
    for uid, _ in train_entries:
        src = os.path.join(orig_wav_dir, f"{uid}.wav")
        dst = os.path.join(train_wav_dir, f"{uid}.wav")
        if os.path.exists(src):
            shutil.copy2(src, dst)

    print("  Copying validation WAVs ...")
    for uid, _ in val_entries:
        src = os.path.join(orig_wav_dir, f"{uid}.wav")
        dst = os.path.join(val_wav_dir, f"{uid}.wav")
        if os.path.exists(src):
            shutil.copy2(src, dst)

    print("  Copying test reference WAVs ...")
    for uid, _ in test_entries:
        src = os.path.join(orig_wav_dir, f"{uid}.wav")
        dst = os.path.join(ref_wav_dir, f"{uid}.wav")
        if os.path.exists(src):
            shutil.copy2(src, dst)

    # Write metadata files
    write_metadata(train_entries, train_meta)
    write_metadata(val_entries, val_meta)
    write_metadata(test_entries, test_texts)

    # Clean up downloads
    shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"\nDone. Train: {len(train_entries)}, Val: {len(val_entries)}, "
          f"Test: {len(test_entries)}")
    print("All audio at 22,050 Hz mono (original LJSpeech format, no resampling).")


if __name__ == "__main__":
    main()
