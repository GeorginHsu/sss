"""VoiceBank+DEMAND speech enhancement data preparation.

Downloads the 16 kHz version from Edinburgh DataShare (or HuggingFace mirror),
organizes into train/val/test splits.

Following common practice (e.g. MetricGAN, MetricGAN+), 10% of training
utterances are randomly held out as a validation set (utterance-level split,
fixed seed for reproducibility).

Output layout:
    FARBENCH_DATA_DIR/
        clean/          — ~10,415 clean training WAV files (16 kHz mono)
        noisy/          — ~10,415 noisy training WAV files (same filenames)
        val_clean/      — ~1,157 clean validation WAV files
        val_noisy/      — ~1,157 noisy validation WAV files
    FARBENCH_TEST_DATA_DIR/
        noisy/          — 824 noisy test WAV files (agent-visible)
        clean/          — 824 clean test reference WAV files (evaluator only)
        test_files.txt  — one filename per line, defines evaluation order
"""

from __future__ import annotations

import os
import random
import shutil
import subprocess
import sys
import zipfile

VAL_RATIO = 0.1
VAL_SEED = 42

# Edinburgh DataShare URLs (48 kHz original)
EDINBURGH_URLS = {
    "clean_trainset": "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/clean_trainset_28spk_wav.zip",
    "noisy_trainset": "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/noisy_trainset_28spk_wav.zip",
    "clean_testset": "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/clean_testset_wav.zip",
    "noisy_testset": "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/noisy_testset_wav.zip",
}

SAMPLE_RATE = 16000
MIN_FILE_SIZE = 1000  # bytes


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


def resample_wav(src: str, dst: str, target_sr: int = 16000) -> None:
    """Resample a WAV file to target sample rate using ffmpeg."""
    subprocess.check_call(
        ["ffmpeg", "-y", "-i", src, "-ar", str(target_sr), "-ac", "1", dst],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def extract_and_resample(
    zip_path: str, extract_dir: str, out_dir: str, target_sr: int = 16000
) -> int:
    """Extract ZIP, resample all WAVs to target_sr, save to out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(extract_dir, exist_ok=True)

    print(f"  Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    # Find all WAV files in extracted directory
    wav_files = []
    for root, _, files in os.walk(extract_dir):
        for f in sorted(files):
            if f.lower().endswith(".wav"):
                wav_files.append(os.path.join(root, f))

    print(f"  Resampling {len(wav_files)} files to {target_sr} Hz ...")
    count = 0
    for wav_path in wav_files:
        fname = os.path.basename(wav_path)
        dst = os.path.join(out_dir, fname)
        resample_wav(wav_path, dst, target_sr)
        count += 1

    # Clean up extracted raw files
    shutil.rmtree(extract_dir, ignore_errors=True)
    return count


def main() -> None:
    data_dir = os.environ["FARBENCH_DATA_DIR"]
    test_data_dir = os.environ["FARBENCH_TEST_DATA_DIR"]
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(test_data_dir, exist_ok=True)

    # Check if already prepared
    train_clean_dir = os.path.join(data_dir, "clean")
    train_noisy_dir = os.path.join(data_dir, "noisy")
    val_clean_dir = os.path.join(data_dir, "val_clean")
    val_noisy_dir = os.path.join(data_dir, "val_noisy")
    test_noisy_dir = os.path.join(test_data_dir, "noisy")
    test_clean_dir = os.path.join(test_data_dir, "clean")
    test_files_txt = os.path.join(test_data_dir, "test_files.txt")

    required_dirs = [train_clean_dir, train_noisy_dir, val_clean_dir, val_noisy_dir,
                     test_noisy_dir, test_clean_dir]
    if all(os.path.isdir(d) and len(os.listdir(d)) > 0 for d in required_dirs):
        if os.path.exists(test_files_txt) and os.path.getsize(test_files_txt) > 0:
            print("Data already prepared, skipping.")
            return

    tmp_dir = os.path.join(data_dir, "_tmp_download")
    os.makedirs(tmp_dir, exist_ok=True)

    # Download all ZIPs
    zip_paths = {}
    for key, url in EDINBURGH_URLS.items():
        zip_path = os.path.join(tmp_dir, f"{key}.zip")
        if not os.path.exists(zip_path) or os.path.getsize(zip_path) < 1_000_000:
            download_file(url, zip_path)
        zip_paths[key] = zip_path

    # Extract and resample each split
    print("Processing training clean set...")
    extract_and_resample(
        zip_paths["clean_trainset"],
        os.path.join(tmp_dir, "raw_clean_train"),
        train_clean_dir,
        SAMPLE_RATE,
    )

    print("Processing training noisy set...")
    extract_and_resample(
        zip_paths["noisy_trainset"],
        os.path.join(tmp_dir, "raw_noisy_train"),
        train_noisy_dir,
        SAMPLE_RATE,
    )

    print("Processing test clean set...")
    extract_and_resample(
        zip_paths["clean_testset"],
        os.path.join(tmp_dir, "raw_clean_test"),
        test_clean_dir,
        SAMPLE_RATE,
    )

    print("Processing test noisy set...")
    extract_and_resample(
        zip_paths["noisy_testset"],
        os.path.join(tmp_dir, "raw_noisy_test"),
        test_noisy_dir,
        SAMPLE_RATE,
    )

    # Split training set into train/val (utterance-level, fixed seed)
    print("Splitting training set into train/val ...")
    os.makedirs(val_clean_dir, exist_ok=True)
    os.makedirs(val_noisy_dir, exist_ok=True)

    all_train_wavs = sorted(f for f in os.listdir(train_clean_dir) if f.endswith(".wav"))
    rng = random.Random(VAL_SEED)
    rng.shuffle(all_train_wavs)
    n_val = int(len(all_train_wavs) * VAL_RATIO)
    val_files = set(all_train_wavs[:n_val])

    for fname in val_files:
        shutil.move(os.path.join(train_clean_dir, fname), os.path.join(val_clean_dir, fname))
        shutil.move(os.path.join(train_noisy_dir, fname), os.path.join(val_noisy_dir, fname))

    # Write test_files.txt
    test_wavs = sorted(f for f in os.listdir(test_noisy_dir) if f.endswith(".wav"))
    with open(test_files_txt, "w") as f:
        for fname in test_wavs:
            f.write(fname + "\n")

    # Clean up downloads
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # Print summary
    n_train = len([f for f in os.listdir(train_clean_dir) if f.endswith(".wav")])
    n_val_actual = len([f for f in os.listdir(val_clean_dir) if f.endswith(".wav")])
    n_test = len(test_wavs)
    print(f"\nDone. Train: {n_train} pairs, Val: {n_val_actual} pairs, Test: {n_test} pairs.")
    print(f"All audio resampled to {SAMPLE_RATE} Hz mono.")


if __name__ == "__main__":
    main()
