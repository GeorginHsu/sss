"""ASVspoof 2021 LA data preparation: download ASVspoof 2019 LA (train/dev) and 2021 LA eval.

Split strategy:
  - Train / Dev: official ASVspoof 2019 LA train and dev sets (no modification).
  - Test: stratified 10% subsample of the full ASVspoof 2021 LA eval set (~181K → ~18K),
    sampled by (label, attack_id) groups with seed=42 to preserve attack-type and
    bonafide/spoof ratio distribution.  The full 181K eval set causes predict.py to
    take 7-10+ minutes per evaluation with wav2vec2 models, making iterative
    development infeasible.  18K keeps eval under 1-2 minutes.

Output layout:
    FARBENCH_DATA_DIR/
        train/flac/          — ASVspoof 2019 LA training audio (.flac, 16 kHz)
        dev/flac/            — ASVspoof 2019 LA development audio
        train_protocol.txt   — train protocol (speaker_id utt_id system_id attack_type label)
        dev_protocol.txt     — dev protocol
    FARBENCH_TEST_DATA_DIR/
        flac/                — ~18,000 subsampled ASVspoof 2021 LA eval audio
        test.txt             — utterance IDs (one per line, defines prediction order)
        test_labels.txt      — ground truth labels for evaluator (utt_id label)
"""

from __future__ import annotations

import collections
import glob
import os
import random
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
import zipfile

# ASVspoof 2019 LA (train + dev + eval audio + protocols)
LA2019_URL = (
    "https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip"
    "?sequence=3&isAllowed=y"
)

# ASVspoof 2021 LA eval audio
LA2021_EVAL_URL = (
    "https://zenodo.org/api/records/4837263/files/"
    "ASVspoof2021_LA_eval.tar.gz/content"
)

# ASVspoof 2021 LA eval keys (labels + metadata)
LA2021_KEYS_URL = "https://www.asvspoof.org/asvspoof2021/LA-keys-full.tar.gz"

# Minimum file counts to verify data integrity
MIN_TRAIN_FLACS = 25000
MIN_DEV_FLACS = 24000
MIN_EVAL_FLACS = 15000  # subsampled from 181K to ~18K

# Eval subsample parameters
EVAL_SAMPLE_SIZE = 18000
EVAL_SAMPLE_SEED = 42


def download_file(url: str, dest: str) -> None:
    """Download a file using wget (robust for large files) with resume support."""
    if os.path.exists(dest):
        local_size = os.path.getsize(dest)
        # Check remote size
        req = urllib.request.Request(url, method="HEAD")
        req.add_header("User-Agent", "FARBench/1.0")
        try:
            resp = urllib.request.urlopen(req, timeout=30)
            remote_size = int(resp.headers.get("Content-Length", 0))
            if remote_size > 0 and local_size >= remote_size:
                print(f"  Already downloaded: {dest} ({local_size:,} bytes)")
                return
        except Exception:
            pass  # Fall through to re-download

    print(f"  Downloading {url}")
    print(f"  -> {dest}")
    ret = subprocess.run(
        ["curl", "-fSL", "-C", "-", "-o", dest, url],
        check=False,
    )
    if ret.returncode != 0:
        raise RuntimeError(f"curl failed (exit {ret.returncode}) for {url}")
    print(f"  Download complete: {dest}")


def extract_zip(archive: str, dest: str) -> None:
    """Extract a ZIP archive with integrity check."""
    print(f"  Extracting {archive} -> {dest}")
    if not zipfile.is_zipfile(archive):
        size = os.path.getsize(archive)
        # Show first bytes to help diagnose (e.g. HTML error page)
        with open(archive, "rb") as f:
            head = f.read(128)
        raise RuntimeError(
            f"{archive} is not a valid zip file "
            f"(size={size:,} bytes, head={head!r})"
        )
    with zipfile.ZipFile(archive, "r") as zf:
        zf.extractall(dest)


def extract_tar(archive: str, dest: str) -> None:
    """Extract a tar.gz archive."""
    print(f"  Extracting {archive} -> {dest}")
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(dest)


def count_flacs(directory: str) -> int:
    """Count .flac files in a directory tree."""
    return len(glob.glob(os.path.join(directory, "**", "*.flac"), recursive=True))


def main() -> None:
    data_dir = os.environ["FARBENCH_DATA_DIR"]
    test_data_dir = os.environ["FARBENCH_TEST_DATA_DIR"]
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(test_data_dir, exist_ok=True)

    # Check if already prepared
    train_flac_dir = os.path.join(data_dir, "train", "flac")
    dev_flac_dir = os.path.join(data_dir, "dev", "flac")
    test_flac_dir = os.path.join(test_data_dir, "flac")
    train_proto = os.path.join(data_dir, "train_protocol.txt")
    dev_proto = os.path.join(data_dir, "dev_protocol.txt")
    test_list = os.path.join(test_data_dir, "test.txt")
    test_labels = os.path.join(test_data_dir, "test_labels.txt")

    all_ready = (
        os.path.isdir(train_flac_dir)
        and count_flacs(train_flac_dir) >= MIN_TRAIN_FLACS
        and os.path.isdir(dev_flac_dir)
        and count_flacs(dev_flac_dir) >= MIN_DEV_FLACS
        and os.path.isfile(train_proto)
        and os.path.isfile(dev_proto)
        and os.path.isdir(test_flac_dir)
        and count_flacs(test_flac_dir) >= MIN_EVAL_FLACS
        and os.path.isfile(test_list)
        and os.path.isfile(test_labels)
    )
    if all_ready:
        print("Data already prepared, skipping.")
        return

    tmp_dir = tempfile.mkdtemp(prefix="asvspoof_prep_")

    try:
        # ── Step 1: ASVspoof 2019 LA (train + dev) ──
        print("\n=== Step 1/3: Downloading ASVspoof 2019 LA ===")
        la2019_zip = os.path.join(tmp_dir, "LA.zip")
        download_file(LA2019_URL, la2019_zip)

        print("\n=== Extracting ASVspoof 2019 LA ===")
        la2019_extract = os.path.join(tmp_dir, "la2019")
        extract_zip(la2019_zip, la2019_extract)

        # Find the extracted LA directory
        la_root = os.path.join(la2019_extract, "LA")
        if not os.path.isdir(la_root):
            # Try to find it
            candidates = glob.glob(os.path.join(la2019_extract, "*", "LA"))
            if candidates:
                la_root = candidates[0]
            else:
                raise FileNotFoundError(
                    f"Cannot find LA directory in {la2019_extract}"
                )

        # Copy protocol files first so we can filter audio by CM protocol IDs.
        # The official LA.zip dev/flac dir contains 142 extra LA_D_A* utterances
        # that belong to the ASV (speaker-verification) protocol, NOT the CM
        # (counter-measure) protocol — they have no bonafide/spoof label and
        # would only confuse a dataloader that enumerates dev/flac directly.
        proto_dir = os.path.join(la_root, "ASVspoof2019_LA_cm_protocols")
        train_proto_src = os.path.join(
            proto_dir, "ASVspoof2019.LA.cm.train.trn.txt"
        )
        dev_proto_src = os.path.join(
            proto_dir, "ASVspoof2019.LA.cm.dev.trl.txt"
        )
        if os.path.isfile(train_proto_src):
            shutil.copy2(train_proto_src, train_proto)
            print(f"  Train protocol: {train_proto}")
        if os.path.isfile(dev_proto_src):
            shutil.copy2(dev_proto_src, dev_proto)
            print(f"  Dev protocol: {dev_proto}")

        def load_proto_ids(path: str) -> set[str]:
            ids: set[str] = set()
            with open(path) as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        ids.add(parts[1])
            return ids

        train_ids = load_proto_ids(train_proto) if os.path.isfile(train_proto) else set()
        dev_ids = load_proto_ids(dev_proto) if os.path.isfile(dev_proto) else set()

        # Copy train audio (filter to CM protocol IDs)
        src_train = os.path.join(la_root, "ASVspoof2019_LA_train", "flac")
        if os.path.isdir(src_train):
            print(f"  Copying train audio: {src_train} -> {train_flac_dir}")
            os.makedirs(train_flac_dir, exist_ok=True)
            for f in os.listdir(src_train):
                if f.endswith(".flac") and os.path.splitext(f)[0] in train_ids:
                    shutil.copy2(os.path.join(src_train, f), train_flac_dir)

        # Copy dev audio (filter to CM protocol IDs — drops 142 ASV-only utts)
        src_dev = os.path.join(la_root, "ASVspoof2019_LA_dev", "flac")
        if os.path.isdir(src_dev):
            print(f"  Copying dev audio: {src_dev} -> {dev_flac_dir}")
            os.makedirs(dev_flac_dir, exist_ok=True)
            for f in os.listdir(src_dev):
                if f.endswith(".flac") and os.path.splitext(f)[0] in dev_ids:
                    shutil.copy2(os.path.join(src_dev, f), dev_flac_dir)

        # Clean up 2019 zip to save space
        os.remove(la2019_zip)
        shutil.rmtree(la2019_extract, ignore_errors=True)

        # ── Step 2: ASVspoof 2021 LA eval audio ──
        print("\n=== Step 2/3: Downloading ASVspoof 2021 LA eval audio ===")
        la2021_tar = os.path.join(tmp_dir, "ASVspoof2021_LA_eval.tar.gz")
        download_file(LA2021_EVAL_URL, la2021_tar)

        print("\n=== Extracting ASVspoof 2021 LA eval audio ===")
        la2021_extract = os.path.join(tmp_dir, "la2021_eval")
        extract_tar(la2021_tar, la2021_extract)

        # Build a lookup of available FLAC files (don't copy yet — subsample first)
        eval_flacs = glob.glob(
            os.path.join(la2021_extract, "**", "*.flac"), recursive=True
        )
        flac_by_id = {}
        for f in eval_flacs:
            fid = os.path.splitext(os.path.basename(f))[0]
            flac_by_id[fid] = f
        print(f"  Found {len(eval_flacs)} eval audio files (will subsample)")

        os.remove(la2021_tar)

        # ── Step 3: ASVspoof 2021 LA eval keys + subsample ──
        print("\n=== Step 3/3: Downloading ASVspoof 2021 LA eval keys ===")
        keys_tar = os.path.join(tmp_dir, "LA-keys-full.tar.gz")
        download_file(LA2021_KEYS_URL, keys_tar)

        print("\n=== Extracting keys ===")
        keys_extract = os.path.join(tmp_dir, "keys")
        extract_tar(keys_tar, keys_extract)

        # Find trial_metadata.txt
        metadata_candidates = glob.glob(
            os.path.join(keys_extract, "**", "trial_metadata.txt"),
            recursive=True,
        )
        # Use the CM metadata (not ASV)
        cm_metadata = None
        for c in metadata_candidates:
            if "CM" in c:
                cm_metadata = c
                break
        if cm_metadata is None and metadata_candidates:
            cm_metadata = metadata_candidates[0]
        if cm_metadata is None:
            raise FileNotFoundError(
                f"Cannot find trial_metadata.txt in {keys_extract}"
            )

        print(f"  Parsing metadata: {cm_metadata}")

        # Parse trial_metadata.txt
        # Format: speaker_id utt_id codec transmission attack_id label trim_flag subset
        all_entries = []  # (utt_id, label, attack_id)
        with open(cm_metadata) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 6:
                    continue
                utt_id = parts[1]
                attack_id = parts[4]
                label = parts[5]  # "bonafide" or "spoof"
                all_entries.append((utt_id, label, attack_id))

        print(f"  Full eval set: {len(all_entries)} utterances")

        # ── Stratified subsample by (label, attack_id) ──
        # Group entries by (label, attack_id) to preserve attack-type distribution
        groups = collections.defaultdict(list)
        for entry in all_entries:
            key = (entry[1], entry[2])  # (label, attack_id)
            groups[key].append(entry)

        rng = random.Random(EVAL_SAMPLE_SEED)
        sample_ratio = EVAL_SAMPLE_SIZE / len(all_entries)
        sampled = []

        for key, entries in sorted(groups.items()):
            n = max(1, round(len(entries) * sample_ratio))
            picked = rng.sample(entries, min(n, len(entries)))
            sampled.extend(picked)

        # Deterministic order
        sampled.sort(key=lambda e: e[0])

        print(f"  Subsampled: {len(sampled)} utterances "
              f"(target {EVAL_SAMPLE_SIZE}, ratio {sample_ratio:.3f})")

        bonafide_count = sum(1 for e in sampled if e[1] == "bonafide")
        spoof_count = sum(1 for e in sampled if e[1] == "spoof")
        print(f"  Bonafide: {bonafide_count}, Spoof: {spoof_count}")

        # Copy only subsampled FLAC files
        os.makedirs(test_flac_dir, exist_ok=True)
        missing_flacs = 0
        for utt_id, _, _ in sampled:
            src = flac_by_id.get(utt_id)
            if src:
                shutil.copy2(src, os.path.join(test_flac_dir, f"{utt_id}.flac"))
            else:
                missing_flacs += 1
        if missing_flacs:
            print(f"  WARNING: {missing_flacs} FLAC files not found in eval archive")

        # Write test.txt (utterance IDs defining prediction order)
        with open(test_list, "w") as f:
            for utt_id, _, _ in sampled:
                f.write(utt_id + "\n")

        # Write test_labels.txt (for evaluator)
        with open(test_labels, "w") as f:
            for utt_id, label, _ in sampled:
                f.write(f"{utt_id} {label}\n")

        # Print attack-type distribution
        attack_dist = collections.Counter(e[2] for e in sampled)
        print(f"  Attack distribution: {dict(sorted(attack_dist.items()))}")

        os.remove(keys_tar)
        shutil.rmtree(keys_extract, ignore_errors=True)
        shutil.rmtree(la2021_extract, ignore_errors=True)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # ── Verify ──
    n_train = count_flacs(train_flac_dir)
    n_dev = count_flacs(dev_flac_dir)
    n_test = count_flacs(test_flac_dir)
    print(f"\n=== Preparation complete ===")
    print(f"  Train audio: {n_train} files")
    print(f"  Dev audio:   {n_dev} files")
    print(f"  Test audio:  {n_test} files")


if __name__ == "__main__":
    main()
