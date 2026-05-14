"""DomainNet Quickdraw data preparation — Domain Generalization setting.

Setting: leave-one-domain-out DG (DomainBed protocol).
  Train/val = 5 source domains (clipart, infograph, painting, real, sketch),
              combined official train+test splits, then 80/20 stratified split.
  Test      = entire Quickdraw domain (train+test combined, all used as test).

Reference: DomainBed (Gulrajani & Lopez-Paz, ICLR 2021) — standard protocol.

IMPORTANT: Uses the CLEANED version of DomainNet-345, which is the version the
cleaned split txt files reference. See IMAGE_URLS below — clipart and painting
require the `/groundtruth/` subpath; the direct `/clipart.zip` and
`/painting.zip` URLs serve incompatible pre-cleaning zips with hash-named files.

Output layout:
    FARBENCH_DATA_DIR/
        train.txt          — "<relative_path> <label>" (source domains, ~330K lines)
        val.txt            — same format (~83K lines)
        domain_labels.txt  — per-sample domain ID for train.txt
                             (0=clipart, 1=infograph, 2=painting, 3=real, 4=sketch)
        class_names.txt    — 345 class names, alphabetical; line i = label i
        images/            — source domain images (<domain>/<class>/<file>.jpg)
                             clipart/, infograph/, painting/, real/, sketch/
    FARBENCH_TEST_DATA_DIR/
        test.txt           — "<relative_path>" only (quickdraw domain, 172,500 lines)
        test_labels.txt    — ground truth labels (evaluator-only)
        images/            — quickdraw/<class>/<key_id>.png  (300×300 RGB PNG)
"""

import collections
import os
import shutil
import urllib.request
import zipfile

import numpy as np

# ── Domain configuration ──────────────────────────────────────────────
SOURCE_DOMAINS = ["clipart", "infograph", "painting", "real", "sketch"]
TARGET_DOMAIN = "quickdraw"
ALL_DOMAINS = SOURCE_DOMAINS + [TARGET_DOMAIN]

DOMAIN_ID = {d: i for i, d in enumerate(SOURCE_DOMAINS)}  # 0-4

# ── URLs ──────────────────────────────────────────────────────────────
BASE_URL = "http://csr.bu.edu/ftp/visda/2019/multi-source"

# Image zip URLs — MUST use the CLEANED version so that the on-disk layout
# matches what the `<domain>_{train,test}.txt` split files reference (which
# is always `<domain>/<class_name>/<file>.jpg`).
#
# For clipart and painting the cleaned zips live under `/groundtruth/`;
# the direct `/clipart.zip` and `/painting.zip` URLs still exist but serve
# the OLD pre-cleaning VisDA 2019 submission zips (hash-named files under
# `<domain>/train/trunkNN/<hash>.jpg`) which are INCOMPATIBLE with the
# cleaned split txt files and will result in 100% missing-path errors at
# training time. We hit exactly this bug on 2026-04-24; the sanity check
# below will now catch it early if the URLs ever drift again.
# Source: http://ai.bu.edu/M3SDA/ — "Download (cleaned version, recommended)".
IMAGE_URLS = {
    "clipart":   f"{BASE_URL}/groundtruth/clipart.zip",    # cleaned
    "infograph": f"{BASE_URL}/infograph.zip",              # direct == cleaned
    "painting":  f"{BASE_URL}/groundtruth/painting.zip",   # cleaned
    "real":      f"{BASE_URL}/real.zip",                   # direct == cleaned
    "sketch":    f"{BASE_URL}/sketch.zip",                 # direct == cleaned
    "quickdraw": f"{BASE_URL}/quickdraw.zip",              # direct == cleaned (0% filter diff)
}

TXT_URL_TEMPLATES = [
    "{base}/domainnet/txt/{domain}_{split}.txt",
    "{base}/txt/{domain}_{split}.txt",
    "{base}/groundtruth/txt/{domain}_{split}.txt",
]

SPLIT_SEED = 42
TRAIN_RATIO = 0.8  # 80% train, 20% val (from source domains)


def _txt_urls(domain: str, split: str) -> list[str]:
    return [t.format(base=BASE_URL, domain=domain, split=split) for t in TXT_URL_TEMPLATES]


def download_file(url: str, dest: str, timeout: int = 1200) -> None:
    print(f"  Downloading {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "FARBench/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        with open(dest, "wb") as f:
            shutil.copyfileobj(resp, f)


def download_with_fallback(urls, dest: str, name: str, timeout: int = 1200) -> None:
    if isinstance(urls, str):
        urls = [urls]
    for url in urls:
        try:
            download_file(url, dest, timeout=timeout)
            return
        except Exception as e:
            print(f"  Failed ({e}), trying next URL...")
    raise RuntimeError(
        f"Failed to download {name} from all URLs.\n"
        f"  Tried: {urls}\n"
        f"  Please download manually and place at: {dest}"
    )


def parse_split_file(path: str) -> list[tuple[str, int]]:
    """Parse a DomainNet split file: 'path label' per line."""
    samples = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.rsplit(" ", 1)
            if len(parts) != 2:
                continue
            rel_path, label = parts[0].strip(), int(parts[1].strip())
            samples.append((rel_path, label))
    return samples


def main():
    data_dir = os.environ["FARBENCH_DATA_DIR"]
    test_data_dir = os.environ["FARBENCH_TEST_DATA_DIR"]
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(test_data_dir, exist_ok=True)

    # ── Idempotency check ─────────────────────────────────────────────
    required = [
        os.path.join(data_dir, "train.txt"),
        os.path.join(data_dir, "val.txt"),
        os.path.join(data_dir, "domain_labels.txt"),
        os.path.join(data_dir, "class_names.txt"),
        os.path.join(test_data_dir, "test.txt"),
        os.path.join(test_data_dir, "test_labels.txt"),
    ]
    if all(os.path.exists(p) and os.path.getsize(p) > 10 for p in required):
        print("DomainNet DG data already prepared, skipping.")
        return

    raw_dir = os.path.join(data_dir, "_raw")
    os.makedirs(raw_dir, exist_ok=True)

    # ── Download split text files for all domains ─────────────────────
    print("=== Downloading split files ===")
    txt_files = {}  # domain -> {"train": path, "test": path}
    for domain in ALL_DOMAINS:
        txt_files[domain] = {}
        for split in ["train", "test"]:
            fname = f"{domain}_{split}.txt"
            dest = os.path.join(raw_dir, fname)
            if not os.path.exists(dest):
                download_with_fallback(_txt_urls(domain, split), dest, fname)
            txt_files[domain][split] = dest

    # ── Download and extract images domain by domain ──────────────────
    # Process one domain at a time to reduce peak disk usage.
    train_img_dir = os.path.join(data_dir, "images")
    test_img_dir = os.path.join(test_data_dir, "images")
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(test_img_dir, exist_ok=True)

    for domain in ALL_DOMAINS:
        # Check if already extracted
        if domain in SOURCE_DOMAINS:
            target_dir = os.path.join(train_img_dir, domain)
        else:
            target_dir = os.path.join(test_img_dir, domain)

        if os.path.isdir(target_dir) and len(os.listdir(target_dir)) > 10:
            print(f"  {domain}/ already extracted, skipping.")
            continue

        zip_path = os.path.join(raw_dir, f"{domain}.zip")
        if not os.path.exists(zip_path):
            print(f"=== Downloading {domain} images ===")
            download_with_fallback(IMAGE_URLS[domain], zip_path, f"{domain}.zip", timeout=1800)

        print(f"=== Extracting {domain} images ===")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(raw_dir)

        # Move extracted domain folder to the right place
        extracted = os.path.join(raw_dir, domain)
        if os.path.isdir(extracted):
            if os.path.exists(target_dir):
                shutil.rmtree(target_dir)
            shutil.move(extracted, target_dir)

        # ── Sanity check: verify the extracted structure matches what the
        # cleaned split txt files reference (<domain>/<class_name>/<file>.jpg).
        # The OLD/uncleaned VisDA 2019 zips extract to <domain>/train/trunkNN/
        # <hash>.jpg which is incompatible — catch that here instead of letting
        # every downstream train.txt lookup silently FileNotFoundError.
        sample_txt = txt_files[domain]["train"]
        with open(sample_txt) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rel_path = line.rsplit(" ", 1)[0]
                break
        expected = os.path.join(target_dir, os.path.relpath(rel_path, domain))
        if not os.path.exists(expected):
            # Diagnose: is it the trunk-structured old zip?
            trunk_markers = [
                os.path.join(target_dir, "train", "trunk00"),
                os.path.join(target_dir, "test", "trunk00"),
            ]
            has_trunk = any(os.path.isdir(t) for t in trunk_markers)
            hint = (
                "\n  This looks like the OLD pre-cleaning VisDA 2019 zip (hash-named "
                "files under <domain>/train/trunkNN/)."
                "\n  For clipart and painting, make sure IMAGE_URLS points to the "
                "`/groundtruth/` subpath. For other domains, double-check the URL."
                if has_trunk else
                "\n  Extracted structure is unrecognized. Inspect {target_dir} manually."
            )
            raise RuntimeError(
                f"Structure mismatch for domain '{domain}': split file references "
                f"'{rel_path}' but '{expected}' does not exist.{hint}"
            )

        # Remove zip to free disk space
        if os.path.exists(zip_path):
            os.remove(zip_path)
            print(f"  Removed {domain}.zip to save space.")

    # ── Parse all split files ─────────────────────────────────────────
    print("\n=== Parsing split files ===")
    source_samples = []  # list of (rel_path, label, domain_name)
    for domain in SOURCE_DOMAINS:
        train_s = parse_split_file(txt_files[domain]["train"])
        test_s = parse_split_file(txt_files[domain]["test"])
        combined = train_s + test_s
        for rel_path, label in combined:
            source_samples.append((rel_path, label, domain))
        print(f"  {domain}: {len(train_s)} train + {len(test_s)} test = {len(combined)} total")

    target_train = parse_split_file(txt_files[TARGET_DOMAIN]["train"])
    target_test = parse_split_file(txt_files[TARGET_DOMAIN]["test"])
    target_samples = target_train + target_test
    print(f"  {TARGET_DOMAIN} (target): {len(target_train)} + {len(target_test)} = {len(target_samples)} total")

    # ── Derive class names from directory structure ───────────────────
    all_labels = set(l for _, l, _ in source_samples) | set(l for _, l in target_samples)
    num_classes = max(all_labels) + 1
    print(f"\n  Number of classes: {num_classes}")

    class_names = [""] * num_classes
    for rel_path, label, _ in source_samples:
        if not class_names[label]:
            parts = rel_path.replace("\\", "/").split("/")
            if len(parts) >= 2:
                class_names[label] = parts[1]
    for i in range(num_classes):
        if not class_names[i]:
            class_names[i] = f"class_{i}"

    # ── Stratified 80/20 train/val split on source domains ────────────
    print("\n=== Splitting source domains into train/val ===")
    rng = np.random.RandomState(SPLIT_SEED)

    # Group by (domain, class) for stratified split
    by_domain_class = collections.defaultdict(list)
    for sample in source_samples:
        rel_path, label, domain = sample
        by_domain_class[(domain, label)].append(sample)

    train_split, val_split = [], []
    for key in sorted(by_domain_class.keys()):
        group = by_domain_class[key]
        indices = list(range(len(group)))
        rng.shuffle(indices)
        n_train = max(1, int(len(indices) * TRAIN_RATIO))
        train_split.extend(group[indices[i]] for i in range(n_train))
        val_split.extend(group[indices[i]] for i in range(n_train, len(indices)))

    rng.shuffle(train_split)
    rng.shuffle(val_split)

    print(f"  Source train: {len(train_split)}")
    print(f"  Source val:   {len(val_split)}")
    print(f"  Target test:  {len(target_samples)}")

    # ── Write train.txt + domain_labels.txt ───────────────────────────
    with open(os.path.join(data_dir, "train.txt"), "w") as f_txt, \
         open(os.path.join(data_dir, "domain_labels.txt"), "w") as f_dom:
        for rel_path, label, domain in train_split:
            f_txt.write(f"{rel_path} {label}\n")
            f_dom.write(f"{DOMAIN_ID[domain]}\n")
    print(f"  Saved train.txt + domain_labels.txt ({len(train_split)} lines)")

    # ── Write val.txt ─────────────────────────────────────────────────
    with open(os.path.join(data_dir, "val.txt"), "w") as f:
        for rel_path, label, _ in val_split:
            f.write(f"{rel_path} {label}\n")
    print(f"  Saved val.txt ({len(val_split)} lines)")

    # ── Write class_names.txt ─────────────────────────────────────────
    with open(os.path.join(data_dir, "class_names.txt"), "w") as f:
        for name in class_names:
            f.write(f"{name}\n")
    print(f"  Saved class_names.txt ({num_classes} classes)")

    # ── Write test.txt (no labels) ────────────────────────────────────
    with open(os.path.join(test_data_dir, "test.txt"), "w") as f:
        for rel_path, _ in target_samples:
            f.write(f"{rel_path}\n")
    print(f"  Saved test.txt ({len(target_samples)} lines)")

    # ── Write test_labels.txt ─────────────────────────────────────────
    with open(os.path.join(test_data_dir, "test_labels.txt"), "w") as f:
        for _, label in target_samples:
            f.write(f"{label}\n")
    print(f"  Saved test_labels.txt ({len(target_samples)} labels)")

    # ── Cleanup raw split files ───────────────────────────────────────
    shutil.rmtree(raw_dir, ignore_errors=True)

    print(f"\n{'='*50}")
    print(f"DomainNet DG data ready (Quickdraw = target domain):")
    print(f"  Train/val (5 source domains): {data_dir}")
    print(f"  Test (Quickdraw):             {test_data_dir}")
    print(f"  Train: {len(train_split)}, Val: {len(val_split)}, Test: {len(target_samples)}")


if __name__ == "__main__":
    main()
