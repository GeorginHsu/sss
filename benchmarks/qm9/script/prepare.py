"""QM9 data preparation with DFT-optimized 3D molecular geometries.

Downloads the full QM9 dataset including 3D atomic coordinates from DFT
optimization (B3LYP/6-31G(2df,p)).  Provides two data formats:

  - CSV  (train.csv, val.csv)      — SMILES + target  (for 2D / fingerprint methods)
  - .pt  (train_3d.pt, val_3d.pt)  — atomic numbers + 3D positions + target
                                      (for geometric deep learning: SchNet, DimeNet, etc.)

Data sources:
  qm9.zip   — DeepChem / MoleculeNet: contains gdb9.sdf (3D structures)
               and gdb9.sdf.csv (quantum-chemical properties)
  uncharacterized.txt — 3,054 molecules that failed consistency checks
                        (excluded following standard practice)

Split: Random 80/10/10, seed=42.

Output layout:
    FARBENCH_DATA_DIR/
        train.csv       — smiles, target  (HOMO-LUMO gap in eV)
        val.csv         — smiles, target
        train_3d.pt     — [{z, pos, smiles, target}, ...]
        val_3d.pt       — [{z, pos, smiles, target}, ...]
    FARBENCH_TEST_DATA_DIR/
        test.csv        — smiles  (no target)
        test_3d.pt      — [{z, pos, smiles}, ...]  (no target)
        test_labels.csv — target  (evaluator only)
"""

from __future__ import annotations

import csv
import math
import os
import shutil
import urllib.request
import zipfile

import numpy as np
import torch
from rdkit import Chem

# ── URLs ──────────────────────────────────────────────────────────────────────

QM9_ZIP_URL = (
    "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/"
    "molnet_publish/qm9.zip"
)
UNCHAR_URL = "https://ndownloader.figshare.com/files/3195389"

# ── Constants ─────────────────────────────────────────────────────────────────

HAR2EV = 27.211386246  # Hartree → eV (NIST CODATA 2018)

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
SPLIT_SEED = 42


# ── Helpers ───────────────────────────────────────────────────────────────────


def _download(url: str, dest: str) -> None:
    print(f"  Downloading {url} …")
    req = urllib.request.Request(url, headers={"User-Agent": "FARBench/1.0"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        with open(dest, "wb") as f:
            shutil.copyfileobj(resp, f)
    size_mb = os.path.getsize(dest) / 1e6
    print(f"  → {dest}  ({size_mb:.1f} MB)")


def _load_excluded(path: str) -> set[int]:
    """Load 1-based molecule indices to exclude (uncharacterized)."""
    excluded: set[int] = set()
    with open(path, "r", encoding="latin-1") as f:
        lines = f.read().split("\n")
    # Skip header lines (first 9) and trailing blank lines
    for line in lines[9:]:
        line = line.strip()
        if not line:
            continue
        try:
            excluded.add(int(line.split()[0]))
        except (ValueError, IndexError):
            continue
    return excluded


def _parse_csv_targets(csv_path: str) -> list[float]:
    """Read HOMO-LUMO gap (in Hartree) from gdb9.sdf.csv.

    CSV columns: mol_id, A, B, C, mu, alpha, homo, lumo, gap, r2, zpve, …
    Gap is at column index 8 (0-based).
    """
    gaps: list[float] = []
    with open(csv_path, "r") as f:
        lines = f.read().split("\n")
    # Skip header (line 0), skip trailing empty
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        try:
            gap_ha = float(parts[8])  # gap in Hartree
            gaps.append(gap_ha * HAR2EV)
        except (IndexError, ValueError):
            gaps.append(float("nan"))
    return gaps


def _parse_sdf(
    sdf_path: str,
    gaps: list[float],
    excluded: set[int],
) -> list[dict]:
    """Parse gdb9.sdf with RDKit, merge with gap targets.

    Returns list of molecule dicts:
        {z: list[int], pos: list[list[float]], smiles: str, gap_ev: float}
    """
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=True)
    molecules: list[dict] = []
    n_skipped_unchar = 0
    n_skipped_parse = 0

    for i, mol in enumerate(suppl):
        mol_id = i + 1  # 1-based index

        if mol_id in excluded:
            n_skipped_unchar += 1
            continue

        if mol is None:
            n_skipped_parse += 1
            continue

        if i >= len(gaps) or math.isnan(gaps[i]) or math.isinf(gaps[i]):
            n_skipped_parse += 1
            continue

        try:
            conf = mol.GetConformer()
            z = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
            pos = conf.GetPositions().tolist()  # [[x, y, z], …]
            smiles = Chem.MolToSmiles(mol)
        except Exception:
            n_skipped_parse += 1
            continue

        molecules.append(
            {
                "z": z,
                "pos": pos,
                "smiles": smiles,
                "gap_ev": gaps[i],
            }
        )

    print(f"  Parsed {len(molecules)} molecules")
    print(f"    skipped (uncharacterized): {n_skipped_unchar}")
    print(f"    skipped (parse error):     {n_skipped_parse}")

    gap_vals = [m["gap_ev"] for m in molecules]
    print(f"  Gap range: [{min(gap_vals):.4f}, {max(gap_vals):.4f}] eV")
    print(f"  Gap mean:  {np.mean(gap_vals):.4f} eV, std: {np.std(gap_vals):.4f} eV")

    return molecules


# ── I/O ───────────────────────────────────────────────────────────────────────


def _save_csv(
    molecules: list[dict], path: str, *, include_target: bool = True
) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if include_target:
            writer.writerow(["smiles", "target"])
            for mol in molecules:
                writer.writerow([mol["smiles"], mol["gap_ev"]])
        else:
            writer.writerow(["smiles"])
            for mol in molecules:
                writer.writerow([mol["smiles"]])
    print(f"  Saved {path}  ({len(molecules)} molecules)")


def _save_labels(molecules: list[dict], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["target"])
        for mol in molecules:
            writer.writerow([mol["gap_ev"]])
    print(f"  Saved {path}  ({len(molecules)} targets)")


def _save_3d(
    molecules: list[dict], path: str, *, include_target: bool = True
) -> None:
    """Save 3D data as a list of dicts.

    Each dict contains:
      z:      int64 tensor of atomic numbers  [N_atoms]
      pos:    float32 tensor of positions      [N_atoms, 3]  (Å)
      smiles: canonical SMILES string
      target: float HOMO-LUMO gap in eV       (only if include_target)
    """
    data_list = []
    for mol in molecules:
        entry = {
            "z": torch.tensor(mol["z"], dtype=torch.long),
            "pos": torch.tensor(mol["pos"], dtype=torch.float32),
            "smiles": mol["smiles"],
        }
        if include_target:
            entry["target"] = mol["gap_ev"]
        data_list.append(entry)

    torch.save(data_list, path)
    print(f"  Saved {path}  ({len(data_list)} molecules, 3D)")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    data_dir = os.environ["FARBENCH_DATA_DIR"]
    test_data_dir = os.environ["FARBENCH_TEST_DATA_DIR"]
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(test_data_dir, exist_ok=True)

    # Check if already prepared
    required_files = [
        os.path.join(data_dir, "train.csv"),
        os.path.join(data_dir, "val.csv"),
        os.path.join(data_dir, "train_3d.pt"),
        os.path.join(data_dir, "val_3d.pt"),
        os.path.join(test_data_dir, "test.csv"),
        os.path.join(test_data_dir, "test_3d.pt"),
        os.path.join(test_data_dir, "test_labels.csv"),
    ]
    if all(os.path.exists(p) and os.path.getsize(p) > 100 for p in required_files):
        print("QM9 data already prepared, skipping.")
        return

    # ── Download ──────────────────────────────────────────────────────────

    raw_dir = os.path.join(data_dir, "_raw")
    os.makedirs(raw_dir, exist_ok=True)

    zip_path = os.path.join(raw_dir, "qm9.zip")
    if not os.path.exists(zip_path):
        _download(QM9_ZIP_URL, zip_path)

    unchar_path = os.path.join(raw_dir, "uncharacterized.txt")
    if not os.path.exists(unchar_path):
        _download(UNCHAR_URL, unchar_path)

    # ── Extract zip ───────────────────────────────────────────────────────

    sdf_path = os.path.join(raw_dir, "gdb9.sdf")
    csv_path = os.path.join(raw_dir, "gdb9.sdf.csv")
    if not os.path.exists(sdf_path) or not os.path.exists(csv_path):
        print("  Extracting qm9.zip …")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(raw_dir)

    # ── Parse ─────────────────────────────────────────────────────────────

    print("Loading excluded molecule list …")
    excluded = _load_excluded(unchar_path)
    print(f"  {len(excluded)} molecules to exclude")

    print("Parsing properties CSV …")
    gaps = _parse_csv_targets(csv_path)
    print(f"  {len(gaps)} property rows")

    print("Parsing 3D structures from SDF …")
    molecules = _parse_sdf(sdf_path, gaps, excluded)

    # ── Split ─────────────────────────────────────────────────────────────

    rng = np.random.RandomState(SPLIT_SEED)
    indices = list(range(len(molecules)))
    rng.shuffle(indices)

    n = len(molecules)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train = [molecules[i] for i in indices[:n_train]]
    val = [molecules[i] for i in indices[n_train : n_train + n_val]]
    test = [molecules[i] for i in indices[n_train + n_val :]]
    print(f"  Split: train={len(train)}, val={len(val)}, test={len(test)}")

    # ── Save CSV (for 2D / fingerprint methods) ───────────────────────────

    print("\nSaving CSV files (SMILES + target) …")
    _save_csv(train, os.path.join(data_dir, "train.csv"), include_target=True)
    _save_csv(val, os.path.join(data_dir, "val.csv"), include_target=True)
    _save_csv(test, os.path.join(test_data_dir, "test.csv"), include_target=False)
    _save_labels(test, os.path.join(test_data_dir, "test_labels.csv"))

    # ── Save 3D (for geometric deep learning) ─────────────────────────────

    print("\nSaving 3D files (atomic numbers + DFT coordinates) …")
    _save_3d(train, os.path.join(data_dir, "train_3d.pt"), include_target=True)
    _save_3d(val, os.path.join(data_dir, "val_3d.pt"), include_target=True)
    _save_3d(
        test, os.path.join(test_data_dir, "test_3d.pt"), include_target=False
    )

    # ── Cleanup raw files ─────────────────────────────────────────────────

    shutil.rmtree(raw_dir)

    print(f"\nQM9 data ready:")
    print(f"  Train/val (CSV): {data_dir}/train.csv, val.csv")
    print(f"  Train/val (3D):  {data_dir}/train_3d.pt, val_3d.pt")
    print(f"  Test:            {test_data_dir}/")


if __name__ == "__main__":
    main()
