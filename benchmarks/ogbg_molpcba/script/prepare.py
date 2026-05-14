"""ogbg-molpcba data preparation: download raw OGB CSV files, parse, split.

Does NOT require the ogb Python package — uses only base image packages
(torch, numpy, pandas) so it can run inside farbench/farbench:base-${FARBENCH_CUDA}.

Split strategy: Official OGB scaffold split (Bemis-Murcko scaffolds).
  Train: 350,343 / Val: 43,793 / Test: 43,793 molecules.
  Scaffold splitting ensures test molecules have novel structural frameworks.

Output layout:
    FARBENCH_DATA_DIR/
        train.pt  — 350,343 molecular graphs with 128-task labels
        val.pt    — 43,793 molecular graphs with 128-task labels
    FARBENCH_TEST_DATA_DIR/
        test.pt        — 43,793 molecular graphs (no labels)
        test_labels.pt — ground truth labels (evaluator only)
"""

import gzip
import io
import os
import shutil
import urllib.request
import zipfile

import numpy as np
import torch

DATA_URL = "http://snap.stanford.edu/ogb/data/graphproppred/csv_mol_download/pcba.zip"
DOWNLOAD_NAME = "pcba"
NUM_TASKS = 128


def download_file(url: str, dest: str) -> None:
    print(f"  Downloading {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "FARBench/1.0"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        with open(dest, "wb") as f:
            shutil.copyfileobj(resp, f)


def read_csv_gz(path: str) -> np.ndarray:
    """Read a gzip CSV file (no header) into a numpy array."""
    with gzip.open(path, "rt") as f:
        lines = f.read().strip().split("\n")
    if not lines or lines == [""]:
        return np.array([], dtype=np.int64)
    # Detect number of columns
    first = lines[0].split(",")
    if len(first) == 1:
        return np.array([int(x) for x in lines], dtype=np.int64)
    else:
        return np.loadtxt(io.StringIO("\n".join(lines)), delimiter=",", dtype=np.int64)


def read_labels_csv_gz(path: str) -> np.ndarray:
    """Read graph-label.csv.gz which may contain empty fields (NaN)."""
    with gzip.open(path, "rt") as f:
        lines = f.read().strip().split("\n")
    n = len(lines)
    ncols = len(lines[0].split(","))
    labels = np.full((n, ncols), np.nan, dtype=np.float32)
    for i, line in enumerate(lines):
        parts = line.split(",")
        for j, val in enumerate(parts):
            val = val.strip()
            if val != "":
                labels[i, j] = float(val)
    return labels


def read_split_idx(path: str) -> np.ndarray:
    """Read a split index CSV (gzip, single column, no header)."""
    with gzip.open(path, "rt") as f:
        lines = f.read().strip().split("\n")
    return np.array([int(x) for x in lines], dtype=np.int64)


def parse_all_graphs(raw_dir: str):
    """Parse all raw OGB CSV files into per-graph lists of tensors.

    Returns: (node_features_list, edge_index_list, edge_attr_list, labels_list)
    Each is a list of N_total tensors, one per graph.
    """
    print("  Reading num-node-list...")
    num_node_list = read_csv_gz(os.path.join(raw_dir, "num-node-list.csv.gz"))
    print("  Reading num-edge-list...")
    num_edge_list = read_csv_gz(os.path.join(raw_dir, "num-edge-list.csv.gz"))
    n_graphs = len(num_node_list)

    print(f"  Total graphs: {n_graphs}")
    print(f"  Total nodes: {num_node_list.sum()}, Total edges (directed): {num_edge_list.sum()}")

    print("  Reading node-feat.csv.gz...")
    all_node_feat = read_csv_gz(os.path.join(raw_dir, "node-feat.csv.gz"))

    print("  Reading edge.csv.gz...")
    all_edge = read_csv_gz(os.path.join(raw_dir, "edge.csv.gz"))

    print("  Reading edge-feat.csv.gz...")
    all_edge_feat = read_csv_gz(os.path.join(raw_dir, "edge-feat.csv.gz"))

    print("  Reading graph-label.csv.gz...")
    all_labels = read_labels_csv_gz(os.path.join(raw_dir, "graph-label.csv.gz"))

    # Split into per-graph tensors
    print("  Splitting into per-graph tensors...")
    node_features_list = []
    edge_index_list = []
    edge_attr_list = []
    labels_list = []

    node_offset = 0
    edge_offset = 0

    for i in range(n_graphs):
        nn = int(num_node_list[i])
        ne = int(num_edge_list[i])

        # Node features: [nn, 9]
        nf = torch.from_numpy(all_node_feat[node_offset:node_offset + nn].copy()).to(torch.int64)
        node_features_list.append(nf)

        # Edge index: OGB adds inverse edges (add_inverse_edge=True)
        if ne > 0:
            raw_edges = all_edge[edge_offset:edge_offset + ne]  # [ne, 2]
            raw_ef = all_edge_feat[edge_offset:edge_offset + ne]  # [ne, 3]
            # Add inverse edges: interleave (u,v) and (v,u)
            inv_edges = raw_edges[:, ::-1]  # swap columns
            edges_combined = np.empty((ne * 2, 2), dtype=np.int64)
            edges_combined[0::2] = raw_edges
            edges_combined[1::2] = inv_edges
            ef_combined = np.empty((ne * 2, 3), dtype=np.int64)
            ef_combined[0::2] = raw_ef
            ef_combined[1::2] = raw_ef  # same features for inverse

            ei = torch.from_numpy(edges_combined.T.copy()).to(torch.int64)  # [2, ne*2]
            ea = torch.from_numpy(ef_combined.copy()).to(torch.int64)       # [ne*2, 3]
        else:
            ei = torch.zeros((2, 0), dtype=torch.int64)
            ea = torch.zeros((0, 3), dtype=torch.int64)

        edge_index_list.append(ei)
        edge_attr_list.append(ea)

        # Labels: [128] float32 with NaN
        lab = torch.from_numpy(all_labels[i].copy())  # already float32
        labels_list.append(lab)

        node_offset += nn
        edge_offset += ne

        if (i + 1) % 100000 == 0:
            print(f"    Processed {i + 1}/{n_graphs} graphs...")

    return node_features_list, edge_index_list, edge_attr_list, labels_list


def save_split(indices, node_features, edge_indices, edge_attrs, labels, path,
               include_labels=True):
    """Save a subset of graphs as a .pt file."""
    data = {
        "node_features": [node_features[i] for i in indices],
        "edge_index": [edge_indices[i] for i in indices],
        "edge_attr": [edge_attrs[i] for i in indices],
    }
    if include_labels:
        data["labels"] = [labels[i] for i in indices]
    torch.save(data, path)
    print(f"  Saved {path} ({len(indices)} graphs)")


def main():
    data_dir = os.environ["FARBENCH_DATA_DIR"]
    test_data_dir = os.environ["FARBENCH_TEST_DATA_DIR"]
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(test_data_dir, exist_ok=True)

    # Skip if already prepared
    required = [
        os.path.join(data_dir, "train.pt"),
        os.path.join(data_dir, "val.pt"),
        os.path.join(test_data_dir, "test.pt"),
        os.path.join(test_data_dir, "test_labels.pt"),
    ]
    if all(os.path.exists(p) and os.path.getsize(p) > 1000 for p in required):
        print("ogbg-molpcba data already prepared, skipping.")
        return

    # Download
    raw_dir = os.path.join(data_dir, "_raw")
    os.makedirs(raw_dir, exist_ok=True)
    zip_path = os.path.join(raw_dir, "pcba.zip")

    if not os.path.exists(zip_path) or os.path.getsize(zip_path) < 1_000_000:
        download_file(DATA_URL, zip_path)

    # Extract
    extract_dir = os.path.join(raw_dir, DOWNLOAD_NAME)
    if not os.path.isdir(os.path.join(extract_dir, "raw")):
        print("  Extracting zip...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(raw_dir)

    csv_raw_dir = os.path.join(extract_dir, "raw")
    split_dir = os.path.join(extract_dir, "split", "scaffold")

    # Read split indices
    print("Reading split indices...")
    train_idx = read_split_idx(os.path.join(split_dir, "train.csv.gz"))
    val_idx = read_split_idx(os.path.join(split_dir, "valid.csv.gz"))
    test_idx = read_split_idx(os.path.join(split_dir, "test.csv.gz"))
    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    # Parse all graphs
    print("Parsing all graphs from raw CSV files...")
    node_features, edge_indices, edge_attrs, labels = parse_all_graphs(csv_raw_dir)

    # Save train (with labels)
    print("Saving train split...")
    save_split(train_idx, node_features, edge_indices, edge_attrs, labels,
               os.path.join(data_dir, "train.pt"), include_labels=True)

    # Save val (with labels)
    print("Saving val split...")
    save_split(val_idx, node_features, edge_indices, edge_attrs, labels,
               os.path.join(data_dir, "val.pt"), include_labels=True)

    # Save test features (no labels — agent sees this)
    print("Saving test split...")
    save_split(test_idx, node_features, edge_indices, edge_attrs, labels,
               os.path.join(test_data_dir, "test.pt"), include_labels=False)

    # Save test labels (evaluator only)
    test_labels = {"labels": [labels[i] for i in test_idx]}
    torch.save(test_labels, os.path.join(test_data_dir, "test_labels.pt"))
    print(f"  Saved test_labels.pt ({len(test_idx)} label vectors)")

    # Print label statistics
    test_lab = torch.stack(test_labels["labels"])
    labeled_mask = test_lab == test_lab  # NaN != NaN
    n_labeled = labeled_mask.sum().item()
    n_positive = (test_lab[labeled_mask] == 1).sum().item()
    print(f"  Test label stats: {n_labeled} labeled entries, "
          f"{n_positive} positive ({100 * n_positive / n_labeled:.2f}%)")

    # Clean up
    shutil.rmtree(raw_dir)
    print(f"\nogbg-molpcba data ready:")
    print(f"  Train/val: {data_dir}")
    print(f"  Test:      {test_data_dir}")


if __name__ == "__main__":
    main()
