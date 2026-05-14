"""METR-LA traffic speed prediction data preparation: download, split, build test windows.

Split strategy: Chronological 70/10/20 split (DCRNN convention, Li et al., ICLR 2018).
  Total: 34,272 timesteps (5-min intervals, Mar 1 - Jun 30, 2012, 207 sensors).
  Train: first 70% (~23,974 timesteps), Val: next 10% (~3,425), Test: last 20% (~6,873).

Output layout:
    FARBENCH_DATA_DIR/
        train.npy     — float32 [N_train, 207] raw speed values (mph)
        val.npy       — float32 [N_val, 207]
        adj_mx.npy    — float32 [207, 207] weighted adjacency matrix
    FARBENCH_TEST_DATA_DIR/
        test_x.npy    — float32 [K, 12, 207] pre-computed input windows
        test_y.npy    — float32 [K, 12, 207] ground truth targets (evaluator only)
        adj_mx.npy    — float32 [207, 207] same adjacency (for agent convenience)
"""

import os
import pickle
import shutil
import urllib.request

import numpy as np
import pandas as pd

# METR-LA data from DCRNN (Google Drive)
METRLA_GDRIVE_ID = "1pAGRfzMx6K9WWsfDcD1NMbIif0T0saFC"
DISTANCES_URL = (
    "https://raw.githubusercontent.com/liyaguang/DCRNN/master/"
    "data/sensor_graph/distances_la_2012.csv"
)
SENSOR_IDS_URL = (
    "https://raw.githubusercontent.com/liyaguang/DCRNN/master/"
    "data/sensor_graph/graph_sensor_ids.txt"
)

# Split ratios (chronological)
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1

# Sliding window parameters
SEQ_LEN = 12   # 1 hour input (12 × 5 min)
PRED_LEN = 12  # 1 hour prediction

# Adjacency matrix parameters (DCRNN standard)
NORMALIZED_K = 0.1  # threshold for Gaussian kernel

MIN_FILE_BYTES = 1024


def download_file(url: str, dest: str) -> None:
    print(f"  Downloading {url[:80]}...")
    req = urllib.request.Request(url, headers={"User-Agent": "FARBench/1.0"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        with open(dest, "wb") as f:
            shutil.copyfileobj(resp, f)


def download_from_gdrive(file_id: str, dest: str) -> None:
    """Download a file from Google Drive using gdown."""
    import gdown
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"  Downloading from Google Drive (id={file_id})...")
    gdown.download(url, dest, quiet=False)


def compute_adjacency_matrix(distances_csv: str, sensor_ids: list) -> np.ndarray:
    """Compute weighted adjacency matrix from sensor distances (DCRNN convention).

    W_ij = exp(-d_ij^2 / sigma^2) if above threshold, else 0.
    sigma^2 is estimated as the variance of all non-zero distances.
    """
    num_sensors = len(sensor_ids)
    sensor_id_to_idx = {int(sid): i for i, sid in enumerate(sensor_ids)}

    # Initialize with infinity (no connection)
    dist_mx = np.full((num_sensors, num_sensors), np.inf, dtype=np.float32)
    np.fill_diagonal(dist_mx, 0.0)

    # Parse distances CSV: from,to,cost
    with open(distances_csv, "r") as f:
        f.readline()  # skip header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != 3:
                continue
            from_id, to_id, dist = int(parts[0]), int(parts[1]), float(parts[2])
            if from_id in sensor_id_to_idx and to_id in sensor_id_to_idx:
                i = sensor_id_to_idx[from_id]
                j = sensor_id_to_idx[to_id]
                dist_mx[i, j] = dist

    # Compute Gaussian kernel adjacency
    finite_mask = ~np.isinf(dist_mx) & (dist_mx > 0)
    sigma_sq = np.var(dist_mx[finite_mask])

    adj_mx = np.exp(-np.square(dist_mx) / sigma_sq)
    adj_mx[adj_mx < NORMALIZED_K] = 0.0
    # Self-loops: set diagonal to 0 (no self-weight in adjacency)
    np.fill_diagonal(adj_mx, 0.0)

    return adj_mx


def create_windows(data: np.ndarray, seq_len: int, pred_len: int) -> tuple:
    """Create sliding windows from time series data.

    Args:
        data: [T, N] time series
        seq_len: input window length
        pred_len: prediction window length

    Returns:
        x: [num_windows, seq_len, N] input windows
        y: [num_windows, pred_len, N] target windows
    """
    total_len = len(data)
    num_windows = total_len - seq_len - pred_len + 1

    x = np.zeros((num_windows, seq_len, data.shape[1]), dtype=np.float32)
    y = np.zeros((num_windows, pred_len, data.shape[1]), dtype=np.float32)

    for i in range(num_windows):
        x[i] = data[i: i + seq_len]
        y[i] = data[i + seq_len: i + seq_len + pred_len]

    return x, y


def main():
    data_dir = os.environ["FARBENCH_DATA_DIR"]
    test_data_dir = os.environ["FARBENCH_TEST_DATA_DIR"]
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(test_data_dir, exist_ok=True)

    # Check if already prepared
    required = [
        os.path.join(data_dir, "train.npy"),
        os.path.join(data_dir, "val.npy"),
        os.path.join(data_dir, "adj_mx.npy"),
        os.path.join(test_data_dir, "test_x.npy"),
        os.path.join(test_data_dir, "test_y.npy"),
    ]
    if all(os.path.exists(p) and os.path.getsize(p) > MIN_FILE_BYTES for p in required):
        print("METR-LA data already prepared, skipping.")
        return

    # ── Download raw files ──
    raw_dir = os.path.join(data_dir, "_raw")
    os.makedirs(raw_dir, exist_ok=True)

    h5_path = os.path.join(raw_dir, "metr-la.h5")
    dist_path = os.path.join(raw_dir, "distances_la_2012.csv")
    ids_path = os.path.join(raw_dir, "graph_sensor_ids.txt")

    # Download metr-la.h5 from Google Drive
    if not os.path.exists(h5_path) or os.path.getsize(h5_path) < 1_000_000:
        download_from_gdrive(METRLA_GDRIVE_ID, h5_path)

    # Download distances and sensor IDs from DCRNN GitHub
    if not os.path.exists(dist_path):
        download_file(DISTANCES_URL, dist_path)
    if not os.path.exists(ids_path):
        download_file(SENSOR_IDS_URL, ids_path)

    # ── Load data ──
    print("  Loading metr-la.h5...")
    df = pd.read_hdf(h5_path)
    print(f"  Loaded: {df.shape[0]} timesteps × {df.shape[1]} sensors")

    # Load sensor IDs (single comma-separated line, defines column ordering)
    with open(ids_path, "r") as f:
        content = f.read().strip()
    # Handle both formats: one-per-line or comma-separated
    if "," in content:
        sensor_ids = [s.strip() for s in content.split(",") if s.strip()]
    else:
        sensor_ids = [line.strip() for line in content.split("\n") if line.strip()]
    print(f"  Sensor IDs: {len(sensor_ids)}")

    # Reorder columns to match sensor_ids ordering
    df_columns = [str(c) for c in df.columns]
    col_order = [df_columns.index(sid) for sid in sensor_ids]
    data = df.values[:, col_order].astype(np.float32)  # [T, 207]

    # Handle missing values (0s → linear interpolation)
    mask = data == 0.0
    if mask.any():
        print(f"  Interpolating {mask.sum()} zero/missing values...")
        data_df = pd.DataFrame(data)
        data_df = data_df.replace(0.0, np.nan)
        data_df = data_df.interpolate(method="linear", axis=0, limit_direction="both")
        data_df = data_df.bfill().ffill()
        data = data_df.values.astype(np.float32)

    T = data.shape[0]
    print(f"  Total timesteps: {T}")

    # ── Chronological split ──
    train_end = int(T * TRAIN_RATIO)
    val_end = int(T * (TRAIN_RATIO + VAL_RATIO))

    train_data = data[:train_end]         # [0, train_end)
    val_data = data[train_end:val_end]    # [train_end, val_end)
    test_data = data[val_end:]            # [val_end, T)

    print(f"  Train: {train_data.shape[0]} steps, Val: {val_data.shape[0]} steps, "
          f"Test: {test_data.shape[0]} steps")

    # ── Compute adjacency matrix ──
    print("  Computing adjacency matrix...")
    adj_mx = compute_adjacency_matrix(dist_path, sensor_ids)
    nonzero = (adj_mx > 0).sum()
    print(f"  Adjacency: {adj_mx.shape}, {nonzero} non-zero entries")

    # ── Save training/validation data (raw time series) ──
    np.save(os.path.join(data_dir, "train.npy"), train_data)
    np.save(os.path.join(data_dir, "val.npy"), val_data)
    np.save(os.path.join(data_dir, "adj_mx.npy"), adj_mx)
    print(f"  Saved train.npy, val.npy, adj_mx.npy to {data_dir}")

    # ── Build pre-computed test windows (for evaluation alignment) ──
    print("  Building test windows...")
    test_x, test_y = create_windows(test_data, SEQ_LEN, PRED_LEN)
    print(f"  Test windows: {test_x.shape[0]} (input {test_x.shape}, target {test_y.shape})")

    np.save(os.path.join(test_data_dir, "test_x.npy"), test_x)
    np.save(os.path.join(test_data_dir, "test_y.npy"), test_y)

    # Copy adjacency to test dir for agent convenience
    np.save(os.path.join(test_data_dir, "adj_mx.npy"), adj_mx)
    print(f"  Saved test_x.npy, test_y.npy, adj_mx.npy to {test_data_dir}")

    # Clean up raw files
    shutil.rmtree(raw_dir)

    print(f"\nMETR-LA data ready:")
    print(f"  Train: {train_data.shape[0]} steps, Val: {val_data.shape[0]} steps")
    print(f"  Test: {test_x.shape[0]} windows ({SEQ_LEN}→{PRED_LEN})")
    print(f"  Speed range: [{data.min():.1f}, {data.max():.1f}] mph")


if __name__ == "__main__":
    main()
