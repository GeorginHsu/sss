"""WeatherBench Z500+T850 data preparation: download ERA5 5.625deg, split, serialize.

Split strategy: standard WeatherBench chronological split (Rasp et al. 2020):
  - Train: 1979-2015 (37 years, 54056 6-hourly steps)
  - Val:   2016 (1 year, 1464 steps — leap year)
  - Test:  2017-2018 (2 years, 2920 steps)

Output layout:
    FARBENCH_DATA_DIR/
        train_data.npy   — float32 [54056, 2, 32, 64]  (Z500, T850)
        val_data.npy     — float32 [1464, 2, 32, 64]
        lat.npy          — float64 [32]  latitude values
        lon.npy          — float64 [64]  longitude values
        norm_stats.json  — global mean/std per variable from train set
    FARBENCH_TEST_DATA_DIR/
        test_inputs.npy  — float32 [2906, 3, 2, 32, 64]  (3-step input windows)
        test_labels.npy  — float32 [2906, 2, 32, 64]     (72h-ahead targets)
        lat.npy          — float64 [32]  (copy for evaluator)
        norm_stats.json  — (copy for evaluator reference)

Data source: WeatherBench (Rasp et al., JAMES 2020) — ERA5 reanalysis at 5.625deg.
  https://github.com/pangeo-data/WeatherBench
  https://mediatum.ub.tum.de/1524895
"""

import glob
import json
import os
import shutil
import urllib.request
import zipfile

import numpy as np

# Download URLs for single-level 5.625deg data from TUM
# NOTE: The HTTPS/NextCloud download URLs (dataserv.ub.tum.de/s/m1524895/download?...)
# are broken (return empty HTML). The FTP server still works (verified Apr 2026).
Z500_URL = (
    "ftp://m1524895:m1524895@dataserv.ub.tum.de"
    "/5.625deg/geopotential_500/geopotential_500_5.625deg.zip"
)
T850_URL = (
    "ftp://m1524895:m1524895@dataserv.ub.tum.de"
    "/5.625deg/temperature_850/temperature_850_5.625deg.zip"
)

# Split boundaries (years inclusive)
TRAIN_YEARS = range(1979, 2016)  # 1979-2015
VAL_YEARS = range(2016, 2017)    # 2016
TEST_YEARS = range(2017, 2019)   # 2017-2018

# Forecast setup
INPUT_STEPS = 3    # 3 consecutive 6-hourly snapshots
FORECAST_HORIZON = 12  # 72h / 6h = 12 steps ahead
STRIDE = 1         # evaluate at every 6-hourly init time

# Expected counts (6-hourly)
# Train: 37 years = 28 non-leap (28*1460) + 9 leap (9*1464) = 54056
# Val: 2016 (leap) = 1464
# Test: 2017-2018 (non-leap) = 2*1460 = 2920
EXPECTED_TRAIN = 54056
EXPECTED_VAL = 1464
EXPECTED_TEST = 2920


def download_file(url: str, dest: str, desc: str = "") -> None:
    """Download a file with progress indication. Supports both HTTP(S) and FTP."""
    print(f"  Downloading {desc or url} ...")
    if url.startswith("ftp://"):
        req = urllib.request.Request(url)
    else:
        req = urllib.request.Request(url, headers={"User-Agent": "FARBench/1.0"})
    with urllib.request.urlopen(req, timeout=1800) as resp:
        total = resp.headers.get("Content-Length")
        total = int(total) if total else None
        downloaded = 0
        with open(dest, "wb") as f:
            while True:
                chunk = resp.read(8192 * 1024)  # 8 MB chunks
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded * 100 / total
                    print(f"\r    {downloaded / 1e6:.0f} / {total / 1e6:.0f} MB ({pct:.1f}%)", end="", flush=True)
                else:
                    print(f"\r    {downloaded / 1e6:.0f} MB downloaded", end="", flush=True)
        print()


def load_variable_from_netcdf(nc_dir: str, var_name: str) -> tuple:
    """Load a variable from yearly NetCDF files, subsample to 6-hourly.

    Returns:
        data_by_year: dict mapping year -> numpy array [N_steps, 32, 64]
        lat: numpy array [32]
        lon: numpy array [64]
    """
    import xarray as xr

    # Find all .nc files in the directory
    nc_files = sorted(glob.glob(os.path.join(nc_dir, "*.nc")))
    if not nc_files:
        raise FileNotFoundError(f"No .nc files found in {nc_dir}")

    print(f"  Found {len(nc_files)} NetCDF files in {nc_dir}")

    # Load first file to get coordinates and variable name
    ds0 = xr.open_dataset(nc_files[0])
    # Detect variable name (could be 'z', 't', 'geopotential', 'temperature', etc.)
    data_vars = list(ds0.data_vars)
    if var_name in data_vars:
        actual_var = var_name
    elif len(data_vars) == 1:
        actual_var = data_vars[0]
    else:
        raise ValueError(f"Cannot find variable '{var_name}' in {data_vars}")

    # Get lat/lon from first file
    lat_name = "latitude" if "latitude" in ds0.dims else "lat"
    lon_name = "longitude" if "longitude" in ds0.dims else "lon"
    lat = ds0[lat_name].values.astype(np.float64)
    lon = ds0[lon_name].values.astype(np.float64)
    ds0.close()

    # Load each file, extract year, subsample to 6-hourly
    data_by_year = {}
    for nc_path in nc_files:
        ds = xr.open_dataset(nc_path)
        times = ds["time"].values

        # Subsample to 6-hourly (00, 06, 12, 18 UTC)
        hours = (times.astype("datetime64[h]") - times.astype("datetime64[D]")).astype(int)
        mask_6h = (hours % 6) == 0
        ds_6h = ds.isel(time=mask_6h)

        arr = ds_6h[actual_var].values.astype(np.float32)  # [T, 32, 64]

        # Determine year from the timestamps
        year_vals = ds_6h["time"].dt.year.values
        unique_years = np.unique(year_vals)

        for yr in unique_years:
            yr_mask = year_vals == yr
            yr_data = arr[yr_mask]
            if yr in data_by_year:
                data_by_year[yr] = np.concatenate([data_by_year[yr], yr_data], axis=0)
            else:
                data_by_year[yr] = yr_data

        ds.close()
        # Print progress
        yr_str = "/".join(str(y) for y in unique_years)
        print(f"    Loaded {os.path.basename(nc_path)}: year(s) {yr_str}, {arr.shape[0]} -> {sum(yr_mask.sum() for yr_mask in [year_vals == y for y in unique_years])} 6h steps")

    return data_by_year, lat, lon


def main():
    data_dir = os.environ["FARBENCH_DATA_DIR"]
    test_data_dir = os.environ["FARBENCH_TEST_DATA_DIR"]
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(test_data_dir, exist_ok=True)

    # Check if already prepared
    required = [
        os.path.join(data_dir, "train_data.npy"),
        os.path.join(data_dir, "val_data.npy"),
        os.path.join(data_dir, "lat.npy"),
        os.path.join(data_dir, "lon.npy"),
        os.path.join(data_dir, "norm_stats.json"),
        os.path.join(test_data_dir, "test_inputs.npy"),
        os.path.join(test_data_dir, "test_labels.npy"),
    ]
    if all(os.path.exists(p) and os.path.getsize(p) > 1000 for p in required):
        print("WeatherBench data already prepared, skipping.")
        return

    raw_dir = os.path.join(data_dir, "_raw")
    os.makedirs(raw_dir, exist_ok=True)

    # --- Download and extract Z500 ---
    z500_zip = os.path.join(raw_dir, "geopotential_500_5.625deg.zip")
    z500_dir = os.path.join(raw_dir, "geopotential_500")

    if not os.path.isdir(z500_dir) or len(glob.glob(os.path.join(z500_dir, "*.nc"))) == 0:
        if not os.path.exists(z500_zip) or os.path.getsize(z500_zip) < 1e6:
            download_file(Z500_URL, z500_zip, "Z500 (geopotential_500)")
        print("  Extracting Z500...")
        with zipfile.ZipFile(z500_zip, "r") as zf:
            zf.extractall(raw_dir)
        # The zip might extract into a subdirectory or flat
        if not os.path.isdir(z500_dir):
            # Try to find the extracted directory
            candidates = glob.glob(os.path.join(raw_dir, "*geopotential*"))
            nc_candidates = glob.glob(os.path.join(raw_dir, "*.nc"))
            if nc_candidates:
                os.makedirs(z500_dir, exist_ok=True)
                for f in nc_candidates:
                    if "geopotential" in os.path.basename(f).lower():
                        shutil.move(f, z500_dir)

    # --- Download and extract T850 ---
    t850_zip = os.path.join(raw_dir, "temperature_850_5.625deg.zip")
    t850_dir = os.path.join(raw_dir, "temperature_850")

    if not os.path.isdir(t850_dir) or len(glob.glob(os.path.join(t850_dir, "*.nc"))) == 0:
        if not os.path.exists(t850_zip) or os.path.getsize(t850_zip) < 1e6:
            download_file(T850_URL, t850_zip, "T850 (temperature_850)")
        print("  Extracting T850...")
        with zipfile.ZipFile(t850_zip, "r") as zf:
            zf.extractall(raw_dir)
        if not os.path.isdir(t850_dir):
            candidates = glob.glob(os.path.join(raw_dir, "*.nc"))
            if candidates:
                os.makedirs(t850_dir, exist_ok=True)
                for f in candidates:
                    if "temperature" in os.path.basename(f).lower():
                        shutil.move(f, t850_dir)

    # --- Load and process Z500 ---
    print("\nLoading Z500 data...")
    z500_by_year, lat, lon = load_variable_from_netcdf(z500_dir, "z")
    print(f"  Z500: {len(z500_by_year)} years loaded, lat={lat.shape}, lon={lon.shape}")

    # --- Load and process T850 ---
    print("\nLoading T850 data...")
    t850_by_year, _, _ = load_variable_from_netcdf(t850_dir, "t")
    print(f"  T850: {len(t850_by_year)} years loaded")

    # --- Combine variables and split by period ---
    def combine_years(years):
        """Stack Z500 and T850 for given years into [N, 2, 32, 64]."""
        z_parts, t_parts = [], []
        for yr in years:
            if yr not in z500_by_year:
                raise ValueError(f"Year {yr} not found in Z500 data")
            if yr not in t850_by_year:
                raise ValueError(f"Year {yr} not found in T850 data")
            z_parts.append(z500_by_year[yr])
            t_parts.append(t850_by_year[yr])
        z_all = np.concatenate(z_parts, axis=0)  # [N, 32, 64]
        t_all = np.concatenate(t_parts, axis=0)  # [N, 32, 64]
        return np.stack([z_all, t_all], axis=1)   # [N, 2, 32, 64]

    print("\nBuilding splits...")
    train_data = combine_years(TRAIN_YEARS)
    val_data = combine_years(VAL_YEARS)
    test_data = combine_years(TEST_YEARS)

    print(f"  Train: {train_data.shape} (expected first dim: {EXPECTED_TRAIN})")
    print(f"  Val:   {val_data.shape} (expected first dim: {EXPECTED_VAL})")
    print(f"  Test:  {test_data.shape} (expected first dim: {EXPECTED_TEST})")

    # --- Save train/val data ---
    np.save(os.path.join(data_dir, "train_data.npy"), train_data)
    np.save(os.path.join(data_dir, "val_data.npy"), val_data)
    np.save(os.path.join(data_dir, "lat.npy"), lat)
    np.save(os.path.join(data_dir, "lon.npy"), lon)
    print(f"  Saved train_data.npy, val_data.npy, lat.npy, lon.npy to {data_dir}")

    # --- Compute normalization stats from training set ---
    z500_train = train_data[:, 0, :, :]  # [N, 32, 64]
    t850_train = train_data[:, 1, :, :]
    norm_stats = {
        "z500_mean": float(z500_train.mean()),
        "z500_std": float(z500_train.std()),
        "t850_mean": float(t850_train.mean()),
        "t850_std": float(t850_train.std()),
    }
    for k, v in norm_stats.items():
        print(f"    {k}: {v:.4f}")

    norm_path = os.path.join(data_dir, "norm_stats.json")
    with open(norm_path, "w") as f:
        json.dump(norm_stats, f, indent=2)

    # Also copy to test_data_dir for evaluator reference
    shutil.copy(norm_path, os.path.join(test_data_dir, "norm_stats.json"))

    # --- Build test windows ---
    # Input: 3 consecutive 6h steps [i, i+1, i+2]
    # Target: step i + 2 + FORECAST_HORIZON (i.e., 72h ahead of the last input step)
    print("\nBuilding test windows...")
    n_test = test_data.shape[0]  # 2920
    windows_in = []
    windows_out = []

    for i in range(0, n_test - (INPUT_STEPS - 1) - FORECAST_HORIZON, STRIDE):
        inp = test_data[i : i + INPUT_STEPS]                    # [3, 2, 32, 64]
        tgt = test_data[i + INPUT_STEPS - 1 + FORECAST_HORIZON] # [2, 32, 64]
        windows_in.append(inp)
        windows_out.append(tgt)

    test_inputs = np.array(windows_in, dtype=np.float32)   # [N_test, 3, 2, 32, 64]
    test_labels = np.array(windows_out, dtype=np.float32)  # [N_test, 2, 32, 64]

    print(f"  Test windows: {test_inputs.shape[0]}")
    print(f"    Input shape:  {test_inputs.shape}")
    print(f"    Target shape: {test_labels.shape}")

    np.save(os.path.join(test_data_dir, "test_inputs.npy"), test_inputs)
    np.save(os.path.join(test_data_dir, "test_labels.npy"), test_labels)

    # Copy lat.npy to test_data_dir for evaluator (latitude-weighted RMSE)
    shutil.copy(os.path.join(data_dir, "lat.npy"), os.path.join(test_data_dir, "lat.npy"))

    print(f"  Saved test_inputs.npy, test_labels.npy, lat.npy to {test_data_dir}")

    # --- Clean up raw data ---
    print("\nCleaning up raw downloads...")
    shutil.rmtree(raw_dir)

    print(f"\nWeatherBench data ready:")
    print(f"  Train/val: {data_dir}")
    print(f"  Test:      {test_data_dir}")
    print(f"  Train: {train_data.shape[0]}, Val: {val_data.shape[0]}, Test windows: {test_inputs.shape[0]}")


if __name__ == "__main__":
    main()
