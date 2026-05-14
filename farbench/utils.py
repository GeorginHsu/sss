"""Utility functions: seeding, file hashing, logging, path validation, GPU selection."""

from __future__ import annotations

import hashlib
import logging
import os
import random
import sys


# ═══════════════════════════════════════════
#  Logging
# ═══════════════════════════════════════════

class _TaskContextFilter(logging.Filter):
    """Injects task_name into every log record under the 'farbench' logger tree."""

    def __init__(self) -> None:
        super().__init__()
        self.task_name: str = "-"

    def filter(self, record: logging.LogRecord) -> bool:
        record.task_name = self.task_name  # type: ignore[attr-defined]
        return True


# Singleton filter — shared across all farbench.* loggers in this process.
_task_filter = _TaskContextFilter()


def set_task_context(task_name: str) -> None:
    """Set the task name that appears in all farbench.* log output."""
    _task_filter.task_name = task_name


_LOG_FORMAT = "[%(asctime)s] %(name)s [%(task_name)s] %(levelname)s: %(message)s"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter(_LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")
        )
        handler.addFilter(_task_filter)
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger


# ═══════════════════════════════════════════
#  Seeding
# ═══════════════════════════════════════════

def set_deterministic(seed: int) -> None:
    """Set global random seeds for reproducibility.

    Called by training scripts, not by the framework core. numpy and
    torch are imported at call time, so ``import farbench.utils`` itself
    never requires these packages.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    import numpy as np
    np.random.seed(seed)

    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ═══════════════════════════════════════════
#  Hardware Info
# ═══════════════════════════════════════════

def collect_hardware_info() -> dict:
    """Collect machine hardware info for leaderboard recording."""
    info: dict = {}

    try:
        import psutil
        info["cpu_count_physical"] = psutil.cpu_count(logical=False)
        info["cpu_count_logical"] = psutil.cpu_count(logical=True)
        info["memory_total_gb"] = round(
            psutil.virtual_memory().total / (1024**3), 1
        )
    except Exception:
        pass

    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass

    try:
        from pynvml import (
            nvmlInit,
            nvmlDeviceGetCount,
            nvmlDeviceGetHandleByIndex,
            nvmlDeviceGetName,
            nvmlDeviceGetMemoryInfo,
            nvmlDeviceGetCudaComputeCapability,
            nvmlSystemGetDriverVersion,
            nvmlSystemGetCudaDriverVersion_v2,
            nvmlShutdown,
        )
        nvmlInit()

        # Driver & CUDA version
        driver_ver = nvmlSystemGetDriverVersion()
        if isinstance(driver_ver, bytes):
            driver_ver = driver_ver.decode("utf-8")
        info["nvidia_driver"] = driver_ver

        cuda_driver_ver = nvmlSystemGetCudaDriverVersion_v2()
        info["cuda_version"] = f"{cuda_driver_ver // 1000}.{(cuda_driver_ver % 1000) // 10}"

        gpu_count = nvmlDeviceGetCount()
        gpus = []
        for i in range(gpu_count):
            handle = nvmlDeviceGetHandleByIndex(i)
            name = nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            mem = nvmlDeviceGetMemoryInfo(handle)
            major, minor = nvmlDeviceGetCudaComputeCapability(handle)
            gpus.append({
                "id": i,
                "name": name,
                "memory_total_gb": round(mem.total / (1024**3), 1),
                "compute_capability": f"{major}.{minor}",
            })
        nvmlShutdown()
        info["gpu_count"] = gpu_count
        info["gpus"] = gpus
    except Exception:
        info["gpu_count"] = 0
        info["gpus"] = []

    # Which GPUs were allocated (FARBENCH_GPUS env)
    farbench_gpus = os.environ.get("FARBENCH_GPUS", "").strip()
    if farbench_gpus:
        info["farbench_gpus"] = farbench_gpus

    return info


# ═══════════════════════════════════════════
#  GPU Selection
# ═══════════════════════════════════════════

def select_best_gpus(count: int = 1) -> list[int]:
    """Select GPUs for Docker passthrough.

    If FARBENCH_GPUS is set by the CLI/script entrypoint, it is an ordered
    allocation of host GPU IDs. Preserve that order so Docker maps, for
    example, host 3,4,1,2 to container CUDA ordinals 0,1,2,3.
    """
    logger = get_logger(__name__)

    farbench_gpus_env = os.environ.get("FARBENCH_GPUS", "").strip()
    if farbench_gpus_env:
        try:
            ordered_ids = [int(x.strip()) for x in farbench_gpus_env.split(",") if x.strip()]
            n = len(ordered_ids) if count == -1 else min(count, len(ordered_ids))
            selected = ordered_ids[:n]
            logger.info(f"FARBENCH_GPUS ordered passthrough: {ordered_ids} -> {selected}")
            return selected
        except ValueError:
            logger.warning(f"FARBENCH_GPUS value '{farbench_gpus_env}' is invalid, ignoring")

    try:
        from pynvml import (
            nvmlInit,
            nvmlDeviceGetCount,
            nvmlDeviceGetHandleByIndex,
            nvmlDeviceGetMemoryInfo,
            nvmlShutdown,
        )
        nvmlInit()
        device_count = nvmlDeviceGetCount()

        gpu_free_mem = []
        for i in range(device_count):
            handle = nvmlDeviceGetHandleByIndex(i)
            mem_info = nvmlDeviceGetMemoryInfo(handle)
            gpu_free_mem.append((i, mem_info.free))
        nvmlShutdown()

        if count == -1:
            count = len(gpu_free_mem)

        gpu_free_mem.sort(key=lambda x: x[1], reverse=True)
        selected = [gpu_id for gpu_id, _ in gpu_free_mem[:count]]

        gpu_info = ", ".join(
            f"GPU {gid}: {free / 1024**3:.1f}GB free"
            for gid, free in gpu_free_mem
        )
        logger.info(f"GPU status: [{gpu_info}]")
        logger.info(f"Selected GPUs: {selected}")
        return selected

    except ImportError:
        logger.warning(
            "pynvml not installed — cannot query GPU memory. "
            "Selecting the first GPU by index."
        )
        n = 1 if count == -1 else count
        return list(range(n))
    except Exception as e:
        logger.warning(
            f"NVML GPU query failed ({e}) — selecting the first GPU by index."
        )
        n = 1 if count == -1 else count
        return list(range(n))


def make_cuda_visible_devices(gpu_count: int | None = 1) -> str:
    """Return a CUDA_VISIBLE_DEVICES string with the best available GPUs."""
    if gpu_count is None:
        gpu_count = -1
    selected = select_best_gpus(gpu_count)
    return ",".join(str(g) for g in selected)


# ═══════════════════════════════════════════
#  Path Validation
# ═══════════════════════════════════════════

def resolve_workspace_path(workspace_path: str, relative_path: str) -> str:
    """Resolve *relative_path* inside *workspace_path* and guard against traversal.

    Returns the absolute path on success.
    Raises ``ValueError`` if the resolved path escapes the workspace.
    """
    abs_path = os.path.realpath(os.path.join(workspace_path, relative_path))
    workspace_real = os.path.realpath(workspace_path)
    if abs_path != workspace_real and not abs_path.startswith(workspace_real + os.sep):
        raise ValueError(
            f"Path traversal blocked: {relative_path} resolves outside workspace"
        )
    return abs_path


# ═══════════════════════════════════════════
#  File Hash
# ═══════════════════════════════════════════

def compute_file_hash(file_path: str) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ═══════════════════════════════════════════
#  Workspace File Operations
# ═══════════════════════════════════════════

def write_files_to_workspace(
    workspace_path: str, files: dict[str, str]
) -> list[str]:
    """Write files to workspace with path traversal protection.

    Returns list of relative paths that were written.
    """
    written = []
    for rel_path, content in files.items():
        abs_path = resolve_workspace_path(workspace_path, rel_path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, "w") as f:
            f.write(content)
        written.append(rel_path)
    return written


# Directories to exclude when listing, snapshotting, or reading workspace files.
# Shared across utils.py, orchestrator.py, and gui/dashboard_api.py to keep
# filtering rules in one place.
WORKSPACE_SKIP_DIRS = frozenset({
    "__pycache__", ".git", ".ipynb_checkpoints",
    ".pip_packages", ".cache", ".huggingface",
})


def list_workspace_files(workspace_path: str) -> list[str]:
    """List all files in workspace as relative paths.

    Skips directories in WORKSPACE_SKIP_DIRS.  Other large directories
    (checkpoints, models, logs) are still listed so the agent knows they exist.
    """
    files = []
    for dirpath, dirnames, filenames in os.walk(workspace_path):
        dirnames[:] = [d for d in dirnames if d not in WORKSPACE_SKIP_DIRS]
        for fname in filenames:
            abs_path = os.path.join(dirpath, fname)
            rel_path = os.path.relpath(abs_path, workspace_path)
            files.append(rel_path)
    return sorted(files)


# Extensions for reading text source files
_TEXT_EXTENSIONS = {
    ".py", ".txt", ".json", ".yaml", ".yml", ".cfg", ".ini", ".toml",
    ".sh", ".md", ".csv", ".tsv", ".log",
}

# Binary / large files to skip
_SKIP_EXTENSIONS = {
    ".pt", ".pth", ".bin", ".pkl", ".npy", ".npz", ".h5", ".hdf5",
    ".onnx", ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff",
    ".zip", ".tar", ".gz", ".bz2", ".xz",
    ".so", ".o", ".a", ".dylib", ".dll", ".exe",
    ".safetensors",
    ".wav", ".mp3", ".flac", ".ogg",
}

# JSON filenames that are output/result artifacts — skip content to save tokens.
# Matched against basename (case-insensitive).
_SKIP_JSON_PATTERNS = {
    "test_pred.json", "test_prediction.json", "test_predictions.json",
    "predictions.json", "pred.json", "preds.json",
    "output.json", "outputs.json", "test_output.json", "test_outputs.json",
    "results.json", "test_results.json", "eval_results.json",
    "submission.json", "submit.json",
    "inference.json", "inference_output.json",
}


def read_workspace_file_contents(
    workspace_path: str,
    file_list: list[str],
    *,
    agent_written_files: set[str] | None = None,
    max_chars_per_file: int = 5000,
    max_total_chars: int = 50000,
) -> dict[str, str]:
    """Read text file contents from workspace for inclusion in observations.

    When *agent_written_files* is provided, ONLY those files are read
    (whitelist mode).  This prevents downloaded models, pip packages, and
    other artifacts from consuming the token budget.  Files not in the
    whitelist are still visible in *workspace_files* so the agent knows
    they exist — just their contents are not sent.

    Falls back to reading all text files when no whitelist is given
    (e.g. iteration 0 before the agent has written anything).
    """
    contents: dict[str, str] = {}
    total_chars = 0

    for rel_path in file_list:
        if total_chars >= max_total_chars:
            break

        # Whitelist mode: only read files the agent explicitly wrote
        if agent_written_files is not None and rel_path not in agent_written_files:
            continue

        ext = os.path.splitext(rel_path)[1].lower()
        if ext in _SKIP_EXTENSIONS:
            continue
        # Only read known text extensions; skip unknown binary formats
        if ext and ext not in _TEXT_EXTENSIONS:
            continue
        # Skip JSON files that are output/result artifacts (large, useless for agent)
        if ext == ".json":
            basename = os.path.basename(rel_path).lower()
            if basename in _SKIP_JSON_PATTERNS:
                continue

        abs_path = os.path.join(workspace_path, rel_path)
        try:
            with open(abs_path, "r", errors="replace") as f:
                text = f.read(max_chars_per_file + 1)
            if len(text) > max_chars_per_file:
                text = text[:max_chars_per_file] + "\n... [truncated]"
            contents[rel_path] = text
            total_chars += len(text)
        except (OSError, UnicodeDecodeError):
            continue

    return contents
