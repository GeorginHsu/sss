"""TrainingTracker: training data collector.

Optional utility for agents to use in their training scripts.
Not required by the framework — agents can use any logging they prefer.

Two usage modes:
1. Embedded: instantiated inside training scripts, writes logs in real time.
2. Analysis: reads logs after training to produce a TrainingSummary.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Optional

from farbench.utils import get_logger


@dataclass
class TrainingSummary:
    """Aggregated statistics from a training run (local to tracker module)."""
    total_epochs: int = 0
    total_steps: int = 0
    training_hours: float = 0.0
    final_loss: float = 0.0
    best_val_metric: Optional[float] = None
    best_val_epoch: Optional[int] = None
    converged: bool = False
    loss_trend: str = "unknown"
    compute_type: str = "gpu"
    gpu_memory_peak_mb: float = 0.0
    gpu_count: int = 0
    cpu_cores: int = 0
    memory_peak_mb: float = 0.0

logger = get_logger(__name__)


class TrainingTracker:

    # ═══════════════════════════════════════════
    #  Embedded mode API (called by training scripts)
    # ═══════════════════════════════════════════

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self._metrics_path = os.path.join(log_dir, "metrics.jsonl")
        self._system_path = os.path.join(log_dir, "system.jsonl")
        self._metrics_file = None
        self._system_file = None
        self._files_closed = False
        self._nvml_initialized = False
        self._gpu_handles: list = []

        # Open files and initialize hardware monitoring.  If anything fails,
        # clean up already-opened handles so we don't leak file descriptors.
        try:
            self._metrics_file = open(self._metrics_path, "a")
            self._system_file = open(self._system_path, "a")
            self._start_time = time.time()
            self._compute_type = self._detect_compute_type()
            self._step_count = 0
            self._epoch_count = 0

            # Initialize NVML once, reuse handles across calls
            if self._compute_type == "gpu":
                self._init_nvml()
        except Exception:
            self._close_files()
            self._shutdown_nvml()
            raise

    def _detect_compute_type(self) -> str:
        import torch
        return "gpu" if torch.cuda.is_available() else "cpu"

    def log_scalar(self, step: int, epoch: int, **kwargs) -> None:
        """Log scalar metrics to metrics.jsonl."""
        record = {
            "step": step,
            "epoch": epoch,
            "timestamp": time.time(),
            **kwargs,
        }
        self._metrics_file.write(json.dumps(record) + "\n")
        self._metrics_file.flush()
        self._step_count = max(self._step_count, step)
        self._epoch_count = max(self._epoch_count, epoch)

    def log_system(self, step: int) -> None:
        """Collect and log system resource usage (adapts to compute_type)."""
        record: dict = {
            "step": step,
            "timestamp": time.time(),
            "compute_type": self._compute_type,
        }

        if self._compute_type == "gpu":
            record.update(self._collect_gpu_stats())
        else:
            record.update(self._collect_cpu_stats())

        self._system_file.write(json.dumps(record) + "\n")
        self._system_file.flush()

    def _init_nvml(self) -> None:
        import pynvml
        pynvml.nvmlInit()
        gpu_count = pynvml.nvmlDeviceGetCount()
        self._gpu_handles = [
            pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(gpu_count)
        ]
        self._nvml_initialized = True

    def _shutdown_nvml(self) -> None:
        if self._nvml_initialized:
            import pynvml
            pynvml.nvmlShutdown()
            self._nvml_initialized = False
            self._gpu_handles = []

    def _collect_gpu_stats(self) -> dict:
        import pynvml
        gpu_memory_mb = []
        gpu_util = []
        for handle in self._gpu_handles:
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_memory_mb.append(mem_info.used / (1024 * 1024))
            gpu_util.append(util_info.gpu / 100.0)
        return {
            "gpu_count": len(self._gpu_handles),
            "gpu_memory_mb": gpu_memory_mb,
            "gpu_util": gpu_util,
        }

    def _collect_cpu_stats(self) -> dict:
        """Collect CPU stats. Prefers cgroup data when running in a container."""
        if self._is_in_container():
            stats = self._collect_cgroup_cpu_stats()
            if stats:
                return stats
        import psutil
        return {
            "cpu_cores": psutil.cpu_count(logical=False) or psutil.cpu_count(),
            "memory_used_mb": psutil.virtual_memory().used / (1024 * 1024),
            "cpu_util": psutil.cpu_percent(interval=0.1) / 100.0,
        }

    def _is_in_container(self) -> bool:
        return (
            os.path.exists("/.dockerenv")
            or os.environ.get("FARBENCH_IN_DOCKER") == "1"
        )

    def _collect_cgroup_cpu_stats(self) -> dict | None:
        try:
            cpu_cores = self._read_cgroup_cpu_cores()
            memory_mb = self._read_cgroup_memory_mb()
            if cpu_cores is not None and memory_mb is not None:
                return {
                    "cpu_cores": cpu_cores,
                    "memory_used_mb": memory_mb,
                    "cpu_util": 0.0,
                }
        except Exception:
            pass
        return None

    def _read_cgroup_cpu_cores(self) -> int | None:
        """Read CPU core limit from cgroup, checking v2 before v1."""
        # cgroup v2: "quota period" e.g. "400000 100000"
        try:
            with open("/sys/fs/cgroup/cpu.max") as f:
                parts = f.read().strip().split()
                if parts[0] == "max":
                    return None  # unlimited
                quota = int(parts[0])
                period = int(parts[1])
                return max(1, quota // period)
        except (FileNotFoundError, IOError):
            pass

        # cgroup v1
        try:
            with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
                quota = int(f.read().strip())
            if quota == -1:
                return None  # unlimited
            with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
                period = int(f.read().strip())
            return max(1, quota // period)
        except (FileNotFoundError, IOError):
            pass

        return None

    def _read_cgroup_memory_mb(self) -> float | None:
        """Read container memory usage in MB, checking cgroup v2 before v1."""
        try:
            with open("/sys/fs/cgroup/memory.current") as f:
                return int(f.read().strip()) / (1024 * 1024)
        except (FileNotFoundError, IOError):
            pass

        try:
            with open("/sys/fs/cgroup/memory/memory.usage_in_bytes") as f:
                return int(f.read().strip()) / (1024 * 1024)
        except (FileNotFoundError, IOError):
            pass

        return None

    def log_image(self, tag: str, image, step: int) -> None:
        """Save image to {log_dir}/visualizations/{tag}_step{step}.png."""
        vis_dir = os.path.join(self.log_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        path = os.path.join(vis_dir, f"{tag}_step{step}.png")
        try:
            import matplotlib.pyplot as plt
            if hasattr(image, "savefig"):
                image.savefig(path)
            else:
                plt.imsave(path, image)
        except Exception as e:
            logger.warning(f"Failed to save image {path}: {e}")

    def finalize(self, **extra_meta) -> None:
        """Close file handles and NVML, write training_meta.json."""
        elapsed_hours = (time.time() - self._start_time) / 3600.0
        meta = {
            "total_epochs": self._epoch_count + 1,
            "total_steps": self._step_count + 1,
            "training_hours": round(elapsed_hours, 4),
            "exit_status": "completed",
            "compute_type": self._compute_type,
            **extra_meta,
        }
        meta_path = os.path.join(self.log_dir, "training_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        self._close_files()
        self._shutdown_nvml()

    def _close_files(self) -> None:
        """Idempotent file handle cleanup."""
        if self._files_closed:
            return
        self._files_closed = True
        for fh in (self._metrics_file, self._system_file):
            if fh is not None:
                try:
                    fh.close()
                except Exception:
                    pass

    def __del__(self):
        self._close_files()
        self._shutdown_nvml()

    # ═══════════════════════════════════════════
    #  Analysis mode API (called by Orchestrator)
    # ═══════════════════════════════════════════

    @staticmethod
    def summarize(training_log_dir: str) -> TrainingSummary:
        """Parse training logs and produce a TrainingSummary.

        Reads: training_meta.json, metrics.jsonl, system.jsonl.
        """
        summary = TrainingSummary()

        # training_meta.json
        meta_path = os.path.join(training_log_dir, "training_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            summary.total_epochs = meta.get("total_epochs", 0)
            summary.total_steps = meta.get("total_steps", 0)
            summary.training_hours = meta.get("training_hours", 0.0)
            summary.compute_type = meta.get("compute_type", "cpu")

        # metrics.jsonl -> loss analysis
        metrics_path = os.path.join(training_log_dir, "metrics.jsonl")
        losses = []
        val_metrics: list[tuple[int, float]] = []

        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if "loss" in record:
                        losses.append(record["loss"])

                    for key, val in record.items():
                        if key.startswith("val_") and isinstance(val, (int, float)):
                            val_metrics.append((record.get("epoch", 0), val))

        if losses:
            summary.final_loss = losses[-1]

            # Convergence: std of last 10% of losses < threshold
            tail = losses[max(0, len(losses) - len(losses) // 10 - 1):]
            if len(tail) >= 2:
                import numpy as np
                std = float(np.std(tail))
                mean = float(np.mean(tail))
                summary.converged = std < 0.01 * abs(mean) if mean != 0 else std < 0.001

            # Loss trend: linear regression slope of last 20%
            tail20 = losses[max(0, len(losses) - len(losses) // 5 - 1):]
            if len(tail20) >= 3:
                import numpy as np
                x = np.arange(len(tail20))
                slope = float(np.polyfit(x, tail20, 1)[0])
                if slope < -0.001:
                    summary.loss_trend = "decreasing"
                elif slope > 0.01:
                    summary.loss_trend = "diverging"
                else:
                    summary.loss_trend = "plateaued"

        if val_metrics:
            # Use last recorded val (not max), because summarize() doesn't
            # know higher_is_better. Actual "best" is determined by MetricEvaluator.
            last_epoch, last_val = val_metrics[-1]
            summary.best_val_metric = last_val
            summary.best_val_epoch = last_epoch

        # system.jsonl -> resource peak tracking
        system_path = os.path.join(training_log_dir, "system.jsonl")
        if os.path.exists(system_path):
            with open(system_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    ctype = record.get("compute_type", summary.compute_type)

                    if ctype == "gpu":
                        summary.gpu_count = max(
                            summary.gpu_count, record.get("gpu_count", 0)
                        )
                        gpu_mem = record.get("gpu_memory_mb", [])
                        if gpu_mem:
                            peak = max(gpu_mem) if isinstance(gpu_mem, list) else gpu_mem
                            summary.gpu_memory_peak_mb = max(
                                summary.gpu_memory_peak_mb, peak
                            )
                    else:
                        summary.cpu_cores = max(
                            summary.cpu_cores, record.get("cpu_cores", 0)
                        )
                        summary.memory_peak_mb = max(
                            summary.memory_peak_mb,
                            record.get("memory_used_mb", 0),
                        )

        return summary


# ═══════════════════════════════════════════
#  Training curve collection (called by Orchestrator)
# ═══════════════════════════════════════════

def collect_training_curves(workspace_dir: str) -> tuple[dict, dict]:
    """Scan workspace/logs/iter_*/ for metrics.jsonl, return data and curve images.

    Returns:
        curves: {iter_label: [{"step": ..., "train_loss": ..., ...}, ...]}
        images: {iter_label: absolute path to curve PNG}
    """
    logs_dir = os.path.join(workspace_dir, "logs")
    if not os.path.isdir(logs_dir):
        return {}, {}

    curves: dict = {}
    images: dict = {}

    iter_dirs = sorted([
        d for d in os.listdir(logs_dir)
        if os.path.isdir(os.path.join(logs_dir, d))
    ])

    # Only use the most recent iter with a metrics.jsonl
    iter_dirs = [d for d in iter_dirs if os.path.exists(os.path.join(logs_dir, d, "metrics.jsonl"))]
    if not iter_dirs:
        return {}, {}
    iter_dirs = [iter_dirs[-1]]

    for iter_label in iter_dirs:
        metrics_path = os.path.join(logs_dir, iter_label, "metrics.jsonl")
        if not os.path.exists(metrics_path):
            continue

        # Read metrics (max 200 entries)
        records = []
        try:
            with open(metrics_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        except Exception:
            continue

        if not records:
            continue

        # Uniform sampling if over 200 entries
        MAX_ENTRIES = 200
        if len(records) > MAX_ENTRIES:
            indices = [int(i * (len(records) - 1) / (MAX_ENTRIES - 1)) for i in range(MAX_ENTRIES)]
            records = [records[i] for i in indices]

        curves[iter_label] = records

        # Generate curve.png with matplotlib
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            steps = [r.get("step", i) for i, r in enumerate(records)]

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            fig.suptitle(iter_label, fontsize=12)

            # Loss subplot
            ax = axes[0]
            if any("train_loss" in r for r in records):
                ax.plot(steps, [r.get("train_loss") for r in records], label="train_loss")
            if any("val_loss" in r for r in records):
                ax.plot(steps, [r.get("val_loss") for r in records], label="val_loss")
            ax.set_xlabel("step")
            ax.set_ylabel("loss")
            ax.set_title("Loss")
            ax.legend()
            ax.grid(True)

            # Accuracy subplot
            ax = axes[1]
            if any("val_acc" in r for r in records):
                ax.plot(steps, [r.get("val_acc") for r in records], label="val_acc")
            if any("train_acc" in r for r in records):
                ax.plot(steps, [r.get("train_acc") for r in records], label="train_acc")
            ax.set_xlabel("step")
            ax.set_ylabel("accuracy")
            ax.set_title("Accuracy")
            ax.legend()
            ax.grid(True)

            plt.tight_layout()

            curve_path = os.path.join(logs_dir, iter_label, "curve.png")
            plt.savefig(curve_path, format="png", dpi=100)
            plt.close(fig)

            images[iter_label] = os.path.abspath(curve_path)

        except Exception as e:
            logger.warning(f"Failed to generate training curve for {iter_label}: {e}")

    return curves, images
