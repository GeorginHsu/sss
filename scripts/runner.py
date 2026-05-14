#!/usr/bin/env python3
"""runner.py — Parallel task runner with GPU pool, per-agent concurrency, retry, and live status.

Modes:
    Batch:  python runner.py --config config.yaml
    Single: python runner.py --task X --preset Y --gpus N

Features:
    - GPU pool: thread-safe allocation of physical GPUs across all agents
    - Per-agent concurrency: limits parallel API calls per provider (avoids 429).
      The configured limit is a hard ceiling and is never raised at runtime,
      so provider rate limits stay honored even when other agents finish first.
    - Retry with configurable exponential backoff
    - Live status: writes experiments/matrix_status.json for dashboard monitoring
    - Graceful shutdown: SIGINT/SIGTERM stops running containers before exit
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
STATUS_FILE = REPO_ROOT / "experiments" / "matrix_status.json"
MATRIX_HISTORY_DIR = REPO_ROOT / "experiments" / "matrix_history"

GREEN  = "\033[32m"
RED    = "\033[31m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
RESET  = "\033[0m"

# Global shutdown flag — set by signal handler, checked by workers.
_shutdown_event = threading.Event()

# Global CUDA suffix — set in main(), read by worker threads.
_cuda_suffix: str = ""


def _signal_handler(signum: int, frame) -> None:
    """Signal workers to stop. Cleanup happens in main() after threads join."""
    print(f"\n{RED}[SHUTDOWN] Received signal {signum}, stopping workers...{RESET}", flush=True)
    _shutdown_event.set()


def _is_task_stopped(tag: str) -> bool:
    """Check if a per-task stop was requested via experiments/.stop/{tag} file."""
    stop_file = REPO_ROOT / "experiments" / ".stop" / tag.replace("/", "__")
    return stop_file.exists()


def stop_task(tag: str) -> None:
    """Request a single task to stop (called externally or from CLI)."""
    stop_dir = REPO_ROOT / "experiments" / ".stop"
    stop_dir.mkdir(parents=True, exist_ok=True)
    (stop_dir / tag.replace("/", "__")).touch()



def _cleanup_task_containers(agent_id: str) -> None:
    """Stop and remove sandbox/eval containers for a specific agent_id.

    Container names follow the pattern: rab-{agent_id_dashed}-sandbox / -eval.
    """
    agent_safe = agent_id.replace("_", "-")
    for suffix in ("sandbox", "eval"):
        name = f"rab-{agent_safe}-{suffix}"
        ret = subprocess.run(
            ["docker", "rm", "-f", name],
            capture_output=True, timeout=15,
        )
        if ret.returncode == 0:
            _log(agent_id, f"Stopped container: {name}", YELLOW)


# ---------------------------------------------------------------------------
# Dynamic Request Queue (restart / add-task)
# ---------------------------------------------------------------------------

REQUEST_DIR = REPO_ROOT / "experiments" / ".requests"


def _write_request(action: str, **fields) -> None:
    """Write a request file atomically for the running batch process to pick up."""
    REQUEST_DIR.mkdir(parents=True, exist_ok=True)
    tag = fields.get("tag", "unknown")
    fname = f"{action}__{tag.replace('/', '__')}.json"
    target = REQUEST_DIR / fname
    tmp = target.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump({"action": action, "requested_at": datetime.now().isoformat(), **fields}, f)
    os.replace(tmp, target)


def _read_and_consume_requests() -> list[dict]:
    """Read all pending request files, delete them, and return the parsed dicts."""
    if not REQUEST_DIR.exists():
        return []
    reqs: list[dict] = []
    for p in sorted(REQUEST_DIR.iterdir()):
        if p.suffix != ".json":
            continue
        try:
            with open(p) as f:
                reqs.append(json.load(f))
        except Exception as e:
            print(f"{YELLOW}WARNING: Bad request file {p.name}: {e}{RESET}", file=sys.stderr)
        # Always consume the file to avoid infinite retry
        try:
            p.unlink()
        except OSError:
            pass
    return reqs


# ---------------------------------------------------------------------------
# GPU Pool
# ---------------------------------------------------------------------------

class GpuPool:
    """Thread-safe GPU pool. acquire() blocks until n GPUs are free."""

    def __init__(self, gpu_ids: list[int]):
        self._available: list[int] = list(gpu_ids)
        self._order: dict[int, int] = {gpu: idx for idx, gpu in enumerate(gpu_ids)}
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._assignments: dict[int, str] = {}

    def acquire(self, n: int, tag: str) -> list[int]:
        with self._cond:
            while len(self._available) < n:
                # Use timeout so shutdown can interrupt the wait
                self._cond.wait(timeout=2)
                if _shutdown_event.is_set():
                    raise RuntimeError("shutdown requested while waiting for GPU")
            gpus = self._available[:n]
            self._available = self._available[n:]
            for g in gpus:
                self._assignments[g] = tag
            return gpus

    def release(self, gpus: list[int]) -> None:
        with self._cond:
            for g in gpus:
                self._assignments.pop(g, None)
            self._available.extend(gpus)
            self._available.sort(key=lambda g: self._order[g])
            self._cond.notify_all()

    def snapshot(self) -> dict[str, str]:
        with self._lock:
            return {str(g): t for g, t in self._assignments.items()}


# ---------------------------------------------------------------------------
# Per-agent Concurrency Control
# ---------------------------------------------------------------------------

class AgentGate:
    """Per-agent concurrency gate with a fixed limit.

    Controls how many experiments for a single agent preset can run concurrently,
    limiting parallel API calls to the same provider to avoid 429 rate-limit errors.
    The limit is whatever the config declared and never changes at runtime.
    """

    def __init__(self, limit: int):
        self._limit = limit
        self._running = 0
        self._cond = threading.Condition()

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def running(self) -> int:
        with self._cond:
            return self._running

    def acquire(self) -> None:
        with self._cond:
            while self._running >= self._limit:
                self._cond.wait(timeout=2)
                if _shutdown_event.is_set():
                    raise RuntimeError("shutdown requested while waiting for agent slot")
            self._running += 1

    def release(self) -> None:
        with self._cond:
            self._running -= 1
            self._cond.notify_all()


class ConcurrencyManager:
    """Manages per-agent concurrency gates.

    Each agent's concurrency limit comes straight from config (max_concurrent)
    and is treated as a hard ceiling. Pending-task counts are tracked purely
    for status/dashboard display and never trigger limit changes.
    """

    def __init__(self, agent_limits: dict[str, int]):
        self._gates: dict[str, AgentGate] = {
            agent: AgentGate(limit) for agent, limit in agent_limits.items()
        }
        self._lock = threading.Lock()
        self._pending: dict[str, int] = {}

    def init_pending(self, agent_task_counts: dict[str, int]) -> None:
        """Set initial pending task counts per agent (status display only)."""
        with self._lock:
            self._pending = dict(agent_task_counts)

    def acquire(self, agent: str) -> None:
        """Block until a concurrency slot is available for this agent."""
        self._gates[agent].acquire()

    def release(self, agent: str) -> None:
        """Release a concurrency slot for this agent."""
        self._gates[agent].release()

    def get_limit(self, agent: str) -> int:
        """Get the concurrency limit for an agent."""
        return self._gates[agent].limit

    def task_done(self, agent: str) -> None:
        """Decrement pending count for an agent (status display only)."""
        with self._lock:
            prev = self._pending.get(agent, 0)
            if prev > 0:
                self._pending[agent] = prev - 1

    def ensure_agent(self, agent: str, limit: int) -> None:
        """Add an agent gate if not already tracked (used when adding tasks for new agents)."""
        with self._lock:
            if agent not in self._gates:
                self._gates[agent] = AgentGate(limit)
                self._pending[agent] = 0

    def snapshot(self) -> dict:
        """Return current state for status monitoring."""
        with self._lock:
            return {
                agent: {
                    "limit": gate.limit,
                    "running": gate.running,
                    "pending": self._pending.get(agent, 0),
                }
                for agent, gate in self._gates.items()
            }


# ---------------------------------------------------------------------------
# Matrix Status History
# ---------------------------------------------------------------------------

def _backup_matrix_status() -> None:
    """Archive the current matrix_status.json before starting a new batch run."""
    if not STATUS_FILE.exists():
        return
    try:
        with open(STATUS_FILE) as f:
            data = json.load(f)
        # Use started_at from the file for a meaningful filename
        started = data.get("started_at", "unknown").replace(":", "").replace("-", "")
        MATRIX_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        dest = MATRIX_HISTORY_DIR / f"matrix_{started}.json"
        # Avoid overwriting if same timestamp already archived
        if not dest.exists():
            shutil.copy2(STATUS_FILE, dest)
            print(f"  Archived previous matrix status → {dest.relative_to(REPO_ROOT)}")
    except Exception as e:
        print(f"  WARNING: Failed to archive matrix status: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Status Writer
# ---------------------------------------------------------------------------

class StatusWriter:
    """Thread-safe atomic writer for matrix_status.json."""

    def __init__(self, path: Path, config: dict, gpu_pool: GpuPool,
                 concurrency_mgr: ConcurrencyManager | None = None,
                 *, write_file: bool = True):
        self._path = path
        self._write_file = write_file
        self._lock = threading.Lock()
        self._gpu_pool = gpu_pool
        self._concurrency_mgr = concurrency_mgr
        self._data: dict = {
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "config": config,
            "summary": {"total": 0, "queued": 0, "running": 0, "done": 0, "failed": 0},
            "runs": {},
        }

    def get_run_status(self, tag: str) -> str:
        with self._lock:
            return self._data["runs"].get(tag, {}).get("status", "")

    def init_runs(self, tags: list[str]) -> None:
        with self._lock:
            for tag in tags:
                self._data["runs"][tag] = {"status": "queued"}
            self._recompute_and_flush()

    def update(self, tag: str, **fields) -> None:
        with self._lock:
            run = self._data["runs"].setdefault(tag, {})
            run.update(fields)
            self._recompute_and_flush()

    def _recompute_and_flush(self) -> None:
        counts = {"total": 0, "queued": 0, "running": 0, "done": 0, "failed": 0}
        for r in self._data["runs"].values():
            counts["total"] += 1
            s = r.get("status", "queued")
            if s in ("queued", "retrying", "preparing"):
                counts["queued"] += 1
            elif s == "running":
                counts["running"] += 1
            elif s == "done":
                counts["done"] += 1
            elif s == "failed":
                counts["failed"] += 1
        self._data["summary"] = counts
        self._data["gpu_pool"] = self._gpu_pool.snapshot()
        if self._concurrency_mgr is not None:
            self._data["agent_concurrency"] = self._concurrency_mgr.snapshot()
        self._data["updated_at"] = datetime.now().isoformat(timespec="seconds")
        if not self._write_file:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(self._data, f, indent=2, default=str)
        os.replace(tmp, self._path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log(tag: str, msg: str, color: str = "") -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] [{tag}] {msg}"
    if color:
        line = color + line + RESET
    print(line, flush=True)


def _read_task_yaml(task: str) -> dict:
    p = REPO_ROOT / "benchmarks" / task / "task.yaml"
    if not p.exists():
        return {}
    with open(p) as f:
        return yaml.safe_load(f) or {}


def _read_max_gpu_count(task: str) -> int:
    return int(_read_task_yaml(task).get("max_gpu_count", 1))


def _compose_env(cuda_suffix: str) -> dict[str, str]:
    """Build env dict for docker compose with RAB_CUDA set."""
    env = os.environ.copy()
    env["RAB_CUDA"] = cuda_suffix  # required — docker-compose.yml references this
    return env


DOCKERHUB_ORG = os.environ.get("DOCKERHUB_ORG", "rabench")

# Mirror of rab.tasks.CUDA_VARIANTS — avoids import cycle from this top-level script.
_CUDA_VARIANTS = {"cu118", "cu128"}


def _validate_cuda_arg(cuda: str) -> None:
    """Fail loudly if --cuda is missing or unknown."""
    if not cuda:
        print(
            f"{RED}ERROR: --cuda is required (one of {sorted(_CUDA_VARIANTS)}){RESET}",
            file=sys.stderr,
        )
        sys.exit(2)
    if cuda not in _CUDA_VARIANTS:
        print(
            f"{RED}ERROR: Unknown --cuda {cuda!r}; expected one of "
            f"{sorted(_CUDA_VARIANTS)}{RESET}",
            file=sys.stderr,
        )
        sys.exit(2)


def _ensure_base_image(cuda_suffix: str) -> None:
    """Ensure rab/base:<cuda_suffix> exists locally; pull from DockerHub if missing."""
    local_image = f"rab/base:{cuda_suffix}"
    hub_image = f"{DOCKERHUB_ORG}/rabench:base-{cuda_suffix}"

    result = subprocess.run(
        ["docker", "image", "inspect", local_image],
        capture_output=True, timeout=10,
    )
    if result.returncode == 0:
        _log("DOCKER", f"Base image ready: {local_image}", GREEN)
        return

    _log("DOCKER", f"Base image {local_image} not found, pulling {hub_image} ...", YELLOW)
    pull = subprocess.run(["docker", "pull", hub_image], timeout=600)
    if pull.returncode != 0:
        print(f"\n{RED}ERROR: Failed to pull base image {hub_image}{RESET}", file=sys.stderr)
        print(f"  Build it manually with docker/Dockerfile.base; see docs/docker.md.", file=sys.stderr)
        sys.exit(1)

    subprocess.run(["docker", "tag", hub_image, local_image], capture_output=True, timeout=10)
    _log("DOCKER", f"Pulled and tagged: {hub_image} → {local_image}", GREEN)


def _discover_tasks() -> list[str]:
    benchmarks = REPO_ROOT / "benchmarks"
    if not benchmarks.is_dir():
        return []
    tasks = []
    for entry in sorted(benchmarks.iterdir()):
        if entry.is_dir() and (entry / "task.yaml").exists():
            tasks.append(entry.name)
    return tasks


def _find_experiment_dir(task: str, agent_id: str) -> str:
    """Find the experiment directory for a running/completed task. Returns relative path or ''."""
    exp_root = REPO_ROOT / "experiments" / task
    if not exp_root.exists():
        return ""
    dirs = sorted(exp_root.glob(f"{agent_id}_*"), reverse=True)
    if not dirs:
        return ""
    return str(dirs[0].relative_to(REPO_ROOT))


def _read_metrics(task: str, agent_id: str) -> dict:
    exp_dir = _find_experiment_dir(task, agent_id)
    if not exp_dir:
        return {}
    results_file = REPO_ROOT / exp_dir / "summary" / "final_results.json"
    if not results_file.exists():
        return {}
    try:
        with open(results_file) as f:
            d = json.load(f)
        return {
            "best_metric": d.get("best_primary_metric"),
            "total_iterations": d.get("total_iterations"),
            "experiment_dir": exp_dir,
        }
    except Exception:
        return {}


def _read_failure_detail(task: str, agent_id: str, stderr_tail: str) -> str:
    """Build a human-readable failure detail from experiment.log and captured stderr.

    Returns the last ~1500 chars of experiment.log (if it exists), otherwise
    falls back to the captured stderr tail.
    """
    exp_dir = _find_experiment_dir(task, agent_id)
    if exp_dir:
        log_file = REPO_ROOT / exp_dir / "experiment.log"
        if log_file.exists():
            try:
                size = log_file.stat().st_size
                with open(log_file, encoding="utf-8", errors="replace") as f:
                    if size > 1500:
                        f.seek(size - 1500)
                        f.readline()  # skip partial line
                    return f.read()[-1500:]
            except Exception:
                pass
    return stderr_tail[-1500:] if stderr_tail else ""


# ---------------------------------------------------------------------------
# Per-task Prepare Lock
# ---------------------------------------------------------------------------

class PrepareLocks:
    """Ensures each task is prepared exactly once, even when multiple agents
    target the same task concurrently.

    The first thread to call prepare() for a given task runs the actual
    docker compose prepare command.  Other threads block until it completes
    and receive the same success/failure result.

    --hub-only is enforced: tasks must pull from DockerHub.
    No local Dockerfile build fallback is allowed.
    """

    def __init__(self, hub_only: bool = True, cuda_suffix: str = ""):
        self._global_lock = threading.Lock()
        self._locks: dict[str, threading.Lock] = {}
        self._results: dict[str, tuple[bool, str]] = {}  # task -> (success, error_msg)
        self._hub_only = hub_only
        self._cuda_suffix = cuda_suffix

    def prepare(self, task: str, cwd: str) -> tuple[bool, str]:
        """Run prepare for task (or wait if another thread is already doing it).

        Returns (success: bool, error_message: str).
        """
        # Get or create per-task lock
        with self._global_lock:
            if task not in self._locks:
                self._locks[task] = threading.Lock()
            lock = self._locks[task]

        with lock:
            # Check if another thread already completed prepare
            if task in self._results:
                return self._results[task]

            # We are the first — run prepare
            cmd = [
                "docker", "compose", "run", "--rm", "rab",
                "tasks", "prepare", task,
            ]
            if self._hub_only:
                cmd.append("--hub-only")
            # Always pass --cuda; whitelist was already validated in main()
            cmd.extend(["--cuda", self._cuda_suffix])

            env = _compose_env(self._cuda_suffix)
            prep = subprocess.run(cmd, cwd=cwd, env=env, stderr=subprocess.PIPE, text=True)
            if prep.returncode == 0:
                self._results[task] = (True, "")
            else:
                err = (prep.stderr or "")[-800:]
                self._results[task] = (False, err)

            return self._results[task]

    def reset(self, task: str) -> None:
        """Clear cached prepare result so a restart can re-prepare if needed."""
        with self._global_lock:
            self._results.pop(task, None)


# ---------------------------------------------------------------------------
# Per-task Cleanup (after all agents finish)
# ---------------------------------------------------------------------------

class TaskCleanup:
    """Track per-task agent completion; clean Docker image + data when all agents are done.

    Resources cleaned:
      - Docker images: local (rab/<task>:*) and hub tag (rabench/rabench:<task>-*)
      - Prepared data: data_dir + test_data_dir from task.yaml
      - Marker file: benchmarks/<task>/.prepared
    """

    def __init__(self, task_agent_counts: dict[str, int], cuda_suffix: str):
        self._lock = threading.Lock()
        # {task: remaining_agent_count}
        self._remaining: dict[str, int] = dict(task_agent_counts)
        self._cuda_suffix = cuda_suffix

    def mark_done(self, task: str) -> None:
        """Called when one agent finishes a task (done or failed).

        When the last agent finishes, fires cleanup in-thread.
        """
        should_clean = False
        with self._lock:
            if task not in self._remaining:
                return
            self._remaining[task] -= 1
            if self._remaining[task] <= 0:
                del self._remaining[task]
                should_clean = True

        if should_clean:
            self._cleanup(task)

    def add_task(self, task: str, n_agents: int) -> None:
        """Register a dynamically added task."""
        with self._lock:
            self._remaining[task] = self._remaining.get(task, 0) + n_agents

    # Mirrors rab.tasks._CUDA_SUFFIX_RE — trailing -cuNNN in Docker tags.
    _CUDA_RE = re.compile(r"-cu\d+$")

    def _cleanup(self, task: str) -> None:
        """Remove Docker images, prepared data, and .prepared marker for a task."""
        _log(f"CLEANUP/{task}", "All agents finished, cleaning up ...", CYAN)
        task_dir = REPO_ROOT / "benchmarks" / task
        cfg = _read_task_yaml(task)

        # 1. Remove Docker images (local tag + hub tag)
        #    Apply CUDA suffix rewriting with the same regex as rab/tasks.py
        #    so that future CUDA variants (cu130, ...) are handled correctly.
        images_to_remove = []
        for raw in (cfg.get("docker_image", ""), cfg.get("docker_hub_image", "")):
            if not raw:
                continue
            if self._cuda_suffix and self._CUDA_RE.search(raw):
                raw = self._CUDA_RE.sub(f"-{self._cuda_suffix}", raw)
            images_to_remove.append(raw)

        for img in images_to_remove:
            ret = subprocess.run(
                ["docker", "rmi", img],
                capture_output=True, text=True, timeout=30,
            )
            if ret.returncode == 0:
                _log(f"CLEANUP/{task}", f"Removed image: {img}", GREEN)
            else:
                _log(f"CLEANUP/{task}",
                     f"Could not remove image {img}: {ret.stderr.strip()}", YELLOW)

        # 2. Remove prepared data directories
        for rel_dir in (cfg.get("data_dir", ""), cfg.get("test_data_dir", "")):
            if not rel_dir:
                continue
            # Resolve relative paths against task_dir (same as schemas.py)
            p = Path(rel_dir) if os.path.isabs(rel_dir) else task_dir / rel_dir
            if p.exists() and p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
                if p.exists():
                    _log(f"CLEANUP/{task}",
                         f"Partially removed data (some files remained): {p}", YELLOW)
                else:
                    _log(f"CLEANUP/{task}", f"Removed data: {p}", GREEN)

        # 3. Remove .prepared marker
        prepared = task_dir / ".prepared"
        if prepared.exists():
            prepared.unlink(missing_ok=True)
            _log(f"CLEANUP/{task}", "Removed .prepared marker", GREEN)

        _log(f"CLEANUP/{task}", "Done", CYAN)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _parse_agent_limits(
    agents: list[str],
    cfg: dict,
    default_concurrent: int,
) -> dict[str, int]:
    """Build {agent_name: max_concurrent} from config.

    Reads per-agent overrides from cfg["agent_config"][agent]["max_concurrent"],
    falls back to default_concurrent.
    """
    agent_config = cfg.get("agent_config", {})
    limits = {}
    for agent in agents:
        per_agent = agent_config.get(agent, {})
        limits[agent] = int(per_agent.get("max_concurrent", default_concurrent))
    return limits


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def run_one(
    agent: str,
    task: str,
    gpu_pool: GpuPool,
    n_gpus: int,
    timestamp: str,
    max_retries: int,
    retry_backoff: list[int],
    status: StatusWriter,
    concurrency_mgr: ConcurrencyManager,
    prepare_locks: PrepareLocks,
    task_cleanup: TaskCleanup | None = None,
) -> None:
    """Worker: prepare -> gate -> GPU -> run -> retry -> release.

    Wrapped in a top-level try/except so that any unhandled exception
    marks the task as failed instead of leaving it stuck in "running".
    """
    tag = f"{agent}/{task}"
    agent_id = f"{agent}_{task}_{timestamp}"

    try:
        _run_one_inner(tag, agent, task, agent_id, gpu_pool, n_gpus,
                       max_retries, retry_backoff, status, concurrency_mgr,
                       prepare_locks)
    except Exception as e:
        # _run_one_inner's finally already calls task_done, but if the
        # exception originated *inside* that finally block, task_done may
        # not have executed. Guard with try/except to be safe.
        _log(tag, f"UNEXPECTED ERROR: {e}", RED)
        try:
            status.update(tag, status="failed", gpus=[], error=f"unexpected: {e}")
        except Exception:
            pass
    finally:
        if task_cleanup is not None:
            task_cleanup.mark_done(task)


def _run_one_inner(
    tag: str,
    agent: str,
    task: str,
    agent_id: str,
    gpu_pool: GpuPool,
    n_gpus: int,
    max_retries: int,
    retry_backoff: list[int],
    status: StatusWriter,
    concurrency_mgr: ConcurrencyManager,
    prepare_locks: PrepareLocks,
) -> None:
    try:
        # ── Check shutdown before starting ──
        if _shutdown_event.is_set():
            status.update(tag, status="failed", error="shutdown requested")
            return

        # ── Prepare (serialized per task — avoids docker name conflicts) ──
        _log(tag, "Preparing task ...", CYAN)
        status.update(tag, status="preparing")
        prep_ok, prep_err = prepare_locks.prepare(task, cwd=str(REPO_ROOT))
        if not prep_ok:
            _log(tag, f"PREPARE FAILED: {prep_err}", RED)
            status.update(tag, status="failed", error=f"prepare failed: {prep_err}")
            return

        # ── Run with retry ──
        for attempt in range(max_retries + 1):
            # Check shutdown between retries
            if _shutdown_event.is_set() or _is_task_stopped(tag):
                _log(tag, "Stopped/shutdown requested, aborting", YELLOW)
                status.update(tag, status="failed", gpus=[], error="stopped")
                return

            # 1) Acquire agent concurrency gate (limits API calls per provider)
            cur_limit = concurrency_mgr.get_limit(agent)
            _log(tag, f"Waiting for {agent} API slot (limit={cur_limit}) ...", YELLOW)
            status.update(tag, status="queued", attempt=attempt, detail="waiting_api_slot")
            concurrency_mgr.acquire(agent)

            # Initialize variables used across try/except/finally to prevent
            # NameError if an exception occurs before they are assigned.
            gpus = []
            started = time.time()
            ret = -1
            stderr_tail = ""
            elapsed = 0.0
            _stop_poller = threading.Event()

            try:
                # 2) Acquire GPU(s) from shared pool
                _log(tag, f"Waiting for {n_gpus} GPU(s) ...", YELLOW)
                status.update(tag, status="queued", attempt=attempt, detail="waiting_gpu")
                gpus = gpu_pool.acquire(n_gpus, tag)
                gpu_str = ",".join(str(g) for g in gpus)
                cuda_ordinal_str = ",".join(str(i) for i in range(len(gpus)))

                started = time.time()
                _log(
                    tag,
                    f"Running on host GPUs [{gpu_str}] -> container CUDA [{cuda_ordinal_str}] "
                    f"- attempt {attempt+1}/{max_retries+1}",
                    GREEN,
                )
                task_cfg = _read_task_yaml(task)
                status.update(
                    tag,
                    status="running",
                    gpus=gpus,
                    agent_id=agent_id,
                    started_at=datetime.now().isoformat(timespec="seconds"),
                    attempt=attempt,
                    detail=None,
                    error=None,
                    time_budget_hours=task_cfg.get("total_time_budget_hours"),
                    max_iterations=task_cfg.get("max_iterations"),
                )

                # Background: poll experiment dir + live_status for dashboard updates
                def _poll_live_status():
                    exp_dir = None
                    for tick in range(int(task_cfg.get("total_time_budget_hours", 6) * 3600 / 5) + 60):
                        if _stop_poller.wait(5):
                            return
                        if exp_dir is None:
                            exp_dir = _find_experiment_dir(task, agent_id)
                            if exp_dir:
                                status.update(tag, experiment_dir=exp_dir)
                        if exp_dir:
                            live_file = REPO_ROOT / exp_dir / "live_status.json"
                            if live_file.exists():
                                try:
                                    with open(live_file) as f:
                                        live = json.load(f)
                                    status.update(
                                        tag,
                                        elapsed_hours=live.get("elapsed_hours"),
                                        total_iterations=live.get("total_iterations"),
                                        best_metric=live.get("best_primary_metric"),
                                        remaining_hours=live.get("remaining_hours"),
                                    )
                                except Exception:
                                    pass
                poller_thread = threading.Thread(target=_poll_live_status, daemon=True)
                poller_thread.start()

                cmd = [
                    "docker", "compose", "run", "--rm",
                    "-e", f"RAB_GPUS={gpu_str}",
                    "-e", f"NVIDIA_VISIBLE_DEVICES={gpu_str}",
                    "rab", "run",
                    "--task", task,
                    "--mode", "api",
                    "--agent-preset", agent,
                    "--agent-id", agent_id,
                ]
                stderr_f = tempfile.TemporaryFile(mode="w+")
                try:
                    proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT),
                                            env=_compose_env(_cuda_suffix),
                                            stderr=stderr_f)
                    # Poll so we can react to shutdown or per-task stop
                    while proc.poll() is None:
                        if _shutdown_event.is_set() or _is_task_stopped(tag):
                            proc.terminate()
                            try:
                                proc.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                proc.kill()
                                proc.wait()
                            # Kill sandbox/eval containers that the harness spawned
                            _cleanup_task_containers(agent_id)
                            break
                        time.sleep(1)
                    ret = proc.returncode if proc.returncode is not None else -1
                    # Read captured stderr tail on failure
                    if ret != 0:
                        try:
                            stderr_f.seek(0, 2)
                            size = stderr_f.tell()
                            stderr_f.seek(max(0, size - 2000))
                            stderr_tail = stderr_f.read()
                        except Exception:
                            pass
                finally:
                    stderr_f.close()
            except Exception as e:
                _log(tag, f"Exception: {e}", RED)
            finally:
                _stop_poller.set()
                if gpus:
                    gpu_pool.release(gpus)
                elapsed = time.time() - started
                # Always release the agent gate after each attempt
                concurrency_mgr.release(agent)

            # If stopped by user or shutdown, don't retry
            if _shutdown_event.is_set() or _is_task_stopped(tag):
                _log(tag, "Stopped by user", YELLOW)
                status.update(tag, status="failed", gpus=[], error="stopped")
                return

            if ret == 0:
                metrics = _read_metrics(task, agent_id)
                _log(tag, f"DONE \u2014 {elapsed/3600:.2f}h, best={metrics.get('best_metric')}", GREEN)
                status.update(
                    tag,
                    status="done",
                    gpus=[],
                    elapsed_hours=round(elapsed / 3600, 3),
                    finished_at=datetime.now().isoformat(timespec="seconds"),
                    detail=None,
                    **metrics,
                )
                return

            err_msg = f"exit {ret}"
            detail = _read_failure_detail(task, agent_id, stderr_tail)
            if detail:
                _log(tag, f"Failure detail:\n{detail}", RED)

            # Defensive cleanup: if the failed attempt left sandbox/eval
            # containers behind (e.g. an uncaught exception skipped
            # env.close()), remove them now so the next attempt can pass
            # the orphaned-container preflight check and start cleanly.
            _cleanup_task_containers(agent_id)

            if attempt < max_retries:
                # Check shutdown before sleeping for retry backoff
                if _shutdown_event.is_set() or _is_task_stopped(tag):
                    continue  # loop back to the top where we check and return
                wait = retry_backoff[min(attempt, len(retry_backoff) - 1)]
                _log(tag, f"FAILED ({err_msg}), retrying in {wait}s ...", YELLOW)
                status.update(
                    tag,
                    status="retrying",
                    gpus=[],
                    error=err_msg,
                    detail=detail or None,
                    next_retry_at=datetime.fromtimestamp(
                        time.time() + wait
                    ).isoformat(timespec="seconds"),
                )
                time.sleep(wait)
            else:
                _log(tag, f"FAILED after {max_retries+1} attempts ({err_msg})", RED)
                status.update(
                    tag,
                    status="failed",
                    gpus=[],
                    elapsed_hours=round(elapsed / 3600, 3),
                    error=err_msg,
                    detail=detail or None,
                )
    finally:
        # Always notify that this task slot is freed (triggers rebalancing)
        concurrency_mgr.task_done(agent)


# ---------------------------------------------------------------------------
# Restart / Add-task CLI handlers
# ---------------------------------------------------------------------------

def _handle_restart_cli(tags: list[str]) -> None:
    """CLI handler for --restart: validate status and write request files."""
    if not STATUS_FILE.exists():
        print(f"{RED}ERROR: No batch status found ({STATUS_FILE}){RESET}", file=sys.stderr)
        sys.exit(1)

    with open(STATUS_FILE) as f:
        data = json.load(f)
    runs = data.get("runs", {})

    for tag in tags:
        if tag not in runs:
            print(f"{RED}{tag} — not found in current batch.{RESET}")
            available = ", ".join(sorted(runs.keys())[:10])
            print(f"  Available: {available}{'...' if len(runs) > 10 else ''}")
            continue
        status = runs[tag].get("status", "unknown")
        if status == "failed":
            _write_request("restart", tag=tag)
            print(f"{GREEN}{tag} — restart requested (was '{status}'){RESET}")
        elif status == "done":
            print(f"{GREEN}{tag} — already completed successfully. No action needed.{RESET}")
        else:
            print(f"{YELLOW}{tag} — currently '{status}'. No action needed.{RESET}")


def _handle_restart_failed_cli() -> None:
    """CLI handler for --restart-failed: restart ALL failed tasks."""
    if not STATUS_FILE.exists():
        print(f"{RED}ERROR: No batch status found ({STATUS_FILE}){RESET}", file=sys.stderr)
        sys.exit(1)

    with open(STATUS_FILE) as f:
        data = json.load(f)
    runs = data.get("runs", {})

    failed = [tag for tag, r in runs.items() if r.get("status") == "failed"]
    if not failed:
        print(f"{GREEN}No failed tasks to restart.{RESET}")
        return

    print(f"Found {len(failed)} failed task(s):")
    for tag in failed:
        _write_request("restart", tag=tag)
        print(f"  {GREEN}{tag} — restart requested{RESET}")


def _handle_add_task_cli(task: str, agents: list[str]) -> None:
    """CLI handler for --add-task: check status, write request files."""
    # Validate task exists on disk
    if not (REPO_ROOT / "benchmarks" / task / "task.yaml").exists():
        print(f"{RED}ERROR: Task '{task}' not found (no benchmarks/{task}/task.yaml){RESET}",
              file=sys.stderr)
        sys.exit(1)

    if not STATUS_FILE.exists():
        print(f"{RED}ERROR: No batch status found ({STATUS_FILE}){RESET}", file=sys.stderr)
        sys.exit(1)

    with open(STATUS_FILE) as f:
        data = json.load(f)
    runs = data.get("runs", {})

    # Default to agents from the running batch config
    if not agents:
        agents = data.get("config", {}).get("agents", [])
    if not agents:
        print(f"{RED}ERROR: No agents specified (use --preset or rely on batch config){RESET}",
              file=sys.stderr)
        sys.exit(1)

    for agent in agents:
        tag = f"{agent}/{task}"
        if tag in runs:
            status = runs[tag].get("status", "unknown")
            if status == "failed":
                answer = input(f"{YELLOW}{tag} is '{status}'. Restart? [y/N]: {RESET}").strip().lower()
                if answer == "y":
                    _write_request("restart", tag=tag)
                    print(f"  {GREEN}Restart requested for {tag}{RESET}")
                else:
                    print(f"  Skipped {tag}")
            elif status == "done":
                metric = runs[tag].get("best_metric")
                print(f"{GREEN}{tag} — already completed (best={metric}). No action needed.{RESET}")
            else:
                print(f"{YELLOW}{tag} — currently '{status}'. No action needed.{RESET}")
        else:
            _write_request("add", tag=tag, agent=agent, task=task)
            print(f"{GREEN}{tag} — add-task requested{RESET}")


def _process_request(
    req: dict,
    gpu_pool: GpuPool,
    gpu_ids: list[int],
    concurrency_mgr: ConcurrencyManager,
    status: StatusWriter,
    prepare_locks: PrepareLocks,
    max_retries: int,
    retry_backoff: list[int],
    default_concurrent: int,
    task_cleanup: TaskCleanup | None = None,
) -> threading.Thread | None:
    """Process a single restart/add request. Returns a new Thread or None."""
    action = req.get("action")
    tag = req.get("tag", "")
    # Use microsecond precision to avoid timestamp collisions on rapid requests
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    if action == "restart":
        current = status.get_run_status(tag)
        if current != "failed":
            _log("MGMT", f"Skip restart {tag}: status is '{current}', not 'failed'", YELLOW)
            return None

        agent, task = tag.split("/", 1)
        n_gpus = _read_max_gpu_count(task)
        if n_gpus > len(gpu_ids):
            n_gpus = len(gpu_ids)

        # Clear per-task stop file so the restarted task doesn't immediately abort
        stop_file = REPO_ROOT / "experiments" / ".stop" / tag.replace("/", "__")
        stop_file.unlink(missing_ok=True)
        # Reset prepare cache so a prepare-failure can be retried
        prepare_locks.reset(task)
        # Bump cleanup counter so the restarted agent is tracked
        if task_cleanup is not None:
            task_cleanup.add_task(task, 1)
        # Reset status
        status.update(tag, status="queued", error=None, detail=None,
                      gpus=[], experiment_dir=None, best_metric=None,
                      elapsed_hours=None, finished_at=None)
        _log("MGMT", f"Restarting {tag}", GREEN)

        t = threading.Thread(
            target=run_one,
            args=(agent, task, gpu_pool, n_gpus, ts,
                  max_retries, retry_backoff, status, concurrency_mgr,
                  prepare_locks, task_cleanup),
            name=f"{tag}/restart",
            daemon=False,
        )
        return t

    if action == "add":
        current = status.get_run_status(tag)
        if current:
            _log("MGMT", f"Skip add {tag}: already exists with status '{current}'", YELLOW)
            return None

        agent = req["agent"]
        task = req["task"]
        # Validate task on disk
        if not (REPO_ROOT / "benchmarks" / task / "task.yaml").exists():
            _log("MGMT", f"Skip add {tag}: task.yaml not found", RED)
            return None

        n_gpus = _read_max_gpu_count(task)
        if n_gpus > len(gpu_ids):
            n_gpus = len(gpu_ids)

        # Ensure agent exists in concurrency manager
        concurrency_mgr.ensure_agent(agent, default_concurrent)
        # Track in cleanup counter
        if task_cleanup is not None:
            task_cleanup.add_task(task, 1)
        # Register in status
        status.init_runs([tag])
        _log("MGMT", f"Adding new task {tag}", GREEN)

        t = threading.Thread(
            target=run_one,
            args=(agent, task, gpu_pool, n_gpus, ts,
                  max_retries, retry_backoff, status, concurrency_mgr,
                  prepare_locks, task_cleanup),
            name=f"{tag}/add",
            daemon=False,
        )
        return t

    _log("MGMT", f"Unknown request action: {action}", RED)
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Register graceful shutdown handlers
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    parser = argparse.ArgumentParser(
        description="RABench runner \u2014 batch or single-task execution with GPU and API concurrency management.",
    )
    parser.add_argument("--config", default=str(REPO_ROOT / "scripts" / "config.yaml"))
    parser.add_argument("--preset", default=None, help="Comma-separated agent presets")
    parser.add_argument("--task", default=None, help="Single task name (single-task mode)")
    parser.add_argument("--tasks", default=None, help="Comma-separated task names")
    parser.add_argument("--stop", default=None,
                        help="Stop a running task: --stop agent/task (e.g. --stop claude/mnist_classification)")
    parser.add_argument("--restart", default=None,
                        help="Restart failed task(s): --restart agent/task[,agent/task2,...] ")
    parser.add_argument("--restart-failed", action="store_true",
                        help="Restart ALL failed tasks in the current batch")
    parser.add_argument("--add-task", default=None, metavar="TASK",
                        help="Add a new task to the running batch (uses --preset for agents, or batch config)")
    parser.add_argument("--gpus", default=None, help="Comma-separated ordered host GPU IDs")
    parser.add_argument("--max-retries", type=int, default=None)
    parser.add_argument("--max-concurrent", type=int, default=None,
                        help="Override max_concurrent_per_agent for all agents")
    parser.add_argument(
        "--cuda",
        default="",
        help="CUDA variant (cu118 or cu128). Required unless set in config.yaml.",
    )
    args = parser.parse_args()

    # ── Stop mode: signal a running task to stop ──
    if args.stop:
        stop_task(args.stop)
        print(f"Stop requested for: {args.stop}")
        print(f"  (wrote {REPO_ROOT / 'experiments' / '.stop' / args.stop.replace('/', '__')})")
        sys.exit(0)

    # ── Restart mode: request restart of failed tasks ──
    if args.restart:
        tags = [t.strip() for t in args.restart.split(",")]
        _handle_restart_cli(tags)
        sys.exit(0)

    # ── Restart-failed mode: restart ALL failed tasks ──
    if args.restart_failed:
        _handle_restart_failed_cli()
        sys.exit(0)

    # ── Add-task mode: add a new task to the running batch ──
    if args.add_task:
        add_agents = args.preset.split(",") if args.preset else []
        _handle_add_task_cli(args.add_task, add_agents)
        sys.exit(0)

    # Load config, CLI overrides
    cfg = {}
    cfg_path = Path(args.config)
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f) or {}

    agents = args.preset.split(",") if args.preset else cfg.get("agents", [])
    max_retries = args.max_retries if args.max_retries is not None else cfg.get("max_retries", 0)
    retry_backoff = cfg.get("retry_backoff", [30, 60, 120])
    default_concurrent = (
        args.max_concurrent if args.max_concurrent is not None
        else cfg.get("max_concurrent_per_agent", 2)
    )
    # CUDA variant: CLI --cuda overrides config cuda: (default cu118 if unset everywhere)
    if not args.cuda:
        args.cuda = cfg.get("cuda", "cu118")
    _validate_cuda_arg(args.cuda)
    global _cuda_suffix
    _cuda_suffix = args.cuda

    # ── Ensure base image is available ──
    _ensure_base_image(args.cuda)

    # ── Single-task mode ──
    if args.task:
        if not agents:
            print("ERROR: --preset required in single-task mode", file=sys.stderr)
            sys.exit(1)
        if not args.gpus:
            print("ERROR: --gpus required in single-task mode", file=sys.stderr)
            sys.exit(1)
        gpu_ids = [int(g) for g in args.gpus.split(",")]
        agent = agents[0]
        task = args.task
        tag = f"{agent}/{task}"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        n_gpus = _read_max_gpu_count(task)
        if n_gpus > len(gpu_ids):
            n_gpus = len(gpu_ids)

        pool = GpuPool(gpu_ids)
        agent_limits = _parse_agent_limits([agent], cfg, default_concurrent)
        cmgr = ConcurrencyManager(agent_limits)
        cmgr.init_pending({agent: 1})
        # Single-task mode does NOT write matrix_status.json — the dashboard
        # Matrix view is reserved for batch runs, so a debug/dev single-task
        # run won't clobber an in-flight batch's status.
        sw = StatusWriter(STATUS_FILE, {
            "gpus": gpu_ids, "agents": [agent], "total_tasks": 1,
            "max_retries": max_retries, "mode": "single",
        }, pool, cmgr, write_file=False)
        sw.init_runs([tag])

        plocks = PrepareLocks(hub_only=True, cuda_suffix=args.cuda)
        cuda_info = f", cuda={args.cuda}" if args.cuda else ""
        print(f"  Single-task: {tag}, GPUs={gpu_ids}, max_concurrent={agent_limits[agent]}{cuda_info}")
        run_one(agent, task, pool, n_gpus, timestamp, max_retries, retry_backoff, sw, cmgr, plocks)
        sys.exit(0 if sw.get_run_status(tag) == "done" else 1)

    # ── Batch mode ──
    if args.gpus:
        gpu_ids = [int(g) for g in args.gpus.split(",")]
    else:
        gpu_ids = cfg.get("gpus", [0])

    if args.tasks:
        tasks = args.tasks.split(",")
    else:
        tasks = cfg.get("tasks") or []
    if not tasks:
        tasks = _discover_tasks()

    if not agents:
        print("ERROR: no agents specified (use --preset or config.yaml agents:)", file=sys.stderr)
        sys.exit(1)
    if not tasks:
        print("ERROR: no tasks found", file=sys.stderr)
        sys.exit(1)

    # Validate task names early to catch typos before GPU allocation
    for t in tasks:
        if not (REPO_ROOT / "benchmarks" / t / "task.yaml").exists():
            print(f"ERROR: task '{t}' not found (no benchmarks/{t}/task.yaml)", file=sys.stderr)
            sys.exit(1)

    # Shuffle and offset task order per agent to minimize same-task collision.
    # Each agent gets the same shuffled list but rotated by a different offset,
    # so at any point in time different agents are working on different tasks.
    # This prevents two agents from running the same task simultaneously —
    # which could cause Docker image conflicts (e.g. one agent deletes/rebuilds
    # an image while another is using it).
    shuffled_tasks = list(tasks)
    random.shuffle(shuffled_tasks)
    n_tasks = len(shuffled_tasks)
    per_agent_tasks: dict[str, list[str]] = {}
    for idx, a in enumerate(agents):
        # Spread agents evenly across the task list using integer division
        offset = (idx * n_tasks // max(len(agents), 1)) % n_tasks
        per_agent_tasks[a] = shuffled_tasks[offset:] + shuffled_tasks[:offset]

    # Round-robin interleave: pick one task per agent in turn so different
    # agents' first tasks are spread apart in the thread start order.
    combos: list[tuple[str, str]] = []
    max_len = max(len(v) for v in per_agent_tasks.values())
    for i in range(max_len):
        for a in agents:
            if i < len(per_agent_tasks[a]):
                combos.append((a, per_agent_tasks[a][i]))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Clear stale stop files from previous runs
    stop_dir = REPO_ROOT / "experiments" / ".stop"
    if stop_dir.exists():
        for f in stop_dir.iterdir():
            f.unlink()

    # Build per-agent concurrency limits and task counts
    agent_limits = _parse_agent_limits(agents, cfg, default_concurrent)
    agent_task_counts = {a: sum(1 for aa, _ in combos if aa == a) for a in agents}

    pool = GpuPool(gpu_ids)
    cmgr = ConcurrencyManager(agent_limits)
    cmgr.init_pending(agent_task_counts)
    plocks = PrepareLocks(hub_only=True, cuda_suffix=args.cuda)
    # Track per-task agent counts for cleanup when all agents finish a task
    task_agent_counts = {t: sum(1 for _, tt in combos if tt == t) for t in tasks}
    tclean = TaskCleanup(task_agent_counts, cuda_suffix=args.cuda)
    _backup_matrix_status()
    sw = StatusWriter(STATUS_FILE, {
        "gpus": gpu_ids, "agents": agents, "total_tasks": len(combos),
        "max_retries": max_retries, "mode": "batch",
        "cuda": args.cuda or "default",
        "agent_concurrency": {a: agent_limits[a] for a in agents},
    }, pool, cmgr)
    sw.init_runs([f"{a}/{t}" for a, t in combos])

    print("=" * 60)
    print(f"  RABench Batch Run")
    print(f"  Agents:")
    for a in agents:
        print(f"    {a}: max_concurrent={agent_limits[a]}, tasks={agent_task_counts[a]}")
    print(f"  Tasks  : {len(tasks)} ({', '.join(tasks[:5])}{'...' if len(tasks) > 5 else ''})")
    print(f"  GPUs   : {gpu_ids} ({len(gpu_ids)} total)")
    if args.cuda:
        print(f"  CUDA   : {args.cuda}")
    print(f"  Combos : {len(combos)} (shuffled to avoid same-task collision)")
    print(f"  Retries: {max_retries} (backoff {retry_backoff}s)")
    print(f"  Time   : {datetime.now()}")
    print("=" * 60)
    # Show first few combos so the user can see the shuffled order
    print(f"\n  Execution order (first 8):")
    for a, t in combos[:8]:
        print(f"    {a}/{t}")
    if len(combos) > 8:
        print(f"    ... ({len(combos) - 8} more)")
    print()

    threads = []
    for agent, task in combos:
        n_gpus = _read_max_gpu_count(task)
        if n_gpus > len(gpu_ids):
            print(f"  WARNING: {task} needs {n_gpus} GPUs, pool has {len(gpu_ids)}, capping.",
                  file=sys.stderr)
            n_gpus = len(gpu_ids)
        t = threading.Thread(
            target=run_one,
            args=(agent, task, pool, n_gpus, timestamp, max_retries, retry_backoff,
                  sw, cmgr, plocks, tclean),
            name=f"{agent}/{task}",
            daemon=False,
        )
        threads.append(t)

    # Clear stale request files from previous runs
    if REQUEST_DIR.exists():
        for f in REQUEST_DIR.iterdir():
            f.unlink(missing_ok=True)

    for t in threads:
        t.start()

    # Management loop: wait for threads while picking up restart/add requests
    while True:
        if not _shutdown_event.is_set():
            for req in _read_and_consume_requests():
                new_t = _process_request(
                    req, pool, gpu_ids, cmgr, sw, plocks,
                    max_retries, retry_backoff, default_concurrent, tclean,
                )
                if new_t is not None:
                    threads.append(new_t)
                    new_t.start()

        # Prune finished threads to avoid unbounded list growth
        threads = [t for t in threads if t.is_alive()]
        if not threads:
            break
        # Sleep briefly before next poll
        time.sleep(2)

    # Summary
    print()
    print("=" * 60)
    print(f"  Batch Complete \u2014 {datetime.now()}")
    print("=" * 60)
    with open(STATUS_FILE) as f:
        final = json.load(f)
    for tag, r in final["runs"].items():
        s = r.get("status", "?")
        color = GREEN if s == "done" else RED if s == "failed" else YELLOW
        metric = r.get("best_metric")
        hours = r.get("elapsed_hours")
        detail = ""
        if metric is not None:
            detail += f"  best={metric}"
        if hours is not None:
            detail += f"  time={hours:.2f}h"
        exp = r.get("experiment_dir", "")
        print(f"{color}  {tag:40s} {s}{detail}{RESET}")
        if exp:
            print(f"    \u2192 {exp}")

    failed = [t for t, r in final["runs"].items() if r.get("status") != "done"]
    if failed:
        print(f"\nFailed: {', '.join(failed)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
