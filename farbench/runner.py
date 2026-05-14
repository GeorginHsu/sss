"""SandboxRunner + EvalDockerRunner.

- SandboxRunner: persistent Docker container for arbitrary command execution.
- EvalDockerRunner: separate container for isolated evaluation.
"""

from __future__ import annotations

import json
import os
import shlex
import sys
import threading
import time
from dataclasses import asdict
from typing import Any, Optional

from farbench.schemas import CommandOutput, ComputeType, EvalResult, TaskConfig
from farbench.utils import get_logger, make_cuda_visible_devices

logger = get_logger(__name__)

_EVAL_USER = "65534:65534"  # nobody:nogroup; must not be able to modify /rab_data test files.


def _cap_cpu_cores(requested: int) -> int:
    """Cap requested CPU cores to the number available on the host."""
    available = os.cpu_count() or requested
    if requested > available:
        logger.warning(
            f"Requested {requested} CPU cores but only {available} available, "
            f"capping to {available}"
        )
        return available
    return requested


# Environment variables for multi-GPU/distributed runtimes (NCCL, Gloo) inside
# containers launched with ``--network=none``. Without these, vLLM's V1 engine
# (and any PyTorch distributed code) hangs indefinitely at startup because
# NCCL/Gloo try to resolve the container hostname via DNS and spin forever on
# SIOCGIFCONF, since the container has no external network interface.
#
# - NCCL_SOCKET_IFNAME / GLOO_SOCKET_IFNAME : force bootstrap over loopback
# - NCCL_IB_DISABLE=1                       : skip InfiniBand probing on non-IB hosts
# - NCCL_P2P_DISABLE=1                      : skip PCIe P2P on Ada/Ampere-class cards
#   without NVLink. NVIDIA's driver rejects PCIe P2P on these SKUs, and NCCL's
#   automatic recovery path occasionally deadlocks during probing. Forcing
#   shared-memory transport is strictly more reliable here and only marginally
#   slower for single-host tensor parallelism.
_DISTRIBUTED_NETWORK_ENV: dict[str, str] = {
    "NCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "NCCL_IB_DISABLE": "1",
    "NCCL_P2P_DISABLE": "1",
}


def _append_experiment_log(experiment_dir: Optional[str], text: str) -> None:
    """Append raw text to <experiment_dir>/experiment.log.

    Used to tee subprocess stdout/stderr alongside structured framework logs.
    Opens with mode 'a' so concurrent writes from logging.FileHandler stay
    line-atomic via O_APPEND.
    """
    if not experiment_dir or not text:
        return
    try:
        with open(
            os.path.join(experiment_dir, "experiment.log"),
            "a",
            encoding="utf-8",
        ) as f:
            f.write(text)
    except OSError:
        pass


def _to_host_path(path: str) -> str:
    """Remap a container-internal path to the host path for Docker bind-mounts.

    When the FARBench harness itself runs inside a container, paths like
    ``/workspace/experiments/...`` must be translated to the corresponding
    host directory so that nested Docker bind-mounts work correctly.
    """
    host_ws = os.environ.get("HOST_WORKSPACE", "")
    if not host_ws:
        return path
    host_ws = host_ws.rstrip("/")
    if path == "/workspace":
        return host_ws
    if path.startswith("/workspace/"):
        return host_ws + path[len("/workspace"):]
    return path


def _image_train_data_path(task_config: TaskConfig) -> str:
    return task_config.data_dir.rstrip("/")


def _image_test_data_path(task_config: TaskConfig) -> str:
    return task_config.test_data_dir.rstrip("/")


def _empty_bind_dir(experiment_dir: str, name: str) -> str:
    """Create an empty host dir used to mask private in-image data paths."""
    path = os.path.join(experiment_dir, ".farbench_empty_mounts", name)
    os.makedirs(path, exist_ok=True)
    return _to_host_path(os.path.abspath(path))


_docker_client: Any = None
_docker_lock = threading.Lock()


def _get_docker_client() -> Any:
    """Return a shared Docker client (lazy-initialized, thread-safe singleton)."""
    global _docker_client
    if _docker_client is None:
        with _docker_lock:
            if _docker_client is None:
                try:
                    import docker as docker_lib
                except ImportError:
                    raise ImportError(
                        "Docker support requires the 'docker' package. "
                        "Install it with: pip install docker"
                    )
                _docker_client = docker_lib.from_env()
    return _docker_client


# ═══════════════════════════════════════════
#  SandboxRunner
# ═══════════════════════════════════════════

class SandboxRunner:
    """Persistent Docker container for agent command execution.

    The container is created on first use and reused across all iterations:
    - Agent file changes are immediately visible via bind-mount.
    - Agent pip installs persist across iterations inside the container.
    - Arbitrary commands can be executed via execute_command().
    - Network access can be disabled for fairness.
    """

    def __init__(self):
        self._container: Any = None
        self._experiment_dir: Optional[str] = None
        self.agent_id: str = "agent"
        self._task_image: str = ""          # e.g. farbench/farbench:qm9-cu118

    def _ensure_image_available(self, task_config: TaskConfig) -> None:
        """Ensure Docker image exists locally."""
        client = _get_docker_client()
        try:
            client.images.get(task_config.docker_image)
            return  # Already available locally
        except Exception as e:
            raise RuntimeError(
                f"Docker image '{task_config.docker_image}' not found locally.\n\n"
                f"Pre-built task images are distributed as Hugging Face dataset "
                f"archives. Prepare the task before running:\n"
                f"  farbench tasks prepare {task_config.name} --cuda <cu118|cu128>\n"
                f"or:\n"
                f"  farbench run --task {task_config.name} --prepare --cuda <cu118|cu128>\n\n"
                f"Original Docker error: {e}"
            ) from e

    def _ensure_container(
        self,
        workspace_path: str,
        experiment_dir: str,
        task_config: TaskConfig,
        device_requests: Any,
        resource_kwargs: dict,
    ) -> None:
        """Create the persistent sandbox container on first use."""
        if self._container is not None:
            return

        import docker as docker_lib

        self._experiment_dir = experiment_dir
        os.makedirs(experiment_dir, exist_ok=True)

        # Ensure the Docker image is available locally. Task preparation loads
        # pre-built images from Hugging Face dataset archives when available.
        self._ensure_image_available(task_config)

        volumes = {
            _to_host_path(os.path.abspath(workspace_path)): {"bind": "/workspace", "mode": "rw"},
            _to_host_path(os.path.abspath(experiment_dir)): {"bind": "/experiments", "mode": "rw"},
            _empty_bind_dir(experiment_dir, "test_data"): {
                "bind": _image_test_data_path(task_config),
                "mode": "ro",
            },
        }

        agent_safe = self.agent_id.replace("_", "-")
        container_name = f"farbench-{agent_safe}-sandbox"

        client = _get_docker_client()

        # Network isolation: disabled by default for fairness
        network_mode = "bridge" if task_config.network_access else "none"

        try:
            self._container = client.containers.run(
                image=task_config.docker_image,
                command=[
                    "bash",
                    "-lc",
                    f"rm -rf /data && ln -s {shlex.quote(_image_train_data_path(task_config))} /data && sleep infinity",
                ],
                name=container_name,
                volumes=volumes,
                working_dir="/workspace",
                device_requests=device_requests,
                shm_size="32g",
                detach=True,
                network_mode=network_mode,
                environment={
                    "FARBENCH_IN_DOCKER": "1",
                    "FARBENCH_DATA_DIR": "/data",
                    # When network is disabled, force offline mode for common ML libraries
                    **({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"} if not task_config.network_access else {}),
                    # NCCL/Gloo: force loopback when network is disabled, else harmless
                    **_DISTRIBUTED_NETWORK_ENV,
                    # Search tools — passed through from host .env if set
                    **{k: os.environ[k] for k in ("GITHUB_TOKEN", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN") if k in os.environ},
                },
                **resource_kwargs,
            )
        except Exception as e:
            if "409" in str(e) or "Conflict" in str(e):
                raise RuntimeError(
                    f"Container name conflict: '{container_name}' already exists.\n\n"
                    f"A previous experiment may not have exited cleanly. Remove it with:\n\n"
                    f"  docker rm -f {container_name}\n"
                ) from e
            raise
        self._task_image = task_config.docker_image
        logger.info(
            f"Sandbox container started: {container_name} "
            f"({self._container.short_id}, image={task_config.docker_image}, "
            f"network={network_mode})"
        )

    def ensure_ready(
        self,
        workspace_path: str,
        experiment_dir: str,
        task_config: TaskConfig,
    ) -> None:
        """Ensure the sandbox container is running."""
        import docker as docker_lib

        device_requests = None
        if task_config.compute_type == ComputeType.GPU:
            gpu_count = task_config.max_gpu_count or -1
            selected_gpus = make_cuda_visible_devices(gpu_count)
            device_requests = [
                docker_lib.types.DeviceRequest(
                    device_ids=selected_gpus.split(","),
                    capabilities=[["gpu"]],
                )
            ]
            logger.info(f"GPU passthrough: {selected_gpus}")

        resource_kwargs: dict = {}
        if task_config.max_cpu_cores:
            resource_kwargs["nano_cpus"] = int(_cap_cpu_cores(task_config.max_cpu_cores) * 1e9)
        if task_config.max_memory_gb:
            resource_kwargs["mem_limit"] = f"{task_config.max_memory_gb}g"

        self._ensure_container(
            workspace_path=workspace_path,
            experiment_dir=experiment_dir,
            task_config=task_config,
            device_requests=device_requests,
            resource_kwargs=resource_kwargs,
        )

    def execute_command(
        self,
        command: str,
        timeout_seconds: int = 300,
        iteration: int = 0,
    ) -> CommandOutput:
        """Execute a shell command in the sandbox container.

        Returns CommandOutput with stdout, stderr, exit_code, timing.
        """
        if self._container is None:
            logger.error("execute_command called before sandbox container initialized")
            return CommandOutput(
                stderr="Sandbox container not initialized",
                exit_code=-1,
            )

        # Strip trailing '&' to prevent background execution.
        # Background processes break exit_code detection and output capture.
        sanitized = command.rstrip()
        if sanitized.endswith("&"):
            sanitized = sanitized[:-1].rstrip()
            logger.warning(f"Stripped trailing '&' from command (background execution not supported)")

        logger.info(f"Executing in sandbox: {sanitized[:200]}")
        start_time = time.time()

        exp_dir = self._experiment_dir
        _append_experiment_log(
            exp_dir,
            f"\n========== [iter {iteration}] sandbox $ {sanitized}\n",
        )

        exec_result: dict = {"exit_code": -1, "stdout": "", "stderr": "", "done": False}

        def _run_exec():
            try:
                resp = self._container.client.api.exec_create(
                    self._container.id,
                    ["bash", "-c", f"set -o pipefail; export PYTHONPATH=/workspace/.pip_packages:$PYTHONPATH; {sanitized}"],
                    workdir="/workspace",
                    stdout=True,
                    stderr=True,
                    environment=[
                        f"FARBENCH_ITERATION={iteration}",
                        "PYTHONWARNINGS=ignore",
                    ],
                )
                exec_id = resp["Id"]
                output = self._container.client.api.exec_start(
                    exec_id, stream=True, demux=True,
                )
                stdout_parts = []
                stderr_parts = []
                for stdout_chunk, stderr_chunk in output:
                    if stdout_chunk:
                        text = stdout_chunk.decode("utf-8", errors="replace")
                        stdout_parts.append(text)
                        sys.stdout.write(text)
                        sys.stdout.flush()
                        _append_experiment_log(exp_dir, text)
                    if stderr_chunk:
                        text = stderr_chunk.decode("utf-8", errors="replace")
                        stderr_parts.append(text)
                        sys.stderr.write(text)
                        sys.stderr.flush()
                        _append_experiment_log(exp_dir, text)

                inspect = self._container.client.api.exec_inspect(exec_id)
                exec_result["exit_code"] = inspect.get("ExitCode", -1)
                exec_result["stdout"] = "".join(stdout_parts)
                exec_result["stderr"] = "".join(stderr_parts)
            except Exception as e:
                logger.error(f"exec error: {e}")
                exec_result["stderr"] = str(e)
            exec_result["done"] = True

        exec_thread = threading.Thread(target=_run_exec, daemon=True)
        exec_thread.start()
        exec_thread.join(timeout=timeout_seconds)

        elapsed = time.time() - start_time

        if not exec_result["done"]:
            logger.warning("Command timed out, killing process in container")
            try:
                self._container.exec_run(["pkill", "-9", "-f", command])
            except Exception as e:
                logger.debug(f"Failed to kill timed-out process: {e}")
            exec_thread.join(timeout=5)
            _append_experiment_log(
                exp_dir,
                f"========== [iter {iteration}] sandbox TIMEOUT after {timeout_seconds}s "
                f"(elapsed={elapsed:.1f}s)\n",
            )
            return CommandOutput(
                stdout=exec_result.get("stdout", ""),
                stderr=exec_result.get("stderr", "") + f"\n[TIMEOUT] Command exceeded {timeout_seconds}s limit. Consider reducing epochs or using a smaller model.",
                exit_code=-1,
                timed_out=True,
                elapsed_seconds=elapsed,
            )

        logger.info(
            f"Command finished in {elapsed:.1f}s, exit_code={exec_result['exit_code']}"
        )
        _append_experiment_log(
            exp_dir,
            f"========== [iter {iteration}] sandbox exit_code={exec_result['exit_code']} "
            f"elapsed={elapsed:.1f}s\n",
        )

        # Detect dead/unreachable container.
        # exit_code -1 means the Docker exec call itself failed (not the
        # command inside).  Check stderr for container-level errors.
        _CONTAINER_DEAD_MARKERS = (
            "No such container",
            "is not running",
            "is restarting",
            "is paused",
            "removal of container",
            "socket is not connected",
        )
        if exec_result["exit_code"] == -1:
            err = exec_result.get("stderr", "")
            if any(marker in err for marker in _CONTAINER_DEAD_MARKERS):
                logger.error(f"Sandbox container is unavailable: {err[:200]}")
                self._container = None
                raise RuntimeError(
                    f"Sandbox container is unavailable ({err[:200]}). "
                    "The experiment cannot continue."
                )

        return CommandOutput(
            stdout=exec_result["stdout"],
            stderr=exec_result["stderr"],
            exit_code=exec_result["exit_code"],
            timed_out=False,
            elapsed_seconds=elapsed,
        )

    def cleanup(self) -> None:
        """Stop and remove the sandbox container. Docker images are never removed."""
        if self._container is not None:
            try:
                self._container.stop(timeout=10)
                self._container.remove(force=True)
            except Exception as e:
                logger.debug(f"Sandbox container cleanup: {e}")
            self._container = None


# ═══════════════════════════════════════════
#  EvalDockerRunner
# ═══════════════════════════════════════════

class EvalDockerRunner:
    """Keeps a Docker eval container alive across all iterations.

    Runs evaluation in an isolated container where:
    - Agent's predict script produces predictions from test inputs
    - System evaluator compares predictions against ground truth
    - Agent never sees test data labels
    """

    def __init__(self):
        self._container: Any = None
        self.agent_id: str = "agent"

    def _ensure_container(
        self,
        workspace_path: str,
        task_config: TaskConfig,
        experiment_dir: str,
        device_requests: Any,
        resource_kwargs: dict | None = None,
    ) -> None:
        if self._container is not None:
            return

        import docker as docker_lib

        farbench_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        os.makedirs(experiment_dir, exist_ok=True)

        volumes: dict = {
            _to_host_path(os.path.abspath(workspace_path)): {"bind": "/workspace", "mode": "ro"},
            _to_host_path(os.path.abspath(task_config.script_dir)): {"bind": "/eval_script", "mode": "ro"},
            _to_host_path(os.path.abspath(experiment_dir)): {"bind": "/experiments", "mode": "rw"},
            _to_host_path(os.path.abspath(farbench_project_root)): {"bind": "/farbench_pkg", "mode": "ro"},
            _empty_bind_dir(experiment_dir, "train_data"): {
                "bind": _image_train_data_path(task_config),
                "mode": "ro",
            },
        }
        agent_safe = self.agent_id.replace("_", "-")
        container_name = f"farbench-{agent_safe}-eval"

        client = _get_docker_client()

        try:
            self._container = client.containers.run(
                image=task_config.docker_image,
                command=["sleep", "infinity"],
                name=container_name,
                volumes=volumes,
                working_dir="/workspace",
                device_requests=device_requests,
                shm_size="32g",
                detach=True,
                network_mode="none",   # eval always network-isolated
                security_opt=["no-new-privileges:true"],
                environment={
                    "FARBENCH_IN_DOCKER": "1",
                    "FARBENCH_DATA_DIR": "/data",
                    "FARBENCH_TEST_DATA_DIR": "/data",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "TMPDIR": "/tmp",
                    "HOME": "/tmp",
                    "XDG_CACHE_HOME": "/tmp/cache",
                    "MPLCONFIGDIR": "/tmp/mplconfig",
                    # Eval container is always --network=none; prevent NCCL/Gloo hang
                    **_DISTRIBUTED_NETWORK_ENV,
                },
                **(resource_kwargs or {}),
            )
        except Exception as e:
            if "409" in str(e) or "Conflict" in str(e):
                raise RuntimeError(
                    f"Container name conflict: '{container_name}' already exists.\n\n"
                    f"A previous experiment may not have exited cleanly. Remove it with:\n\n"
                    f"  docker rm -f {container_name}\n"
                ) from e
            raise
        link_result = self._container.exec_run([
            "bash",
            "-lc",
            f"rm -rf /data && ln -s {shlex.quote(_image_test_data_path(task_config))} /data",
        ])
        if link_result.exit_code != 0:
            raise RuntimeError(
                f"Failed to link eval /data to {_image_test_data_path(task_config)}: "
                f"{link_result.output.decode('utf-8', errors='replace')}"
            )
        logger.info(
            f"Eval container started: {container_name} "
            f"({self._container.short_id}, image={task_config.docker_image})"
        )

    def evaluate(
        self,
        checkpoint_path: str,
        predict_script: str,
        task_config: TaskConfig,
        workspace_path: str,
        output_dir: str,
    ) -> EvalResult:
        """Run evaluation: agent's predict script + system evaluator."""
        import docker as docker_lib

        experiment_dir = os.path.dirname(output_dir)

        device_requests = None
        if task_config.compute_type == ComputeType.GPU:
            gpu_count = task_config.max_gpu_count or -1
            selected_gpus = make_cuda_visible_devices(gpu_count)
            device_requests = [
                docker_lib.types.DeviceRequest(
                    device_ids=selected_gpus.split(","),
                    capabilities=[["gpu"]],
                )
            ]

        resource_kwargs: dict = {}
        if task_config.max_cpu_cores:
            resource_kwargs["nano_cpus"] = int(_cap_cpu_cores(task_config.max_cpu_cores) * 1e9)
        if task_config.max_memory_gb:
            resource_kwargs["mem_limit"] = f"{task_config.max_memory_gb}g"

        self._ensure_container(workspace_path, task_config, experiment_dir, device_requests, resource_kwargs)

        iter_name = os.path.basename(output_dir)
        container_output_dir = f"/experiments/{iter_name}"

        # Resolve checkpoint path relative to experiment dir
        checkpoint_rel = os.path.relpath(
            os.path.join(workspace_path, checkpoint_path),
            experiment_dir,
        ) if not os.path.isabs(checkpoint_path) else os.path.relpath(checkpoint_path, experiment_dir)

        # If checkpoint is inside workspace, use /workspace mount
        if checkpoint_path and not os.path.isabs(checkpoint_path):
            container_checkpoint = f"/workspace/{checkpoint_path}"
        else:
            container_checkpoint = f"/experiments/{checkpoint_rel}"

        mkdir_result = self._container.exec_run(["mkdir", "-p", container_output_dir])
        if mkdir_result.exit_code != 0:
            raise RuntimeError(
                f"Failed to create eval output dir {container_output_dir}: "
                f"{mkdir_result.output.decode('utf-8', errors='replace')}"
            )
        chmod_result = self._container.exec_run(["chmod", "a+rwx", container_output_dir])
        if chmod_result.exit_code != 0:
            raise RuntimeError(
                f"Failed to make eval output dir writable {container_output_dir}: "
                f"{chmod_result.output.decode('utf-8', errors='replace')}"
            )
        tc_dict = asdict(task_config)
        tc_dict["data_dir"] = "/data"
        tc_dict["test_data_dir"] = "/data"
        tc_dict["script_dir"] = "/eval_script"
        tc_dict["task_dir"] = ""

        environment = {
            "FARBENCH_EVAL_CHECKPOINT": container_checkpoint,
            "FARBENCH_EVAL_WORKSPACE": "/workspace",
            "FARBENCH_EVAL_OUTPUT": f"{container_output_dir}/eval_result.json",
            "FARBENCH_EVAL_SCRIPT_DIR": "/eval_script",
            "FARBENCH_TASK_CONFIG_JSON": json.dumps(tc_dict),
            "FARBENCH_PREDICT_SCRIPT": f"/workspace/{predict_script}",
            # /farbench_pkg: FARBench framework; /workspace/.pip_packages: agent-installed packages
            "PYTHONPATH": "/farbench_pkg:/workspace/.pip_packages",
            # Eval container is always network-isolated; force offline mode
            # so HF/transformers use local cache instead of hanging on network calls
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "TMPDIR": "/tmp",
            "HOME": "/tmp",
            "XDG_CACHE_HOME": "/tmp/cache",
            "MPLCONFIGDIR": "/tmp/mplconfig",
            # NCCL/Gloo: force loopback for vLLM/distributed startup under --network=none
            **_DISTRIBUTED_NETWORK_ENV,
        }

        logger.info(
            f"Running evaluation via docker exec in {self._container.name} "
            f"(checkpoint={checkpoint_path}, predict={predict_script})"
        )

        _append_experiment_log(
            experiment_dir,
            f"\n========== [{iter_name}] eval_harness $ python -m farbench.eval_harness "
            f"(checkpoint={checkpoint_path}, predict={predict_script})\n",
        )
        eval_start = time.time()

        exec_result: dict = {"exit_code": -1, "output": ""}

        try:
            exec_kwargs = {
                "container": self._container.id,
                "cmd": [
                    "bash",
                    "-c",
                    "export PYTHONPATH=/farbench_pkg:/workspace/.pip_packages:$PYTHONPATH; "
                    "exec python -m farbench.eval_harness",
                ],
                "environment": {k: v for k, v in environment.items() if k != "PYTHONPATH"},
                "workdir": "/workspace",
                "user": _EVAL_USER,
            }
            resp = self._container.client.api.exec_create(**exec_kwargs)
            exec_id = resp["Id"]
            output = self._container.client.api.exec_start(exec_id, stream=True)
            captured_parts = []
            for chunk in output:
                if chunk:
                    text = chunk.decode("utf-8", errors="replace")
                    captured_parts.append(text)
                    sys.stdout.write(text)
                    sys.stdout.flush()
                    _append_experiment_log(experiment_dir, text)
            exec_result["output"] = "".join(captured_parts)
            inspect = self._container.client.api.exec_inspect(exec_id)
            exec_result["exit_code"] = inspect.get("ExitCode", -1)
        except Exception as e:
            logger.error(f"Eval exec error: {e}")
            _append_experiment_log(experiment_dir, f"[eval_harness exec error] {e}\n")

        _append_experiment_log(
            experiment_dir,
            f"========== [{iter_name}] eval_harness exit_code={exec_result['exit_code']} "
            f"elapsed={time.time() - eval_start:.1f}s\n",
        )

        exit_code = exec_result["exit_code"]

        # Always save full eval output to file for debugging (success or failure)
        captured_full = exec_result.get("output", "")
        if captured_full:
            eval_log_path = os.path.join(output_dir, "eval_harness_output.log")
            try:
                with open(eval_log_path, "w", encoding="utf-8") as f:
                    f.write(captured_full)
            except OSError:
                pass

        if exit_code != 0:
            # Include captured container output so the agent can diagnose
            # Show last 2000 chars (was 1000 — often too short for stack traces)
            log_tail = captured_full[-2000:] if len(captured_full) > 2000 else captured_full
            raise RuntimeError(
                f"Evaluation failed in container {self._container.name} "
                f"(exit {exit_code}):\n{log_tail}"
            )

        eval_result_path = os.path.join(output_dir, "eval_result.json")
        with open(eval_result_path, "r") as f:
            data = json.load(f)
        result = EvalResult(**data)

        # Append eval harness container output so the agent can see the
        # full evaluation process (Phase 1 predict + Phase 2 scoring).
        captured = exec_result.get("output", "").strip()
        if captured:
            existing = result.eval_log.strip() if result.eval_log else ""
            tail = captured[-2000:] if len(captured) > 2000 else captured
            result.eval_log = f"{existing}\n{tail}".strip() if existing else tail

        return result

    def cleanup(self) -> None:
        if self._container is not None:
            try:
                self._container.stop(timeout=10)
                self._container.remove(force=True)
            except Exception as e:
                logger.debug(f"Eval container cleanup: {e}")
            self._container = None
