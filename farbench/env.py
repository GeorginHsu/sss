"""ResearchEnv: Gym-style environment interface.

Agent writes all code from scratch, executes commands, submits for evaluation.
"""

from __future__ import annotations

import copy
import os
from typing import Optional

from farbench.orchestrator import EnvironmentOrchestrator
from farbench.runner import SandboxRunner
from farbench.schemas import (
    Action,
    AgentMetadata,
    Observation,
    TaskConfig,
)
from farbench.storage import ExperimentStore
from farbench.tasks import TaskRegistry
from farbench.utils import get_logger

logger = get_logger(__name__)


def _prepared_cuda_variant(task_config: TaskConfig) -> str:
    from farbench.tasks import _infer_cuda_variant

    if task_config.docker_image:
        try:
            return _infer_cuda_variant(task_config.docker_image)
        except ValueError:
            return ""
    return ""


def _apply_runtime_cuda_variant(task_config: TaskConfig) -> None:
    """Rewrite the task image tag according to FARBENCH_CUDA, if configured."""
    cuda_suffix = os.environ.get("FARBENCH_CUDA", "").strip()
    if not cuda_suffix:
        return

    from farbench.tasks import _apply_cuda_variant, _validate_cuda

    _validate_cuda(cuda_suffix)
    old = task_config.docker_image
    if not old:
        return
    new = _apply_cuda_variant(old, cuda_suffix)
    if new != old:
        logger.info(f"CUDA variant (docker_image): {old} -> {new}")
        task_config.docker_image = new


def _ensure_task_prepared(task_config: TaskConfig, task_name: str) -> None:
    from farbench.tasks import TaskPreparer

    cuda_variant = _prepared_cuda_variant(task_config)
    if TaskPreparer(task_config).check_status(cuda_variant=cuda_variant):
        return
    raise RuntimeError(
        f"Task is not prepared under the current FARBench data policy. "
        f"Run: farbench tasks prepare {task_name}"
        + (f" --cuda {cuda_variant}" if cuda_variant else "")
    )


class ResearchEnv:
    """Gym-style environment interface for FARBench.

    Usage::

        env = ResearchEnv()
        obs = env.reset(task_name="mnist_classification")
        while True:
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
            if done:
                break
        env.close()
    """

    def __init__(
        self,
        benchmarks_dir: str = "benchmarks",
        experiments_dir: str = "experiments",
    ):
        self.benchmarks_dir = os.path.abspath(benchmarks_dir)
        self.experiments_dir = os.path.abspath(experiments_dir)
        self._registry = TaskRegistry(self.benchmarks_dir)
        self._registry.discover()

        self._orchestrator: Optional[EnvironmentOrchestrator] = None
        self._task_config: Optional[TaskConfig] = None
        self._store: Optional[ExperimentStore] = None

    @property
    def task_config(self) -> Optional[TaskConfig]:
        """Current task configuration (set after reset)."""
        return self._task_config

    @property
    def experiment_store(self) -> Optional["ExperimentStore"]:
        """Current experiment store (set after reset)."""
        return self._store

    def reset(
        self,
        task_name: str,
        agent_id: str = "default",
    ) -> Observation:
        """Initialize a new experiment session."""
        self._task_config = copy.copy(self._registry.get(task_name))

        # Honor FARBENCH_CUDA env var: rewrite docker image tags to match runtime CUDA variant.
        # task.yaml stores canonical cu118 tags; FARBENCH_CUDA=cu128 swaps at runtime.
        _apply_runtime_cuda_variant(self._task_config)

        _ensure_task_prepared(self._task_config, task_name)

        self._store = ExperimentStore(
            base_dir=self.experiments_dir,
            task_name=task_name,
            agent_id=agent_id,
        )

        sandbox_runner = SandboxRunner()

        self._orchestrator = EnvironmentOrchestrator(
            task_config=self._task_config,
            experiment_store=self._store,
            sandbox_runner=sandbox_runner,
        )

        obs = self._orchestrator.initialize()
        logger.info(f"Environment reset: task={task_name}, agent={agent_id}")
        return obs

    def resume(self, experiment_dir: str) -> Observation:
        """Re-attach to an existing experiment directory and continue running.

        Mirrors reset() but attaches to the on-disk experiment instead of
        creating a new one. The task_name and agent_id are recovered from
        `<experiment_dir>/config.json`; the last iter_NNN/ is archived as
        iter_NNN.aborted/ (treated as half-finished) before the next
        iteration resumes.
        """
        experiment_dir = os.path.abspath(experiment_dir)

        store = ExperimentStore.attach(experiment_dir)
        self._task_config = copy.copy(self._registry.get(store.task_name))

        # Honor FARBENCH_CUDA (same rewrite as reset) in case the host CUDA variant
        # differs from when the experiment first started.
        _apply_runtime_cuda_variant(self._task_config)

        _ensure_task_prepared(self._task_config, store.task_name)

        self._store = store
        sandbox_runner = SandboxRunner()
        self._orchestrator = EnvironmentOrchestrator(
            task_config=self._task_config,
            experiment_store=self._store,
            sandbox_runner=sandbox_runner,
        )

        obs = self._orchestrator.resume()
        logger.info(
            f"Environment resumed: task={store.task_name}, agent={store.agent_id}, "
            f"dir={experiment_dir}"
        )
        return obs

    def step(
        self,
        action: Action,
        agent_metadata: Optional[AgentMetadata] = None,
    ) -> tuple[Observation, float, bool, dict]:
        """Execute one iteration.

        Args:
            action: Action dataclass with files_to_write, command,
                    submit_eval, and/or done flag.
            agent_metadata: Optional LLM usage info.
        """
        if self._orchestrator is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        return self._orchestrator.run_iteration(action, agent_metadata)

    def status(self) -> dict:
        """Return current environment status."""
        if self._orchestrator is None:
            return {"status": "idle"}
        return {
            "status": "running",
            "task_name": self._task_config.name if self._task_config else None,
            "current_iteration": self._orchestrator.current_iteration,
            "remaining_time_budget_hours": round(
                self._orchestrator.remaining_budget_hours(), 4
            ),
            "remaining_iterations": self._orchestrator.remaining_iterations(),
            "max_iterations": self._task_config.max_iterations if self._task_config else None,
        }

    def mark_error(self, reason: str = "") -> None:
        """Record that the experiment terminated due to an error."""
        if self._orchestrator is not None:
            self._orchestrator.mark_error(reason)

    def close(self):
        """End the experiment and generate final summary."""
        if self._orchestrator is not None:
            if self._orchestrator.current_iteration > 0 or self._orchestrator._terminal_reason is not None:
                results = self._orchestrator.finalize()
                logger.info(f"Experiment closed: {results}")

            self._orchestrator.cleanup()
            self._orchestrator = None
