"""Experiment path management and JSON I/O."""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

from farbench.utils import get_logger

logger = get_logger(__name__)


# Trailing pattern stamped by ExperimentStore.__init__:
#   _YYYYMMDD_HHMMSS_NNNNNN   (NNNNNN = 6-digit microsecond remainder)
_STORE_TS_RE = re.compile(r"_\d{8}_\d{6}_\d{6}$")


def _strip_store_timestamp(dir_name: str) -> str:
    """Strip the ExperimentStore timestamp suffix to recover the agent_id.

    Example:
      gpt54_bigcodebench_codegen_20260423_200414_20260424_030426_352709
        -> gpt54_bigcodebench_codegen_20260423_200414
    """
    stripped = _STORE_TS_RE.sub("", dir_name)
    if stripped == dir_name:
        raise ValueError(
            f"Directory name does not end with an ExperimentStore timestamp "
            f"(_YYYYMMDD_HHMMSS_NNNNNN): {dir_name!r}"
        )
    return stripped


def _validate_path_segment(value: str, label: str) -> None:
    """Reject values that would escape the intended directory layout."""
    if not value or value in {".", ".."} or os.path.isabs(value):
        raise ValueError(f"{label} must be a non-empty relative path segment")
    if "/" in value or "\\" in value:
        raise ValueError(f"{label} must not contain path separators: {value!r}")


class ExperimentStore:
    """Path management and persistence for experiment data."""

    def __init__(self, base_dir: str, task_name: str, agent_id: str) -> None:
        _validate_path_segment(task_name, "task_name")
        _validate_path_segment(agent_id, "agent_id")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        timestamp += f"_{int(time.time() * 1_000_000) % 1_000_000:06d}"
        self.agent_id: str = agent_id
        self.task_name: str = task_name
        self.experiment_dir: str = os.path.join(
            base_dir, task_name, f"{agent_id}_{timestamp}"
        )
        self.workspace_dir: str = os.path.join(self.experiment_dir, "workspace")

    @classmethod
    def attach(cls, experiment_dir: str) -> "ExperimentStore":
        """Build a store attached to an EXISTING experiment directory.

        Used by resume: the directory + its timestamp are inherited, nothing
        new is created. agent_id is read from config.json so container naming
        (farbench-{agent_id_dashed}-*) matches the original run.
        """
        experiment_dir = os.path.abspath(experiment_dir)
        if not os.path.isdir(experiment_dir):
            raise FileNotFoundError(
                f"Experiment directory not found: {experiment_dir}"
            )

        config_path = os.path.join(experiment_dir, "config.json")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(
                f"config.json missing — cannot attach to {experiment_dir}"
            )
        with open(config_path) as f:
            config = json.load(f)
        task_name = config.get("task_name")
        if not task_name:
            raise ValueError(f"config.json has no task_name: {config_path}")

        # Recover agent_id from the directory name by stripping the trailing
        # ExperimentStore timestamp suffix (_YYYYMMDD_HHMMSS_NNNNNN).
        base = os.path.basename(experiment_dir)
        agent_id = _strip_store_timestamp(base)

        obj = cls.__new__(cls)
        obj.agent_id = agent_id
        obj.task_name = task_name
        obj.experiment_dir = experiment_dir
        obj.workspace_dir = os.path.join(experiment_dir, "workspace")
        return obj

    # ── Directory paths ──

    def iteration_dir(self, iteration: int) -> str:
        return os.path.join(self.experiment_dir, f"iter_{iteration:03d}")

    def summary_dir(self) -> str:
        return os.path.join(self.experiment_dir, "summary")

    # ── File paths ──

    def obs_path(self, iteration: int) -> str:
        return os.path.join(self.iteration_dir(iteration), "obs.json")

    def action_path(self, iteration: int) -> str:
        return os.path.join(self.iteration_dir(iteration), "action.json")

    def command_output_path(self, iteration: int) -> str:
        return os.path.join(self.iteration_dir(iteration), "command_output.json")

    def eval_result_path(self, iteration: int) -> str:
        return os.path.join(self.iteration_dir(iteration), "eval_result.json")

    def conversation_path(self, iteration: int) -> str:
        return os.path.join(self.iteration_dir(iteration), "llm_conversation.json")

    # ── Directory creation ──

    def ensure_dirs(self, iteration: int) -> None:
        """Create all directories needed for an iteration."""
        os.makedirs(self.iteration_dir(iteration), exist_ok=True)

    def ensure_workspace(self) -> None:
        os.makedirs(self.workspace_dir, exist_ok=True)

    def ensure_summary(self) -> None:
        os.makedirs(self.summary_dir(), exist_ok=True)

    # ── JSON I/O ──

    def save_json(self, path: str, data: dict[str, Any]) -> None:
        """Atomic JSON write (write to tmp file, then rename)."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_path = path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, path)
