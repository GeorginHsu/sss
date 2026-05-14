"""Core data structures (dataclasses), single source of truth.

Agents write all code from scratch, execute arbitrary commands in a
Docker sandbox, and explicitly request evaluation when ready.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════
#  Enums
# ═══════════════════════════════════════════

class ComputeType(str, Enum):
    """Hardware compute type for task execution."""
    GPU = "gpu"
    CPU = "cpu"


class TerminalReason(str, Enum):
    """Why an experiment ended."""
    TIME_BUDGET_EXHAUSTED = "time_budget_exhausted"
    ITERATION_LIMIT_REACHED = "iteration_limit_reached"
    AGENT_DONE = "agent_done"
    ERROR = "error"


class AgentMode(str, Enum):
    """CLI agent interaction mode."""
    DEMO = "demo"
    INTERACTIVE = "interactive"
    API = "api"


# ═══════════════════════════════════════════
#  Agent Action
# ═══════════════════════════════════════════

@dataclass
class EvalSubmission:
    """Agent's request to evaluate a checkpoint."""
    checkpoint_path: str = ""      # relative path inside workspace
    predict_script: str = ""       # relative path to prediction script


@dataclass
class Action:
    """What the agent wants to do in one iteration.

    All fields are optional — the agent can do any combination:
      - Write files only (explore data structure, set up code)
      - Execute a command only (run training, inspect output)
      - Submit for evaluation only (test a previously trained model)
      - Any combination of the above
      - Set done=True to terminate the experiment
    """
    # Reasoning: plain text string in Observation-Analysis-Decision (O-A-D) format.
    # Full text is preserved in iteration history for agent learning and reproducibility.
    reasoning: Any = ""
    files_to_write: dict[str, str] = field(default_factory=dict)
    packages_to_install: list[str] = field(default_factory=list)  # pip packages installed before command
    command: Optional[str] = None
    submit_eval: Optional[EvalSubmission] = None
    done: bool = False
    done_reason: str = ""  # optional explanation when done=True

    def to_dict(self) -> dict:
        d = asdict(self)
        if d["submit_eval"] is None:
            del d["submit_eval"]
        if not d["packages_to_install"]:
            del d["packages_to_install"]
        if not d["done_reason"]:
            del d["done_reason"]
        return d

    @classmethod
    def from_dict(cls, data: dict) -> Action:
        action = cls()
        action.reasoning = data.get("Reasoning", data.get("reasoning", ""))
        action.files_to_write = data.get("files_to_write", {})
        action.packages_to_install = data.get("packages_to_install", [])
        action.command = data.get("command")
        action.done = data.get("done", False)
        action.done_reason = data.get("done_reason", "")
        sub = data.get("submit_eval")
        if sub and isinstance(sub, dict):
            action.submit_eval = EvalSubmission(**sub)
        return action


# ═══════════════════════════════════════════
#  Command Output
# ═══════════════════════════════════════════

@dataclass
class CommandOutput:
    """Result of executing a command in the sandbox."""
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    timed_out: bool = False
    elapsed_seconds: float = 0.0


# ═══════════════════════════════════════════
#  Evaluation
# ═══════════════════════════════════════════

@dataclass
class EvalResult:
    """Evaluation result from running predictions on the test set."""
    metrics: dict[str, float] = field(default_factory=dict)
    primary_metric_name: str = ""
    primary_metric_value: float = 0.0
    inference_time_ms: float = 0.0
    eval_log: str = ""


# ═══════════════════════════════════════════
#  Token Usage (value object)
# ═══════════════════════════════════════════

@dataclass
class TokenUsage:
    """Token counts for a single LLM call.  Travels as a unit through the
    entire tracking chain: API → Agent → Orchestrator → storage."""
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0

    @property
    def total(self) -> int:
        # thinking_tokens is a breakdown of output_tokens (already included),
        # not an independent bucket, so we do NOT add it here.
        return (self.input_tokens + self.output_tokens
                + self.cache_read_tokens + self.cache_creation_tokens)

    def __iadd__(self, other: "TokenUsage") -> "TokenUsage":
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.thinking_tokens += other.thinking_tokens
        self.cache_read_tokens += other.cache_read_tokens
        self.cache_creation_tokens += other.cache_creation_tokens
        return self

    def to_dict(self) -> dict:
        return asdict(self)


# ═══════════════════════════════════════════
#  Iteration Record
# ═══════════════════════════════════════════

@dataclass
class IterationRecord:
    """Summary of a single iteration (for history tracking)."""
    iteration: int = 0
    command: Optional[str] = None
    files_written: list[str] = field(default_factory=list)
    command_output_summary: str = ""       # truncated stdout for history
    error_summary: str = ""                # truncated stderr (only on failure)
    eval_submitted: bool = False
    eval_result: Optional[EvalResult] = None  # full eval result (if eval was submitted)
    eval_error_log: str = ""               # eval container stderr on failure
    reward: float = 0.0                    # signed delta vs previous best
    elapsed_seconds: float = 0.0
    description: str = ""
    token_usage: TokenUsage = field(default_factory=TokenUsage)


@dataclass
class AgentMetadata:
    """Optional LLM usage info reported by the agent."""
    llm_calls: int = 0
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    model_name: str = ""
    latency_seconds: float = 0.0
    role: str = ""
    # Full LLM conversation log (messages sent + raw response)
    conversation: Optional[dict] = None


# ═══════════════════════════════════════════
#  Observation
# ═══════════════════════════════════════════

@dataclass
class Observation:
    """Complete information the agent receives after each iteration.

    In the autonomous model, agents see:
    - Task description (on reset)
    - Workspace file listing (not contents — agent reads via commands)
    - Command output from the last executed command
    - Eval result if an evaluation was submitted
    - Time and iteration budget remaining
    - History of past iterations
    """

    # Task context (only set on reset)
    task_description: Optional[str] = None

    # Workspace state: list of relative file paths
    workspace_files: list[str] = field(default_factory=list)

    # Workspace file contents (key=relative path, value=content, may be truncated)
    workspace_file_contents: dict[str, str] = field(default_factory=dict)

    # Current iteration
    iteration: int = 0

    # Command execution result (from this iteration)
    command_output: Optional[CommandOutput] = None

    # Evaluation result (from this iteration, if eval was submitted)
    eval_result: Optional[EvalResult] = None

    # Budget tracking
    remaining_time_budget_hours: Optional[float] = None
    remaining_iterations: Optional[int] = None

    # History of past iterations
    history: list[IterationRecord] = field(default_factory=list)

    # Error info
    error: Optional[str] = None

    # Best evaluation result so far
    best_eval_result: Optional[EvalResult] = None

    # Training curves: {iter_N: [{"step": ..., "train_loss": ..., ...}, ...]}
    training_curves: dict[str, list] = field(default_factory=dict)

    # Training curve images: {iter_N: file path to PNG on disk}
    training_curve_images: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


# ═══════════════════════════════════════════
#  TaskConfig
# ═══════════════════════════════════════════

@dataclass
class TaskConfig:
    """Task configuration, loaded from task.yaml."""

    name: str = ""
    description: str = ""
    # domain accepts str (single) or list[str] (multi-category) in task.yaml;
    # from_yaml/from_dict normalize to list[str] so consumers can treat it uniformly.
    domain: list[str] = field(default_factory=list)
    subdomain: str = ""

    compute_type: ComputeType = ComputeType.GPU

    # Paths
    task_dir: str = ""
    script_dir: str = ""
    data_dir: str = ""          # in-image train data path, e.g. /rab_data/<task>
    test_data_dir: str = ""     # in-image test data path, hidden from the agent
    docker_image: str = ""

    # Evaluation
    primary_metric: str = ""
    higher_is_better: bool = True
    evaluator_class: str = ""

    # Evaluation contract: documents what the agent must produce
    eval_contract: dict = field(default_factory=dict)

    # Time budget (core constraint — wall-clock time)
    total_time_budget_hours: float = 10.0

    # Iteration budget (hard cap on number of agent iterations)
    max_iterations: int = 30

    # Network access (disabled by default for fairness)
    network_access: bool = False

    # Max allowed command_timeout per action (seconds).
    # Default: derived from total_time_budget_hours (see from_yaml).
    max_command_timeout: Optional[int] = None

    # Resource constraints
    max_gpu_count: int = 4
    per_gpu_memory_gb: Optional[float] = None  # per-GPU VRAM; None = auto-detect
    max_cpu_cores: int = 24
    max_memory_gb: float = 48  # CPU RAM in GB

    # Upper bound (in billions) on total model parameters the agent may use
    # (pretrained + any added/trained weights).  Surfaced in the agent's
    # system prompt; exceeding this triggers a score penalty at eval time.
    # Default 1.0 matches the legacy global cap used for small-model tasks;
    # LLM-centric tasks (e.g. aime_math_rl with Qwen3-4B) should override in
    # task.yaml, e.g. `max_model_params_billion: 5.0`.
    max_model_params_billion: float = 1.0

    # Optional task-specific hints appended to the agent's system prompt.
    # Use this instead of hardcoding task-name checks in agent_prompt.py.
    agent_hints: str = ""

    def validate(self) -> None:
        """Validate required fields and value constraints.

        Raises ValueError with a descriptive message on the first violation.
        Call after from_yaml() / from_dict() to catch config errors early.
        """
        # Required string fields
        for field_name in ("name", "primary_metric", "evaluator_class"):
            if not getattr(self, field_name):
                raise ValueError(
                    f"TaskConfig.{field_name} is required but empty"
                )

        if not self.description.strip():
            raise ValueError("TaskConfig.description must not be blank")

        # higher_is_better must be bool
        if not isinstance(self.higher_is_better, bool):
            raise ValueError(
                f"TaskConfig.higher_is_better must be bool, "
                f"got {type(self.higher_is_better).__name__}"
            )

        # Time budget must be positive
        if self.total_time_budget_hours <= 0:
            raise ValueError(
                f"TaskConfig.total_time_budget_hours must be > 0, "
                f"got {self.total_time_budget_hours}"
            )

        # Iteration budget must be positive int
        if not isinstance(self.max_iterations, int) or self.max_iterations < 1:
            raise ValueError(
                f"TaskConfig.max_iterations must be a positive int, "
                f"got {self.max_iterations}"
            )

        # max_gpu_count must be positive int
        if not isinstance(self.max_gpu_count, int) or self.max_gpu_count < 1:
            raise ValueError(
                f"TaskConfig.max_gpu_count must be a positive int, "
                f"got {self.max_gpu_count}"
            )

        # max_model_params_billion must be a positive number
        if (
            not isinstance(self.max_model_params_billion, (int, float))
            or isinstance(self.max_model_params_billion, bool)
            or self.max_model_params_billion <= 0
        ):
            raise ValueError(
                f"TaskConfig.max_model_params_billion must be a positive number, "
                f"got {self.max_model_params_billion!r}"
            )
        self.max_model_params_billion = float(self.max_model_params_billion)

        # compute_type must be a valid enum
        if not isinstance(self.compute_type, ComputeType):
            raise ValueError(
                f"TaskConfig.compute_type must be a ComputeType enum, "
                f"got {self.compute_type!r}"
            )

        for field_name in ("data_dir", "test_data_dir"):
            path = getattr(self, field_name)
            if not path:
                raise ValueError(f"TaskConfig.{field_name} is required but empty")
            if not os.path.isabs(path):
                raise ValueError(
                    f"TaskConfig.{field_name} must be an absolute in-image path, "
                    f"got {path!r}"
                )
            if not path.rstrip("/").startswith("/rab_data/"):
                raise ValueError(
                    f"TaskConfig.{field_name} must point under /rab_data, "
                    f"got {path!r}"
                )
        if self.data_dir.rstrip("/") == self.test_data_dir.rstrip("/"):
            raise ValueError(
                "TaskConfig.data_dir and TaskConfig.test_data_dir must be separate "
                "in-image paths"
            )
        if not self.docker_image:
            raise ValueError("TaskConfig.docker_image is required but empty")
        if not re.search(r"-cu\d+$", self.docker_image):
            raise ValueError(
                "TaskConfig.docker_image must end with a CUDA variant suffix "
                f"(for example -cu118), got {self.docker_image!r}"
            )

    @classmethod
    def from_yaml(cls, yaml_path: str) -> TaskConfig:
        """Load config from task.yaml.

        Unknown keys fail immediately. This keeps task.yaml aligned with the
        released framework schema instead of silently carrying stale fields.
        """
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"task.yaml '{yaml_path}' must contain a mapping")

        task_dir = os.path.dirname(os.path.abspath(yaml_path))
        config = cls()
        unknown_fields: list[str] = []

        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                unknown_fields.append(key)

        if unknown_fields:
            joined = ", ".join(sorted(unknown_fields))
            raise ValueError(f"Unknown field(s) in {yaml_path}: {joined}")

        # Coerce compute_type string → enum
        if isinstance(config.compute_type, str):
            config.compute_type = ComputeType(config.compute_type)

        # Normalize domain: accept str or list[str] in YAML, store as list[str]
        if isinstance(config.domain, str):
            config.domain = [config.domain] if config.domain else []
        elif config.domain is None:
            config.domain = []
        else:
            config.domain = [d for d in config.domain if d]

        # Resolve framework paths to absolute host paths. Data paths are
        # in-image /rab_data paths and are validated, not host-resolved.
        config.task_dir = task_dir
        if config.script_dir:
            config.script_dir = os.path.join(task_dir, config.script_dir)

        # Default max_command_timeout to total_time_budget_hours (in seconds)
        if config.max_command_timeout is None:
            config.max_command_timeout = int(config.total_time_budget_hours * 3600)

        config.validate()
        return config

    @classmethod
    def from_dict(cls, data: dict) -> TaskConfig:
        """Reconstruct a TaskConfig from a plain dict (e.g. JSON-deserialized)."""
        config = cls()
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)
        if isinstance(config.compute_type, str):
            config.compute_type = ComputeType(config.compute_type)
        if isinstance(config.domain, str):
            config.domain = [config.domain] if config.domain else []
        elif config.domain is None:
            config.domain = []
        if config.max_command_timeout is None:
            config.max_command_timeout = int(config.total_time_budget_hours * 3600)
        return config
