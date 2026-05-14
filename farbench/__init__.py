"""FARBench — core framework package."""

from farbench.agent import LLMAgent, APIError
from farbench.agent_prompt import build_agent_prompt
from farbench.env import ResearchEnv
from farbench.evaluator import MetricEvaluatorBase
from farbench.runner import SandboxRunner, EvalDockerRunner
from farbench.schemas import (
    Action,
    AgentMetadata,
    AgentMode,
    CommandOutput,
    ComputeType,
    EvalResult,
    EvalSubmission,
    IterationRecord,
    Observation,
    TaskConfig,
    TerminalReason,
    TokenUsage,
)

__all__ = [
    # Agent
    "LLMAgent",
    "APIError",
    # Environment
    "ResearchEnv",
    "build_agent_prompt",
    # Runners
    "SandboxRunner",
    "EvalDockerRunner",
    # Evaluator
    "MetricEvaluatorBase",
    # Enums
    "AgentMode",
    "ComputeType",
    "TerminalReason",
    # Data structures
    "Action",
    "AgentMetadata",
    "CommandOutput",
    "EvalResult",
    "EvalSubmission",
    "IterationRecord",
    "Observation",
    "TaskConfig",
    "TokenUsage",
]
