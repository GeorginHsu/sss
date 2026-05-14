"""MetricEvaluatorBase — abstract base class for task evaluators.

The agent's predict.py runs first and produces a predictions JSON file.
The evaluator loads ground truth and compares against predictions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from farbench.schemas import EvalResult, TaskConfig

__all__ = ["EvalResult", "MetricEvaluatorBase"]


class MetricEvaluatorBase(ABC):
    """Abstract base class; each task implements a subclass.

    Evaluators compare agent-produced predictions against ground truth.
    They never import agent code (model.py, etc.).
    """

    @abstractmethod
    def evaluate(
        self,
        predictions_path: str,
        test_data_dir: str,
        task_config: TaskConfig,
    ) -> EvalResult:
        """Compare predictions file against ground truth test data.

        Args:
            predictions_path: Path to JSON file produced by agent's predict.py.
            test_data_dir: Directory containing ground truth test data.
            task_config: Task configuration.

        Returns:
            EvalResult with computed metrics.
        """
