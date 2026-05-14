"""EnvironmentOrchestrator: orchestrates the full lifecycle of a single experiment.

Agent writes code from scratch, executes commands, and submits for evaluation.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from typing import Any, Optional

from farbench.constants import (
    PIP_INSTALL_TIMEOUT,
    STDERR_ERROR_TAIL,
    STDERR_HISTORY_TAIL,
    STDOUT_HISTORY_TAIL,
)
from farbench.runner import SandboxRunner, EvalDockerRunner
from farbench.schemas import (
    Action,
    AgentMetadata,
    CommandOutput,
    EvalResult,
    IterationRecord,
    Observation,
    TaskConfig,
    TerminalReason,
    TokenUsage,
)
from farbench.storage import ExperimentStore
from farbench.tracker import collect_training_curves
from farbench.utils import (
    WORKSPACE_SKIP_DIRS,
    collect_hardware_info,
    get_logger,
    list_workspace_files,
    read_workspace_file_contents,
    resolve_workspace_path,
    set_task_context,
    write_files_to_workspace,
)

logger = get_logger(__name__)

_StepReturn = tuple[Observation, float, bool, dict]


class EnvironmentOrchestrator:
    """Orchestrates the full lifecycle of a single experiment.

    Iteration flow:
    1. Write files (if any) to workspace
    2. Execute command (if any) in sandbox
    3. Submit for evaluation (if requested)
    4. Return observation with results
    """

    def __init__(
        self,
        task_config: TaskConfig,
        experiment_store: ExperimentStore,
        sandbox_runner: SandboxRunner,
    ) -> None:
        self.task_config = task_config
        self.store = experiment_store
        self.sandbox_runner = sandbox_runner
        self.current_iteration: int = 0
        self.history: list[IterationRecord] = []
        self._eval_runner: Optional[EvalDockerRunner] = None
        self._eval_submissions_used: int = 0
        self._start_time: float = 0.0  # set in initialize()
        # Cumulative wall-clock spent inside LLM API calls. Excluded from the
        # task's time budget so that a slow reasoning model doesn't eat into
        # training/eval time. Seeded from live_status.json on resume.
        self._llm_elapsed_seconds: float = 0.0
        self._best_eval_result: Optional[EvalResult] = None
        self._terminal_reason: Optional[TerminalReason] = None
        self._termination_detail: str = ""

        # Inject agent_id into runners for container naming
        self.sandbox_runner.agent_id = experiment_store.agent_id
        self._check_no_orphaned_containers()

    # ── Initialization ──

    def initialize(self) -> Observation:
        """Initialize experiment: create empty workspace, start sandbox."""
        self._start_time = time.time()
        set_task_context(self.task_config.name)
        self.store.ensure_workspace()

        # Pre-create common directories so agents don't waste iterations
        # on "directory not found" errors for standard ML workflow paths.
        for subdir in ("checkpoints", "logs"):
            os.makedirs(
                os.path.join(self.store.workspace_dir, subdir), exist_ok=True,
            )

        # Set up experiment-level log file
        self._setup_file_logging()

        # Save experiment config (including hardware snapshot at start time)
        self.store.save_json(
            os.path.join(self.store.experiment_dir, "config.json"),
            {
                "task_name": self.task_config.name,
                "domain": self.task_config.domain,
                "primary_metric": self.task_config.primary_metric,
                "higher_is_better": self.task_config.higher_is_better,
                "total_time_budget_hours": self.task_config.total_time_budget_hours,
                "max_iterations": self.task_config.max_iterations,
                "compute_type": self.task_config.compute_type.value,
                "network_access": self.task_config.network_access,
                "hardware_info": collect_hardware_info(),
            },
        )

        # Start sandbox container — clean up resources on failure
        try:
            self.sandbox_runner.ensure_ready(
                workspace_path=self.store.workspace_dir,
                experiment_dir=self.store.experiment_dir,
                task_config=self.task_config,
            )
        except Exception as e:
            logger.error(f"Failed to start sandbox: {e}")
            self.cleanup()
            raise

        obs = self._build_observation(
            iteration=0,
            task_description=self.task_config.description,
        )

        # Save initial observation
        self.store.ensure_dirs(0)
        self.store.save_json(self.store.obs_path(0), obs.to_dict())

        logger.info(
            f"Experiment initialized: {self.store.experiment_dir}, "
            f"time_budget={self.task_config.total_time_budget_hours}h, "
            f"max_iterations={self.task_config.max_iterations}"
        )
        return obs

    # ── Resume from an existing experiment directory ──

    def resume(self) -> Observation:
        """Attach to an existing experiment and prepare to continue.

        Contract (matching the design agreed with the user):
          * Reads config.json / live_status.json to recover time + token state.
          * Scans iter_NNN/ directories to rebuild history.
          * The LAST iter_NNN is assumed to be aborted (no valid command_output
            or obs.json persisted). It is renamed to iter_NNN.aborted/ so the
            next iteration reuses that number. If it turns out the last iter
            was actually complete, we still treat it as aborted and re-run the
            next decision; this is the simpler "always drop last" policy the
            user asked for.
          * Starts a FRESH sandbox container bind-mounted to the existing
            workspace directory (so pip packages, checkpoints, code changes
            all persist to the new run).
          * Appends a `resumes` audit record to config.json.

        Returns the observation that feeds into the next agent.act() call
        (i.e. the obs.json of the last SURVIVING iter).
        """
        set_task_context(self.task_config.name)

        # ── 1. Load config + live_status ─────────────────────────────────
        config_path = os.path.join(self.store.experiment_dir, "config.json")
        with open(config_path) as f:
            config = json.load(f)

        live_path = os.path.join(self.store.experiment_dir, "live_status.json")
        live_status: dict = {}
        if os.path.isfile(live_path):
            try:
                with open(live_path) as f:
                    live_status = json.load(f)
            except Exception as e:
                logger.warning(f"live_status.json unreadable, treating as empty: {e}")

        # Refuse to resume a fully-finalized experiment.
        if os.path.isfile(os.path.join(self.store.summary_dir(), "final_results.json")):
            raise RuntimeError(
                f"Experiment already finalized (summary/final_results.json "
                f"exists): {self.store.experiment_dir}"
            )

        # ── 2. Discover iter directories ─────────────────────────────────
        # Only count top-level iter_NNN (ignore already-archived .aborted).
        all_iters = sorted(
            int(d[len("iter_"):]) for d in os.listdir(self.store.experiment_dir)
            if d.startswith("iter_")
               and d[len("iter_"):].isdigit()
               and os.path.isdir(os.path.join(self.store.experiment_dir, d))
        )
        if not all_iters:
            raise RuntimeError(
                f"No iter_* directories under {self.store.experiment_dir}; "
                f"nothing to resume from."
            )

        # Policy: always drop the last iter_NNN (treat as aborted), no matter
        # whether it looks complete. Simpler than per-file sniffing, and agrees
        # with "最后一个 iter 是烂尾的".
        # iter_000 is special: it only holds the initial obs.json. If it's the
        # ONLY iter present, there's nothing useful to resume from — re-running
        # from scratch would be cheaper. Bail out rather than do something weird.
        if len(all_iters) <= 1:
            raise RuntimeError(
                f"Experiment has only {len(all_iters)} iter_* dir(s); "
                f"there is no completed iteration to resume after. "
                f"Start a new run instead."
            )

        aborted_iter = all_iters[-1]
        last_good_iter = all_iters[-2]
        aborted_path = os.path.join(
            self.store.experiment_dir, f"iter_{aborted_iter:03d}",
        )
        aborted_archive = aborted_path + ".aborted"
        # If a prior resume already archived this slot, append a numeric suffix.
        if os.path.exists(aborted_archive):
            i = 2
            while os.path.exists(f"{aborted_archive}.{i}"):
                i += 1
            aborted_archive = f"{aborted_archive}.{i}"
        os.rename(aborted_path, aborted_archive)
        logger.info(
            f"[resume] archived presumed-aborted iter_{aborted_iter:03d} "
            f"-> {os.path.basename(aborted_archive)}"
        )

        # ── 3. Rebuild history from iter_001..iter_{last_good_iter} ─────
        # iter_000 only holds the initial obs, skip it.
        per_iter_tokens = {
            int(e["iteration"]): e for e in live_status.get("per_iteration_tokens", [])
            if isinstance(e, dict) and "iteration" in e
        }
        self.history = []
        for n in all_iters:
            if n == 0 or n > last_good_iter:
                continue
            self.history.append(self._rebuild_iter_record(n, per_iter_tokens))

        # ── 4. Rebuild derived state ────────────────────────────────────
        self.current_iteration = last_good_iter
        self._eval_submissions_used = sum(1 for r in self.history if r.eval_submitted)
        self._best_eval_result = None
        for r in self.history:
            if r.eval_result is not None:
                self._update_best(r.eval_result)

        # Time budget: skip the gap between abort and resume. We pretend the
        # experiment started exactly `elapsed_hours` seconds ago; so the
        # benchmark's "total_time_budget_hours" clock only counts time the
        # agent actually spent doing work.
        elapsed_hours = float(live_status.get("elapsed_hours") or 0.0)
        self._start_time = time.time() - elapsed_hours * 3600.0
        # Seed LLM time from previous run so budget math stays consistent.
        # Missing field (older experiments) → 0, no bookkeeping loss other
        # than the prior run's API time counting against budget once.
        self._llm_elapsed_seconds = (
            float(live_status.get("llm_elapsed_hours") or 0.0) * 3600.0
        )

        # ── 5. Re-wire runners / logging / sandbox ──────────────────────
        self._setup_file_logging()

        # Start a FRESH sandbox container. No orphaned-container check here —
        # resume explicitly expects its own agent_id's sandbox to be gone
        # already (the shell wrapper removes it before invoking us).
        try:
            self.sandbox_runner.ensure_ready(
                workspace_path=self.store.workspace_dir,
                experiment_dir=self.store.experiment_dir,
                task_config=self.task_config,
            )
        except Exception as e:
            logger.error(f"[resume] failed to start sandbox: {e}")
            self.cleanup()
            raise

        # ── 6. Audit: append a resumes[] entry to config.json ───────────
        resumes = list(config.get("resumes", []))
        resumes.append({
            "at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "resumed_after_iter": last_good_iter,
            "aborted_iter_archived_as": os.path.basename(aborted_archive),
            "elapsed_hours_at_resume": elapsed_hours,
        })
        config["resumes"] = resumes
        self.store.save_json(config_path, config)

        # ── 7. Return the observation the agent will act on next ───────
        # It's the obs.json of the last surviving iter; rewrite its `history`
        # so it matches our just-rebuilt record list (identical content, but
        # guarantees consistency even if older runs had bugs).
        last_obs_path = self.store.obs_path(last_good_iter)
        if not os.path.isfile(last_obs_path):
            raise RuntimeError(
                f"[resume] iter_{last_good_iter:03d}/obs.json missing — cannot "
                f"resume cleanly. Delete the experiment and start over, or "
                f"manually archive iter_{last_good_iter:03d}/ and retry."
            )

        # Build a fresh obs so file_contents / workspace_files reflect the CURRENT
        # on-disk workspace (agent may have left useful artifacts there).
        obs = self._build_observation(iteration=last_good_iter)

        logger.info(
            f"[resume] attached to {self.store.experiment_dir}: "
            f"history={len(self.history)} iters, "
            f"current_iteration={self.current_iteration}, "
            f"elapsed={elapsed_hours:.2f}h, "
            f"remaining_budget={self.remaining_budget_hours():.2f}h"
        )
        return obs

    def _rebuild_iter_record(
        self, n: int, per_iter_tokens: dict[int, dict],
    ) -> IterationRecord:
        """Reconstruct an IterationRecord from disk artifacts of iter_NNN/."""
        iter_dir = self.store.iteration_dir(n)

        action_data: dict = {}
        action_path = self.store.action_path(n)
        if os.path.isfile(action_path):
            try:
                with open(action_path) as f:
                    action_data = json.load(f) or {}
            except Exception as e:
                logger.warning(f"[resume] bad action.json for iter_{n:03d}: {e}")

        cmd_out_data: dict = {}
        cmd_path = self.store.command_output_path(n)
        if os.path.isfile(cmd_path):
            try:
                with open(cmd_path) as f:
                    cmd_out_data = json.load(f) or {}
            except Exception as e:
                logger.warning(f"[resume] bad command_output.json for iter_{n:03d}: {e}")

        eval_data: Optional[dict] = None
        eval_path = self.store.eval_result_path(n)
        if os.path.isfile(eval_path):
            try:
                with open(eval_path) as f:
                    eval_data = json.load(f)
            except Exception as e:
                logger.warning(f"[resume] bad eval_result.json for iter_{n:03d}: {e}")

        stdout = cmd_out_data.get("stdout", "") or ""
        stderr = cmd_out_data.get("stderr", "") or ""
        exit_code = cmd_out_data.get("exit_code", 0) or 0
        cmd_summary = stdout[-STDOUT_HISTORY_TAIL:] if len(stdout) > STDOUT_HISTORY_TAIL else stdout
        err_summary = ""
        if exit_code != 0 and stderr:
            err_summary = stderr[-STDERR_HISTORY_TAIL:] if len(stderr) > STDERR_HISTORY_TAIL else stderr

        eval_result_obj: Optional[EvalResult] = None
        if eval_data:
            try:
                eval_result_obj = EvalResult(**{
                    k: v for k, v in eval_data.items()
                    if k in EvalResult.__dataclass_fields__
                })
            except Exception as e:
                logger.warning(f"[resume] could not parse eval_result for iter_{n:03d}: {e}")

        # Token usage: prefer the per-iteration record stashed in live_status.json.
        tu_data = per_iter_tokens.get(n) or {}
        tu = TokenUsage(
            input_tokens=int(tu_data.get("input_tokens") or 0),
            output_tokens=int(tu_data.get("output_tokens") or 0),
            thinking_tokens=int(tu_data.get("thinking_tokens") or 0),
            cache_read_tokens=int(tu_data.get("cache_read_tokens") or 0),
            cache_creation_tokens=int(tu_data.get("cache_creation_tokens") or 0),
        )

        return IterationRecord(
            iteration=n,
            command=action_data.get("command"),
            files_written=list((action_data.get("files_to_write") or {}).keys()),
            command_output_summary=cmd_summary,
            error_summary=err_summary,
            eval_submitted=bool(action_data.get("submit_eval")),
            eval_result=eval_result_obj,
            eval_error_log="",
            reward=0.0,
            elapsed_seconds=float(cmd_out_data.get("elapsed_seconds") or 0.0),
            description=str(action_data.get("reasoning") or ""),
            token_usage=tu,
        )

    # ── Iteration execution ──

    def run_iteration(
        self,
        action: Action,
        agent_metadata: Optional[AgentMetadata] = None,
    ) -> _StepReturn:
        """Execute one full iteration."""

        # Accumulate LLM API wall-clock so it is excluded from the task budget
        # (see remaining_budget_hours). Done first so both the done-path and
        # the normal iteration benefit consistently.
        if agent_metadata is not None and agent_metadata.latency_seconds:
            self._llm_elapsed_seconds += float(agent_metadata.latency_seconds)

        # Agent signals done — save the final action/conversation before returning
        if action.done:
            self._terminal_reason = TerminalReason.AGENT_DONE
            self._termination_detail = action.done_reason or ""
            # Save done action into the summary directory (not a new iter_ dir,
            # which would inflate _iter_dirs counts and create a phantom entry).
            summary_dir = self.store.summary_dir()
            os.makedirs(summary_dir, exist_ok=True)
            self.store.save_json(
                os.path.join(summary_dir, "done_action.json"), action.to_dict(),
            )
            if agent_metadata and agent_metadata.conversation:
                self.store.save_json(
                    os.path.join(summary_dir, "done_conversation.json"),
                    agent_metadata.conversation,
                )
            obs = self._build_observation(iteration=self.current_iteration)
            info: dict = {"terminal_reason": TerminalReason.AGENT_DONE.value}
            if action.done_reason:
                info["done_reason"] = action.done_reason
                logger.info(f"Agent done: {action.done_reason}")
            return obs, 0.0, True, info

        self.current_iteration += 1
        iteration = self.current_iteration
        self.store.ensure_dirs(iteration)
        logger.info(f"=== Iteration {iteration} ===")

        iter_start = time.time()
        command_output: Optional[CommandOutput] = None
        eval_result: Optional[EvalResult] = None
        files_written: list[str] = []
        error: Optional[str] = None

        # Save agent action
        self.store.save_json(
            self.store.action_path(iteration),
            action.to_dict(),
        )

        # Save LLM conversation log
        if agent_metadata and agent_metadata.conversation:
            self.store.save_json(
                self.store.conversation_path(iteration),
                agent_metadata.conversation,
            )

        # Step A: Write files
        if action.files_to_write:
            try:
                files_written = write_files_to_workspace(
                    self.store.workspace_dir, action.files_to_write,
                )
                logger.info(f"Files written: {files_written}")
            except ValueError as e:
                error = f"File write error: {e}"
                logger.warning(error)

        # Step B: Install packages to a shared workspace directory so both
        # the sandbox AND eval containers can use them.  The eval container is
        # network-isolated and never runs pip itself, so we install to
        # /workspace/.pip_packages (visible to both via bind-mount) and inject
        # PYTHONPATH in every container exec call.
        #
        # IMPORTANT: After --target install, remove torch/nvidia/triton packages
        # from .pip_packages.  pip --target ignores system site-packages, so it
        # re-installs the whole torch ecosystem (often a newer CUDA version).
        # PYTHONPATH puts .pip_packages first, so the new torch shadows the
        # CUDA-compatible one from the base image, causing cuda.is_available()=False.
        if action.packages_to_install and not error:
            pkgs = " ".join(action.packages_to_install)
            install_cmd = (
                f"pip install --quiet --target /workspace/.pip_packages {pkgs} && "
                f"rm -rf /workspace/.pip_packages/torch* "
                f"/workspace/.pip_packages/nvidia* "
                f"/workspace/.pip_packages/triton* "
                f"2>/dev/null; true"
            )
            logger.info(f"Installing packages: {pkgs}")
            install_out = self.sandbox_runner.execute_command(
                command=install_cmd,
                timeout_seconds=PIP_INSTALL_TIMEOUT,
                iteration=iteration,
            )
            if install_out and install_out.exit_code != 0:
                error = f"Package install failed: {install_out.stderr[-STDERR_ERROR_TAIL:]}"
                logger.warning(error)

        # Step C: Execute command
        if action.command and not error:
            remaining_seconds = int(self.remaining_budget_hours() * 3600)
            timeout = min(remaining_seconds, self.task_config.max_command_timeout)
            if timeout <= 0:
                error = "Time budget exhausted"
                command_output = CommandOutput(
                    stderr="Time budget exhausted", exit_code=-1,
                )
            else:
                command_output = self.sandbox_runner.execute_command(
                    command=action.command,
                    timeout_seconds=timeout,
                    iteration=iteration,
                )

            if command_output:
                self.store.save_json(
                    self.store.command_output_path(iteration),
                    asdict(command_output),
                )
                # If the command failed and eval was requested in the same
                # action, skip the eval — the checkpoint almost certainly
                # does not exist yet.
                if command_output.exit_code != 0 and action.submit_eval:
                    error = (
                        f"Command failed (exit {command_output.exit_code}), "
                        f"skipping eval submission in the same action"
                    )
                    logger.warning(error)

        # Step D: Submit evaluation
        if action.submit_eval and not error:
            if not action.submit_eval.checkpoint_path:
                error = "submit_eval.checkpoint_path is required"
                logger.warning(error)
            elif not action.submit_eval.predict_script:
                error = "submit_eval.predict_script is required"
                logger.warning(error)
            elif not self._validate_eval_path(action.submit_eval.checkpoint_path, must_be_file=False):
                error = (
                    "submit_eval.checkpoint_path must be an existing relative "
                    "path inside the workspace"
                )
                logger.warning(error)
            elif not self._validate_eval_path(action.submit_eval.predict_script, must_be_file=True):
                error = (
                    "submit_eval.predict_script must be an existing file path "
                    "inside the workspace"
                )
                logger.warning(error)
            else:
                self._eval_submissions_used += 1
                try:
                    eval_result = self._run_evaluation(
                        checkpoint_path=action.submit_eval.checkpoint_path,
                        predict_script=action.submit_eval.predict_script,
                        output_dir=self.store.iteration_dir(iteration),
                    )
                    self.store.save_json(
                        self.store.eval_result_path(iteration),
                        asdict(eval_result),
                    )
                    logger.info(
                        f"Eval result: {eval_result.primary_metric_name}="
                        f"{eval_result.primary_metric_value}"
                    )
                except Exception as e:
                    error = f"Evaluation failed: {e}"
                    logger.warning(error)

        # Build observation
        elapsed = time.time() - iter_start
        obs = self._build_observation(
            iteration=iteration,
            command_output=command_output,
            eval_result=eval_result,
            error=error,
        )

        # Build iteration record
        cmd_summary = ""
        err_summary = ""
        if command_output:
            # Truncate stdout for history (keep last 500 chars)
            out = command_output.stdout
            cmd_summary = out[-STDOUT_HISTORY_TAIL:] if len(out) > STDOUT_HISTORY_TAIL else out
            # Keep stderr tail on failure for agent learning
            if command_output.exit_code != 0 and command_output.stderr:
                err = command_output.stderr
                err_summary = err[-STDERR_HISTORY_TAIL:] if len(err) > STDERR_HISTORY_TAIL else err

        # Capture eval error log (from the error string set above)
        eval_err_log = ""
        if error and "Evaluation failed" in error:
            eval_err_log = error

        # Compute reward as signed delta vs previous best, then update best.
        # - First eval: reward = 0 (establishes baseline)
        # - Subsequent evals: reward = signed improvement
        reward = 0.0
        if eval_result:
            current = eval_result.primary_metric_value
            if self._best_eval_result is None:
                reward = 0.0
            else:
                prev_best = self._best_eval_result.primary_metric_value
                if self.task_config.higher_is_better:
                    reward = current - prev_best
                else:
                    reward = prev_best - current
            self._update_best(eval_result)

        # Reasoning: store as plain text (O-A-D format).
        # Full text is preserved in history for agent learning and reproducibility.
        reasoning_summary = str(action.reasoning or "")

        record = IterationRecord(
            iteration=iteration,
            command=action.command,
            files_written=files_written,
            command_output_summary=cmd_summary,
            error_summary=err_summary,
            eval_submitted=action.submit_eval is not None,
            eval_result=eval_result,
            eval_error_log=eval_err_log,
            reward=reward,
            elapsed_seconds=elapsed,
            description=reasoning_summary,
            token_usage=agent_metadata.token_usage if agent_metadata else TokenUsage(),
        )
        self.history.append(record)
        obs.history = list(self.history)

        self.store.save_json(self.store.obs_path(iteration), obs.to_dict())

        # Save cumulative live status (tokens, best metric, etc.) for dashboard
        self._save_live_status(iteration)

        # Snapshot workspace text files for reproducibility
        self._snapshot_workspace(iteration)

        terminal_reason = self._check_done()
        done = terminal_reason is not None
        if terminal_reason:
            self._terminal_reason = terminal_reason

        info: dict = {
            "files_written": files_written,
            "eval_result": asdict(eval_result) if eval_result else None,
            "elapsed_seconds": elapsed,
        }
        if terminal_reason:
            info["terminal_reason"] = terminal_reason.value
        if agent_metadata is not None:
            meta_dict = asdict(agent_metadata)
            meta_dict.pop("conversation", None)  # saved separately
            info["agent_metadata"] = meta_dict

        logger.info(
            f"Iteration {iteration}: reward={reward:.4f}, done={done}, "
            f"remaining_time={self.remaining_budget_hours():.3f}h, "
            f"remaining_iters={self.remaining_iterations()}"
        )
        return obs, reward, done, info

    # ── Evaluation ──

    def _run_evaluation(
        self,
        checkpoint_path: str,
        predict_script: str,
        output_dir: str,
    ) -> EvalResult:
        """Run evaluation in isolated container."""
        if self._eval_runner is None:
            self._eval_runner = EvalDockerRunner()
            self._eval_runner.agent_id = self.store.agent_id

        return self._eval_runner.evaluate(
            checkpoint_path=checkpoint_path,
            predict_script=predict_script,
            task_config=self.task_config,
            workspace_path=self.store.workspace_dir,
            output_dir=output_dir,
        )

    def _validate_eval_path(self, path: str, *, must_be_file: bool) -> bool:
        """Validate an eval submission path before passing it to Docker."""
        if os.path.isabs(path):
            return False
        try:
            abs_path = resolve_workspace_path(self.store.workspace_dir, path)
        except ValueError:
            return False
        if must_be_file:
            return os.path.isfile(abs_path)
        return os.path.exists(abs_path)

    def _update_best(self, eval_result: EvalResult) -> None:
        """Update best eval result if the new result is better."""
        if self._best_eval_result is None:
            self._best_eval_result = eval_result
            return

        current = eval_result.primary_metric_value
        best = self._best_eval_result.primary_metric_value
        if self.task_config.higher_is_better:
            if current > best:
                self._best_eval_result = eval_result
        else:
            if current < best:
                self._best_eval_result = eval_result

    # ── Finalize ──

    def finalize(self) -> dict[str, Any]:
        """Generate experiment summary and leaderboard entry."""
        self.store.ensure_summary()
        summary_dir = self.store.summary_dir()

        # Trajectory
        self.store.save_json(
            os.path.join(summary_dir, "trajectory.json"),
            {"iterations": [asdict(r) for r in self.history]},
        )

        # Final results
        total_elapsed_hours = (time.time() - self._start_time) / 3600.0
        llm_elapsed_hours = self._llm_elapsed_seconds / 3600.0
        effective_elapsed_hours = max(total_elapsed_hours - llm_elapsed_hours, 0.0)
        eval_submissions_used = self._eval_submissions_used

        total_usage = TokenUsage()
        for r in self.history:
            total_usage += r.token_usage

        _reason_value = self._terminal_reason.value if self._terminal_reason else "unknown"
        final_results = {
            "task_name": self.task_config.name,
            "primary_metric": self.task_config.primary_metric,
            "higher_is_better": self.task_config.higher_is_better,
            "terminal_reason": _reason_value,
            "termination_detail": self._termination_detail or None,
            "total_iterations": self.current_iteration,
            "max_iterations": self.task_config.max_iterations,
            "total_elapsed_hours": round(total_elapsed_hours, 4),
            "llm_elapsed_hours": round(llm_elapsed_hours, 4),
            "effective_elapsed_hours": round(effective_elapsed_hours, 4),
            "eval_submissions_used": eval_submissions_used,
            "best_metrics": (
                self._best_eval_result.metrics if self._best_eval_result else {}
            ),
            "best_primary_metric": (
                self._best_eval_result.primary_metric_value
                if self._best_eval_result else 0.0
            ),
            **{f"total_{k}": v for k, v in total_usage.to_dict().items()},
            "total_tokens": total_usage.total,
        }
        self.store.save_json(
            os.path.join(summary_dir, "final_results.json"), final_results,
        )

        # Leaderboard entry
        leaderboard = {
            "task_name": self.task_config.name,
            "agent_id": self.store.agent_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "terminal_reason": _reason_value,
            "termination_detail": self._termination_detail or None,
            "total_iterations": self.current_iteration,
            "total_elapsed_hours": round(total_elapsed_hours, 4),
            "llm_elapsed_hours": round(llm_elapsed_hours, 4),
            "effective_elapsed_hours": round(effective_elapsed_hours, 4),
            "time_budget_hours": self.task_config.total_time_budget_hours,
            "max_iterations": self.task_config.max_iterations,
            # Budget utilization is computed against effective (non-LLM) time.
            "budget_utilization_pct": round(
                effective_elapsed_hours / self.task_config.total_time_budget_hours * 100, 1
            ) if self.task_config.total_time_budget_hours > 0 else 0,
            "eval_submissions_used": eval_submissions_used,
            "best_metrics": (
                self._best_eval_result.metrics if self._best_eval_result else {}
            ),
            "best_primary_metric": (
                self._best_eval_result.primary_metric_value
                if self._best_eval_result else 0.0
            ),
            "compute_type": self.task_config.compute_type.value,
            "hardware_info": collect_hardware_info(),
            "trajectory_summary": [
                {
                    "iteration": r.iteration,
                    "eval_submitted": r.eval_submitted,
                    **(r.eval_result.metrics if r.eval_result else {}),
                    "elapsed_seconds": r.elapsed_seconds,
                }
                for r in self.history
            ],
        }
        self.store.save_json(
            os.path.join(summary_dir, "leaderboard_entry.json"), leaderboard,
        )

        # RL-friendly episode transitions (uses reward stored in IterationRecord)
        exp_dir = self.store.experiment_dir
        episodes_path = os.path.join(summary_dir, "episodes.jsonl")
        with open(episodes_path, "w") as f:
            for idx, record in enumerate(self.history):
                i = record.iteration
                is_last = (idx == len(self.history) - 1)

                # Determine terminal reason
                terminal_reason = None
                if is_last:
                    if effective_elapsed_hours >= self.task_config.total_time_budget_hours:
                        terminal_reason = TerminalReason.TIME_BUDGET_EXHAUSTED
                    elif self.current_iteration >= self.task_config.max_iterations:
                        terminal_reason = TerminalReason.ITERATION_LIMIT_REACHED
                    else:
                        terminal_reason = TerminalReason.AGENT_DONE

                transition = {
                    "iteration": i,
                    "state": os.path.relpath(self.store.obs_path(i - 1), exp_dir),
                    "action": os.path.relpath(self.store.action_path(i), exp_dir),
                    "reward": round(record.reward, 6),
                    "next_state": os.path.relpath(self.store.obs_path(i), exp_dir),
                    "done": is_last,
                }
                if terminal_reason:
                    transition["terminal_reason"] = terminal_reason.value
                f.write(json.dumps(transition, ensure_ascii=False) + "\n")

        return final_results

    # ── Internal helpers ──

    def _setup_file_logging(self) -> None:  # noqa: D401
        """Add a FileHandler to persist all log output to experiment dir."""
        import logging as _logging
        from farbench.utils import _task_filter, _LOG_FORMAT

        log_path = os.path.join(self.store.experiment_dir, "experiment.log")
        fh = _logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(
            _logging.Formatter(_LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")
        )
        fh.addFilter(_task_filter)
        # Attach to the root 'farbench' logger so all farbench.* loggers write here
        farbench_logger = _logging.getLogger("farbench")
        farbench_logger.addHandler(fh)
        self._log_file_handler = fh

    def _snapshot_workspace(self, iteration: int) -> None:
        """Save a copy of all workspace text files into the iteration dir."""
        import shutil

        snapshot_dir = os.path.join(
            self.store.iteration_dir(iteration), "workspace_snapshot"
        )
        os.makedirs(snapshot_dir, exist_ok=True)

        workspace = self.store.workspace_dir
        # Use the shared skip set, plus checkpoints/logs (binary/generated data)
        snapshot_skip = WORKSPACE_SKIP_DIRS | {"checkpoints", "logs"}
        for dirpath, dirnames, filenames in os.walk(workspace):
            dirnames[:] = [d for d in dirnames if d not in snapshot_skip]
            for fname in filenames:
                # Only snapshot text source files (skip .pt, .pth, etc.)
                ext = os.path.splitext(fname)[1].lower()
                if ext in {".pt", ".pth", ".bin", ".pkl", ".npy", ".npz",
                           ".h5", ".hdf5", ".onnx", ".so", ".o"}:
                    continue
                src = os.path.join(dirpath, fname)
                if os.path.islink(src):
                    continue
                rel = os.path.relpath(src, workspace)
                dst = os.path.join(snapshot_dir, rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                try:
                    shutil.copy2(src, dst)
                except OSError:
                    pass

    def _save_live_status(self, iteration: int) -> None:
        """Save cumulative usage stats for live dashboard monitoring.

        Written after every iteration so the dashboard can show real-time
        token counts, elapsed time, best metric, etc. without waiting
        for finalize().
        """
        total_usage = TokenUsage()
        for r in self.history:
            total_usage += r.token_usage
        elapsed_hours = (time.time() - self._start_time) / 3600.0 if self._start_time else 0.0
        llm_elapsed_hours = self._llm_elapsed_seconds / 3600.0

        status = {
            "total_iterations": iteration,
            **{f"total_{k}": v for k, v in total_usage.to_dict().items()},
            "total_tokens": total_usage.total,
            "best_primary_metric": (
                self._best_eval_result.primary_metric_value
                if self._best_eval_result else None
            ),
            "best_metrics": (
                self._best_eval_result.metrics
                if self._best_eval_result else {}
            ),
            "eval_submissions_used": self._eval_submissions_used,
            "elapsed_hours": round(elapsed_hours, 4),
            # LLM wall-clock excluded from budget — persisted so resume
            # continues with the right budget math.
            "llm_elapsed_hours": round(llm_elapsed_hours, 4),
            "effective_elapsed_hours": round(
                max(elapsed_hours - llm_elapsed_hours, 0.0), 4,
            ),
            "remaining_hours": round(self.remaining_budget_hours(), 4),
            "time_budget_hours": self.task_config.total_time_budget_hours,
            "remaining_iterations": self.remaining_iterations(),
            "max_iterations": self.task_config.max_iterations,
            # Per-iteration token breakdown for charts
            "per_iteration_tokens": [
                {
                    "iteration": r.iteration,
                    **r.token_usage.to_dict(),
                    "eval_submitted": r.eval_submitted,
                    "eval_result": r.eval_result.metrics if r.eval_result else None,
                }
                for r in self.history
            ],
        }
        self.store.save_json(
            os.path.join(self.store.experiment_dir, "live_status.json"),
            status,
        )

    def _build_observation(
        self,
        iteration: int,
        *,
        task_description: Optional[str] = None,
        command_output: Optional[CommandOutput] = None,
        eval_result: Optional[EvalResult] = None,
        error: Optional[str] = None,
    ) -> Observation:
        workspace_files = list_workspace_files(self.store.workspace_dir)

        # Only send contents of files modified in the last N iterations.
        # Stale one-time scripts are excluded to save tokens.
        # Binary/model files are already filtered by read_workspace_file_contents
        # (via _SKIP_EXTENSIONS and _TEXT_EXTENSIONS).
        # Agent can still see all filenames in workspace_files and use 'cat' if needed.
        RECENT_FILE_WINDOW = 5

        agent_files: set[str] | None = None
        if self.history:
            agent_files = set()
            for rec in self.history[-RECENT_FILE_WINDOW:]:
                agent_files.update(rec.files_written)

        file_contents = read_workspace_file_contents(
            self.store.workspace_dir, workspace_files,
            agent_written_files=agent_files,
        )
        curves, training_curve_images = collect_training_curves(self.store.workspace_dir)

        return Observation(
            task_description=task_description,
            workspace_files=workspace_files,
            workspace_file_contents=file_contents,
            iteration=iteration,
            command_output=command_output,
            eval_result=eval_result,
            remaining_time_budget_hours=round(self.remaining_budget_hours(), 4),
            remaining_iterations=self.remaining_iterations(),
            history=list(self.history),
            error=error,
            best_eval_result=self._best_eval_result,
            training_curves=curves,
            training_curve_images=training_curve_images,
        )

    def mark_error(self, reason: str = "") -> None:
        """Record that the experiment terminated due to an error."""
        self._terminal_reason = TerminalReason.ERROR
        self._termination_detail = reason

    def _check_done(self) -> Optional[TerminalReason]:
        """Check if experiment should terminate. Returns reason or None."""
        if self.remaining_budget_hours() <= 0:
            return TerminalReason.TIME_BUDGET_EXHAUSTED
        if self.current_iteration >= self.task_config.max_iterations:
            return TerminalReason.ITERATION_LIMIT_REACHED
        return None

    def remaining_budget_hours(self) -> float:
        if self._start_time == 0:
            return self.task_config.total_time_budget_hours
        wall_seconds = time.time() - self._start_time
        # LLM API call time is excluded from the task budget.
        effective_seconds = max(wall_seconds - self._llm_elapsed_seconds, 0.0)
        return max(
            self.task_config.total_time_budget_hours - effective_seconds / 3600.0,
            0.0,
        )

    def remaining_iterations(self) -> int:
        return max(self.task_config.max_iterations - self.current_iteration, 0)

    def cleanup(self) -> None:
        if hasattr(self, "sandbox_runner") and self.sandbox_runner is not None:
            self.sandbox_runner.cleanup()
        if self._eval_runner is not None:
            self._eval_runner.cleanup()
            self._eval_runner = None
        # Close experiment log file handler
        if hasattr(self, "_log_file_handler") and self._log_file_handler:
            import logging as _logging
            _logging.getLogger("farbench").removeHandler(self._log_file_handler)
            self._log_file_handler.close()
            self._log_file_handler = None

    def _check_no_orphaned_containers(self) -> None:  # noqa: D401
        """Before starting, verify no containers from a previous crashed run exist."""
        from farbench.runner import _get_docker_client
        try:
            client = _get_docker_client()
        except Exception:
            return

        agent_safe = self.store.agent_id.replace("_", "-")
        sandbox_name = f"farbench-{agent_safe}-sandbox"
        eval_name = f"farbench-{agent_safe}-eval"

        found = []
        for name in (sandbox_name, eval_name):
            try:
                client.containers.get(name)
                found.append(name)
            except Exception:
                pass

        if found:
            remove_cmd = "  " + "\n  ".join(f"docker rm -f {n}" for n in found)
            raise RuntimeError(
                f"Existing container(s) found for task={self.task_config.name}, "
                f"agent={self.store.agent_id}:\n"
                f"  {', '.join(found)}\n\n"
                f"This means either:\n"
                f"  1. A previous experiment did not exit cleanly (orphaned), or\n"
                f"  2. Another experiment with the same agent_id is still running.\n\n"
                f"If you are sure no experiment is running, remove them:\n\n"
                f"{remove_cmd}\n"
            )
