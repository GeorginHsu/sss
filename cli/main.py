"""FARBench CLI — command-line interface."""

from __future__ import annotations

import json
import os
import re
import sys
import time
from typing import TYPE_CHECKING

import click

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _load_project_env() -> None:
    """Load repo-local .env for local CLI commands; docker compose also reads it."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"), override=False)


_load_project_env()

from farbench.schemas import AgentMode  # noqa: E402
from farbench.utils import get_logger  # noqa: E402
from cli.dashboard import register_dashboard  # noqa: E402
from cli.demo import register_demo  # noqa: E402
from cli.results import register_results  # noqa: E402
from cli.tasks import register_tasks  # noqa: E402

logger = get_logger("farbench.cli")

if TYPE_CHECKING:
    from farbench.llm import ProviderProfile, ResolvedProvider


@click.group()
def cli():
    """FARBench CLI."""
    pass


register_tasks(cli)
register_results(cli)
register_demo(cli)
register_dashboard(cli)


# ═══════════════════════════════════════════════════════════════════════════
#  Core experiment runner
# ═══════════════════════════════════════════════════════════════════════════

def _resolve_agent_provider(
    preset: str,
) -> "ResolvedProvider":
    """Resolve a provider preset into endpoint/model/key settings."""
    from farbench.llm import resolve_provider

    try:
        return resolve_provider(preset)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc


def _resolve_stored_agent_provider(
    stored_agent: dict,
) -> "ResolvedProvider":
    """Resolve the exact non-secret provider settings saved by a run."""
    from farbench.llm import resolve_stored_provider

    try:
        return resolve_stored_provider(stored_agent)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc


def _configure_run_environment(
    *,
    gpus: str,
    cuda: str,
) -> str:
    """Apply per-run environment settings consumed by lower framework layers."""
    cuda = (cuda or os.environ.get("FARBENCH_CUDA") or "cu118").strip()
    if cuda:
        from farbench.tasks import _validate_cuda

        try:
            _validate_cuda(cuda)
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
        os.environ["FARBENCH_CUDA"] = cuda

    gpus = (gpus or "").strip()
    if gpus:
        # FARBENCH_GPUS is used by farbench.utils.select_best_gpus(); NVIDIA_VISIBLE_DEVICES
        # is also set for child processes and non-Docker local runs.
        os.environ["FARBENCH_GPUS"] = gpus
        os.environ["NVIDIA_VISIBLE_DEVICES"] = gpus

    return cuda


def _make_default_agent_id(*, task: str, mode: str, agent_preset: str) -> str:
    """Generate descriptive IDs for experiment dirs and Docker containers."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if agent_preset:
        prefix = agent_preset
    elif mode == AgentMode.DEMO:
        prefix = "demo"
    elif mode == AgentMode.INTERACTIVE:
        prefix = "interactive"
    else:
        prefix = "agent"
    raw = f"{prefix}_{task}_{timestamp}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_") or f"agent_{timestamp}"


def _agent_config_for_store(
    provider: "ResolvedProvider",
    *,
    temperature: float,
    max_tokens: int,
) -> dict:
    """Build non-secret agent config for experiment config.json."""
    cfg = provider.public_dict()
    cfg.update(
        {
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
    )
    return cfg


def _persist_agent_config(store, agent_config: dict | None) -> None:
    """Append non-secret agent config to an experiment config.json."""
    if not store or not agent_config:
        return
    config_path = os.path.join(store.experiment_dir, "config.json")
    try:
        with open(config_path) as f:
            config = json.load(f)
        config["agent"] = agent_config
        store.save_json(config_path, config)
    except Exception as exc:
        logger.warning(f"Could not persist agent config: {exc}")


def _prepare_task_for_run(
    *,
    task: str,
    benchmarks_dir: str,
    cuda: str,
    force: bool,
) -> None:
    """Prepare one task before `farbench run` enters the environment."""
    from farbench.tasks import TaskPreparer, TaskRegistry

    registry = TaskRegistry(benchmarks_dir)
    registry.discover()
    config = registry.get(task)

    click.echo(f"Preparing task '{task}' (cuda={cuda}) ...")
    result = TaskPreparer(config).prepare(
        force=force,
        cuda_suffix=cuda,
    )
    if result.success:
        click.echo(f"Task '{task}' prepared: {', '.join(result.steps_completed)}")
        return

    for err in result.errors:
        click.echo(f"  Error: {err}", err=True)
    raise click.ClickException(f"Task '{task}' preparation failed.")


def _run_experiment(
    task: str,
    agent_id: str,
    agent,
    *,
    mode: str = AgentMode.API,
    benchmarks_dir: str = "benchmarks",
    experiments_dir: str = "experiments",
    resume_from: str | None = None,
    agent_config: dict | None = None,
) -> dict | None:
    """Run a single experiment and return final_results dict (or None on failure).

    Shared by ``farbench run`` and ``farbench resume``. When ``resume_from`` is
    provided, we attach to that existing experiment directory instead of
    starting a new one.
    """
    from farbench.agent import APIError
    from farbench.env import ResearchEnv

    env = ResearchEnv(
        benchmarks_dir=benchmarks_dir,
        experiments_dir=experiments_dir,
    )
    try:
        if resume_from:
            obs = env.resume(resume_from)
            # Seed the live agent's usage counters from live_status.json so that
            # final/leaderboard totals reflect pre-abort spend too.
            if agent is not None:
                _seed_agent_usage_from_live_status(agent, resume_from)
            click.echo(
                f"\nResuming experiment: task={env.task_config.name}, "
                f"agent_id={env.experiment_store.agent_id}"
            )
            click.echo(f"Resumed from: {resume_from}")
            # Surface resume provenance so the user/auditor can see at a glance:
            # how many iters already succeeded, how many resume attempts exist
            # in total, and which iter_NNN/ the next run will create. This is
            # printed once at startup (not every iteration) to stay unobtrusive.
            try:
                with open(os.path.join(resume_from, "config.json")) as _cf:
                    _cfg = json.load(_cf)
                _last_good = int(env.status().get("current_iteration") or 0)
                _resumes = _cfg.get("resumes", [])
                click.echo(
                    f"Resume state: {_last_good} iter(s) already completed, "
                    f"resume attempts so far: {len(_resumes)}, "
                    f"next iteration will be iter_{_last_good + 1:03d}"
                )
            except Exception as _e:
                logger.debug(f"Could not surface resume provenance: {_e}")
        else:
            obs = env.reset(task_name=task, agent_id=agent_id)
            _persist_agent_config(env.experiment_store, agent_config)
            click.echo(f"\nStarting experiment: task={task}, agent_id={agent_id}")
    except (RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Time budget: {obs.remaining_time_budget_hours}h, max iterations: {env.task_config.max_iterations}")

    # Tolerate transient LLM output parse failures: feed the error back to
    # the agent via obs.error so it can self-correct on the next iteration,
    # instead of crashing the whole experiment (which would trigger the
    # outer retry mechanism and leak the sandbox container).
    MAX_JSON_PARSE_FAILURES = 5
    json_parse_failures = 0

    # Start the local iteration counter aligned with env's internal state.
    # For a fresh run: current_iteration is 0 → first loop prints "Iteration 1".
    # For a resume: current_iteration equals `last_good_iter` (already completed),
    # so the first loop prints the NEXT iter number (e.g. if iter_001 is the
    # last good one and iter_002 was aborted, this resumes at "Iteration 2",
    # matching the `iter_002/` directory that orchestrator will actually create).
    iteration = int(env.status().get("current_iteration") or 0)
    try:
        while True:
            iteration += 1
            click.echo(f"\n--- Iteration {iteration} ---")

            metadata = None

            if mode == AgentMode.DEMO:
                action = _demo_action(obs)
            elif mode == AgentMode.API:
                try:
                    action, metadata = agent.act(obs, env.task_config)
                    json_parse_failures = 0  # reset on any successful parse
                except APIError as e:
                    logger.error(f"API error: {e}")
                    env.mark_error(f"APIError: {e}")
                    click.echo(f"\nExperiment aborted (error): APIError: {e}")
                    break
                except ValueError as e:
                    # LLM output couldn't be parsed as JSON. Feed the error
                    # back via obs.error so the agent self-corrects next
                    # iteration. Skip env.step() so iteration budget is not
                    # consumed — only 1 extra LLM call is spent.
                    json_parse_failures += 1
                    logger.warning(
                        f"Agent output parse error "
                        f"({json_parse_failures}/{MAX_JSON_PARSE_FAILURES}): {e}"
                    )
                    if json_parse_failures >= MAX_JSON_PARSE_FAILURES:
                        logger.error(
                            "Too many consecutive JSON parse failures, "
                            "ending experiment."
                        )
                        env.mark_error(f"Too many JSON parse failures: {e}")
                        click.echo(
                            f"\nExperiment aborted (error): "
                            f"Too many JSON parse failures: {e}"
                        )
                        break
                    obs.error = (
                        f"Your previous response could not be parsed as valid JSON.\n"
                        f"Parser error: {e}\n\n"
                        f"REMINDER: You MUST respond with a SINGLE well-formed "
                        f"JSON object. Either raw JSON:\n"
                        f'  {{"reasoning": "...", "command": "...", ...}}\n'
                        f"Or wrapped in a ```json ... ``` fence. Do NOT include "
                        f"multiple JSON objects or prose outside the JSON. "
                        f"Please try again."
                    )
                    continue
            else:
                action = _interactive_action(obs)

            obs, reward, done, info = env.step(action, agent_metadata=metadata)

            if action.done:
                click.echo("\nAgent requested stop.")
                if action.done_reason:
                    click.echo(f"Reason: {action.done_reason}")
                break

            click.echo(f"Reward: {reward:.4f}")
            if obs.eval_result:
                click.echo(f"Eval: {obs.eval_result.metrics}")
            if obs.command_output:
                click.echo(f"Command exit: {obs.command_output.exit_code}")
            if obs.error:
                click.echo(f"Error: {obs.error}")
            click.echo(f"Remaining: time={obs.remaining_time_budget_hours:.3f}h, iterations={obs.remaining_iterations}")

            # Show token usage (per-iteration + cumulative)
            if metadata:
                tu = metadata.token_usage
                click.echo(
                    f"Tokens (this iter): "
                    f"in={tu.input_tokens:,}  out={tu.output_tokens:,}"
                    f"{'  think=' + f'{tu.thinking_tokens:,}' if tu.thinking_tokens else ''}"
                    f"{'  cached=' + f'{tu.cache_read_tokens:,}' if tu.cache_read_tokens else ''}"
                    f"  total={tu.total:,}"
                )
            if agent is not None:
                usage = agent.usage_summary()
                tu = usage.total_usage
                click.echo(
                    f"Tokens (cumulative): "
                    f"in={tu.input_tokens:,}  out={tu.output_tokens:,}"
                    f"{'  think=' + f'{tu.thinking_tokens:,}' if tu.thinking_tokens else ''}"
                    f"{'  cached=' + f'{tu.cache_read_tokens:,}' if tu.cache_read_tokens else ''}"
                    f"  total={tu.total:,}"
                )

            if done:
                reason = info.get("terminal_reason", "unknown")
                click.echo(f"\nExperiment complete ({reason})")
                break
    except Exception as e:
        logger.error(f"Experiment aborted due to error: {e}")
        env.mark_error(f"{type(e).__name__}: {e}")
        click.echo(f"\nExperiment aborted (error): {type(e).__name__}: {e}")

    # Usage summary
    if agent is not None:
        usage = agent.usage_summary()
        tu = usage.total_usage
        click.echo(f"\n{'=' * 50}")
        click.echo(f"LLM Usage Summary ({usage.model})")
        click.echo(f"  Calls:           {usage.total_calls}")
        click.echo(f"  Input tokens:    {tu.input_tokens:,}")
        click.echo(f"  Output tokens:   {tu.output_tokens:,}")
        if tu.thinking_tokens:
            click.echo(f"  Thinking tokens: {tu.thinking_tokens:,}")
        if tu.cache_read_tokens:
            click.echo(f"  Cache read:      {tu.cache_read_tokens:,}")
        click.echo(f"  Total tokens:    {tu.total:,}")
        click.echo(f"  Total latency:   {usage.total_latency_seconds:.1f}s")
        _save_usage(env, usage)

    try:
        env.close()
    except Exception as e:
        logger.warning(f"Error during finalize: {e}")

    # Read final results after closing (finalize() writes final_results.json)
    final_results = _read_final_results(env)
    return final_results


def _seed_agent_usage_from_live_status(agent, experiment_dir: str) -> None:
    """Pre-populate agent.usage_summary totals from a resumed experiment.

    Reads <experiment_dir>/live_status.json and forwards the cumulative
    token counters to agent.seed_usage() so leaderboards after a resume
    reflect total spend (pre-abort + post-resume).
    """
    live_path = os.path.join(experiment_dir, "live_status.json")
    if not os.path.isfile(live_path):
        return
    try:
        with open(live_path) as f:
            live = json.load(f)
    except Exception as e:
        click.echo(f"Warning: could not read live_status.json for seeding: {e}",
                   err=True)
        return
    try:
        agent.seed_usage(
            total_calls=live.get("total_iterations", 0) or 0,
            total_input_tokens=live.get("total_input_tokens", 0) or 0,
            total_output_tokens=live.get("total_output_tokens", 0) or 0,
            total_thinking_tokens=live.get("total_thinking_tokens", 0) or 0,
            total_cache_read_tokens=live.get("total_cache_read_tokens", 0) or 0,
            total_cache_creation_tokens=live.get("total_cache_creation_tokens", 0) or 0,
        )
    except AttributeError:
        # Older agent types without seed_usage() — silently skip, usage just
        # restarts from zero for this resumed run.
        pass


def _read_final_results(env: "ResearchEnv") -> dict | None:
    """Read final_results.json from the experiment directory."""
    try:
        store = env.experiment_store
        if store is None:
            return None
        path = os.path.join(store.summary_dir(), "final_results.json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    except Exception as e:
        click.echo(f"Warning: could not read final results: {e}", err=True)
    return None


def _save_usage(env: "ResearchEnv", usage: "UsageSummary") -> None:
    """Save LLM usage summary to the experiment's summary directory."""
    try:
        store = env.experiment_store
        if store is None:
            return
        summary_dir = store.summary_dir()
        os.makedirs(summary_dir, exist_ok=True)
        usage_path = os.path.join(summary_dir, "llm_usage.json")
        store.save_json(usage_path, usage.to_dict())
    except Exception as e:
        click.echo(f"Warning: could not save usage: {e}", err=True)


def _build_agent_kwargs(
    base_url: str,
    model: str,
    api_key: str,
    provider_profile: "ProviderProfile",
    temperature: float,
    max_tokens: int,
) -> dict:
    return {
        "base_url": base_url,
        "model": model,
        "api_key": api_key,
        "provider_profile": provider_profile,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }


def _create_agent(
    base_url: str,
    model: str,
    api_key: str,
    provider_profile: "ProviderProfile",
    temperature: float,
    max_tokens: int,
) -> "LLMAgent":
    """Create an LLMAgent instance."""
    from farbench.agent import LLMAgent
    return LLMAgent(
        base_url=base_url,
        model=model,
        api_key=api_key,
        provider_profile=provider_profile,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _demo_action(obs: "Observation") -> "Action":
    from farbench.schemas import Action
    if obs.iteration == 0:
        return Action(
            files_to_write={
                "hello.py": "print('Hello from FARBench v2!')\nprint('Workspace is ready.')\n",
            },
            command="python hello.py",
        )
    return Action(done=True)


def _interactive_action(obs: "Observation") -> "Action":
    from farbench.schemas import Action
    if obs.eval_result:
        click.echo(f"Last eval: {obs.eval_result.metrics}")
    if obs.error:
        click.echo(f"Error: {obs.error}")
    command = click.prompt(
        "\nEnter command to execute (or 'done' to stop)",
        default="", show_default=False,
    )
    if command.strip().lower() == "done":
        return Action(done=True)
    return Action(command=command if command.strip() else None)


# ═══════════════════════════════════════════════════════════════════════════
#  farbench run — single experiment
# ═══════════════════════════════════════════════════════════════════════════

@cli.command()
@click.option("--task", required=True, help="Task name")
@click.option("--mode", default=AgentMode.DEMO.value,
              type=click.Choice([m.value for m in AgentMode]),
              help="demo: simple test, interactive: human input, api: remote LLM agent")
@click.option(
    "--agent-id",
    default="",
    help="Agent identifier. Defaults to <preset>_<task>_<timestamp>.",
)
@click.option("--benchmarks-dir", default="benchmarks")
@click.option("--experiments-dir", default="experiments")
@click.option("--gpus", default="", help="Comma-separated host GPU IDs for this run")
@click.option(
    "--cuda",
    default="",
    envvar="FARBENCH_CUDA",
    help="CUDA variant (cu118 or cu128). Defaults to FARBENCH_CUDA or cu118.",
)
@click.option("--prepare", is_flag=True, help="Prepare the task before running")
@click.option("--force-prepare", is_flag=True, help="Re-prepare even if already marked ready")
@click.option("--agent-preset", "--preset", "agent_preset", default="", help=(
    "LLM provider preset, e.g. claude, gemini, gpt54, kimi."
))
@click.option("--agent-temperature", default=0.1, type=float, help="API mode temperature")
@click.option("--agent-max-tokens", default=32768, type=int, help="API mode max_tokens")
def run(
    task, mode, agent_id, benchmarks_dir, experiments_dir, gpus, cuda,
    prepare, force_prepare,
    agent_preset,
    agent_temperature, agent_max_tokens,
):
    """Run a single experiment."""
    cuda = _configure_run_environment(gpus=gpus, cuda=cuda)
    if force_prepare and not prepare:
        raise click.ClickException("--force-prepare is only valid together with --prepare.")
    if prepare:
        _prepare_task_for_run(
            task=task,
            benchmarks_dir=benchmarks_dir,
            cuda=cuda,
            force=force_prepare,
        )

    provider = None
    agent_config = None
    if mode == AgentMode.API:
        provider = _resolve_agent_provider(agent_preset)
        agent_config = _agent_config_for_store(
            provider,
            temperature=agent_temperature,
            max_tokens=agent_max_tokens,
        )

    if not agent_id:
        agent_id = _make_default_agent_id(
            task=task,
            mode=mode,
            agent_preset=provider.preset if provider else agent_preset,
        )

    agent = None
    if mode == AgentMode.API:
        agent_kwargs = _build_agent_kwargs(
            provider.base_url, provider.model, provider.api_key, provider.profile,
            agent_temperature, agent_max_tokens,
        )
        agent = _create_agent(**agent_kwargs)

    result = _run_experiment(
        task=task,
        agent_id=agent_id,
        agent=agent,
        mode=mode,
        benchmarks_dir=benchmarks_dir,
        experiments_dir=experiments_dir,
        agent_config=agent_config,
    )
    if result is None or result.get("terminal_reason") == "error":
        sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════
#  farbench resume — continue an aborted experiment
# ═══════════════════════════════════════════════════════════════════════════

@cli.command()
@click.option("--experiment-dir", required=True,
              help="Absolute or relative path to the experiment directory to resume")
@click.option("--benchmarks-dir", default="benchmarks")
@click.option("--gpus", default="", help="Comma-separated host GPU IDs for this resume")
@click.option(
    "--cuda",
    default="",
    envvar="FARBENCH_CUDA",
    help="CUDA variant (cu118 or cu128). Defaults to FARBENCH_CUDA or cu118.",
)
@click.option("--agent-preset", "--preset", "agent_preset", default="",
              help="LLM provider preset. If empty, read from config.json or inferred.")
@click.option("--agent-temperature", default=None, type=float)
@click.option("--agent-max-tokens", default=None, type=int)
def resume(
    experiment_dir, benchmarks_dir, gpus, cuda,
    agent_preset,
    agent_temperature, agent_max_tokens,
):
    """Resume a previously-aborted experiment from its last complete iteration.

    The task name and agent_id are recovered from
    <experiment-dir>/config.json; the trailing iter_NNN is treated as aborted
    and archived as iter_NNN.aborted/ before the next iteration starts.
    """
    experiment_dir = os.path.abspath(experiment_dir)
    if not os.path.isdir(experiment_dir):
        raise click.ClickException(f"Not a directory: {experiment_dir}")
    _configure_run_environment(gpus=gpus, cuda=cuda)

    config_path = os.path.join(experiment_dir, "config.json")
    if not os.path.isfile(config_path):
        raise click.ClickException(
            f"config.json missing in {experiment_dir}; cannot infer preset."
        )
    with open(config_path) as f:
        run_config = json.load(f)
    stored_agent = run_config.get("agent") or {}
    if not isinstance(stored_agent, dict):
        stored_agent = {}

    agent_temperature = (
        float(agent_temperature)
        if agent_temperature is not None
        else float(stored_agent.get("temperature", 0.1))
    )
    agent_max_tokens = (
        int(agent_max_tokens)
        if agent_max_tokens is not None
        else int(stored_agent.get("max_tokens", 32768))
    )
    explicit_agent_preset = bool(agent_preset)

    if not agent_preset:
        agent_preset = stored_agent.get("preset", "")

    # Backward compatibility for older experiments: recover preset from the
    # directory name if config.json predates the `agent` block.
    if not agent_preset:
        task_name = run_config.get("task_name", "")
        if not task_name:
            raise click.ClickException("task_name missing in config.json")
        dirname = os.path.basename(experiment_dir)
        marker = f"_{task_name}_"
        if marker not in dirname:
            raise click.ClickException(
                f"Cannot infer preset: task '{task_name}' not in dir name "
                f"{dirname!r}. Pass --agent-preset explicitly."
            )
        agent_preset = dirname.split(marker, 1)[0]
        click.echo(f"Inferred preset: {agent_preset}")

    if explicit_agent_preset:
        provider = _resolve_agent_provider(agent_preset)
    elif stored_agent.get("base_url") and stored_agent.get("model"):
        provider = _resolve_stored_agent_provider(stored_agent)
    else:
        provider = _resolve_agent_provider(agent_preset)

    agent_kwargs = _build_agent_kwargs(
        provider.base_url, provider.model, provider.api_key, provider.profile,
        agent_temperature, agent_max_tokens,
    )
    agent = _create_agent(**agent_kwargs)

    # Recover experiments-dir from the passed-in experiment_dir: two levels up
    # (experiments/<task>/<exp>), which is what _run_experiment uses when it
    # constructs ResearchEnv for resume. (Not strictly necessary since resume
    # passes experiment_dir through, but kept consistent with run's layout.)
    task_dir_parent = os.path.dirname(os.path.dirname(experiment_dir))

    result = _run_experiment(
        task="",  # unused on the resume path
        agent_id="",  # unused
        agent=agent,
        mode=AgentMode.API,
        benchmarks_dir=benchmarks_dir,
        experiments_dir=task_dir_parent,
        resume_from=experiment_dir,
    )
    if result is None or result.get("terminal_reason") == "error":
        sys.exit(1)


if __name__ == "__main__":
    cli()
