"""Task-management CLI commands."""

from __future__ import annotations

import sys

import click


def register_tasks(root: click.Group) -> None:
    """Register `farbench tasks ...` commands on the root CLI group."""

    @root.group()
    def tasks():
        """Task management commands."""
        pass

    @tasks.command("list")
    @click.option("--benchmarks-dir", default="benchmarks", help="Benchmarks directory")
    def tasks_list(benchmarks_dir):
        """List all available tasks."""
        from farbench.tasks import TaskRegistry

        registry = TaskRegistry(benchmarks_dir)
        registry.discover()
        all_tasks = registry.list_all()

        if not all_tasks:
            click.echo("No tasks found.")
            return

        click.echo(
            f"{'Name':<30} {'Compute':<8} {'Metric':<15} "
            f"{'Budget':<10} {'Description'}"
        )
        click.echo("-" * 100)
        for task in all_tasks:
            click.echo(
                f"{task['name']:<30} {task['compute_type']:<8} "
                f"{task['primary_metric']:<15} "
                f"{task['total_time_budget_hours']:<10} "
                f"{task['description'][:40]}"
            )

    @tasks.command("info")
    @click.argument("task_name")
    @click.option("--benchmarks-dir", default="benchmarks")
    def tasks_info(task_name, benchmarks_dir):
        """Show detailed task info."""
        from farbench.tasks import TaskPreparer, TaskRegistry, _infer_cuda_variant

        registry = TaskRegistry(benchmarks_dir)
        registry.discover()
        config = registry.get(task_name)

        click.echo(f"Name:               {config.name}")
        click.echo(f"Description:        {config.description.strip()}")
        click.echo(f"Compute:            {config.compute_type}")
        click.echo(
            "Primary metric:     "
            f"{config.primary_metric} "
            f"({'higher' if config.higher_is_better else 'lower'} is better)"
        )
        click.echo(f"Time budget:        {config.total_time_budget_hours}h")
        click.echo(f"Max iterations:     {config.max_iterations}")
        click.echo(f"Network access:     {config.network_access}")
        click.echo(f"Eval contract:      {config.eval_contract}")
        click.echo(f"Docker image:       {config.docker_image}")

        cuda_variant = ""
        if config.docker_image:
            cuda_variant = _infer_cuda_variant(config.docker_image)
        prepared = TaskPreparer(config).check_status(cuda_variant=cuda_variant)
        click.echo(f"Prepared:           {'Yes' if prepared else 'No'}")

    @tasks.command("prepare")
    @click.argument("task_name")
    @click.option("--benchmarks-dir", default="benchmarks")
    @click.option("--force", is_flag=True, help="Force re-prepare")
    @click.option(
        "--cuda",
        "cuda_suffix",
        default="",
        envvar="FARBENCH_CUDA",
        help="CUDA variant (cu118 or cu128).",
    )
    def tasks_prepare(task_name, benchmarks_dir, force, cuda_suffix):
        """Prepare a task from the published Hugging Face image archive."""
        from farbench.tasks import TaskPreparer, TaskRegistry

        registry = TaskRegistry(benchmarks_dir)
        registry.discover()
        config = registry.get(task_name)

        if cuda_suffix:
            from farbench.tasks import _validate_cuda

            try:
                _validate_cuda(cuda_suffix)
            except ValueError as exc:
                raise click.ClickException(str(exc)) from exc

        preparer = TaskPreparer(config)
        result = preparer.prepare(
            force=force,
            cuda_suffix=cuda_suffix,
        )

        if result.success:
            click.echo(f"Task '{task_name}' prepared successfully.")
            click.echo(f"Steps: {', '.join(result.steps_completed)}")
            click.echo(f"Time: {result.total_time_minutes:.1f} minutes")
            return

        click.echo(f"Task '{task_name}' preparation FAILED.", err=True)
        for err in result.errors:
            click.echo(f"  Error: {err}", err=True)
        sys.exit(1)
