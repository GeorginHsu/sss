"""Experiment result CLI command."""

from __future__ import annotations

import json
import os

import click


def register_results(root: click.Group) -> None:
    """Register `farbench results` on the root CLI group."""

    @root.command()
    @click.option("--experiment", required=True, help="Experiment directory path")
    def results(experiment):
        """Show experiment results."""
        summary_path = os.path.join(experiment, "summary", "final_results.json")
        if not os.path.exists(summary_path):
            click.echo("No results found. Experiment may not be complete.")
            return

        with open(summary_path) as f:
            data = json.load(f)
        click.echo(json.dumps(data, indent=2))
