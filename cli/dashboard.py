"""Dashboard CLI command."""

from __future__ import annotations

import socket
import sys

import click


def register_dashboard(root: click.Group) -> None:
    """Register `farbench dashboard` on the root CLI group."""

    @root.command()
    @click.option("--port", default=8501, type=int, help="Dashboard port")
    @click.option("--host", default="0.0.0.0", help="Dashboard host")
    @click.option("--experiments-dir", default="experiments", help="Experiments directory")
    def dashboard(port, host, experiments_dir):
        """Launch the experiment dashboard (web UI)."""
        try:
            import uvicorn
            from fastapi import FastAPI
        except ImportError:
            click.echo("Install dependencies: pip install fastapi uvicorn", err=True)
            sys.exit(1)

        from gui.dashboard_api import create_dashboard_router

        app = FastAPI(title="FARBench Dashboard")
        app.include_router(create_dashboard_router(experiments_dir=experiments_dir))

        click.echo("=" * 60)
        click.echo("  FARBench Dashboard")
        click.echo("=" * 60)
        click.echo(f"  Local:   http://localhost:{port}/dashboard")
        if host == "0.0.0.0":
            try:
                probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                probe.connect(("8.8.8.8", 80))
                lan_ip = probe.getsockname()[0]
                probe.close()
                click.echo(f"  Network: http://{lan_ip}:{port}/dashboard")
            except Exception:
                pass
        click.echo("=" * 60)

        uvicorn.run(app, host=host, port=port)
