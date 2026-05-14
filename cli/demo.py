"""Terminal trajectory replay command."""

from __future__ import annotations

import json
import os
import re
import textwrap
import time

import click


def _read_json_file(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _one_line(value):
    return " ".join(str(value or "").strip().split())


def _shorten(value, limit=120):
    text = _one_line(value)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _wrap_lines(text, width, indent="  ", max_lines=2):
    text = _one_line(text)
    if not text:
        return []

    body_width = max(24, width - len(indent))
    lines = textwrap.wrap(
        text,
        width=body_width,
        break_long_words=False,
        break_on_hyphens=False,
    )
    if max_lines and len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(".") + "..."
    return [indent + line for line in lines]


def _format_seconds(seconds):
    if seconds is None:
        return None
    try:
        seconds = float(seconds)
    except (TypeError, ValueError):
        return None
    if seconds >= 3600:
        return f"{seconds / 3600:.2f}h"
    if seconds >= 60:
        return f"{seconds / 60:.1f}m"
    return f"{seconds:.1f}s"


def _format_hours(hours):
    try:
        hours = float(hours)
    except (TypeError, ValueError):
        hours = 0.0
    return f"{hours:.2f}h"


def _format_metric(value):
    if value is None:
        return "-"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(value) < 1:
        return f"{value:.4f}"
    return f"{value:.3f}"


def _metric_from_eval(eval_result):
    if not eval_result:
        return None
    if eval_result.get("primary_metric_value") is not None:
        return (
            eval_result.get("primary_metric_name") or "metric",
            eval_result.get("primary_metric_value"),
        )
    metrics = eval_result.get("metrics") or {}
    if not metrics:
        return None
    if "accuracy" in metrics:
        return "accuracy", metrics["accuracy"]
    name, value = next(iter(metrics.items()))
    return name, value


def _is_better(value, best, higher_is_better):
    if best is None:
        return True
    if value is None:
        return False
    try:
        value = float(value)
        best = float(best)
    except (TypeError, ValueError):
        return False
    return value > best if higher_is_better else value < best


def _split_reasoning(text):
    text = str(text or "").strip()
    if not text:
        return {}

    matches = list(re.finditer(r"\b(Observation|Analysis|Decision|Plan|Reflection):", text))
    if not matches:
        return {"Summary": text}

    sections = {}
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        sections[match.group(1)] = text[start:end].strip()
    return sections


def _extract_error_head(error_log):
    for line in str(error_log or "").splitlines():
        line = line.strip()
        if not line:
            continue
        if "Traceback" in line:
            continue
        if "Error" in line or "failed" in line.lower() or "Exception" in line:
            return line
    return _shorten(error_log, 120)


def _iter_dirs(experiment):
    try:
        names = os.listdir(experiment)
    except OSError:
        return []
    rows = []
    for name in names:
        if not re.fullmatch(r"iter_\d{3}", name):
            continue
        rows.append((int(name.split("_", 1)[1]), os.path.join(experiment, name)))
    return [path for _, path in sorted(rows)]


def _rows_from_iteration_dirs(experiment):
    rows = []
    for iter_dir in _iter_dirs(experiment):
        iteration = int(os.path.basename(iter_dir).split("_", 1)[1])
        action = _read_json_file(os.path.join(iter_dir, "action.json")) or {}
        command_output = _read_json_file(os.path.join(iter_dir, "command_output.json")) or {}
        eval_result = _read_json_file(os.path.join(iter_dir, "eval_result.json"))

        stdout = command_output.get("stdout") or ""
        stderr = command_output.get("stderr") or ""
        files_to_write = action.get("files_to_write") or {}
        row = {
            "iteration": iteration,
            "command": action.get("command"),
            "files_written": list(files_to_write),
            "command_output_summary": (stdout + ("\n" + stderr if stderr else "")).strip(),
            "error_summary": stderr if command_output.get("exit_code") else "",
            "eval_submitted": bool(action.get("submit_eval")) or eval_result is not None,
            "eval_result": eval_result,
            "eval_error_log": "",
            "elapsed_seconds": command_output.get("elapsed_seconds"),
            "description": action.get("reasoning") or "",
            "_action": action,
        }
        rows.append(row)
    return rows


def _load_demo_rows(experiment):
    trajectory = _read_json_file(os.path.join(experiment, "summary", "trajectory.json")) or {}
    rows = list(trajectory.get("iterations") or [])
    if not rows:
        rows = _rows_from_iteration_dirs(experiment)

    enriched = []
    for row in rows:
        row = dict(row)
        iteration = row.get("iteration")
        if iteration is not None:
            action_path = os.path.join(experiment, f"iter_{int(iteration):03d}", "action.json")
            action = _read_json_file(action_path)
            if action:
                row["_action"] = action
                row["command"] = row.get("command") or action.get("command")
                if not row.get("files_written") and action.get("files_to_write"):
                    row["files_written"] = list(action.get("files_to_write") or {})
        enriched.append(row)
    return enriched


def _new_best_indices(rows, higher_is_better):
    if any("_demo_new_best" in row for row in rows):
        return {idx for idx, row in enumerate(rows) if row.get("_demo_new_best")}

    best = None
    indices = set()
    for idx, row in enumerate(rows):
        metric = _metric_from_eval(row.get("eval_result"))
        if not metric:
            continue
        _, value = metric
        if _is_better(value, best, higher_is_better):
            best = value
            indices.add(idx)
    return indices


def _annotate_best_progress(rows, higher_is_better):
    best = None
    best_iter = None
    annotated = []
    for row in rows:
        row = dict(row)
        metric = _metric_from_eval(row.get("eval_result"))
        row["_demo_new_best"] = False
        if metric:
            _, value = metric
            if _is_better(value, best, higher_is_better):
                best = value
                best_iter = row.get("iteration")
                row["_demo_new_best"] = True
        row["_demo_best_value"] = best
        row["_demo_best_iter"] = best_iter
        annotated.append(row)
    return annotated


def _select_demo_rows(rows, max_steps, show_all, higher_is_better):
    if show_all or max_steps <= 0 or len(rows) <= max_steps:
        return rows

    new_best = _new_best_indices(rows, higher_is_better)
    last_idx = len(rows) - 1
    ranked = []
    for idx, row in enumerate(rows):
        rank = 9
        if idx == 0 or idx == last_idx or idx in new_best:
            rank = 0
        if row.get("eval_submitted") and not row.get("eval_result") and row.get("eval_error_log"):
            rank = min(rank, 0)
        elif row.get("files_written"):
            rank = min(rank, 1)
        elif row.get("eval_submitted"):
            rank = min(rank, 2)
        elif row.get("command"):
            rank = min(rank, 3)
        ranked.append((rank, idx, row))

    selected = sorted(ranked, key=lambda item: (item[0], item[1]))[:max_steps]
    selected_by_idx = {idx: item for item in selected for idx in [item[1]]}
    for mandatory_idx in (0, last_idx):
        if mandatory_idx in selected_by_idx:
            continue
        removable = [
            item for item in selected_by_idx.values()
            if item[1] not in (0, last_idx)
        ]
        if removable:
            worst = max(removable, key=lambda item: (item[0], item[1]))
            selected_by_idx.pop(worst[1], None)
        selected_by_idx[mandatory_idx] = (0, mandatory_idx, rows[mandatory_idx])
    return [row for _, _, row in sorted(selected_by_idx.values(), key=lambda item: item[1])]


def _best_metric_summary(rows, higher_is_better):
    best_name = None
    best_value = None
    best_iter = None
    for row in rows:
        metric = _metric_from_eval(row.get("eval_result"))
        if not metric:
            continue
        name, value = metric
        if _is_better(value, best_value, higher_is_better):
            best_name = name
            best_value = value
            best_iter = row.get("iteration")
    return best_name, best_value, best_iter


def _print_demo_kv(label, value, width, fg=None, max_lines=2):
    if value is None or value == "":
        return
    label_width = 11
    prefix = click.style(f"{label:<{label_width}} ", fg=fg, bold=True)
    lines = _wrap_lines(value, width, indent="", max_lines=max_lines)
    if not lines:
        return
    click.echo(prefix + lines[0])
    for line in lines[1:]:
        click.echo(" " * (label_width + 1) + line)


def _print_demo_row(row, ordinal, total, width, higher_is_better, best_state):
    iteration = int(row.get("iteration") or ordinal)
    title = f"[{ordinal:02d}/{total:02d}] iter_{iteration:03d}"
    elapsed = _format_seconds(row.get("elapsed_seconds"))
    if elapsed:
        title += f"  {elapsed}"
    click.secho(title, fg="cyan", bold=True)

    sections = _split_reasoning(row.get("description") or row.get("reasoning"))
    for label in ("Observation", "Analysis", "Decision", "Plan", "Reflection", "Summary"):
        if label in sections:
            _print_demo_kv(label.upper(), sections[label], width, fg="white", max_lines=2)

    action = row.get("_action") or {}
    files = row.get("files_written") or []
    packages = action.get("packages_to_install") or []
    command = row.get("command") or action.get("command")
    submit_eval = action.get("submit_eval")

    action_bits = []
    if files:
        action_bits.append("write " + ", ".join(files[:5]) + (" ..." if len(files) > 5 else ""))
    if packages:
        action_bits.append("install " + ", ".join(packages[:4]) + (" ..." if len(packages) > 4 else ""))
    if command:
        action_bits.append("run " + _shorten(command, 150))
    if submit_eval:
        ckpt = submit_eval.get("checkpoint_path") or "checkpoint"
        pred = submit_eval.get("predict_script") or "predict.py"
        action_bits.append(f"submit {ckpt} via {pred}")
    elif row.get("eval_submitted"):
        action_bits.append("submit evaluation")
    if action_bits:
        _print_demo_kv("ACTION", " | ".join(action_bits), width, fg="yellow", max_lines=3)

    output = row.get("command_output_summary") or row.get("error_summary")
    if output:
        _print_demo_kv("SANDBOX", output, width, fg="blue", max_lines=2)

    metric = _metric_from_eval(row.get("eval_result"))
    if row.get("eval_submitted"):
        if metric:
            name, value = metric
            is_new_best = bool(row.get("_demo_new_best"))
            if is_new_best:
                best_state["value"] = value
                best_state["iteration"] = iteration
            best_iter = row.get("_demo_best_iter") or best_state.get("iteration")
            suffix = "new best" if is_new_best else f"best iter_{int(best_iter):03d}"
            _print_demo_kv(
                "EVAL",
                f"{name}={_format_metric(value)} ({suffix})",
                width,
                fg="green",
                max_lines=1,
            )
        else:
            _print_demo_kv(
                "EVAL",
                "failed: " + _extract_error_head(row.get("eval_error_log")),
                width,
                fg="red",
                max_lines=2,
            )

    usage = row.get("token_usage") or {}
    if usage:
        tokens = [
            f"in={usage.get('input_tokens', 0)}",
            f"out={usage.get('output_tokens', 0)}",
            f"think={usage.get('thinking_tokens', 0)}",
        ]
        cache = usage.get("cache_read_tokens")
        if cache:
            tokens.append(f"cache={cache}")
        click.secho(f"{'TOKENS':<11} " + " ".join(tokens), fg="bright_black")


def register_demo(root: click.Group) -> None:
    """Register `farbench demo` on the root CLI group."""

    @root.command("demo")
    @click.argument("experiment", type=click.Path(exists=True, file_okay=False, dir_okay=True))
    @click.option("--all", "show_all", is_flag=True, help="Replay every recorded iteration.")
    @click.option(
        "--max-steps",
        default=12,
        show_default=True,
        help="Maximum keyframes to show unless --all is used.",
    )
    @click.option("--delay", default=0.12, show_default=True, help="Pause between frames in seconds.")
    @click.option("--width", default=96, show_default=True, help="Text wrap width.")
    def demo(experiment, show_all, max_steps, delay, width):
        """Replay an experiment trajectory as a terminal agent-workflow demo."""
        experiment = os.path.abspath(experiment)
        config = _read_json_file(os.path.join(experiment, "config.json")) or {}
        final = _read_json_file(os.path.join(experiment, "summary", "final_results.json")) or {}
        rows = _load_demo_rows(experiment)
        if not rows:
            raise click.ClickException("No trajectory rows found in this experiment.")

        higher_is_better = bool(final.get("higher_is_better", config.get("higher_is_better", True)))
        rows = _annotate_best_progress(rows, higher_is_better)
        shown_rows = _select_demo_rows(rows, max_steps, show_all, higher_is_better)
        metric_name, best_value, best_iter = _best_metric_summary(rows, higher_is_better)

        line = "=" * min(width, 110)
        click.secho(line, fg="bright_black")
        click.secho("FARBench CLI Demo", bold=True)
        click.echo(f"Experiment: {os.path.basename(experiment)}")
        click.echo(f"Task:       {final.get('task_name') or config.get('task_name') or '-'}")
        click.echo("Workflow:   task.yaml -> observation -> agent -> action -> agent sandbox -> eval sandbox")
        click.echo(
            "Budget:     "
            f"{final.get('total_iterations', len(rows))}/"
            f"{config.get('max_iterations', final.get('max_iterations', len(rows)))} iterations, "
            f"{_format_hours(final.get('total_elapsed_hours'))} elapsed"
        )
        if metric_name:
            click.echo(f"Best:       {metric_name}={_format_metric(best_value)} at iter_{int(best_iter):03d}")
        view = "all iterations" if show_all else f"{len(shown_rows)} keyframes from {len(rows)} iterations"
        click.echo(f"View:       {view}")
        click.secho(line, fg="bright_black")

        best_state = {"value": None, "iteration": None}
        for idx, row in enumerate(shown_rows, 1):
            if idx > 1:
                click.echo("")
            _print_demo_row(row, idx, len(shown_rows), width, higher_is_better, best_state)
            if delay > 0 and idx < len(shown_rows):
                time.sleep(delay)

        click.secho(line, fg="bright_black")
        click.echo("Tip: use --all for the complete trajectory, or --delay 0 for instant playback.")
