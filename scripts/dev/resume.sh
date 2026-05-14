#!/usr/bin/env bash
# ============================================================================
# resume.sh — continue an aborted FARBench experiment
#
# Usage:
#   bash scripts/dev/resume.sh --experiment <experiment_name> [--gpus 1,2,3,4] [--cuda cu118]
#
# Where <experiment_name> is the folder name under experiments/<task>/,
# e.g. gpt54_bigcodebench_codegen_20260423_200414_20260424_030426_352709
# (giving an absolute path is also accepted).
#
# What it does, in order:
#   1. Locate experiments/<task>/<experiment_name>/ by scanning experiments/.
#   2. Read config.json to find the task.
#   3. Force-remove any lingering sandbox/eval containers for the experiment's
#      agent_id — the old run may have crashed with containers still up.
#   4. Honour --cuda / --gpus by passing them through to
#      `python -m cli.main resume`.
#   5. Invoke `python -m cli.main resume --experiment-dir <abs path>`, which
#      archives the last iter_NNN (treated as aborted), rebuilds history
#      from disk, starts a fresh sandbox, and continues looping.
#
# The trailing iter (assumed half-finished) is moved to iter_NNN.aborted/
# by farbench/orchestrator.py so you can inspect what went wrong after the fact.
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."

EXPERIMENT=""
CUDA=""
GPUS=""

require_value() {
    if [[ $# -lt 2 || -z "${2:-}" || "${2:-}" == --* ]]; then
        echo "ERROR: $1 requires a value." >&2
        exit 2
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --experiment|--experiment-dir|-e)
            require_value "$@"
            EXPERIMENT="$2"; shift 2 ;;
        --cuda)
            require_value "$@"
            CUDA="$2"; shift 2 ;;
        --gpus|-g)
            require_value "$@"
            # Comma-separated GPU IDs, e.g. "1,2,3,4". Overrides FARBENCH_GPUS
            # from .env. Match run.sh's --gpus semantics so users have one
            # mental model for both fresh runs and resumes.
            GPUS="$2"; shift 2 ;;
        -h|--help)
            sed -n '1,32p' "$0"
            exit 0 ;;
        *)
            echo "Unknown arg: $1" >&2
            exit 2 ;;
    esac
done

if [[ -z "$EXPERIMENT" ]]; then
    echo "[resume] --experiment <name> is required" >&2
    exit 2
fi

# ── Load .env (HF_TOKEN, provider keys, FARBENCH_CUDA, FARBENCH_GPUS default) ──────
if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

# CLI --cuda overrides .env's FARBENCH_CUDA
if [[ -n "$CUDA" ]]; then
    export FARBENCH_CUDA="$CUDA"
fi

# CLI --gpus overrides .env's FARBENCH_GPUS. Loud echo so the user can verify
# before sandbox starts.
if [[ -n "$GPUS" ]]; then
    export FARBENCH_GPUS="$GPUS"
    echo "[resume] FARBENCH_GPUS overridden by --gpus: $FARBENCH_GPUS"
else
    echo "[resume] FARBENCH_GPUS = ${FARBENCH_GPUS:-<unset, framework will default to all>}" \
         "(no --gpus given; will use whatever .env/env says — DOUBLE-CHECK this" \
         "is what you want, especially if other resumes/runs are sharing GPUs)"
fi

# ── Resolve experiment directory ─────────────────────────────────────────
if [[ "$EXPERIMENT" == */* ]] || [[ -d "$EXPERIMENT" ]]; then
    # Looks like a path
    EXP_DIR="$(readlink -f "$EXPERIMENT")"
else
    # Just a name — glob experiments/*/<name>
    MATCHES=(experiments/*/"$EXPERIMENT")
    # Filter to actual directories
    REAL=()
    for m in "${MATCHES[@]}"; do
        [[ -d "$m" ]] && REAL+=("$m")
    done
    if [[ ${#REAL[@]} -eq 0 ]]; then
        echo "[resume] No experiment named $EXPERIMENT under experiments/*/" >&2
        exit 2
    fi
    if [[ ${#REAL[@]} -gt 1 ]]; then
        echo "[resume] Ambiguous experiment name, matched:" >&2
        printf '  %s\n' "${REAL[@]}" >&2
        echo "[resume] Pass the full relative path instead." >&2
        exit 2
    fi
    EXP_DIR="$(readlink -f "${REAL[0]}")"
fi

if [[ ! -f "$EXP_DIR/config.json" ]]; then
    echo "[resume] $EXP_DIR is missing config.json" >&2
    exit 2
fi

echo "[resume] experiment_dir = $EXP_DIR"
TASK_NAME="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("task_name",""))' "$EXP_DIR/config.json")"
echo "[resume] task           = ${TASK_NAME:-<missing>}"

# ── Derive agent_id from the dir name (strip _YYYYMMDD_HHMMSS_NNNNNN) ────
DIRNAME="$(basename "$EXP_DIR")"
AGENT_ID="$(python3 -c '
import re, sys
m = re.sub(r"_\d{8}_\d{6}_\d{6}$", "", sys.argv[1])
print(m)
' "$DIRNAME")"
echo "[resume] agent_id       = $AGENT_ID"

# ── Force-remove orphaned containers for this agent_id ───────────────────
AGENT_SAFE="${AGENT_ID//_/-}"
for suffix in sandbox eval; do
    NAME="farbench-${AGENT_SAFE}-${suffix}"
    if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
        echo "[resume] removing orphan container: $NAME"
        docker rm -f "$NAME" >/dev/null
    fi
done

# ── Reclaim ownership of experiment dir from container's root ────────────
# Sandbox/eval containers write /experiments as root (PID 1 in-container),
# so iter_NNN/ shows up root-owned on the host. `farbench resume` then tries to
# os.rename(iter_NNN -> iter_NNN.aborted), which requires WRITE permission
# on the *parent* directory — the host user doesn't have it, and we get
# PermissionError. Fix by chowning the tree back to the host user via a
# throwaway alpine container (which runs as root and so can chown anything).
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"
DIR_OWNER="$(stat -c '%u' "$EXP_DIR")"
if [[ "$DIR_OWNER" != "$HOST_UID" ]]; then
    echo "[resume] experiment dir is owned by uid=$DIR_OWNER (container root); " \
         "chowning to $HOST_UID:$HOST_GID so os.rename works"
    docker run --rm \
        -v "$EXP_DIR:/target" \
        alpine:3.19 \
        chown -R "$HOST_UID:$HOST_GID" /target
fi

# ── Hand off to the CLI ──────────────────────────────────────────────────
echo "[resume] launching farbench resume ..."
CLI_ARGS=(resume --experiment-dir "$EXP_DIR")
[[ -n "$GPUS" ]] && CLI_ARGS+=(--gpus "$GPUS")
[[ -n "$CUDA" ]] && CLI_ARGS+=(--cuda "$CUDA")
exec python3 -m cli.main "${CLI_ARGS[@]}"
