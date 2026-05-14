#!/usr/bin/env bash
# ============================================================================
# run.sh — single-task wrapper around `farbench run`
#
# Public behavior:
#   - single task only
#   - docker compose wrapper only
#
# Examples:
#   bash scripts/run.sh --task mnist_classification --preset claude --gpus 0
#   bash scripts/run.sh --task qm9 --preset gpt54 --gpus 2,3 --cuda cu128
#
# The wrapper loads the selected task image from Hugging Face and uses it as
# the harness image. Task images are already built from the FARBench base.
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

HF_IMAGE_DATASET_REPO="${FARBENCH_HF_IMAGE_DATASET_REPO:-FARBenchAnonymous/FARBench}"
HF_IMAGE_CACHE_DIR="${FARBENCH_HF_IMAGE_CACHE_DIR:-farbench-images}"
HF_IMAGE_REPOSITORY="${FARBENCH_HF_IMAGE_REPOSITORY:-farbench/farbench}"
TASK=""
GPUS=""
CUDA="${FARBENCH_CUDA:-}"
MODE_SET=0
PREPARE_SET=0
ARGS=()

usage() {
    cat <<'EOF'
Usage:
  bash scripts/run.sh --task TASK --gpus GPU_IDS [options]

Required:
  --task TASK                Benchmark task name
  --gpus GPU_IDS             Comma-separated host GPU IDs, e.g. 0 or 0,1

Common options:
  --preset NAME              LLM provider preset, e.g. claude, gemini, gpt54
  --cuda TAG                 CUDA variant: cu118 or cu128
  --prepare                  Forwarded to `farbench run --prepare`
                             Default: enabled
  --mode MODE                Forwarded to `farbench run` (default: api)

Additional options are forwarded to `farbench run` inside the runtime container.
EOF
}

require_value() {
    if [[ $# -lt 2 || -z "${2:-}" || "${2:-}" == --* ]]; then
        echo "ERROR: $1 requires a value." >&2
        exit 2
    fi
}

hf_download_cmd() {
    if command -v hf >/dev/null 2>&1; then
        printf '%s\n' "hf"
    elif command -v huggingface-cli >/dev/null 2>&1; then
        printf '%s\n' "huggingface-cli"
    else
        echo "ERROR: Hugging Face CLI not found. Install huggingface_hub." >&2
        exit 2
    fi
}

ensure_hf_image() {
    local image="$1"
    local archive_name="$2"
    local label="$3"
    local archive_path="${HF_IMAGE_CACHE_DIR}/${archive_name}"

    if docker image inspect "$image" >/dev/null 2>&1; then
        echo "[run] Using local ${label} image: $image"
        return 0
    fi

    mkdir -p "$HF_IMAGE_CACHE_DIR"
    if [[ ! -f "$archive_path" ]]; then
        local hf_cli
        hf_cli="$(hf_download_cmd)"
        echo "[run] Downloading ${label} image archive: ${HF_IMAGE_DATASET_REPO}/${archive_name}"
        if ! "$hf_cli" download "$HF_IMAGE_DATASET_REPO" "$archive_name" \
            --repo-type dataset \
            --local-dir "$HF_IMAGE_CACHE_DIR"; then
            echo "ERROR: failed to download ${archive_name} from ${HF_IMAGE_DATASET_REPO}." >&2
            echo "FARBench release runs require published image archives; local build is developer-only." >&2
            exit 2
        fi
    else
        echo "[run] Using cached ${label} image archive: $archive_path"
    fi

    echo "[run] Loading Docker image archive: $archive_path"
    local load_output
    if ! load_output="$(docker load -i "$archive_path" 2>&1)"; then
        echo "$load_output" >&2
        echo "ERROR: docker load failed for $archive_path" >&2
        exit 2
    fi

    if ! docker image inspect "$image" >/dev/null 2>&1; then
        local loaded_image
        loaded_image="$(printf '%s\n' "$load_output" | sed -n 's/^Loaded image: //p' | tail -n 1)"
        if [[ -n "$loaded_image" ]] && docker image inspect "$loaded_image" >/dev/null 2>&1; then
            docker tag "$loaded_image" "$image"
        fi
    fi

    if ! docker image inspect "$image" >/dev/null 2>&1; then
        echo "$load_output" >&2
        echo "ERROR: expected Docker image not available after load: $image" >&2
        exit 2
    fi
}

ensure_release_images() {
    local task="$1"
    local cuda="$2"
    local task_image="${HF_IMAGE_REPOSITORY}:${task}-${cuda}"

    ensure_hf_image "$task_image" "${task}-${cuda}.docker.tar.gz" "task"

    export FARBENCH_HARNESS_IMAGE="$task_image"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task)
            require_value "$@"
            TASK="$2"
            ARGS+=("$1" "$2")
            shift 2
            ;;
        --gpus)
            require_value "$@"
            GPUS="$2"
            ARGS+=("$1" "$2")
            shift 2
            ;;
        --preset)
            require_value "$@"
            ARGS+=("--agent-preset" "$2")
            shift 2
            ;;
        --cuda)
            require_value "$@"
            CUDA="$2"
            ARGS+=("$1" "$2")
            shift 2
            ;;
        --mode)
            require_value "$@"
            MODE_SET=1
            ARGS+=("$1" "$2")
            shift 2
            ;;
        --prepare)
            PREPARE_SET=1
            ARGS+=("$1")
            shift
            ;;
        --config|--tasks|--stop|--restart|--restart-failed|--add-task|--max-concurrent|--max-retries|--agent-pipeline)
            echo "ERROR: $1 is not supported by the release wrapper. Use one task per command." >&2
            exit 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ -z "$TASK" ]]; then
    echo "ERROR: --task is required." >&2
    usage >&2
    exit 2
fi

if [[ -z "$GPUS" ]]; then
    echo "ERROR: --gpus is required." >&2
    usage >&2
    exit 2
fi

if [[ $MODE_SET -eq 0 ]]; then
    ARGS+=("--mode" "api")
fi

if [[ $PREPARE_SET -eq 0 ]]; then
    ARGS+=("--prepare")
fi

export FARBENCH_GPUS="$GPUS"
export FARBENCH_CUDA="${CUDA:-cu118}"
case "$FARBENCH_CUDA" in
    cu118|cu128) ;;
    *)
        echo "ERROR: unsupported CUDA variant '$FARBENCH_CUDA'. Expected cu118 or cu128." >&2
        exit 2
        ;;
esac

if [[ " ${ARGS[*]} " != *" --cuda "* ]]; then
    ARGS+=("--cuda" "$FARBENCH_CUDA")
fi

ensure_release_images "$TASK" "$FARBENCH_CUDA"

exec docker compose run --rm \
    -e "FARBENCH_GPUS=$FARBENCH_GPUS" \
    -e "NVIDIA_VISIBLE_DEVICES=$FARBENCH_GPUS" \
    -e "FARBENCH_CUDA=$FARBENCH_CUDA" \
    -e "FARBENCH_HARNESS_IMAGE=$FARBENCH_HARNESS_IMAGE" \
    farbench run "${ARGS[@]}"
