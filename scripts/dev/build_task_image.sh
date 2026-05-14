#!/usr/bin/env bash
# ============================================================================
# build_task_image.sh — Developer-only FARBench task image builder
#
# Release users should not run this script. Public runs load pre-built Docker
# archives from the FARBench Hugging Face dataset via scripts/run.sh.
#
# What this script does for each task:
#   1. Build the matching FARBench base image if it is missing.
#   2. Build the task dependency image from benchmarks/<task>/docker/Dockerfile.
#   3. Run benchmarks/<task>/script/prepare.py into host staging directories.
#   4. Bake staged train/test data into /rab_data/<task>{,_test}/ in the image.
#   5. Optionally export farbench-images/<task>-<cuda>.docker.tar.gz.
#
# Examples:
#   bash scripts/dev/build_task_image.sh mnist_classification --cuda cu118
#   bash scripts/dev/build_task_image.sh mnist_classification --cuda cu128 --archive
#   bash scripts/dev/build_task_image.sh --all --cuda cu118 --archive
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."

CUDA=""
ARCHIVE=false
FORCE_PREPARE=false
TASKS=()
DATA_ROOT="${FARBENCH_DEV_DATA_DIR:-farbench_data}"
ARCHIVE_DIR="${FARBENCH_HF_IMAGE_CACHE_DIR:-farbench-images}"
IMAGE_REPOSITORY="${FARBENCH_HF_IMAGE_REPOSITORY:-farbench/farbench}"

if [[ -z "${HF_TOKEN:-}" && -f .env ]]; then
    HF_TOKEN="$(grep '^HF_TOKEN=' .env 2>/dev/null | head -1 | sed 's/^HF_TOKEN=//' | tr -d '"' | tr -d "'" || true)"
fi
if [[ -z "${HF_TOKEN:-}" && -f .env ]]; then
    HF_TOKEN="$(grep '^HUGGING_FACE_HUB_TOKEN=' .env 2>/dev/null | head -1 | sed 's/^HUGGING_FACE_HUB_TOKEN=//' | tr -d '"' | tr -d "'" || true)"
fi

usage() {
    cat <<'EOF'
Usage:
  bash scripts/dev/build_task_image.sh TASK [TASK ...] --cuda cu118|cu128 [options]
  bash scripts/dev/build_task_image.sh --all --cuda cu118|cu128 [options]

Options:
  --archive          Export farbench-images/<task>-<cuda>.docker.tar.gz
  --force-prepare    Delete staged data and rerun prepare.py
  --cuda TAG         CUDA variant: cu118 or cu128

Environment:
  FARBENCH_DEV_DATA_DIR       Repo-relative staging root. Default: farbench_data
  FARBENCH_HF_IMAGE_CACHE_DIR Archive output dir. Default: farbench-images
  FARBENCH_HF_IMAGE_REPOSITORY Image repository/tag prefix. Default: farbench/farbench
  HF_TOKEN                    Optional token for private dataset/model downloads
EOF
}

all_tasks() {
    find benchmarks -mindepth 2 -maxdepth 2 -name task.yaml -printf '%h\n' \
        | sed 's#^benchmarks/##' \
        | sort
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --all)
            mapfile -t TASKS < <(all_tasks)
            shift
            ;;
        --archive)
            ARCHIVE=true
            shift
            ;;
        --force-prepare)
            FORCE_PREPARE=true
            shift
            ;;
        --cuda)
            if [[ $# -lt 2 || -z "${2:-}" || "${2:-}" == --* ]]; then
                echo "ERROR: --cuda requires cu118 or cu128." >&2
                exit 2
            fi
            CUDA="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --*)
            echo "ERROR: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
        *)
            TASKS+=("$1")
            shift
            ;;
    esac
done

case "$CUDA" in
    cu118) BASE_NVIDIA_IMAGE="nvidia/cuda:11.8.0-runtime-ubuntu22.04" ;;
    cu128) BASE_NVIDIA_IMAGE="nvidia/cuda:12.8.1-runtime-ubuntu22.04" ;;
    "") echo "ERROR: --cuda is required." >&2; usage >&2; exit 2 ;;
    *) echo "ERROR: unsupported CUDA variant '$CUDA'. Expected cu118 or cu128." >&2; exit 2 ;;
esac

if [[ ${#TASKS[@]} -eq 0 ]]; then
    echo "ERROR: at least one task or --all is required." >&2
    usage >&2
    exit 2
fi

if [[ "$DATA_ROOT" = /* || "$DATA_ROOT" == *".."* || "$DATA_ROOT" == *" "* ]]; then
    echo "ERROR: FARBENCH_DEV_DATA_DIR must be a repo-relative path without spaces or '..'." >&2
    exit 2
fi

echo "[dev-build] Strict task.yaml validation"
python - <<'PY'
from farbench.tasks import TaskRegistry

registry = TaskRegistry("benchmarks")
registry.discover()
print(f"validated {len(registry.list_all())} tasks")
PY

build_base() {
    local base_tag="$1"
    local dockerfile="$2"
    local image="${IMAGE_REPOSITORY}:base-${base_tag}"

    if docker image inspect "$image" >/dev/null 2>&1; then
        echo "[dev-build] Using local base image: $image"
        return
    fi

    echo "[dev-build] Building base image: $image"
    docker build --network=host \
        -t "$image" \
        -f "$dockerfile" \
        --build-arg "BASE_IMAGE=$BASE_NVIDIA_IMAGE" \
        --build-arg "CUDA_TAG=$CUDA" \
        .
}

task_base_tag() {
    local task="$1"
    case "$task" in
        vlabench_manipulation) printf 'py312-%s\n' "$CUDA" ;;
        *) printf '%s\n' "$CUDA" ;;
    esac
}

task_base_dockerfile() {
    local task="$1"
    case "$task" in
        vlabench_manipulation) printf 'docker/Dockerfile.base.py312\n' ;;
        *) printf 'docker/Dockerfile.base\n' ;;
    esac
}

data_ready() {
    local train_dir="$1"
    local test_dir="$2"
    [[ -d "$train_dir" && -d "$test_dir" ]] || return 1
    find "$train_dir" -type f -print -quit 2>/dev/null | grep -q . || return 1
    find "$test_dir" -type f -print -quit 2>/dev/null | grep -q . || return 1
}

verify_image_data() {
    local image="$1"
    local task="$2"
    local train_path="/rab_data/${task}"
    local test_path="/rab_data/${task}_test"

    docker run --rm --user 65534:65534 "$image" python -c "
import os
train = '$train_path'
test = '$test_path'
if not os.path.isdir(train) or not os.path.isdir(test):
    raise SystemExit('missing /rab_data train/test directories')
if not any(files for _root, _dirs, files in os.walk(train)):
    raise SystemExit('empty train data directory')
if not any(files for _root, _dirs, files in os.walk(test)):
    raise SystemExit('empty test data directory')
if os.access(test, os.W_OK):
    raise SystemExit('test data is writable by eval user 65534:65534')
"
}

SUCCEEDED=()
FAILED=()

for task in "${TASKS[@]}"; do
    echo
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[dev-build] Task: $task ($CUDA)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    task_dir="benchmarks/${task}"
    dockerfile="${task_dir}/docker/Dockerfile"
    prepare_py="${task_dir}/script/prepare.py"
    final_image="${IMAGE_REPOSITORY}:${task}-${CUDA}"
    deps_image="${IMAGE_REPOSITORY}:${task}-${CUDA}-deps"
    train_dir="${DATA_ROOT}/${task}"
    test_dir="${DATA_ROOT}/${task}_test"

    if [[ ! -f "${task_dir}/task.yaml" ]]; then
        echo "[dev-build] ERROR: missing ${task_dir}/task.yaml" >&2
        FAILED+=("$task")
        continue
    fi
    if [[ ! -f "$dockerfile" ]]; then
        echo "[dev-build] ERROR: missing $dockerfile" >&2
        FAILED+=("$task")
        continue
    fi
    if [[ ! -f "$prepare_py" ]]; then
        echo "[dev-build] ERROR: missing $prepare_py; release images must contain baked /rab_data." >&2
        FAILED+=("$task")
        continue
    fi

    base_tag="$(task_base_tag "$task")"
    base_dockerfile="$(task_base_dockerfile "$task")"

    if ! build_base "$base_tag" "$base_dockerfile"; then
        FAILED+=("$task")
        continue
    fi

    echo "[dev-build] Building task dependency image: $deps_image"
    if ! docker build --network=host \
        -t "$deps_image" \
        -f "$dockerfile" \
        --build-arg "BASE_TAG=$base_tag" \
        --build-arg "PYTORCH_CUDA=$CUDA" \
        ${HF_TOKEN:+--build-arg "HF_TOKEN=$HF_TOKEN"} \
        "$task_dir"; then
        FAILED+=("$task")
        continue
    fi

    if $FORCE_PREPARE; then
        echo "[dev-build] Removing staged data for $task"
        rm -rf "$train_dir" "$test_dir"
    fi

    if data_ready "$train_dir" "$test_dir"; then
        echo "[dev-build] Using staged data: $train_dir and $test_dir"
    else
        echo "[dev-build] Running prepare.py into staged data"
        mkdir -p "$train_dir" "$test_dir"
        abs_task_dir="$(cd "$task_dir" && pwd)"
        abs_train_dir="$(cd "$train_dir" && pwd)"
        abs_test_dir="$(cd "$test_dir" && pwd)"

        if ! docker run --rm --network=host \
            -v "${abs_task_dir}:${abs_task_dir}:ro" \
            -v "${abs_train_dir}:${abs_train_dir}:rw" \
            -v "${abs_test_dir}:${abs_test_dir}:rw" \
            -e "FARBENCH_DATA_DIR=${abs_train_dir}" \
            -e "FARBENCH_TEST_DATA_DIR=${abs_test_dir}" \
            ${HF_TOKEN:+-e "HF_TOKEN=$HF_TOKEN"} \
            ${HF_TOKEN:+-e "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN"} \
            -e "HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}" \
            "$deps_image" \
            python "${abs_task_dir}/script/prepare.py"; then
            FAILED+=("$task")
            continue
        fi
    fi

    if ! data_ready "$train_dir" "$test_dir"; then
        echo "[dev-build] ERROR: prepare.py did not produce non-empty train/test data." >&2
        FAILED+=("$task")
        continue
    fi

    echo "[dev-build] Baking staged data into image: $final_image"
    tmpfile="$(mktemp /tmp/farbench-image.XXXXXX.Dockerfile)"
    cat > "$tmpfile" <<EOF
FROM ${deps_image}
COPY ${DATA_ROOT}/${task}/ /rab_data/${task}/
COPY ${DATA_ROOT}/${task}_test/ /rab_data/${task}_test/
RUN chmod -R a+rX /rab_data/${task} /rab_data/${task}_test \\
    && chmod -R a-w /rab_data/${task}_test
EOF

    if ! docker build -t "$final_image" -f "$tmpfile" .; then
        rm -f "$tmpfile"
        FAILED+=("$task")
        continue
    fi
    rm -f "$tmpfile"

    echo "[dev-build] Verifying baked /rab_data policy"
    if ! verify_image_data "$final_image" "$task"; then
        FAILED+=("$task")
        continue
    fi

    if $ARCHIVE; then
        mkdir -p "$ARCHIVE_DIR"
        archive_path="${ARCHIVE_DIR}/${task}-${CUDA}.docker.tar.gz"
        echo "[dev-build] Exporting archive: $archive_path"
        docker save "$final_image" | gzip -c > "$archive_path"
    fi

    SUCCEEDED+=("$task")
done

echo
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[dev-build] Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [[ ${#SUCCEEDED[@]} -gt 0 ]]; then
    echo "Succeeded: ${SUCCEEDED[*]}"
fi
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "Failed: ${FAILED[*]}" >&2
    exit 1
fi
