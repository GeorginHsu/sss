#!/usr/bin/env bash
# ============================================================================
# clean.sh — Clean up Docker containers, markers, and optionally all data
#
# Usage:
#   bash scripts/dev/clean.sh          # clean containers + .prepared markers
#   bash scripts/dev/clean.sh --all    # also remove experiments/ and Docker images
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."

ALL=false
[[ "${1:-}" == "--all" ]] && ALL=true

# 1. Remove FARBench containers
echo "==> Cleaning FARBench containers..."
CONTAINERS=$(docker ps -a --filter "name=farbench-" --format "{{.Names}}" 2>/dev/null || true)
if [[ -n "$CONTAINERS" ]]; then
    for c in $CONTAINERS; do
        echo "  Removing $c"
        docker rm -f "$c" 2>/dev/null || true
    done
else
    echo "  No containers found"
fi

# 2. Remove .prepared markers
echo "==> Removing .prepared markers..."
find benchmarks/ -name ".prepared" -delete 2>/dev/null || true

if $ALL; then
    # 3. Remove experiments (use Docker to handle root-owned files)
    echo "==> Removing experiments/..."
    # Use whichever farbench/farbench:base-* variant is present to chown-free-delete
    BASE_IMG=""
    for tag in "${FARBENCH_CUDA:-cu118}" cu118 cu128; do
        if docker image inspect "farbench/farbench:base-${tag}" &>/dev/null; then
            BASE_IMG="farbench/farbench:base-${tag}"
            break
        fi
    done
    if [[ -d experiments/ ]] && [[ -n "$BASE_IMG" ]]; then
        docker run --rm -v "$(pwd)/experiments:/experiments" "$BASE_IMG" \
            rm -rf /experiments/*
    else
        rm -rf experiments/ 2>/dev/null || true
    fi

    # 4. Remove Docker images (local + hub)
    echo "==> Removing FARBench Docker images..."
    IMAGES=$(docker images --format "{{.Repository}}:{{.Tag}}" 2>/dev/null \
        | grep -E "^farbench/" || true)
    if [[ -n "$IMAGES" ]]; then
        for img in $IMAGES; do
            echo "  Removing $img"
            docker rmi "$img" 2>/dev/null || true
        done
    fi
fi

echo "==> Done"
