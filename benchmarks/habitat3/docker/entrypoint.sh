#!/bin/bash
# Bridge the image-baked H3 dataset tree at /rab_data/habitat3 into the paths
# habitat-lab expects (/opt/habitat-lab-src/data) and the agent's writable
# working tree (/workspace/data).
#
# FARBench launches TWO containers per iteration with DIFFERENT bind mounts:
#   • sandbox  : -v FARBENCH_DATA_DIR      → /data
#   • eval     : -v FARBENCH_TEST_DATA_DIR → /data
# Neither of those mounts contains the heavy 25 GB scene/episode/robot/object
# tree — that lives at /rab_data/habitat3 (baked by the Dockerfile via
# `habitat_sim.utils.datasets_download`). All the asset symlinks created by
# the downloader are *absolute*, so as long as habitat-lab can find a path
# starting at "data/scene_datasets/hssd-hab/...", it follows the symlink to
# /rab_data/habitat3/versioned_data/... and reads the real bytes.
#
# So the only job of this entrypoint is to make `data/...` resolve to
# `/rab_data/habitat3/...` from cwd=`/opt/habitat-lab-src` AND from cwd=
# `/workspace`. We do that with two symlinks. We also copy in a tiny
# physics config that habitat-sim wants in `data/`.

set -e

DATA_SRC="/rab_data/habitat3"

# ── Hard-fail if the image is broken ─────────────────────────────────────────
# Better to die in entrypoint than to let the agent waste an iteration
# debugging a missing-asset error from deep inside habitat-sim.
if [ ! -d "$DATA_SRC" ]; then
    echo "[entrypoint] FATAL: $DATA_SRC missing — image is broken." >&2
    exit 1
fi
for required in \
    "scene_datasets/hssd-hab/hssd-hab.scene_dataset_config.json" \
    "hab3_bench_assets/episode_datasets/medium_medium.json.gz" \
    "humanoids/humanoid_data" \
    "objects/ycb" \
    "robots/hab_spot_arm" \
    "datasets/hssd/rearrange/train/social_rearrange.json.gz" \
    "datasets/hssd/rearrange/val/social_rearrange.json.gz" ; do
    if [ ! -e "$DATA_SRC/$required" ]; then
        echo "[entrypoint] FATAL: required asset missing: $DATA_SRC/$required" >&2
        echo "[entrypoint] Rebuild the image: docker build --no-cache -t farbench/farbench:habitat3-cu118 ..." >&2
        exit 1
    fi
done

# ── Bridge habitat-lab's expected `data/` directory ──────────────────────────
# habitat-lab resolves data paths relative to cwd. Two cwds matter:
#   • /opt/habitat-lab-src   — the "official" cwd, used by all sample scripts
#   • /workspace             — agent's working dir (FARBench bind-mount)
#
# We make `data/` at both locations point to /rab_data/habitat3.
ln -sfn "$DATA_SRC" /opt/habitat-lab-src/data
mkdir -p /workspace
# /workspace can be a read-only mount in eval container — `ln -sfn` is a
# no-op if the symlink already exists from a previous run, and silently
# fails (|| true) if /workspace is read-only.
ln -sfn "$DATA_SRC" /workspace/data 2>/dev/null || true

# ── Drop the default physics config habitat-sim looks for ────────────────────
# (It's tiny — just timestep, gravity, friction. Habitat will autogenerate
# a sane default if missing, but having it here matches the shipped config.)
PHYS_CFG="$DATA_SRC/default.physics_config.json"
if [ ! -f "$PHYS_CFG" ]; then
    # Try to write it; if /rab_data is read-only at runtime (it shouldn't be —
    # it's part of the image's own filesystem) this is best-effort.
    echo '{"physics_simulator":"bullet","timestep":0.008,"gravity":[0,-9.8,0],"friction_coefficient":0.4,"restitution_coefficient":0.1,"rigid object paths":["objects"]}' \
        > "$PHYS_CFG" 2>/dev/null || true
fi

echo "[entrypoint] Habitat data bridged: /opt/habitat-lab-src/data -> $DATA_SRC"
echo "[entrypoint] Episode dataset:      $DATA_SRC/datasets/hssd/rearrange/{train,val}/social_rearrange.json.gz"
echo "[entrypoint] Scene dataset:        $DATA_SRC/scene_datasets/hssd-hab/"

exec "$@"
