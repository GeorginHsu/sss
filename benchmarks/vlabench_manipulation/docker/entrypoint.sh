#!/bin/bash
# Bridge the VLABench package's hard-coded asset path to wherever the heavy
# 12 GB obj/ + scenes/ subtrees actually live in this container.
#
# VLABench's Python code computes
#     xml_root = os.path.join(os.getenv("VLABENCH_ROOT"), "assets")
# and reads things like
#     $xml_root/robots/franka_emika_panda/panda.xml      (small, IN IMAGE)
#     $xml_root/base/...                                  (small, IN IMAGE)
#     $xml_root/obj/meshes/...                            (12 GB, see SOURCES below)
#     $xml_root/scenes/default/...                        (12 GB, see SOURCES below)
#
# FARBench launches TWO containers per iteration with DIFFERENT bind mounts:
#   • sandbox  : -v FARBENCH_DATA_DIR      → /data   ⇒ /data/vlabench_root/assets/{obj,scenes}
#   • eval     : -v FARBENCH_TEST_DATA_DIR → /data   ⇒ /data has NO vlabench_root/
#
# Originally we only handled the sandbox case, which broke iteration 1 evaluation
# the moment the agent's predict.py did `from VLABench.envs import load_env`.
# Both containers can ALWAYS see the image-baked /rab_data/... copy (12 GB COPY
# at image-build time), so we use that as the
# fallback. Same content as /data/vlabench_root/assets/ — they're literally the
# bytes packaged in the image under /rab_data.
set -e

# ── Make container hostname resolve to 127.0.0.1 ────────────────────────────
# FARBench task containers run with `network_access: false` → docker
# `--network=none`, which strips DNS and does NOT add an entry for the
# container's auto-generated hostname (e.g. `a5d94fbf3cac`) to /etc/hosts.
# As a result torch.distributed launchers (torchrun, accelerate, mp.spawn)
# emit
#   [W socket.cpp:755] [c10d] The IPv6 network addresses of (<container-id>,
#   <port>) cannot be retrieved (gai error: -3 - Temporary failure in name
#   resolution).
# and exponential-backoff retry for 1–2 min before falling back. Adding a
# 127.0.0.1 → hostname row makes `socket.gethostbyname(socket.gethostname())`
# resolve immediately, silencing the warnings across all c10d socket layers
# (rdzv store, ProcessGroup TCPStore, NCCL bootstrap). Idempotent: re-running
# is a no-op (grep guard) and each container starts with a fresh /etc/hosts
# anyway. Failure (e.g. /etc/hosts read-only on a future docker version) is
# non-fatal — we proceed without the mapping.
HOSTNAME_SHORT="$(hostname)"
if ! grep -qE "[[:space:]]${HOSTNAME_SHORT}([[:space:]]|$)" /etc/hosts 2>/dev/null; then
    {
        echo "127.0.0.1 ${HOSTNAME_SHORT}"
        echo "::1 ${HOSTNAME_SHORT}"
    } >> /etc/hosts 2>/dev/null \
        && echo "[entrypoint] mapped ${HOSTNAME_SHORT} -> 127.0.0.1, ::1 in /etc/hosts" \
        || echo "[entrypoint] WARN: could not write /etc/hosts (c10d may emit hostname-resolution warnings)" >&2
fi

ASSETS_DIR="/opt/VLABench/VLABench/assets"
mkdir -p "$ASSETS_DIR"

# Probed in priority order. First existing dir wins.
SOURCES=(
    "/data/vlabench_root/assets"                              # sandbox
    "/rab_data/vlabench_manipulation/vlabench_root/assets"    # eval (image-baked, also valid in sandbox)
)

for sub in obj scenes; do
    DST="$ASSETS_DIR/$sub"
    SRC=""
    for cand in "${SOURCES[@]}"; do
        if [ -d "$cand/$sub" ]; then
            SRC="$cand/$sub"
            break
        fi
    done

    if [ -z "$SRC" ]; then
        echo "[entrypoint] WARN: $sub not found in any of: ${SOURCES[*]/%//${sub}}" >&2
        continue
    fi

    if [ -e "$DST" ] && [ ! -L "$DST" ]; then
        # Image left a real dir behind (shouldn't happen with the new
        # Dockerfile, but stay defensive across rebuilds).
        rm -rf "$DST"
    fi
    ln -sfn "$SRC" "$DST"
    echo "[entrypoint] linked $DST -> $SRC"
done

exec "$@"
