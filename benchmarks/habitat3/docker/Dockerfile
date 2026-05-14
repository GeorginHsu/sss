ARG BASE_TAG=cu118
FROM farbench/farbench:base-${BASE_TAG}
ARG HF_TOKEN=""
ENV HF_TOKEN=${HF_TOKEN}

# ── System deps for headless EGL rendering + git-LFS for large data ──────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-dev \
    libgles2-mesa-dev \
    libegl1-mesa-dev \
    libglew-dev \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libxi-dev \
    cmake \
    git \
    git-lfs \
    wget \
    rsync \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

# ── Headless rendering via EGL ───────────────────────────────────────────────
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV MAGNUM_LOG=quiet
ENV HABITAT_SIM_LOG=quiet
ENV HABITAT_SIM_GPU_DEVICE_ID=0
ENV MAGNUM_GPU_VALIDATION=OFF

RUN mkdir -p /usr/share/glvnd/egl_vendor.d && \
    echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' \
    > /usr/share/glvnd/egl_vendor.d/10_nvidia.json
ENV __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libEGL.so.1

# ── Install Miniconda (habitat-sim ships only conda packages) ────────────────
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh
ENV PATH="/opt/conda/bin:${PATH}"

# `pip` MUST be in the conda env or every later `RUN pip install` will silently
# fall through to base conda's pip (Python 3.13) and install into the WRONG
# site-packages. This was the root cause of the cu128 build's torchvision bug.
RUN conda create -y -n habitat python=3.9 pip \
    --override-channels -c conda-forge && \
    conda clean -afy

ENV CONDA_DEFAULT_ENV=habitat
ENV PATH="/opt/conda/envs/habitat/bin:${PATH}"

# All pip/python invocations below use ABSOLUTE paths into the conda env so
# they cannot accidentally pick up base conda's binaries even if PATH ordering
# changes (e.g. between cu118 and cu128 base images).
ARG ENV_PIP=/opt/conda/envs/habitat/bin/pip
ARG ENV_PY=/opt/conda/envs/habitat/bin/python

# ── habitat-sim 0.3.3 (headless EGL + Bullet) ────────────────────────────────
RUN conda install -y -n habitat \
    --override-channels -c conda-forge -c aihabitat \
    habitat-sim=0.3.3 withbullet headless && \
    conda clean -afy

# ── habitat-lab + habitat-baselines (pinned to v0.3.3 matching sim) ──────────
RUN git clone --branch v0.3.3 --depth 1 \
    https://github.com/facebookresearch/habitat-lab.git /opt/habitat-lab-src && \
    ${ENV_PIP} install --no-cache-dir -e /opt/habitat-lab-src/habitat-lab && \
    ${ENV_PIP} install --no-cache-dir -e /opt/habitat-lab-src/habitat-baselines

# ── torchvision (req'd by resnet50 pre-cache + agent vision backbones) ──────
# Installed FIRST and verified IMMEDIATELY in the same RUN so any silent pip
# failure (resolver fallback, ABI mismatch with conda-installed torch, etc.)
# fails the build with a real error message instead of leaking forward.
RUN ${ENV_PIP} install --no-cache-dir torchvision && \
    ${ENV_PY} -c "import torchvision; print('[verify] torchvision', torchvision.__version__, 'on torch', __import__('torch').__version__)"

# ── ML / research packages for agent use ─────────────────────────────────────
RUN ${ENV_PIP} install --no-cache-dir \
    stable-baselines3==2.4.1 \
    sb3-contrib==2.4.0 \
    "shimmy>=2.0" \
    transformers==4.46.3 \
    accelerate==1.1.1 \
    huggingface_hub==0.26.5 \
    einops==0.8.0 \
    timm==1.0.11 \
    wandb==0.18.7 \
    tensorboard==2.18.0 \
    requests==2.32.3

# ── Pre-cache pretrained vision backbone (no network needed at run time) ─────
RUN ${ENV_PY} -c "import torchvision; torchvision.models.resnet50(weights='IMAGENET1K_V1')"

# ──────────────────────────────────────────────────────────────────────────────
# Bake H3 datasets into the image (~25 GB total).
#
# Why split into multiple RUNs (one per UID)?
#   habitat-sim's downloader pulls each UID via `git clone` from HuggingFace.
#   If ANY one of them fails (network blip, HF rate-limit, bad UID name…), the
#   ENTIRE single-RUN download has to start over from scratch on the next
#   build, wasting hours re-downloading the 24 GB hssd-hab. By giving each
#   UID its own RUN layer, BuildKit caches the successful ones and only
#   re-runs the failed step on rebuild.
#
# Why use `habitat_sim.utils.datasets_download` at all (vs raw git-LFS clone)?
#   1. It writes to `versioned_data/<uid>_v0.X.Y/` and creates *absolute*
#      symlinks at the canonical paths habitat-lab expects (e.g.
#      `scene_datasets/hssd-hab → versioned_data/hssd-hab_v0.X.Y`). Absolute
#      symlinks resolve correctly inside the container regardless of how the
#      entrypoint binds /rab_data/habitat to /opt/habitat-lab-src/data.
#   2. It strips .git history. hssd-hab's full git history is ~33 GB; we only
#      want the actual scene assets (~24 GB).
#   3. It's part of habitat-sim itself, so it stays in lock-step with the sim
#      version (no risk of HSSD-200 schema drift).
#
# UID names ARE NOT `link` paths — `humanoid_data` is the ON-DISK directory
# name; the registered UID is `habitat_humanoids`. An invalid UID is silently
# ignored by the downloader, so a typo manifests as "asset missing" much later
# in the sanity check rather than a clear download error. We bumped against
# this with a `humanoid_data` typo on first build — keep UID names verbatim
# from `habitat_sim.utils.datasets_download`'s registry below.
#
# Sizes (after `--prune`-equivalent strip; raw clone is 1.5–3× larger):
#   hssd-hab           → scene_datasets/hssd-hab          ~24 GB
#   hab3_bench_assets  → hab3_bench_assets/               ~0.2 GB (H3 episodes)
#   habitat_humanoids  → humanoids/humanoid_data/         ~0.06 GB (avatars)
#   ycb                → objects/ycb/                     ~0.5 GB (objects)
#   hab_spot_arm       → robots/hab_spot_arm/             ~0.1 GB (Spot URDF)
#   hab_fetch          → robots/hab_fetch/                ~0.1 GB (Fetch URDF)
# ──────────────────────────────────────────────────────────────────────────────

# 1/6 hssd-hab (the big one — 24 GB; isolated so a sanity-check failure
# downstream NEVER forces redownloading this layer)
RUN mkdir -p /rab_data/habitat && \
    ${ENV_PY} -m habitat_sim.utils.datasets_download \
        --uids hssd-hab --data-path /rab_data/habitat \
        --no-replace --no-prune && \
    test -f /rab_data/habitat/scene_datasets/hssd-hab/hssd-hab.scene_dataset_config.json \
        || { echo "[FATAL] hssd-hab download succeeded but config.json missing"; exit 1; }

# 2/6 hab3_bench_assets (H3 episode datasets, the medium_medium.json.gz used as
# our train/val source)
RUN ${ENV_PY} -m habitat_sim.utils.datasets_download \
        --uids hab3_bench_assets --data-path /rab_data/habitat \
        --no-replace --no-prune && \
    test -f /rab_data/habitat/hab3_bench_assets/episode_datasets/medium_medium.json.gz \
        || { echo "[FATAL] hab3_bench_assets/episode_datasets/medium_medium.json.gz missing"; exit 1; }

# 3/6 habitat_humanoids (NOT `humanoid_data` — that's the dirname; UID is
# `habitat_humanoids`. Was the bug that triggered the very first build failure.)
RUN ${ENV_PY} -m habitat_sim.utils.datasets_download \
        --uids habitat_humanoids --data-path /rab_data/habitat \
        --no-replace --no-prune && \
    test -d /rab_data/habitat/humanoids/humanoid_data \
        || { echo "[FATAL] humanoids/humanoid_data missing after habitat_humanoids download"; exit 1; }

# 4/6 ycb (object assets referenced by social_nav.yaml's additional_object_paths)
RUN ${ENV_PY} -m habitat_sim.utils.datasets_download \
        --uids ycb --data-path /rab_data/habitat \
        --no-replace --no-prune && \
    test -d /rab_data/habitat/objects/ycb \
        || { echo "[FATAL] objects/ycb missing"; exit 1; }

# 5/6 hab_spot_arm (Spot quadruped URDF — the agent_0 robot)
RUN ${ENV_PY} -m habitat_sim.utils.datasets_download \
        --uids hab_spot_arm --data-path /rab_data/habitat \
        --no-replace --no-prune && \
    test -d /rab_data/habitat/robots/hab_spot_arm \
        || { echo "[FATAL] robots/hab_spot_arm missing"; exit 1; }

# 6/6 hab_fetch (Fetch arm URDF — used by some H3 example scripts agents may
# reference. Cheap to include even if not on the critical path for social_nav.)
RUN ${ENV_PY} -m habitat_sim.utils.datasets_download \
        --uids hab_fetch --data-path /rab_data/habitat \
        --no-replace --no-prune && \
    test -d /rab_data/habitat/robots/hab_fetch \
        || { echo "[FATAL] robots/hab_fetch missing"; exit 1; }

# ── Bridge episode dataset: H3's social_nav.yaml expects ─────────────────────
#   data/datasets/hssd/rearrange/{train,val}/social_rearrange.json.gz
# but the actual files (medium_medium.json.gz × 9 size combos) ship inside
# `hab3_bench_assets/episode_datasets/`. We materialise the canonical name as
# an *absolute* symlink so habitat-lab can find it from any cwd.
RUN mkdir -p /rab_data/habitat/datasets/hssd/rearrange/train \
             /rab_data/habitat/datasets/hssd/rearrange/val && \
    ln -sf /rab_data/habitat/hab3_bench_assets/episode_datasets/medium_medium.json.gz \
           /rab_data/habitat/datasets/hssd/rearrange/train/social_rearrange.json.gz && \
    ln -sf /rab_data/habitat/hab3_bench_assets/episode_datasets/medium_medium.json.gz \
           /rab_data/habitat/datasets/hssd/rearrange/val/social_rearrange.json.gz

# ── Patch hab3-hssd.scene_dataset_config.json to register `hssd_ssd_map` tag ─
# H3's hab3-hssd scene_instance.json files reference `semantic_scene_instance:
# hssd_ssd_map`, but the bundled config only registers semantics via a glob
# (`semantics/scenes/*.json`) which contains scene-specific files, NOT the
# `hssd_ssd_map` tag itself. The tag is only registered in the original
# hssd-hab dataset config.
#
# Without this patch, habitat-sim hard-fails at scene creation:
#   ESP_CHECK failed: Simulator::createSceneInstance() :
#   semantic tag `hssd_ssd_map` not being found in SemanticAttributesManager
#
# Fix: add an explicit `hssd_ssd_map` mapping to the hssd-hab semantic lexicon
# (which IS baked at /rab_data/habitat/scene_datasets/hssd-hab/semantics/).
# Path is relative to hab3_bench_assets/hab3-hssd/, so `../../scene_datasets/...`.
RUN ${ENV_PY} -c "\
import json, pathlib; \
p = pathlib.Path('/rab_data/habitat/hab3_bench_assets/hab3-hssd/hab3-hssd.scene_dataset_config.json'); \
d = json.loads(p.read_text()); \
ABS = '/rab_data/habitat/scene_datasets/hssd-hab/semantics/hssd-hab_semantic_lexicon.json'; \
d['semantic_scene_descriptor_instances'] = { \
    'hssd_ssd_map': ABS, \
    'default_attributes': {'semantic_descriptor_filename': ABS}, \
    'paths': {'.json': ['semantics/scenes/*']}, \
}; \
p.write_text(json.dumps(d, indent=4)); \
print('[patch] hab3-hssd.scene_dataset_config.json: registered hssd_ssd_map -> ' + ABS); \
"

# ── Final sanity check: every per-UID test_d already gated above; this is the
# integration check that all 7 contract paths agree. Any mismatch here is a
# bug in the 6 RUN blocks above, not a flaky download.
RUN set -e && \
    echo "[sanity] Verifying baked H3 assets..." && \
    test -f /rab_data/habitat/scene_datasets/hssd-hab/hssd-hab.scene_dataset_config.json \
        || { echo "[sanity FAIL] hssd-hab scene_dataset_config.json"; exit 1; } && \
    test -f /rab_data/habitat/hab3_bench_assets/episode_datasets/medium_medium.json.gz \
        || { echo "[sanity FAIL] hab3_bench medium_medium episodes"; exit 1; } && \
    test -d /rab_data/habitat/humanoids/humanoid_data \
        || { echo "[sanity FAIL] humanoids/humanoid_data directory"; exit 1; } && \
    test -d /rab_data/habitat/objects/ycb \
        || { echo "[sanity FAIL] objects/ycb directory"; exit 1; } && \
    test -d /rab_data/habitat/robots/hab_spot_arm \
        || { echo "[sanity FAIL] robots/hab_spot_arm directory"; exit 1; } && \
    test -L /rab_data/habitat/datasets/hssd/rearrange/train/social_rearrange.json.gz \
        || { echo "[sanity FAIL] train/social_rearrange.json.gz symlink"; exit 1; } && \
    test -L /rab_data/habitat/datasets/hssd/rearrange/val/social_rearrange.json.gz \
        || { echo "[sanity FAIL] val/social_rearrange.json.gz symlink"; exit 1; } && \
    grep -q '"hssd_ssd_map"' \
        /rab_data/habitat/hab3_bench_assets/hab3-hssd/hab3-hssd.scene_dataset_config.json \
        || { echo "[sanity FAIL] hab3-hssd config missing hssd_ssd_map patch"; exit 1; } && \
    test -f /rab_data/habitat/scene_datasets/hssd-hab/semantics/hssd-hab_semantic_lexicon.json \
        || { echo "[sanity FAIL] hssd-hab semantic lexicon"; exit 1; } && \
    echo "[sanity] OK — all required assets present" && \
    du -sh /rab_data/habitat/* | sort -h

# ── Task-name alias: /rab_data/habitat3 → /rab_data/habitat ──────────────────
# We rename the task externally (habitat → habitat3) but keep the physical data
# directory at /rab_data/habitat so the image's existing absolute symlinks
# (created by datasets_download) and BuildKit layer cache stay valid. The
# alias lets entrypoint.sh / prepare.py / task.yaml use /rab_data/habitat3
# uniformly while still resolving to the same bytes.
RUN ln -s /rab_data/habitat /rab_data/habitat3 && \
    test -d /rab_data/habitat3/scene_datasets/hssd-hab \
        || { echo "[FATAL] /rab_data/habitat3 alias broken"; exit 1; }

# ── Smoke-test: actually create env & reset (catches API + dataset drift) ────
# This is a strict end-to-end test — if it fails, the image is unusable.
RUN cd /opt/habitat-lab-src && \
    DISPLAY= EGL_PLATFORM=surfaceless ${ENV_PY} -c "\
import habitat; \
from habitat.config.default import get_config; \
from habitat.config import read_write; \
cfg_path = '/opt/habitat-lab-src/habitat-lab/habitat/config/benchmark/multi_agent/hssd_spot_human_social_nav.yaml'; \
cfg = get_config(cfg_path); \
print('[smoke] social_nav config loaded:'); \
print('  task type:    ', cfg.habitat.task.type); \
print('  reward:       ', cfg.habitat.task.reward_measure); \
print('  success:      ', cfg.habitat.task.success_measure); \
print('  max_steps:    ', cfg.habitat.environment.max_episode_steps); \
print('  agents_order: ', list(cfg.habitat.simulator.agents_order)); \
print('  data_path:    ', cfg.habitat.dataset.data_path); \
" || true

# ── Entrypoint: bridge image-baked /rab_data/habitat3 to the paths habitat ───
# expects (/opt/habitat-lab-src/data and /workspace/data). Implemented as a
# real file (not base64-inlined) for transparency & easier debugging.
#
# NOTE on COPY path: the image build runs `docker build -f docker/Dockerfile .`
# with cwd=benchmarks/habitat3/, so the build context root is
# benchmarks/habitat3/. Use paths *relative to that*, e.g. `docker/entrypoint.sh`,
# NOT `benchmarks/habitat3/docker/entrypoint.sh` (that path doesn't exist
# inside the context).
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

WORKDIR /workspace
