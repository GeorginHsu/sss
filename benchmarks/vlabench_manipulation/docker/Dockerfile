# vlabench_manipulation requires Python 3.12 (lerobot[pi]==0.5.1 hard
# requirement). scripts/dev/FARBench image archive build pipeline maps this task to
# farbench/farbench:base-py312-${CUDA} built from docker/Dockerfile.base.py312.
ARG BASE_TAG=py312-cu118
ARG PYTORCH_CUDA=cu118
FROM farbench/farbench:base-${BASE_TAG}
ARG PYTORCH_CUDA=cu118
ARG HF_TOKEN=""

# ── System deps for MuJoCo EGL headless rendering ──────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-dev \
    libgles2-mesa-dev \
    libegl1-mesa-dev \
    libglew-dev \
    libsm6 \
    libxrender1 \
    libxext6 \
    libxi-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Headless rendering via EGL ──────────────────────────────────────────────
ENV MUJOCO_GL=egl
ENV PYOPENGL_PLATFORM=egl

# ── Pin torch to lerobot[pi]==0.5.1's requirement (torch>=2.7,<2.11) ────────
# The base image already has torch==2.7.1+cu118; this RUN is a no-op safety
# net in case the base ever drifts. Use the explicit cu118 wheel index so
# pip never picks a newer CUDA wheel (which would raise the host driver
# requirement and silently break the cu118 image).
RUN python -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/${PYTORCH_CUDA} \
        torch==2.7.1 torchvision==0.22.1

# ── lerobot[pi] (Pi0/Pi0.5) — forces transformers==5.3.0 + safetensors==0.5.2 ─
# `[pi]` extra is REQUIRED. Without it, transformers floats at whatever
# diffusers/accelerate ask for and `PI0Policy.from_pretrained` crashes
# with `TypeError: 'NoneType' object is not subscriptable` at
# modeling_pi0.py:359 because the paligemma config drops out of
# CONFIG_MAPPING.
RUN python -m pip install --no-cache-dir \
        "lerobot[pi]==0.5.1"

# ── VLABench / sim / training extras not in the base ────────────────────────
# `openai` is a HARD import dependency of VLABench.utils.gpt_utils — it is
# only `import`ed at module load (never called); network is disabled at
# runtime. Removing it crashes load_env() / Evaluator at task registration.
RUN python -m pip install --no-cache-dir \
        mujoco==3.2.2 \
        dm_control==1.0.22 \
        gym==0.26.2 \
        gymnasium==0.29.1 \
        open3d==0.19.0 \
        mediapy==1.2.0 \
        colorlog==6.10.1 \
        colorama==0.4.6 \
        openai==2.33.0 \
        gdown==6.0.0 \
        imageio==2.37.3 \
        deepspeed==0.18.9 \
        peft==0.19.1 \
        bitsandbytes==0.49.2 \
        zarr==2.18.3 \
        pyquaternion==0.9.9 \
        open-clip-torch==3.3.0 \
        huggingface_hub==1.12.0 \
        wandb==0.26.1 \
        requests==2.33.1 \
        "git+https://github.com/motion-planning/rrt-algorithms.git@e51d95ee489a225220d6ae2a764c4111f6ba7d85#egg=rrt-algorithms"

# ── VLABench package ─────────────────────────────────────────────────────────
RUN git clone --depth 1 https://github.com/OpenMOSS/VLABench.git /opt/VLABench
RUN python -m pip install --no-cache-dir -e /opt/VLABench

# ── Drop the heavy obj/ + scenes/ dirs (entrypoint relinks at runtime) ──────
# VLABench's git ships small but ESSENTIAL subdirs under assets/ (notably
# `robots/franka_emika_panda/panda.xml` and `base/`). The 12 GB obj/ +
# scenes/ come from upstream zip releases and are baked into
# /rab_data/vlabench_manipulation/vlabench_root/ at prepare time, then
# bind-mounted at /data/vlabench_root/ at sandbox runtime. The entrypoint
# relinks /opt/VLABench/VLABench/assets/{obj,scenes} →
# /data/vlabench_root/assets/{...} (or to the image-baked /rab_data/...
# fallback in the eval container). Do NOT remove the entire assets dir.
RUN rm -rf /opt/VLABench/VLABench/assets/obj \
           /opt/VLABench/VLABench/assets/scenes

# ── Patch 1: skip composite tasks (need rrt_algorithms + openai, unused) ────
RUN sed -i \
    's/from VLABench.tasks.hierarchical_tasks.composite import \*/# composite tasks skipped/' \
    /opt/VLABench/VLABench/tasks/hierarchical_tasks/__init__.py

# ── Patch 2: make rrt_algorithms import optional ────────────────────────────
# rrt_algorithms.rrt submodule is missing from the installed package.
# Primitive tasks (add_condiment, select_fruit, select_book) never call RRT
# at inference time, so a stub is sufficient.
RUN sed -i \
    's/^from rrt_algorithms\.rrt\.rrt import RRT/try:\n    from rrt_algorithms.rrt.rrt import RRT\nexcept ImportError:\n    RRT = None/' \
    /opt/VLABench/VLABench/algorithms/motion_planning/rrt.py && \
    sed -i \
    's/^from rrt_algorithms\.rrt\.rrt_star import RRTStar/try:\n    from rrt_algorithms.rrt.rrt_star import RRTStar\nexcept ImportError:\n    RRTStar = None/' \
    /opt/VLABench/VLABench/algorithms/motion_planning/rrt.py

# ── Patch 3: lazily register robots/tasks when load_env is called ───────────
# Avoid heavy package-level imports during prepare-time helpers, but keep
# runtime registration automatic for predict.py and evaluation.
RUN python - <<'PY'
from pathlib import Path

path = Path("/opt/VLABench/VLABench/envs/__init__.py")
text = path.read_text()
needle = "    # load config\n"
replacement = (
    "    import VLABench.robots\n"
    "    import VLABench.tasks\n\n"
    "    # load config\n"
)
if "import VLABench.robots" not in text:
    text = text.replace(needle, replacement, 1)
path.write_text(text)
PY

# ── Patch 4: keep official Evaluator imports lightweight ────────────────────
# Upstream VLABench/evaluation/__init__.py imports all bundled policy
# backends (including OpenVLA, which would force a non-lerobot transformers
# version). Export only the official base Evaluator from the package roots
# the agents and the task documentation use.
RUN python - <<'PY'
from pathlib import Path

eval_init = Path("/opt/VLABench/VLABench/evaluation/__init__.py")
eval_init.write_text(
    '"""Runtime-safe VLABench evaluation exports for FARBench."""\n'
    "from VLABench.evaluation.evaluator.base import Evaluator\n\n"
    '__all__ = ["Evaluator"]\n'
)

evaluator_init = Path("/opt/VLABench/VLABench/evaluation/evaluator/__init__.py")
evaluator_init.write_text(
    '"""Runtime-safe Evaluator package export."""\n'
    "from VLABench.evaluation.evaluator.base import Evaluator\n\n"
    '__all__ = ["Evaluator"]\n'
)
PY

# ── Patch 5: skip depth/segmentation/robot_mask render when require_pcd=False ─
# Default `LM4ManipDMEnv.get_observation(require_pcd=False)` renders, for
# every camera, four buffers (rgb + depth + segmentation + a derived
# robot_mask) — empirically 1271 ms per call on a 4-camera scene, vs.
# ~480 ms for rgb alone. The official `Evaluator.evaluate(...)` only reads
# `obs["ee_state"]` (L139) and `obs["rgb"]` (L145, visulization=True path);
# depth/seg/mask are unused at evaluation time and only consumed by the
# `require_pcd=True` branch. Skipping their render in the require_pcd=False
# path cuts a 60-episode eval from ~6–7 h to ~2.5–3.5 h with NO change in
# observable Evaluator behavior. Agents that want depth/seg can call
# `env.get_observation(require_pcd=True)` to opt in (same code path as
# upstream).
RUN python - <<'PY'
from pathlib import Path

path = Path("/opt/VLABench/VLABench/envs/dm_env.py")
src = path.read_text()
old = (
    "    def get_observation(self, require_pcd=True):\n"
    "        observation = dict()\n"
    "        multi_view_rgb, multi_view_depth, multi_view_seg = [], [], []\n"
    "        instrinsic_matrixs, extrinsic_matrixs = [], []\n"
    "        for cam in range(self.physics.model.ncam):\n"
    "            multi_view_rgb.append(self.render(camera_id=cam, **self.render_options))\n"
    "            multi_view_depth.append(self.render(camera_id=cam, **self.render_options, depth=True))\n"
    "            multi_view_seg.append(self.render(camera_id=cam, **self.render_options, segmentation=True))\n"
    "            instrinsic, extrinsic = self.get_camera_matrix(cam, **self.render_options)\n"
    "            instrinsic_matrixs.append(instrinsic)\n"
    "            extrinsic_matrixs.append(extrinsic)\n"
)
new = (
    "    def get_observation(self, require_pcd=True):\n"
    "        # FARBench Patch 5: skip depth/seg/robot_mask render when\n"
    "        # require_pcd=False (Evaluator default). Saves ~1.1 s per env\n"
    "        # step (~3x speedup on the 60-episode eval) without changing\n"
    "        # any observable Evaluator behavior; pcd path unchanged.\n"
    "        observation = dict()\n"
    "        multi_view_rgb, multi_view_depth, multi_view_seg = [], [], []\n"
    "        instrinsic_matrixs, extrinsic_matrixs = [], []\n"
    "        for cam in range(self.physics.model.ncam):\n"
    "            multi_view_rgb.append(self.render(camera_id=cam, **self.render_options))\n"
    "            if require_pcd:\n"
    "                multi_view_depth.append(self.render(camera_id=cam, **self.render_options, depth=True))\n"
    "                multi_view_seg.append(self.render(camera_id=cam, **self.render_options, segmentation=True))\n"
    "            instrinsic, extrinsic = self.get_camera_matrix(cam, **self.render_options)\n"
    "            instrinsic_matrixs.append(instrinsic)\n"
    "            extrinsic_matrixs.append(extrinsic)\n"
)
if old not in src:
    raise SystemExit("Patch 5 anchor not found — VLABench upstream changed get_observation; refresh the patch.")
src = src.replace(old, new)

# Move depth/seg/mask field assignment under `if require_pcd:` so they
# only get materialised when actually rendered.
old2 = (
    '        observation["rgb"] = np.array(multi_view_rgb)\n'
    '        observation["depth"] = np.array(multi_view_depth)\n'
    '        observation["segmentation"] = np.array(multi_view_seg)\n'
    '        observation["robot_mask"] = np.where((observation["segmentation"][..., 0] <= 72)&(observation["segmentation"][..., 0] > 0), 0, 1).astype(np.uint8)\n'
)
new2 = (
    '        observation["rgb"] = np.array(multi_view_rgb)\n'
    '        if require_pcd:\n'
    '            observation["depth"] = np.array(multi_view_depth)\n'
    '            observation["segmentation"] = np.array(multi_view_seg)\n'
    '            observation["robot_mask"] = np.where((observation["segmentation"][..., 0] <= 72)&(observation["segmentation"][..., 0] > 0), 0, 1).astype(np.uint8)\n'
)
if old2 not in src:
    raise SystemExit("Patch 5 second anchor not found — VLABench upstream changed observation assembly; refresh the patch.")
src = src.replace(old2, new2)
path.write_text(src)
print("Patch 5 applied: get_observation(require_pcd=False) skips depth/seg/mask render")
PY

# Static (AST-only) verification of Patch 5 — see comments on the test
# at the matching commit for why we DON'T call load_env() at build time.
RUN python - <<'PY'
import ast
src = open('/opt/VLABench/VLABench/envs/dm_env.py').read()
tree = ast.parse(src)  # SyntaxError → fail build immediately

# Pre-patch the get_observation function has 1 `if require_pcd:` (the pcd
# generation guard at the bottom). Post-Patch-5 we add 2 more (gating
# depth/seg render and gating depth/seg/mask field assignment), so a
# passing build must see >= 3.
require_pcd_count = 0
get_obs_found = False
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == 'get_observation':
        get_obs_found = True
        for child in ast.walk(node):
            if (isinstance(child, ast.If)
                    and isinstance(child.test, ast.Name)
                    and child.test.id == 'require_pcd'):
                require_pcd_count += 1
assert get_obs_found, "Patch 5: get_observation function not found in dm_env.py"
assert require_pcd_count >= 3, (
    f"Patch 5: expected >= 3 `if require_pcd:` branches in get_observation "
    f"(2 added by patch + 1 original pcd guard); found {require_pcd_count}."
)
print(f'Patch 5 verified (AST): get_observation has {require_pcd_count} require_pcd guards')
PY

# Smoke test: the Evaluator base path imports cleanly.
RUN python - <<'PY'
from VLABench.evaluation import Evaluator as E0
from VLABench.evaluation.evaluator import Evaluator as E1
from VLABench.evaluation.evaluator.base import Evaluator as E2
assert E0 is E2 is E1
print("VLABench Evaluator import smoke test passed")
PY

# Smoke test: VLABench task registration chain doesn't crash on missing
# openai (it would, if we ever drop the openai pin above).
RUN python - <<'PY'
import VLABench.robots
from VLABench.utils.gpt_utils import query_gpt4_v
print("VLABench task-registration import smoke test passed")
PY

# ── VLABENCH_ROOT — entrypoint relinks obj/ + scenes/ here at runtime ───────
ENV VLABENCH_ROOT=/opt/VLABench/VLABench

# ── Pre-download lerobot/pi0_base into the HF cache (~14.5 GB) ──────────────
# Network is disabled in the agent sandbox at runtime — the agent must
# load Pi0 by canonical HF id (`PI0Policy.from_pretrained("lerobot/pi0_base")`)
# and HF Hub resolves it against the cache populated here at build time.
# `lerobot/pi0_base` is currently `gated: false` (Gemma license applies
# to downstream use). Do NOT bake HF tokens into the image.
RUN HF_HUB_DISABLE_PROGRESS_BARS=1 python -c "\
import os; \
from huggingface_hub import snapshot_download; \
token = os.environ.get('HF_TOKEN') or None; \
p = snapshot_download(repo_id='lerobot/pi0_base', token=token); \
print(f'lerobot/pi0_base cached at {p}')"

# Pi0's policy_preprocessor.json references the PaliGemma tokenizer by
# canonical HF id. `snapshot_download("lerobot/pi0_base")` does not populate
# that separate repo cache, so cache only the tokenizer/processor metadata
# needed by AutoTokenizer and LeRobot's tokenizer_processor. Do not download
# google/paligemma model weights; Pi0 already carries the model weights.
# This Google repo is gated on Hugging Face, so FARBench image archive build pipeline must
# pass an HF_TOKEN whose account has accepted the PaliGemma/Gemma license.
# The token is only used during image build; runtime remains offline.
RUN HF_HUB_DISABLE_PROGRESS_BARS=1 python - <<'PY'
import os
from huggingface_hub import snapshot_download

token = os.environ.get("HF_TOKEN") or None
if not token:
    raise SystemExit(
        "HF_TOKEN is required to cache google/paligemma-3b-pt-224 tokenizer "
        "assets. Add HF_TOKEN=... to .env or export it before running "
        "scripts/dev/FARBench image archive build pipeline."
    )

p = snapshot_download(
    repo_id="google/paligemma-3b-pt-224",
    token=token,
    allow_patterns=[
        "config.json",
        "tokenizer*",
        "special_tokens_map.json",
        "added_tokens.json",
        "processor_config.json",
        "preprocessor_config.json",
        "image_processor_config.json",
    ],
)
print(f"google/paligemma-3b-pt-224 tokenizer assets cached at {p}")
PY

# End-to-end smoke test: load PI0Policy and its official LeRobot processors
# from cached files only. Catches four classes of build-time regressions:
#   1. Cached files corrupt / incomplete → from_pretrained raises during load
#   2. lerobot import path drift (e.g. `lerobot.policies.pi0` renamed)
#   3. transformers/lerobot version skew → the paligemma CONFIG_MAPPING crash
#   4. Missing PaliGemma tokenizer cache → make_pre_post_processors fails
# CPU-only; HF_HUB_OFFLINE=1 forces no network so this proves the cache is
# fully self-contained.
RUN HF_HUB_OFFLINE=1 python - <<'PY'
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi0 import PI0Policy
from transformers import AutoTokenizer

policy = PI0Policy.from_pretrained("lerobot/pi0_base")
AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224", local_files_only=True)
make_pre_post_processors(
    policy.config,
    "lerobot/pi0_base",
    preprocessor_overrides={"device_processor": {"device": "cpu"}},
)
print("PI0Policy + make_pre_post_processors offline smoke test passed")
PY

# ── Entrypoint: maps hostname to loopback + symlinks vlabench_root assets ───
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

WORKDIR /workspace
