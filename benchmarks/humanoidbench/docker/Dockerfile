ARG BASE_TAG=cu118
FROM farbench/farbench:base-${BASE_TAG}

ARG HF_TOKEN=""
ENV HF_TOKEN=${HF_TOKEN}

# ── System deps for MuJoCo headless rendering ────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-dev \
    libgles2-mesa-dev \
    libegl1-mesa-dev \
    libglew-dev \
    libglfw3 \
    libosmesa6 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libxi-dev \
    libx11-6 \
    ffmpeg \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# ── Headless MuJoCo rendering via EGL ────────────────────────────────────────
ENV MUJOCO_GL=egl
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV PYOPENGL_PLATFORM=egl

# ── Core MuJoCo + dm_control + Gymnasium stack ───────────────────────────────
RUN pip install --no-cache-dir \
    "mujoco==3.1.6" \
    "dm_control==1.0.20" \
    "gymnasium[other]>=0.29,<1.2" \
    "numpy<2.1" \
    "scipy" \
    "imageio" \
    "opencv-python-headless"

# ── Clone HumanoidBench at a pinned commit and install editable ──────────────
# Pin a specific commit for reproducibility. The original code used `main`,
# but `main` is a moving pointer — rebuilding the image months later could
# pick up upstream changes to obs/action spaces or reward functions and
# silently break previously trained policies. The SHA below is the upstream
# main HEAD as of 2025-09-18 (commit message: "Update LICENSE", code stable
# for ~7 months at pinning time). To bump: pick a new SHA from
# https://github.com/carlosferrazza/humanoid-bench/commits/main, then
# rebuild + repush this image and document the change.
ARG HUMANOID_BENCH_REF=cb1189039151c8aadaaa987b442da54383c87fab
# `git clone --branch` only accepts branch/tag refs, so for a SHA we have to
# clone first (full history is small, ~few MB) then `checkout <sha>`.
RUN git clone https://github.com/carlosferrazza/humanoid-bench.git /opt/humanoid-bench \
    && cd /opt/humanoid-bench \
    && git checkout ${HUMANOID_BENCH_REF} \
    && pip install --no-cache-dir -e /opt/humanoid-bench

# ── ML / RL packages for the agent ───────────────────────────────────────────
RUN pip install --no-cache-dir \
    "torch" \
    "stable-baselines3>=2.3" \
    "sb3-contrib>=2.3" \
    "tensorboard" \
    "tqdm" \
    "pyyaml" \
    "wandb"

# ── Bake static task configs into the image ──────────────────────────────────
# These files describe the eval protocol (10 tasks × 20 episodes × fixed seeds)
# and are copied verbatim at `prepare` time into FARBENCH_DATA_DIR / FARBENCH_TEST_DATA_DIR.
# Keep them in one place so the host `prepare.py` stays a tiny file-copy step.
#
# COPY happens BEFORE the sanity check below so the check can read the JSON
# directly — single source of truth, no chance of the hardcoded list drifting
# from the actual eval protocol.
RUN mkdir -p /opt/farbench/humanoidbench
COPY docker/fixtures/eval_config.json   /opt/farbench/humanoidbench/eval_config.json
COPY docker/fixtures/dataset_info.json  /opt/farbench/humanoidbench/dataset_info.json

# ── Sanity check: import + assert every eval env matches eval_config.json ────
# HARD FAIL gate (no `|| echo` swallow). This is the single source of truth
# for the obs_dim / act_dim / success_bar / info["success"] facts that the
# task.yaml and evaluator depend on. If an upstream humanoid-bench bump
# changes any of them (e.g. observation re-formatting, action_space drift,
# new success_bar values, or h1hand env id rename), THIS step fails and the
# image is never published — far better than letting silent drift produce
# `success_rate=0` for 8/10 tasks at runtime.
#
# Catches:
#   - env id registration drift (NameNotFound)
#   - MJCF / asset path drift (only surfaces at reset())
#   - obs_dim drift between humanoid-bench commits
#   - act_dim drift (e.g. body or hand DOF count change)
#   - success_bar drift (per-task class attribute)
#   - info_has_success drift (push/package only)
#
# Drives every check from /opt/farbench/humanoidbench/eval_config.json which was
# COPY'd in the previous layer — single source of truth for the runtime
# evaluator AND the build-time gate.
RUN python - <<'PY'
import json
import sys
import warnings
warnings.filterwarnings("ignore")

import gymnasium as gym
import humanoid_bench  # registers h1*/h1hand*/g1* env ids on import
from humanoid_bench import env as hb_env_module
from humanoid_bench.env import TASKS as HB_TASKS

with open("/opt/farbench/humanoidbench/eval_config.json") as f:
    cfg = json.load(f)
eval_tasks = cfg["tasks"]
expected_act_dim = int(cfg.get("action_dim", 61))
print(f"[sanity] expected action_dim = {expected_act_dim}")
print(f"[sanity] checking {len(eval_tasks)} env ids vs eval_config.json")

failures = []
for spec in eval_tasks:
    tid = spec["task_id"]
    try:
        env = gym.make(tid)
        obs, info_reset = env.reset(seed=0)
        act = env.action_space.sample()
        step_out = env.step(act)
        info_step = step_out[-1]

        # Real measurements
        real_obs_dim = int(env.observation_space.shape[0])
        real_act_dim = int(env.action_space.shape[0])
        real_info_has_success = "success" in info_step
        env.close()

        # Cross-check vs eval_config.json
        errors = []
        if real_act_dim != expected_act_dim:
            errors.append(f"act_dim {real_act_dim} != expected {expected_act_dim}")
        if real_obs_dim != int(spec["obs_dim"]):
            errors.append(f"obs_dim {real_obs_dim} != cfg {spec['obs_dim']}")
        if real_info_has_success != bool(spec["info_has_success"]):
            errors.append(
                f"info_has_success {real_info_has_success} != cfg {spec['info_has_success']}"
            )

        # success_bar: cross-check against humanoid-bench Task class attr
        # via TASKS dict ('walk' -> Walk, 'h1hand-walk-v0' -> 'walk').
        task_short = tid.removeprefix("h1hand-").removesuffix("-v0")
        TaskCls = HB_TASKS.get(task_short)
        if TaskCls is None:
            errors.append(f"unknown task short name '{task_short}' in HB_TASKS")
        else:
            real_bar = float(getattr(TaskCls, "success_bar"))
            if real_bar != float(spec["success_bar"]):
                errors.append(
                    f"success_bar {real_bar} != cfg {spec['success_bar']}"
                )

        if errors:
            print(f"[sanity] FAIL {tid:<30s} {'; '.join(errors)}")
            failures.append((tid, "; ".join(errors)))
        else:
            print(
                f"[sanity] OK   {tid:<30s} obs={real_obs_dim:>4d} act={real_act_dim} "
                f"bar={spec['success_bar']:>5} info_succ={real_info_has_success}"
            )
    except Exception as exc:
        print(f"[sanity] FAIL {tid}: {type(exc).__name__}: {exc}")
        failures.append((tid, repr(exc)))

if failures:
    print(f"\n[sanity] {len(failures)} env id(s) failed:", file=sys.stderr)
    for tid, msg in failures:
        print(f"  - {tid}: {msg}", file=sys.stderr)
    sys.exit(1)
print(
    f"\n[sanity] all {len(eval_tasks)} eval envs match eval_config.json "
    f"(obs_dim, act_dim={expected_act_dim}, success_bar, info_has_success)."
)
PY

WORKDIR /workspace
