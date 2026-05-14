ARG BASE_TAG=cu118
FROM farbench/farbench:base-${BASE_TAG}
ARG HF_TOKEN=""
ENV HF_TOKEN=${HF_TOKEN}

# ── Minigrid + RL dependencies ────────────────────────────────────────────────
# gymnasium:  RL environment API (required by minigrid)
# minigrid:   grid-world environments
# stable-baselines3: PPO implementation (agent may choose to use this or write custom PPO)
# shimmy:     compatibility layer for older gym envs (some minigrid wrappers need it)
RUN pip install --no-cache-dir \
    gymnasium \
    minigrid \
    stable-baselines3 \
    shimmy>=0.2.1

WORKDIR /workspace
