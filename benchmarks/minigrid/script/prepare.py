"""Minigrid PPO data preparation.

Minigrid is a procedurally-generated RL environment — there is no static
dataset to download.  This script only writes the evaluation config that
defines which environments, seeds, and episode counts the evaluator expects.

Output layout:
    FARBENCH_DATA_DIR/          — empty (agent collects its own experience)
    FARBENCH_TEST_DATA_DIR/
        eval_config.json   — evaluation specification
"""

from __future__ import annotations

import json
import os


# ── Evaluation config ────────────────────────────────────────────────────────
# 10 evaluation environments (same as train envs), 50 episodes each = 500 total.
EVAL_CONFIG = {
    "eval_envs": [
        "MiniGrid-DoorKey-8x8-v0",
        "MiniGrid-FourRooms-v0",
        "MiniGrid-KeyCorridorS3R3-v0",
        "MiniGrid-LockedRoom-v0",
        "MiniGrid-Dynamic-Obstacles-8x8-v0",
        "MiniGrid-LavaCrossingS9N2-v0",
        "MiniGrid-UnlockPickup-v0",
        "MiniGrid-MultiRoom-N4-S5-v0",
        "MiniGrid-LavaGapS7-v0",
        "MiniGrid-DistShift1-v0",
    ],
    "train_envs": [
        "MiniGrid-DoorKey-8x8-v0",
        "MiniGrid-FourRooms-v0",
        "MiniGrid-KeyCorridorS3R3-v0",
        "MiniGrid-LockedRoom-v0",
        "MiniGrid-Dynamic-Obstacles-8x8-v0",
        "MiniGrid-LavaCrossingS9N2-v0",
        "MiniGrid-UnlockPickup-v0",
        "MiniGrid-MultiRoom-N4-S5-v0",
        "MiniGrid-LavaGapS7-v0",
        "MiniGrid-DistShift1-v0",
    ],
    "episodes_per_env": 50,
    "seed_start": 0,
    "seed_end": 49,
    "max_steps_multiplier": 1,  # use env default max_steps
}


def main() -> None:
    data_dir = os.environ["FARBENCH_DATA_DIR"]
    test_data_dir = os.environ["FARBENCH_TEST_DATA_DIR"]
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(test_data_dir, exist_ok=True)

    # ── Write placeholder so data_dir is non-empty for validation ──────────
    readme_path = os.path.join(data_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(
            "Minigrid is a procedurally-generated RL environment.\n"
            "No static training data — the agent collects experience online.\n"
        )

    # ── Write eval config ────────────────────────────────────────────────────
    eval_config_path = os.path.join(test_data_dir, "eval_config.json")
    with open(eval_config_path, "w") as f:
        json.dump(EVAL_CONFIG, f, indent=2)
    print(f"[prepare] Eval config written to {eval_config_path}")

    # ── Summary ──────────────────────────────────────────────────────────────
    eval_envs = EVAL_CONFIG["eval_envs"]
    n_envs = len(eval_envs)
    n_eps = EVAL_CONFIG["episodes_per_env"]
    print(f"\n[prepare] Minigrid task ready.")
    print(f"  Train envs: {EVAL_CONFIG['train_envs']}")
    print(f"  Eval envs:  {EVAL_CONFIG['eval_envs']}")
    print(f"  Episodes:   {n_eps} x {n_envs} = {n_eps * n_envs}")
    print(f"\n  NOTE: No training data to download.")
    print(f"  The agent collects experience by interacting with Minigrid environments.")


if __name__ == "__main__":
    main()
