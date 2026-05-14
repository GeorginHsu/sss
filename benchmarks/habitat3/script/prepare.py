"""Habitat 3.0 Social Navigation data preparation.

All heavy assets (HSSD-200 scenes, hab3_bench episodes, humanoid avatars,
YCB objects, Spot/Fetch URDFs — ~25 GB total) are baked into the Docker
image at build time via `habitat_sim.utils.datasets_download` (see
docker/Dockerfile). The container entrypoint bridges them at
`/opt/habitat-lab-src/data` and `/workspace/data`.

This is a pure RL task — there is no static test set. `eval_config.json`
exists only to satisfy `farbench/env.py`'s framework-level "test_data_dir
must contain eval_config.json" check. The agent's predict.py does NOT
read it (task.yaml tells agents to hardcode split/num_episodes/seed).

Outputs:
    $FARBENCH_DATA_DIR/.prepared            (empty sentinel for hub prepare)
    $FARBENCH_TEST_DATA_DIR/eval_config.json (placeholder; satisfies framework)
    $FARBENCH_TEST_DATA_DIR/.prepared       (empty sentinel)
"""

from __future__ import annotations

import json
import os


# ── Eval config (placeholder — see module docstring) ─────────────────────────
# Documented here for human reference. The agent's predict.py is told (in
# task.yaml) to hardcode these values, NOT to read this file.
EVAL_CONFIG = {
    "split": "val",
    "num_episodes": 50,
    "seed": 42,
    "primary_metric": "nav_seek_success",
}


def _touch(path: str) -> None:
    open(path, "w").close()


def main() -> None:
    data_dir = os.environ["FARBENCH_DATA_DIR"]
    test_data_dir = os.environ["FARBENCH_TEST_DATA_DIR"]
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(test_data_dir, exist_ok=True)

    # 1. Eval config — read by the agent's predict.py at /data/eval_config.json
    #    inside the eval container.
    eval_config_path = os.path.join(test_data_dir, "eval_config.json")
    with open(eval_config_path, "w") as f:
        json.dump(EVAL_CONFIG, f, indent=2)

    # 2. Sentinel files. These are necessary for two consumers:
    #    • Build-time checks can assert preparation completed.
    #    • The image archive pipeline must not bake an empty task-data tree.
    _touch(os.path.join(data_dir, ".prepared"))
    _touch(os.path.join(test_data_dir, ".prepared"))

    print("[prepare] Habitat 3.0 Social-Nav task ready.")
    print(f"[prepare]   Data dir:      {data_dir}")
    print(f"[prepare]   Test data dir: {test_data_dir}")
    print(f"[prepare]   Eval config:   {eval_config_path}")
    print(f"[prepare]   Eval split:    {EVAL_CONFIG['split']}")
    print(f"[prepare]   Episodes:      {EVAL_CONFIG['num_episodes']}")
    print(f"[prepare]   Seed:          {EVAL_CONFIG['seed']}")
    print("[prepare] All heavy assets (~25 GB) are baked into the image at /rab_data/habitat3;")
    print("[prepare] no host-side data staging is performed.")


if __name__ == "__main__":
    main()
