"""HumanoidBench task preparation (thin file-copy layer).

All heavy lifting (MuJoCo, humanoid-bench install, eval protocol JSON) is
baked into the Docker image at build time. This script only:
  1. Copies `/opt/farbench/humanoidbench/dataset_info.json`  → FARBENCH_DATA_DIR/
  2. Copies `/opt/farbench/humanoidbench/eval_config.json`   → FARBENCH_TEST_DATA_DIR/
  3. Writes `.prepared` sentinels.

Online RL has no static training dataset, so FARBENCH_DATA_DIR only holds metadata.
"""

from __future__ import annotations

import os
import shutil
import sys


IMAGE_FIXTURE_DIR = "/opt/farbench/humanoidbench"
DATASET_INFO_SRC = os.path.join(IMAGE_FIXTURE_DIR, "dataset_info.json")
EVAL_CONFIG_SRC = os.path.join(IMAGE_FIXTURE_DIR, "eval_config.json")


def _copy_fixture(src: str, dst_dir: str) -> str:
    if not os.path.isfile(src):
        raise RuntimeError(
            f"[prepare] Fixture missing inside the image: {src}. "
            "Rebuild the task image so docker/fixtures/*.json are baked in."
        )
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, os.path.basename(src))
    shutil.copyfile(src, dst)
    return dst


def _mark_done(path: str) -> None:
    open(os.path.join(path, ".prepared"), "w").close()


def main() -> None:
    data_dir = os.environ["FARBENCH_DATA_DIR"]
    test_data_dir = os.environ["FARBENCH_TEST_DATA_DIR"]

    dataset_info_dst = _copy_fixture(DATASET_INFO_SRC, data_dir)
    eval_config_dst = _copy_fixture(EVAL_CONFIG_SRC, test_data_dir)
    _mark_done(data_dir)
    _mark_done(test_data_dir)

    print(f"[prepare] dataset_info -> {dataset_info_dst}")
    print(f"[prepare] eval_config  -> {eval_config_dst}")
    print("[prepare] HumanoidBench ready (online RL, no static training data).")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[prepare] ERROR: {exc}", file=sys.stderr)
        raise
