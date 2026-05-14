"""ScreenSpot-Pro task preparation (thin file-copy layer).

All heavy lifting (training-mix download from HF, ScreenSpot-Pro test set
download, Qwen2.5-VL-3B base model download) is baked into the Docker image
at build time under /opt/farbench/screenspot_pro/. This script only:

  /opt/farbench/screenspot_pro/train_data/    -> FARBENCH_DATA_DIR/
  /opt/farbench/screenspot_pro/test_data/     -> FARBENCH_TEST_DATA_DIR/
  /opt/farbench/screenspot_pro/base_model/    -> FARBENCH_DATA_DIR/base_model/

Then writes `.prepared` sentinels.

Why no logic here? Because the training mix (~28GB unified_train.jsonl +
images/) is too large + too custom to (re)derive at task-prepare time, and
network is OFF inside the container. Build the image once, prepare is fast
and deterministic forever after.
"""

from __future__ import annotations

import os
import shutil
import sys


IMAGE_FIXTURE_DIR = "/opt/farbench/screenspot_pro"
TRAIN_SRC = os.path.join(IMAGE_FIXTURE_DIR, "train_data")
TEST_SRC = os.path.join(IMAGE_FIXTURE_DIR, "test_data")
BASE_MODEL_SRC = os.path.join(IMAGE_FIXTURE_DIR, "base_model")


def _copytree(src: str, dst: str) -> None:
    """Idempotent copytree: skips files that already exist at dst with the
    same size. Lets `prepare` re-run cheaply without re-copying ~30GB.
    """
    if not os.path.isdir(src):
        raise RuntimeError(
            f"[prepare] Fixture missing inside the image: {src}. "
            f"Rebuild the image so /opt/farbench/screenspot_pro/{os.path.basename(src)} "
            "is baked in."
        )
    os.makedirs(dst, exist_ok=True)
    n_copied = 0
    n_skipped = 0
    for root, _dirs, files in os.walk(src):
        rel = os.path.relpath(root, src)
        target_root = os.path.join(dst, rel) if rel != "." else dst
        os.makedirs(target_root, exist_ok=True)
        for fname in files:
            s = os.path.join(root, fname)
            d = os.path.join(target_root, fname)
            try:
                if os.path.exists(d) and os.path.getsize(d) == os.path.getsize(s):
                    n_skipped += 1
                    continue
            except OSError:
                pass
            shutil.copyfile(s, d)
            n_copied += 1
    print(f"[prepare] {os.path.basename(src)}: copied={n_copied} skipped={n_skipped}")


def _mark_done(path: str) -> None:
    open(os.path.join(path, ".prepared"), "w").close()


def main() -> None:
    data_dir = os.environ["FARBENCH_DATA_DIR"]
    test_data_dir = os.environ["FARBENCH_TEST_DATA_DIR"]

    # Training data (jsonl + images/) -> FARBENCH_DATA_DIR/
    _copytree(TRAIN_SRC, data_dir)

    # Base model -> FARBENCH_DATA_DIR/base_model/
    if os.path.isdir(BASE_MODEL_SRC):
        _copytree(BASE_MODEL_SRC, os.path.join(data_dir, "base_model"))
    else:
        print("[prepare] WARNING: no base_model under /opt/farbench/screenspot_pro; "
              "agents must download Qwen2.5-VL-3B themselves (network is off!).",
              file=sys.stderr)

    # Test data -> FARBENCH_TEST_DATA_DIR/
    _copytree(TEST_SRC, test_data_dir)

    _mark_done(data_dir)
    _mark_done(test_data_dir)
    print(f"[prepare] data_dir       = {data_dir}")
    print(f"[prepare] test_data_dir  = {test_data_dir}")
    print("[prepare] ScreenSpot-Pro ready.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[prepare] ERROR: {exc}", file=sys.stderr)
        raise
