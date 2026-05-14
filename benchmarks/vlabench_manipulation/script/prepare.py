"""VLABench data preparation.

Mirrors the bigcodebench / aime_math_rl convention: every byte the agent
needs lives under ``$FARBENCH_DATA_DIR`` (which is bind-mounted into the
sandbox at ``/data``).  prepare.py is the *only* place that may touch
host paths outside the FARBench tree; the agent and the evaluator never see
anything outside ``/data``.

Outputs (all under FARBENCH_DATA_DIR / FARBENCH_TEST_DATA_DIR):

    FARBENCH_DATA_DIR/
        vlabench_root/                    # mujoco scene assets (mesh + xml)
            assets/...
            .prepared
        vlabench_train/                   # staged training episodes
            train/<task_dir>/episode_*.hdf5
            meta/                         # dataset metadata
            schema.json                   # observation/action schema
            dataset_info.json             # CONTAINER-relative paths only
            .prepared
    FARBENCH_TEST_DATA_DIR/
        eval_config.json                  # benchmark task list + seeds

Source of the training episodes is the FARBench Hugging Face dataset:

    https://huggingface.co/datasets/USC-PSI-Lab/FARBench/tree/main/vlabench/train

``prepare.py`` downloads the FARBench VLABench training subtree, stages the
actual training tree into ``FARBENCH_DATA_DIR/vlabench_train/``, validates the
sample HDF5 schema, creates missing companion metadata when the HF subtree
only contains episode files, and removes repository/cache files that the agent
should not see.
``$VLABENCH_LOCAL_TRAIN_DATA`` is still supported as an offline override for
pre-downloaded copies of the same ``vlabench/train`` tree. If the HF repo is
private, set ``HF_TOKEN``, ``HUGGINGFACE_HUB_TOKEN``, or
``HUGGING_FACE_HUB_TOKEN`` before preparing.

Idempotency: a downstream re-run of ``farbench tasks prepare
vlabench_manipulation`` does NOT re-copy anything if
``vlabench_train/.prepared`` already exists.  Force a re-stage with::

    rm -rf <FARBENCH_DATA_DIR>/vlabench_train
"""

from __future__ import annotations

import glob
import json
import os
import shlex
import shutil
import subprocess
import sys
import zipfile

from huggingface_hub import snapshot_download


# ── Evaluation config ────────────────────────────────────────────────────────
# 3 representative primitive tasks, 20 episodes each = 60 total episodes.
# Chosen to cover: tool use (add_condiment), semantic selection (select_fruit,
# select_book). Fixed seed ensures reproducibility across runs.
EVAL_CONFIG = {
    "benchmark_tasks": ["add_condiment", "select_fruit", "select_book"],
    "episodes_per_task": 20,
    "max_steps_per_episode": 250,
    "seed": 42,
    # Use VLABench's official evaluation-track protocol rather than naked
    # load_env(task) random initialization. The agent's predict.py should pass
    # this through to the installed VLABench/LeRobot evaluation code.
    "eval_track": "track_1_in_distribution",
    "eval_track_path": "/opt/VLABench/VLABench/configs/evaluation/tracks/track_1_in_distribution.json",
    "evaluation_protocol": "vlabench_official_track",
}

TRAIN_TASKS = [
    "add_condiment",
    "insert_flower",
    "select_book",
    "select_chemistry_tube",
    "select_drink",
    "select_fruit",
    "select_mahjong",
    "select_painting",
    "select_poker",
    "select_toy",
]
HF_DATASET_REPO = "USC-PSI-Lab/FARBench"
HF_DATASET_SUBDIR = "vlabench/train"
HF_DATASET_URL = (
    f"https://huggingface.co/datasets/{HF_DATASET_REPO}/tree/main/"
    f"{HF_DATASET_SUBDIR}"
)
HF_ALLOW_PATTERNS = [
    f"{HF_DATASET_SUBDIR}/**",
    "vlabench/schema.json",
    "vlabench/meta/**",
]
HF_IGNORE_PATTERNS = [
    "**/.git/**",
    "**/.cache/**",
    "**/__pycache__/**",
    "**/.pytest_cache/**",
    "**/.ipynb_checkpoints/**",
    "**/.DS_Store",
    "**/.gitattributes",
    "**/.gitignore",
    "**/README.md",
]
EXPECTED_VIEW_COUNT = 2
EXPECTED_ACTION_DIM = 7
EXPECTED_STATE_DIM = 8
EXPECTED_SCHEMA_VERSION = "farbench-vla-unified-v1"
EXPECTED_SOURCE_DATASET_FRAGMENT = "libero"
MIN_TRAIN_TASK_DIRS = 40
MIN_TRAIN_EPISODES = 1600
TRAIN_ROOT_KEEP_ENTRIES = {
    "train",
    "meta",
    "schema.json",
    "dataset_info.json",
    ".prepared",
}
ASSET_FILES = {
    "obj.zip": "https://drive.google.com/uc?id=1ldEMZua2OzXHJTYTCP0IGVU1aFYBCMu-",
    "scene.zip": "https://drive.google.com/uc?id=1KdReRkibJClBHHD32jz_wTkaBzhEJ9Kw",
}

# NOTE on the base model:
#   lerobot/pi0_base is NOT staged here. It is pre-downloaded into the
#   image's HF cache at Docker build time (see docker/Dockerfile, the
#   "Pre-download lerobot/pi0_base" RUN step). Agents access it via the
#   canonical HF id `lerobot/pi0_base` from the py312 venv:
#       PI0Policy.from_pretrained("lerobot/pi0_base")
#   This mirrors the aime_math_rl pattern (Qwen/Qwen3-4B in HF cache) and
#   removes prepare.py's old dependency on a 14 GB host-local snapshot at
#   /data/jiajunxu/models/lerobot/pi0_base. If the cache is missing the
#   model (e.g. someone overwrote it), rebuild the task image rather than
#   patching prepare.py to re-download — the build is the single source of
#   truth for what's in the image.


def _sentinel(path: str) -> bool:
    return os.path.exists(os.path.join(path, ".prepared"))


def _mark_done(path: str) -> None:
    sentinel = os.path.join(path, ".prepared")
    if os.path.exists(sentinel):
        return
    open(sentinel, "w").close()


def _write_json_file(
    path: str,
    payload: dict[str, object],
    *,
    allow_compatible_existing_dataset_info: bool = False,
) -> None:
    """Write JSON, preserving idempotency on read-only prepared data trees."""
    existing_payload: object | None = None
    if os.path.exists(path):
        try:
            with open(path) as f:
                existing_payload = json.load(f)
            if existing_payload == payload:
                return
        except Exception:
            existing_payload = None

    try:
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
    except PermissionError as exc:
        if (
            allow_compatible_existing_dataset_info
            and isinstance(existing_payload, dict)
            and _compatible_existing_dataset_info(existing_payload)
        ):
            print(
                f"[prepare] Reusing existing read-only dataset_info.json at {path}; "
                "it is compatible with the current task contract."
            )
            return
        raise RuntimeError(
            f"[prepare] Cannot write {path}. Existing prepared data appears to be "
            "owned by another user; fix file ownership or remove the prepared tree "
            "before regenerating it."
        ) from exc


def _compatible_existing_dataset_info(payload: dict[str, object]) -> bool:
    return (
        payload.get("training_data_prepared") is True
        and payload.get("container_path") == "/data/vlabench_train"
        and payload.get("schema_version") == EXPECTED_SCHEMA_VERSION
        and payload.get("file_pattern") == "train/task_*/episode_*.hdf5"
        and isinstance(payload.get("validation_summary"), dict)
    )


# ── Base model (LeRobot Pi0) ────────────────────────────────────────────────
# Removed `_prepare_base_model`: the previous version copied a 14 GB host-local
# Pi0 snapshot into FARBENCH_DATA_DIR. We now bake the model into the image's HF
# cache at build time (see docker/Dockerfile) and let agents load it via the
# canonical HF id. See the module-level NOTE for rationale.


def _download_assets(vlabench_root: str) -> None:
    """Prepare VLABench assets without importing the full task registry."""
    assets_dir = os.path.join(vlabench_root, "assets")
    local_asset_dir = os.environ.get("VLABENCH_LOCAL_ASSET_DIR", "").strip()

    os.makedirs(assets_dir, exist_ok=True)

    if _looks_like_assets_dir(assets_dir):
        print(f"[prepare] Reusing already extracted assets at {assets_dir}")
        return

    if local_asset_dir:
        if _looks_like_assets_dir(local_asset_dir):
            print(f"[prepare] Reusing local extracted assets: {local_asset_dir}")
            _copy_tree(local_asset_dir, assets_dir)
            return

        if _extract_local_asset_archives(local_asset_dir, assets_dir):
            print(f"[prepare] Extracted local asset archives from {local_asset_dir}")
            return

        raise RuntimeError(
            "VLABENCH_LOCAL_ASSET_DIR is set but does not contain extracted assets "
            f"or obj.zip + scene.zip/scenes.zip archives: {local_asset_dir}"
        )

    print("[prepare] Downloading VLABench assets from official release links ...")
    for archive_name, url in ASSET_FILES.items():
        archive_path = os.path.join(assets_dir, archive_name)
        _download_google_drive_file(url, archive_path)
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(assets_dir)
        os.remove(archive_path)

    if not _looks_like_assets_dir(assets_dir):
        raise RuntimeError(
            f"VLABench asset download finished but {assets_dir} does not look complete."
        )


def _looks_like_assets_dir(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    required_dirs = [
        os.path.join(path, "obj", "meshes"),
        os.path.join(path, "obj", "assets", "textures"),
        os.path.join(path, "scenes", "default"),
    ]
    if not all(os.path.isdir(p) for p in required_dirs):
        return False

    has_mesh_xml = bool(glob.glob(os.path.join(path, "obj", "meshes", "*", "*.xml")))
    has_scene_xml = bool(glob.glob(os.path.join(path, "scenes", "*", "*.xml")))
    return has_mesh_xml and has_scene_xml


def _copy_tree(src: str, dst: str) -> None:
    for entry in os.listdir(src):
        src_path = os.path.join(src, entry)
        dst_path = os.path.join(dst, entry)
        if os.path.isdir(src_path):
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
        else:
            shutil.copy2(src_path, dst_path)


def _extract_local_asset_archives(source_dir: str, assets_dir: str) -> bool:
    archive_candidates = {
        "obj.zip": os.path.join(source_dir, "obj.zip"),
        "scene.zip": None,
    }
    for name in ("scene.zip", "scenes.zip"):
        candidate = os.path.join(source_dir, name)
        if os.path.exists(candidate):
            archive_candidates["scene.zip"] = candidate
            break

    if not archive_candidates["scene.zip"] or not os.path.exists(archive_candidates["obj.zip"]):
        return False

    for archive_path in archive_candidates.values():
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(assets_dir)
    return _looks_like_assets_dir(assets_dir)


def _download_google_drive_file(url: str, output_path: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "gdown", url, "-O", output_path],
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to download asset archive from {url} (exit {result.returncode})"
        )


def _looks_like_training_dataset(path: str) -> bool:
    return not _training_dataset_errors(path, inspect_hdf5=True)


def _has_training_episodes(path: str) -> bool:
    return bool(glob.glob(os.path.join(path, "train", "task_*", "episode_*.hdf5")))


def _has_split_episodes(path: str) -> bool:
    return bool(glob.glob(os.path.join(path, "task_*", "episode_*.hdf5")))


def _training_source_episode_files(path: str) -> list[str]:
    if _has_training_episodes(path):
        return sorted(glob.glob(os.path.join(path, "train", "task_*", "episode_*.hdf5")))
    return sorted(glob.glob(os.path.join(path, "task_*", "episode_*.hdf5")))


def _training_source_task_dirs(path: str) -> list[str]:
    if _has_training_episodes(path):
        return sorted(glob.glob(os.path.join(path, "train", "task_*")))
    return sorted(glob.glob(os.path.join(path, "task_*")))


def _looks_like_training_source(path: str) -> bool:
    """Accept either a full dataset root or the train split root itself."""
    if not os.path.isdir(path):
        return False
    return (
        len(_training_source_task_dirs(path)) >= MIN_TRAIN_TASK_DIRS
        and len(_training_source_episode_files(path)) >= MIN_TRAIN_EPISODES
    )


def _preserve_reusable_hf_download(train_root: str) -> str | None:
    """Move a complete failed-run HF download out of train_root before clearing."""
    existing_download_root = os.path.join(train_root, "_hf_download")
    if not _find_training_tree(existing_download_root):
        return None

    reuse_root = os.path.join(os.path.dirname(train_root), "_vlabench_train_hf_reuse")
    shutil.rmtree(reuse_root, ignore_errors=True)
    shutil.move(existing_download_root, reuse_root)
    print(f"[prepare] Reusing completed HF download from previous failed run: {reuse_root}")
    return reuse_root


def _training_dataset_errors(path: str, *, inspect_hdf5: bool) -> list[str]:
    """Validate the FARBench VLABench training tree.

    The training data for this FARBench task is the unified HDF5 conversion of
    LeRobot LIBERO episodes. It is intentionally different from the live
    VLABench simulator observation/action surface used during evaluation.
    """
    errors: list[str] = []
    if not os.path.isdir(path):
        return [f"missing directory: {path}"]

    train_dir = os.path.join(path, "train")
    if not os.path.isdir(train_dir):
        errors.append("missing train/ directory")

    task_dirs = sorted(glob.glob(os.path.join(train_dir, "task_*")))
    if len(task_dirs) < MIN_TRAIN_TASK_DIRS:
        errors.append(
            f"expected at least {MIN_TRAIN_TASK_DIRS} task_* directories, "
            f"found {len(task_dirs)}"
        )

    hdf5_files = sorted(glob.glob(os.path.join(train_dir, "task_*", "episode_*.hdf5")))
    if not hdf5_files:
        errors.append("missing train/task_*/episode_*.hdf5 files")
    elif len(hdf5_files) < MIN_TRAIN_EPISODES:
        errors.append(
            f"expected at least {MIN_TRAIN_EPISODES} episode_*.hdf5 files, "
            f"found {len(hdf5_files)}"
        )

    schema_path = os.path.join(path, "schema.json")
    if not os.path.isfile(schema_path):
        errors.append("missing schema.json")

    tasks_path = os.path.join(path, "meta", "tasks.json")
    if not os.path.isfile(tasks_path):
        errors.append("missing meta/tasks.json")

    source_info_path = os.path.join(path, "meta", "dataset_info.json")
    if os.path.isfile(source_info_path):
        try:
            with open(source_info_path) as f:
                json.load(f)
        except Exception as exc:  # noqa: BLE001 - surface malformed metadata.
            errors.append(f"cannot parse meta/dataset_info.json: {exc}")

    if not inspect_hdf5 or not hdf5_files:
        return errors

    try:
        import h5py  # type: ignore
    except Exception as exc:  # noqa: BLE001 - dependency failure is actionable.
        return errors + [f"cannot import h5py to validate sample episode: {exc}"]

    sample_path = hdf5_files[0]
    try:
        with h5py.File(sample_path, "r") as f:
            for key in [
                "observation/rgb",
                "observation/state",
                "action",
                "task_index",
                "instruction",
                "source_dataset",
            ]:
                if key not in f:
                    errors.append(f"sample episode missing dataset {key}")

            if "observation/rgb" in f:
                rgb_shape = tuple(f["observation/rgb"].shape)
                if len(rgb_shape) != 5:
                    errors.append(f"observation/rgb should be 5-D, got {rgb_shape}")
                else:
                    if rgb_shape[1] != EXPECTED_VIEW_COUNT:
                        errors.append(
                            "observation/rgb should contain "
                            f"{EXPECTED_VIEW_COUNT} camera views, got {rgb_shape[1]}"
                        )
                    if rgb_shape[-1] != 3:
                        errors.append(f"observation/rgb should be RGB, got shape {rgb_shape}")

            if "observation/state" in f:
                state_shape = tuple(f["observation/state"].shape)
                if len(state_shape) != 2:
                    errors.append(f"observation/state should be 2-D, got {state_shape}")
                elif state_shape[-1] != EXPECTED_STATE_DIM:
                    errors.append(
                        f"observation/state should be {EXPECTED_STATE_DIM}-dim, "
                        f"got {state_shape[-1]}"
                    )

            if "action" in f:
                action_shape = tuple(f["action"].shape)
                if len(action_shape) != 2:
                    errors.append(f"action should be 2-D, got {action_shape}")
                elif action_shape[-1] != EXPECTED_ACTION_DIM:
                    errors.append(
                        f"action should be {EXPECTED_ACTION_DIM}-dim, "
                        f"got {action_shape[-1]}"
                    )

            source_dataset = _read_h5_scalar_text(f, "source_dataset").lower()
            if not source_dataset:
                errors.append("source_dataset is empty")
            elif EXPECTED_SOURCE_DATASET_FRAGMENT not in source_dataset:
                errors.append(
                    "sample episode source_dataset should identify the "
                    f"converted LIBERO source, got: {source_dataset}"
                )

            attr_action_dim = f.attrs.get("action_dim")
            if attr_action_dim is not None and int(attr_action_dim) != EXPECTED_ACTION_DIM:
                errors.append(
                    f"sample episode attr action_dim should be {EXPECTED_ACTION_DIM}, "
                    f"got {attr_action_dim}"
                )

            attr_state_dim = f.attrs.get("state_dim")
            if attr_state_dim is not None and int(attr_state_dim) != EXPECTED_STATE_DIM:
                errors.append(
                    f"sample episode attr state_dim should be {EXPECTED_STATE_DIM}, "
                    f"got {attr_state_dim}"
                )

            schema_version = str(f.attrs.get("schema_version", ""))
            if schema_version and schema_version != EXPECTED_SCHEMA_VERSION:
                errors.append(
                    f"sample episode schema_version should be {EXPECTED_SCHEMA_VERSION}, "
                    f"got {schema_version}"
                )
    except Exception as exc:  # noqa: BLE001 - corrupt HDF5 should fail prepare.
        errors.append(f"cannot inspect sample episode {sample_path}: {exc}")

    return errors


def _read_h5_scalar_text(h5_file, key: str) -> str:
    if key not in h5_file:
        return ""
    value = h5_file[key][()]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "item"):
        try:
            value = value.item()
            if isinstance(value, bytes):
                return value.decode("utf-8", errors="replace")
        except Exception:
            pass
    return str(value)


# ── Stage training data into FARBENCH_DATA_DIR ────────────────────────────────────
def _stage_training_data(train_root: str) -> str:
    """Stage VLABench training data into ``train_root``.

    Returns a human-readable source label for the FARBench manifest.
    """
    sentinel = os.path.join(train_root, ".prepared")

    if os.path.exists(sentinel) and _looks_like_training_dataset(train_root):
        print(f"[prepare] Reusing already-staged training data at {train_root}")
        return f"already staged from {HF_DATASET_URL}"

    local_src = os.environ.get("VLABENCH_LOCAL_TRAIN_DATA", "").strip()
    if local_src:
        return _stage_training_data_from_local(local_src, train_root)

    return _stage_training_data_from_hf(train_root)


def _stage_training_data_from_hf(train_root: str) -> str:
    """Download USC-PSI-Lab/FARBench/vlabench training files and stage them."""
    reuse_download_root = _preserve_reusable_hf_download(train_root)
    _clear_training_root(train_root)
    download_root = reuse_download_root or os.path.join(train_root, "_hf_download")
    if not reuse_download_root:
        os.makedirs(download_root, exist_ok=True)

        print(f"[prepare] Downloading VLABench train data from {HF_DATASET_URL}")
        token = (
            os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGINGFACE_HUB_TOKEN")
            or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            or None
        )
        snapshot_download(
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            allow_patterns=HF_ALLOW_PATTERNS,
            ignore_patterns=HF_IGNORE_PATTERNS,
            local_dir=download_root,
            token=token,
        )

    source_tree = _find_training_tree(download_root)
    if not source_tree:
        errors = _training_dataset_errors(
            os.path.join(download_root, HF_DATASET_SUBDIR),
            inspect_hdf5=False,
        )
        raise RuntimeError(
            "[prepare] Hugging Face download finished, but no valid VLABench "
            "training tree was found.\n"
            f"  repo: {HF_DATASET_REPO}\n"
            f"  subdir: {HF_DATASET_SUBDIR}\n"
            f"  expected: train/task_*/episode_*.hdf5 plus schema/meta metadata\n"
            f"  errors: {errors}"
        )

    _move_training_tree(source_tree, train_root)
    shutil.rmtree(download_root, ignore_errors=True)
    _ensure_companion_metadata(train_root, source_label=HF_DATASET_URL)
    _cleanup_training_root(train_root)
    _finalize_training_root(train_root)
    return HF_DATASET_URL


def _stage_training_data_from_local(local_src: str, train_root: str) -> str:
    """Stage a pre-downloaded local copy of the HF vlabench/train tree."""
    source_tree = _find_training_tree(local_src)
    if not source_tree:
        errors = _training_dataset_errors(local_src, inspect_hdf5=False)
        raise RuntimeError(
            "[prepare] VLABENCH_LOCAL_TRAIN_DATA does not contain a valid "
            "VLABench training tree.\n"
            f"  source: {local_src}\n"
            f"  expected: train/task_*/episode_*.hdf5 plus schema/meta metadata\n"
            f"  errors: {errors}"
        )

    src_abs = os.path.abspath(source_tree)
    dst_abs = os.path.abspath(train_root)
    if src_abs == dst_abs:
        _ensure_companion_metadata(train_root, source_label=f"local override {source_tree}")
        _cleanup_training_root(train_root)
        _finalize_training_root(train_root)
        return f"local override {source_tree}"

    _clear_training_root(train_root)
    _copy_training_tree(source_tree, train_root)
    _ensure_companion_metadata(train_root, source_label=f"local override {source_tree}")
    _cleanup_training_root(train_root)
    _finalize_training_root(train_root)
    return f"local override {source_tree}"


def _find_training_tree(root: str) -> str | None:
    """Find a valid training tree, accepting both repo-root and subdir inputs."""
    if not root or not os.path.isdir(root):
        return None

    direct_candidates = [
        root,
        os.path.join(root, "vlabench"),
        os.path.join(root, HF_DATASET_SUBDIR),
        os.path.join(root, "vlabench_train"),
        os.path.join(root, "vlabench", "train"),
    ]
    for candidate in direct_candidates:
        if _looks_like_training_source(candidate):
            return candidate

    root_abs = os.path.abspath(root)
    for current, dirs, _files in os.walk(root_abs):
        rel = os.path.relpath(current, root_abs)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        if depth > 4:
            dirs[:] = []
            continue
        if _looks_like_training_source(current):
            return current
    return None


def _copy_training_tree(local_src: str, train_root: str) -> None:
    """Copy a validated local training tree into ``train_root``."""

    # Free-space guard: refuse to start a large local copy if the destination
    # filesystem can't hold it. Better to fail fast in prepare than to produce
    # a half-copied tree and have the agent's HDF5 reader die mid-run.
    src_bytes = _dir_size(local_src)
    free_bytes = shutil.disk_usage(os.path.dirname(train_root) or "/").free
    # 1.10× safety margin: rsync's transient temp files and fs metadata
    # add a few percent on top of the raw source size.
    if free_bytes < int(src_bytes * 1.10):
        raise RuntimeError(
            f"[prepare] Not enough space to stage training data:\n"
            f"  source ({local_src}): {src_bytes / 1e9:.1f} GB\n"
            f"  free at destination ({train_root}): {free_bytes / 1e9:.1f} GB\n"
            f"  required (1.10× source): {src_bytes * 1.10 / 1e9:.1f} GB\n"
            "Either free space on the destination filesystem or set "
            "FARBENCH_DATA_DIR to a different mount."
        )

    print(
        f"[prepare] Staging training data: {local_src} → {train_root} "
        f"({src_bytes / 1e9:.1f} GB, this can take 5–15 min on a single SSD)"
    )

    # Prefer rsync — gives progress, atomic per-file replace, and resumes
    # cleanly if interrupted (subsequent run will only copy missing files,
    # then we mark .prepared). Fall back to shutil.copytree on minimal
    # systems without rsync.
    # Exclude developer / VCS junk from local clones or downloaded snapshots.
    # Keep this list narrow: anything excluded here MUST be inferred from a
    # file the agent actually reads, or we'll hide a real dependency.
    exclude_patterns = [
        ".git",
        ".gitattributes",
        ".gitignore",
        ".gitmodules",
        ".github",
        "__pycache__",
        ".pytest_cache",
        ".DS_Store",
        ".ipynb_checkpoints",
    ]

    rsync_bin = shutil.which("rsync")
    dst_base = (
        os.path.join(train_root, "train")
        if _has_split_episodes(local_src) and not _has_training_episodes(local_src)
        else train_root
    )
    os.makedirs(dst_base, exist_ok=True)
    if rsync_bin:
        # Note the trailing slashes: "src/ → dst/" copies CONTENTS into dst,
        # not "dst/<basename(src)>/". Both must end in "/".
        rsync_cmd = [
            rsync_bin,
            "-aH",                  # archive + preserve hardlinks within src
            "--info=progress2",     # human-readable progress on stdout
            "--no-inc-recursive",   # accurate progress requires up-front scan
        ]
        for pat in exclude_patterns:
            rsync_cmd.extend(["--exclude", pat])
        rsync_cmd.extend([
            os.path.join(local_src, ""),
            os.path.join(dst_base, ""),
        ])
        print(f"[prepare] $ {' '.join(shlex.quote(a) for a in rsync_cmd)}")
        result = subprocess.run(rsync_cmd, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                f"rsync staging failed (exit {result.returncode}); "
                f"partial tree at {train_root} may need manual cleanup."
            )
    else:
        print("[prepare] rsync not found — falling back to shutil.copytree (no progress, slower)")
        ignore = shutil.ignore_patterns(*exclude_patterns)
        for entry in os.listdir(local_src):
            if entry in exclude_patterns:
                continue
            src_path = os.path.join(local_src, entry)
            dst_path = os.path.join(dst_base, entry)
            if os.path.isdir(src_path):
                shutil.copytree(src_path, dst_path, dirs_exist_ok=True, ignore=ignore)
            else:
                shutil.copy2(src_path, dst_path)

def _clear_training_root(train_root: str) -> None:
    """Remove stale contents created by earlier prepare versions."""
    if os.path.basename(os.path.abspath(train_root)) != "vlabench_train":
        raise RuntimeError(f"refusing to clear unexpected training root: {train_root}")
    os.makedirs(train_root, exist_ok=True)
    for entry in os.listdir(train_root):
        path = os.path.join(train_root, entry)
        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path)
        else:
            os.remove(path)


def _move_contents(src: str, dst: str) -> None:
    os.makedirs(dst, exist_ok=True)
    for entry in os.listdir(src):
        src_path = os.path.join(src, entry)
        dst_path = os.path.join(dst, entry)
        if os.path.exists(dst_path):
            if os.path.isdir(dst_path) and not os.path.islink(dst_path):
                shutil.rmtree(dst_path)
            else:
                os.remove(dst_path)
        shutil.move(src_path, dst_path)


def _move_training_tree(src: str, dst: str) -> None:
    """Move either a dataset root or a bare train split into ``dst``."""
    if _has_split_episodes(src) and not _has_training_episodes(src):
        split_dst = os.path.join(dst, "train")
        os.makedirs(split_dst, exist_ok=True)
        _move_contents(src, split_dst)
        return

    _move_contents(src, dst)


def _cleanup_training_root(train_root: str) -> None:
    """Keep only files the agent needs at runtime."""
    for entry in list(os.listdir(train_root)):
        if entry in TRAIN_ROOT_KEEP_ENTRIES:
            continue
        path = os.path.join(train_root, entry)
        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path)
        else:
            os.remove(path)


def _finalize_training_root(train_root: str) -> None:
    errors = _training_dataset_errors(train_root, inspect_hdf5=True)
    if errors:
        raise RuntimeError(
            f"Staging finished but {train_root} does not pass the VLABench "
            f"training-data validation:\n  - " + "\n  - ".join(errors)
        )

    _mark_done(train_root)
    print(f"[prepare] ✓ Staged and validated: {train_root}")


def _ensure_companion_metadata(train_root: str, *, source_label: str) -> None:
    """Create schema/meta files when the HF subtree contains only episodes."""
    os.makedirs(os.path.join(train_root, "meta"), exist_ok=True)

    hdf5_files = sorted(glob.glob(os.path.join(train_root, "train", "task_*", "episode_*.hdf5")))
    task_dirs = sorted(glob.glob(os.path.join(train_root, "train", "task_*")))
    task_records: list[dict[str, object]] = []
    source_dataset = ""
    sample_attrs: dict[str, object] = {}

    try:
        import h5py  # type: ignore
    except Exception:
        h5py = None  # type: ignore

    if h5py and hdf5_files:
        for task_dir in task_dirs:
            task_files = sorted(glob.glob(os.path.join(task_dir, "episode_*.hdf5")))
            instruction = ""
            task_index: int | None = None
            if task_files:
                with h5py.File(task_files[0], "r") as f:
                    instruction = _read_h5_scalar_text(f, "instruction")
                    if "task_index" in f:
                        task_index = int(f["task_index"][()])
                    if not source_dataset:
                        source_dataset = _read_h5_scalar_text(f, "source_dataset")
                    if not sample_attrs:
                        sample_attrs = {
                            "fps": float(f.attrs["fps"]) if "fps" in f.attrs else None,
                            "schema_version": str(f.attrs.get("schema_version", EXPECTED_SCHEMA_VERSION)),
                            "state_dim": int(f.attrs["state_dim"]) if "state_dim" in f.attrs else EXPECTED_STATE_DIM,
                            "action_dim": int(f.attrs["action_dim"]) if "action_dim" in f.attrs else EXPECTED_ACTION_DIM,
                            "view_names_json": str(f.attrs.get("view_names_json", '["image", "image2"]')),
                        }
            task_records.append(
                {
                    "task_dir": os.path.basename(task_dir),
                    "task_index": task_index,
                    "num_episodes": len(task_files),
                    "instruction": instruction,
                }
            )

    schema_path = os.path.join(train_root, "schema.json")
    if not os.path.isfile(schema_path):
        schema = {
            "schema_version": EXPECTED_SCHEMA_VERSION,
            "format": "FARBench unified HDF5 episode",
            "source_dataset": source_dataset or "lerobot/libero",
            "datasets": {
                "observation/rgb": {
                    "dtype": "uint8",
                    "shape": ["T", EXPECTED_VIEW_COUNT, "H", "W", 3],
                    "view_names": ["image", "image2"],
                },
                "observation/state": {
                    "dtype": "float32",
                    "shape": ["T", EXPECTED_STATE_DIM],
                },
                "action": {
                    "dtype": "float32",
                    "shape": ["T", EXPECTED_ACTION_DIM],
                },
                "task_index": {"dtype": "int64", "shape": []},
                "instruction": {"dtype": "utf8", "shape": []},
                "source_dataset": {"dtype": "utf8", "shape": []},
            },
            "attrs": {
                "fps": "float",
                "schema_version": EXPECTED_SCHEMA_VERSION,
                "state_dim": EXPECTED_STATE_DIM,
                "action_dim": EXPECTED_ACTION_DIM,
                "num_frames": "int",
                "view_names_json": '["image", "image2"]',
            },
        }
        with open(schema_path, "w") as f:
            json.dump(schema, f, indent=2)

    tasks_path = os.path.join(train_root, "meta", "tasks.json")
    if not os.path.isfile(tasks_path):
        with open(tasks_path, "w") as f:
            json.dump(
                {
                    "num_tasks": len(task_records),
                    "task_index_source": "HDF5 task_index scalar",
                    "tasks": task_records,
                },
                f,
                indent=2,
            )

    source_info_path = os.path.join(train_root, "meta", "dataset_info.json")
    if not os.path.isfile(source_info_path):
        with open(source_info_path, "w") as f:
            json.dump(
                {
                    "source": source_label,
                    "source_dataset": source_dataset or "lerobot/libero",
                    "format": "FARBench unified HDF5 conversion of LeRobot LIBERO episodes",
                    "schema_version": sample_attrs.get("schema_version", EXPECTED_SCHEMA_VERSION),
                    "num_task_dirs": len(task_dirs),
                    "num_episode_files": len(hdf5_files),
                    "sample_attrs": sample_attrs,
                },
                f,
                indent=2,
            )


def _dir_size(path: str) -> int:
    """Sum of regular-file sizes under ``path`` (bytes). Symlinks not followed."""
    total = 0
    for cur, _dirs, files in os.walk(path):
        for f in files:
            fp = os.path.join(cur, f)
            try:
                if not os.path.islink(fp):
                    total += os.path.getsize(fp)
            except OSError:
                continue
    return total


def _training_dataset_summary(train_root: str) -> dict[str, object]:
    hdf5_files = sorted(glob.glob(os.path.join(train_root, "train", "task_*", "episode_*.hdf5")))
    task_dirs = sorted(glob.glob(os.path.join(train_root, "train", "task_*")))
    summary: dict[str, object] = {
        "num_task_dirs": len(task_dirs),
        "num_episode_files": len(hdf5_files),
        "sample_episode": None,
    }

    if not hdf5_files:
        return summary

    try:
        import h5py  # type: ignore
        with h5py.File(hdf5_files[0], "r") as f:
            rel_sample = os.path.relpath(hdf5_files[0], train_root)
            summary.update(
                {
                    "sample_episode": rel_sample,
                    "rgb_shape": list(f["observation/rgb"].shape),
                    "state_shape": list(f["observation/state"].shape),
                    "action_shape": list(f["action"].shape),
                    "source_dataset": _read_h5_scalar_text(f, "source_dataset"),
                    "attrs": {
                        "fps": float(f.attrs["fps"]) if "fps" in f.attrs else None,
                        "action_dim": int(f.attrs["action_dim"]) if "action_dim" in f.attrs else None,
                        "state_dim": int(f.attrs["state_dim"]) if "state_dim" in f.attrs else None,
                        "num_frames": int(f.attrs["num_frames"]) if "num_frames" in f.attrs else None,
                        "view_names_json": str(f.attrs["view_names_json"]) if "view_names_json" in f.attrs else None,
                    },
                }
            )
    except Exception as exc:  # noqa: BLE001 - validation already caught hard failures.
        summary["sample_read_error"] = str(exc)
    return summary


def _write_dataset_info(
    train_root: str,
    *,
    source_label: str,
    summary: dict[str, object],
) -> None:
    """Write the manifest the agent reads at iteration 1.

    All paths in the manifest are CONTAINER-relative (under /data), never
    host paths — the agent has no way to access host paths and embedding
    them here only invites the kind of confusion the previous version of
    this file caused.
    """
    info_path = os.path.join(train_root, "dataset_info.json")
    payload: dict[str, object] = {
        "training_data_prepared": True,
        "default_dataset": "vlabench_train",
        "source": source_label,
        "hf_repo_id": HF_DATASET_REPO,
        "hf_subdir": HF_DATASET_SUBDIR,
        "format": "FARBench unified HDF5 episodes converted from LeRobot LIBERO",
        "training_data_role": "auxiliary offline manipulation imitation data",
        "container_path": "/data/vlabench_train",
        "schema_version": "farbench-vla-unified-v1",
        "file_pattern": "train/task_*/episode_*.hdf5",
        "alignment_notes": [
            (
                "This is converted LeRobot LIBERO data, not VLABench's "
                "official primitive fine-tuning dataset."
            ),
            (
                "Training code must adapt the two-camera, 8-D state, 7-D "
                "EE-action convention to VLABench runtime observations and "
                "actions during evaluation."
            ),
        ],
        "cleanup_policy": (
            "prepare.py keeps train/, meta/, schema.json, dataset_info.json, "
            "and .prepared only; HF cache, README files, git metadata, and "
            "stale nested datasets are removed from /data/vlabench_train."
        ),
        "reference_vlabench_eval_tasks": TRAIN_TASKS,
        "minimum_training_task_dirs": MIN_TRAIN_TASK_DIRS,
        "minimum_training_episodes": MIN_TRAIN_EPISODES,
        "required_episode_datasets": {
            "observation/rgb": {
                "dtype": "uint8",
                "shape": ["T", EXPECTED_VIEW_COUNT, "H", "W", 3],
                "notes": "Two LIBERO camera streams. Read H/W from the file; do not assume the live simulator observation dict has the same image keys.",
            },
            "observation/state": {
                "dtype": "float32",
                "shape": ["T", EXPECTED_STATE_DIM],
                "notes": "Proprioceptive state exported by the data converter.",
            },
            "action": {
                "dtype": "float32",
                "shape": ["T", EXPECTED_ACTION_DIM],
                "notes": (
                    "Demonstration action. The official VLABench Evaluator "
                    "accepts EE-control policy outputs and converts them to "
                    "robot qpos internally. If predict.py bypasses that "
                    "Evaluator and calls env.step(...) directly, it must "
                    "adapt this 7-D EE convention to the 9-D Franka action spec "
                    "(7 joints + 2 fingers)."
                ),
            },
            "task_index": {"dtype": "int64", "shape": []},
            "instruction": {"dtype": "utf8", "shape": []},
            "source_dataset": {
                "dtype": "utf8",
                "shape": [],
                "notes": "Expected to identify the converted LIBERO source.",
            },
        },
        "required_episode_attrs": {
            "fps": "float",
            "view_names_json": "JSON array of two camera/view names",
            "state_dim": EXPECTED_STATE_DIM,
            "action_dim": EXPECTED_ACTION_DIM,
            "num_frames": "int",
        },
        "companion_files": {
            "schema": "/data/vlabench_train/schema.json",
            "task_metadata": "/data/vlabench_train/meta/tasks.json",
            "source_metadata": "/data/vlabench_train/meta/dataset_info.json",
        },
        "validation_summary": summary,
    }

    os.makedirs(train_root, exist_ok=True)
    _write_json_file(
        info_path,
        payload,
        allow_compatible_existing_dataset_info=True,
    )


def main() -> None:
    """Main entry point for data preparation.

    Layout produced (all under FARBENCH_DATA_DIR, mounted as /data inside the
    sandbox; bigcodebench / aime_math_rl follow the same convention):

        $FARBENCH_DATA_DIR/
            vlabench_root/assets/...     # mujoco scene assets
            vlabench_root/.prepared
            vlabench_train/train/<task>/episode_*.hdf5
            vlabench_train/meta/...
            vlabench_train/schema.json
            vlabench_train/dataset_info.json
            vlabench_train/.prepared
        $FARBENCH_TEST_DATA_DIR/
            eval_config.json
    """
    farbench_data_dir = os.environ["FARBENCH_DATA_DIR"]
    farbench_test_dir = os.environ["FARBENCH_TEST_DATA_DIR"]

    # 1) VLABench mujoco assets — already worked, leave alone
    vlabench_root = os.path.join(farbench_data_dir, "vlabench_root")
    _download_assets(vlabench_root)
    _mark_done(vlabench_root)

    # 2) Pi0 base model lives in the image's HF cache (see Dockerfile NOTE)

    # 3) Eval config (test side)
    os.makedirs(farbench_test_dir, exist_ok=True)
    eval_config_path = os.path.join(farbench_test_dir, "eval_config.json")
    _write_json_file(eval_config_path, EVAL_CONFIG)

    # 4) Stage training episodes into FARBENCH_DATA_DIR/vlabench_train. For offline
    # rebuilds, VLABENCH_LOCAL_TRAIN_DATA may point to a pre-downloaded copy of
    # the same tree.
    train_root = os.path.join(farbench_data_dir, "vlabench_train")
    os.makedirs(train_root, exist_ok=True)

    source_label = _stage_training_data(train_root)
    summary = _training_dataset_summary(train_root)

    # 5) Manifest. At this point training data is guaranteed to be staged:
    #    _stage_training_data fails fast on missing/invalid sources.
    _write_dataset_info(
        train_root,
        source_label=source_label,
        summary=summary,
    )

    print(f"[prepare] ✓ Assets prepared:    {vlabench_root}")
    print(f"[prepare] ✓ Base model:         lerobot/pi0_base (pre-cached in image HF cache)")
    print(f"[prepare] ✓ Eval config:        {eval_config_path}")
    print(f"[prepare] ✓ Dataset info:       {os.path.join(train_root, 'dataset_info.json')}")
    print(f"[prepare] ✓ Training episodes:  {train_root}  (container path: /data/vlabench_train)")


if __name__ == "__main__":
    main()
