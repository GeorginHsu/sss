"""TaskRegistry + TaskPreparer + PrepareResult."""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass, field

from farbench.schemas import TaskConfig
from farbench.utils import get_logger

logger = get_logger(__name__)

# CUDA variants distributed for task images. Update this whitelist only when
# matching Hugging Face dataset archives have been published.
CUDA_VARIANTS: frozenset[str] = frozenset({"cu118", "cu128"})

_CUDA_SUFFIX_RE = re.compile(r"-cu\d+$")
_DOCKER_LOAD_IMAGE_RE = re.compile(r"Loaded image:\s*(\S+)")
_DOCKER_LOAD_IMAGE_ID_RE = re.compile(r"Loaded image ID:\s*(\S+)")
_DEFAULT_HF_IMAGE_DATASET_REPO = "FARBenchAnonymous/FARBench"
_DEFAULT_HF_IMAGE_REPOSITORY = "farbench/farbench"
_DEFAULT_HF_IMAGE_CACHE_DIR = "farbench-images"
_DATA_POLICY_VERSION = 2


def _validate_cuda(cuda: str) -> None:
    """Raise ValueError if `cuda` is not in the whitelist."""
    if not cuda:
        raise ValueError("CUDA variant required (got empty string)")
    if cuda not in CUDA_VARIANTS:
        raise ValueError(
            f"Unknown CUDA variant {cuda!r}; expected one of {sorted(CUDA_VARIANTS)}"
        )


def _apply_cuda_variant(tag: str, cuda: str) -> str:
    """Rewrite the trailing `-cuNNN` in a Docker tag to match `cuda`.

    Pass-through on empty input (no tag configured).
    Raises ValueError if the tag has no trailing `-cuNNN`.
    """
    if not tag:
        return tag
    if not _CUDA_SUFFIX_RE.search(tag):
        raise ValueError(
            f"tag {tag!r} is missing CUDA variant suffix (expected trailing -cuNNN)"
        )
    return _CUDA_SUFFIX_RE.sub(f"-{cuda}", tag)


def _infer_cuda_variant(tag: str) -> str:
    """Infer the trailing CUDA variant from a Docker tag."""
    match = _CUDA_SUFFIX_RE.search(tag or "")
    if not match:
        raise ValueError(
            f"tag {tag!r} is missing CUDA variant suffix (expected trailing -cuNNN)"
        )
    return match.group()[1:]


def _hf_image_dataset_repo() -> str:
    return os.environ.get("FARBENCH_HF_IMAGE_DATASET_REPO", _DEFAULT_HF_IMAGE_DATASET_REPO)


def _hf_image_repository() -> str:
    return os.environ.get("FARBENCH_HF_IMAGE_REPOSITORY", _DEFAULT_HF_IMAGE_REPOSITORY)


def _hf_image_cache_dir() -> str:
    return os.path.abspath(
        os.environ.get("FARBENCH_HF_IMAGE_CACHE_DIR", _DEFAULT_HF_IMAGE_CACHE_DIR)
    )


def _hf_download_command() -> list[str]:
    for executable in ("hf", "huggingface-cli"):
        if shutil.which(executable):
            return [executable, "download"]
    raise RuntimeError(
        "Hugging Face CLI not found. Install huggingface_hub so that "
        "`hf download` or `huggingface-cli download` is available."
    )


@dataclass
class PrepareResult:
    success: bool
    steps_completed: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    data_size_gb: float = 0.0
    docker_image_id: str = ""
    total_time_minutes: float = 0.0


class TaskRegistry:
    """Scans the benchmarks/ directory to discover all tasks."""

    def __init__(self, benchmarks_dir: str = "benchmarks"):
        self.benchmarks_dir = os.path.abspath(benchmarks_dir)
        self._tasks: dict[str, TaskConfig] = {}

    def discover(self) -> None:
        """Scan all subdirectories for task.yaml and fail on any load error."""
        self._tasks.clear()
        if not os.path.isdir(self.benchmarks_dir):
            raise FileNotFoundError(f"Benchmarks dir not found: {self.benchmarks_dir}")

        errors: list[str] = []
        for entry in sorted(os.listdir(self.benchmarks_dir)):
            task_dir = os.path.join(self.benchmarks_dir, entry)
            yaml_path = os.path.join(task_dir, "task.yaml")
            if os.path.isdir(task_dir) and os.path.isfile(yaml_path):
                try:
                    config = TaskConfig.from_yaml(yaml_path)
                    if config.name != entry:
                        errors.append(
                            f"Task name mismatch in {yaml_path}: "
                            f"name={config.name!r}, directory={entry!r}"
                        )
                        continue
                    if config.name in self._tasks:
                        errors.append(
                            f"Duplicate task name {config.name!r} in {yaml_path}"
                        )
                        continue
                    self._tasks[config.name] = config
                    logger.info(f"Discovered task: {config.name} ({config.compute_type})")
                except Exception as e:
                    errors.append(f"Failed to load task from {yaml_path}: {e}")

        if errors:
            details = "\n".join(f"- {msg}" for msg in errors)
            raise RuntimeError(f"Task discovery failed:\n{details}")

    def get(self, task_name: str) -> TaskConfig:
        if task_name not in self._tasks:
            raise KeyError(
                f"Task '{task_name}' not found. "
                f"Available: {list(self._tasks.keys())}"
            )
        return self._tasks[task_name]

    def list_all(self) -> list[dict]:
        return [
            {
                "name": c.name,
                "domain": c.domain,
                "subdomain": c.subdomain,
                "description": c.description.strip().split("\n")[0],
                "compute_type": c.compute_type,
                "primary_metric": c.primary_metric,
                "network_access": c.network_access,
                "total_time_budget_hours": c.total_time_budget_hours,
            }
            for c in self._tasks.values()
        ]


class TaskPreparer:
    """Prepares a task before first run from a published HF image archive."""

    def __init__(self, task_config: TaskConfig):
        self.task_config = task_config

    @staticmethod
    def _prepare_timeout_seconds() -> int:
        value = os.environ.get("FARBENCH_PREPARE_TIMEOUT_SECONDS", "14400")
        try:
            return max(1, int(value))
        except ValueError:
            logger.warning(f"Invalid FARBENCH_PREPARE_TIMEOUT_SECONDS={value!r}; using 14400")
            return 14400

    def prepare(self, force: bool = False, cuda_suffix: str = "") -> PrepareResult:
        """Run the full preparation pipeline.

        Load the published task Docker image archive from the Hugging Face
        dataset and verify its packaged train/test data is present in-image.

        Args:
            force: Re-prepare even if already prepared.
            cuda_suffix: CUDA variant suffix (e.g. "cu128"). When set,
                ``docker_image`` has its trailing ``-cuNNN`` rewritten to
                ``-{cuda_suffix}``. task.yaml stores ``-cu118`` canonically;
                runtime swaps to ``-cu128`` when requested.
        """
        # task.yaml stores canonical cu118 tags; regex swap handles cu128.
        if cuda_suffix:
            _validate_cuda(cuda_suffix)
            old = self.task_config.docker_image
            new = _apply_cuda_variant(old, cuda_suffix)
            if new != old:
                logger.info(f"CUDA variant (docker_image): {old} -> {new}")
                self.task_config.docker_image = new

        if not cuda_suffix and self.task_config.docker_image:
            cuda_suffix = _infer_cuda_variant(self.task_config.docker_image)

        start = time.time()
        result = PrepareResult(success=True)
        task_name = self.task_config.name

        logger.info("=" * 50)
        logger.info(f"Starting preparation for task: {task_name}")
        logger.info("Mode: HF prebuilt image archive")
        logger.info("=" * 50)

        if not force and self.check_status(cuda_variant=cuda_suffix):
            logger.info(f"Task {task_name} already prepared. Use --force to re-prepare.")
            result.steps_completed.append("already_prepared")
            return result

        if not self.task_config.docker_image:
            result.errors.append("docker_image is required in task.yaml")
            result.success = False
            return result

        try:
            if self._docker_image_exists(self.task_config.docker_image):
                logger.info(
                    f"[{task_name}] Using local Docker image "
                    f"{self.task_config.docker_image}"
                )
                result.steps_completed.append("docker_image_available")
            else:
                archive_name = self._hf_image_filename(cuda_suffix)
                logger.info(
                    f"[{task_name}] Loading pre-built image from Hugging Face dataset "
                    f"{_hf_image_dataset_repo()} ({archive_name}) ..."
                )
                archive_path = self._download_hf_image_archive(cuda_suffix)
                result.steps_completed.append("download_hf_image")
                self._load_hf_image_archive(archive_path, cuda_suffix)
                result.steps_completed.append("load_hf_image")

            logger.info(f"[{task_name}] Verifying in-image train/test data paths ...")
            self._verify_image_data_paths()
            result.steps_completed.append("verify_image_data")
            logger.info(f"[{task_name}] In-image data paths verified")
        except Exception as e:
            archive_path = self._hf_image_archive_path(cuda_suffix)
            error = str(e)
            if os.path.isfile(archive_path):
                error = (
                    f"{error} Cached archive: {archive_path}. If this cache was "
                    "created before the FARBench release, delete it and retry."
                )
            logger.error(f"[{task_name}] Task image preparation failed: {error}")
            result.errors.append(f"task_image_prepare_failed: {error}")
            result.success = False
            return result

        # ── Verify + mark ready ──────────────────────────────────────
        logger.info(f"[{task_name}] Verifying task configuration ...")
        verify_errors = self.verify()
        if verify_errors:
            result.errors.extend(verify_errors)
            result.success = False
            logger.error(f"[{task_name}] Verification failed: {verify_errors}")
            return result
        result.steps_completed.append("verify")
        logger.info(f"[{task_name}] Verification passed")

        self.mark_ready(cuda_variant=cuda_suffix)
        result.steps_completed.append("mark_ready")
        logger.info(f"[{task_name}] Task marked as ready")

        result.total_time_minutes = (time.time() - start) / 60.0
        logger.info(
            f"Task {self.task_config.name} prepared in {result.total_time_minutes:.1f} minutes"
        )
        return result

    # ── Hugging Face image archive helpers ───────────────────────

    def _hf_image_filename(self, cuda_variant: str) -> str:
        """Return the dataset archive name for this task/CUDA variant."""
        _validate_cuda(cuda_variant)
        return f"{self.task_config.name}-{cuda_variant}.docker.tar.gz"

    def _hf_image_archive_path(self, cuda_variant: str) -> str:
        """Return the local cache path for this task/CUDA image archive."""
        return os.path.join(_hf_image_cache_dir(), self._hf_image_filename(cuda_variant))

    def _expected_hf_loaded_image(self, cuda_variant: str) -> str:
        """Return the expected image tag embedded in the downloaded archive."""
        _validate_cuda(cuda_variant)
        return f"{_hf_image_repository()}:{self.task_config.name}-{cuda_variant}"

    def _download_hf_image_archive(self, cuda_variant: str) -> str:
        """Download the pre-built Docker image archive from Hugging Face."""
        filename = self._hf_image_filename(cuda_variant)
        cache_dir = _hf_image_cache_dir()
        archive_path = self._hf_image_archive_path(cuda_variant)
        os.makedirs(cache_dir, exist_ok=True)

        if os.path.isfile(archive_path):
            logger.info(f"Using cached HF image archive: {archive_path}")
            return archive_path

        repo_id = _hf_image_dataset_repo()
        logger.info(f"Downloading HF image archive: {repo_id}/{filename}")
        cmd = [
            *_hf_download_command(),
            repo_id,
            filename,
            "--repo-type",
            "dataset",
            "--local-dir",
            cache_dir,
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self._prepare_timeout_seconds(),
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"hf download failed (exit {result.returncode}): "
                f"{result.stderr[-500:]}"
            )

        if not os.path.isfile(archive_path):
            raise RuntimeError(
                f"HF download completed but archive was not found at {archive_path}"
            )
        return archive_path

    def _docker_image_exists(self, image: str) -> bool:
        result = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0

    def _load_hf_image_archive(self, archive_path: str, cuda_variant: str) -> None:
        """Load a Docker image archive and tag it as task_config.docker_image."""
        local_image = self.task_config.docker_image
        if not local_image:
            raise ValueError("docker_image not specified in task.yaml")

        logger.info(f"Loading Docker image archive: {archive_path}")
        load_result = subprocess.run(
            ["docker", "load", "-i", archive_path],
            capture_output=True,
            text=True,
            timeout=self._prepare_timeout_seconds(),
        )
        if load_result.returncode != 0:
            raise RuntimeError(
                f"docker load failed (exit {load_result.returncode}): "
                f"{load_result.stderr[-500:]}"
            )

        combined_output = "\n".join([load_result.stdout or "", load_result.stderr or ""])
        loaded_images = _DOCKER_LOAD_IMAGE_RE.findall(combined_output)
        loaded_image_ids = _DOCKER_LOAD_IMAGE_ID_RE.findall(combined_output)
        candidate_images = [
            *loaded_images,
            *loaded_image_ids,
            self._expected_hf_loaded_image(cuda_variant),
            local_image,
        ]

        source_image = ""
        for image in dict.fromkeys(candidate_images):
            if image and self._docker_image_exists(image):
                source_image = image
                break
        if not source_image:
            raise RuntimeError(
                "docker load succeeded, but no loaded image tag could be found. "
                f"Expected one of: {candidate_images}"
            )

        if source_image != local_image:
            logger.info(f"Tagging loaded image {source_image} -> {local_image}")
            tag_result = subprocess.run(
                ["docker", "tag", source_image, local_image],
                capture_output=True,
                text=True,
            )
            if tag_result.returncode != 0:
                raise RuntimeError(
                    f"docker tag failed (exit {tag_result.returncode}): "
                    f"{tag_result.stderr[-300:]}"
                )

    def _image_data_paths(self) -> tuple[str, str]:
        """Return the train/test data paths packaged inside task images."""
        return (
            self.task_config.data_dir.rstrip("/"),
            self.task_config.test_data_dir.rstrip("/"),
        )

    def _verify_image_data_paths(self) -> None:
        """Verify the published task image contains non-empty train/test dirs."""
        image = self.task_config.docker_image
        train_path, test_path = self._image_data_paths()
        q_train = shlex.quote(train_path)
        q_test = shlex.quote(test_path)
        script = (
            f"test -d {q_train} && "
            f"test -n \"$(find {q_train} -mindepth 1 -maxdepth 1 -print -quit)\" && "
            f"test -d {q_test} && "
            f"test -n \"$(find {q_test} -mindepth 1 -maxdepth 1 -print -quit)\""
        )
        result = subprocess.run(
            ["docker", "run", "--rm", "--entrypoint", "/bin/sh", image, "-lc", script],
            capture_output=True,
            text=True,
            timeout=self._prepare_timeout_seconds(),
        )
        if result.returncode != 0:
            details = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(
                f"Image {image!r} must contain non-empty {train_path!r} and "
                f"{test_path!r}. {details}"
            )

        permission_script = f"test -r {q_test} && test -x {q_test} && test ! -w {q_test}"
        permission_result = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--entrypoint",
                "/bin/sh",
                "--user",
                "65534:65534",
                image,
                "-lc",
                permission_script,
            ],
            capture_output=True,
            text=True,
            timeout=self._prepare_timeout_seconds(),
        )
        if permission_result.returncode != 0:
            details = (permission_result.stderr or permission_result.stdout or "").strip()
            raise RuntimeError(
                f"Image {image!r} test data path {test_path!r} must be readable "
                f"but not writable by a non-root eval process. {details}"
            )

    # ── Status helpers ────────────────────────────────────────────

    def check_status(self, cuda_variant: str = "") -> bool:
        """Check whether the task has already been prepared for the given variant.

        A marker without a ``cuda_variant`` field (pre-cleanup format) is
        always treated as a mismatch to force re-prepare under the new schema.
        """
        prepared_path = os.path.join(self.task_config.task_dir, ".prepared")
        if not os.path.exists(prepared_path):
            return False

        try:
            with open(prepared_path) as f:
                meta = json.load(f)
        except Exception as e:
            logger.info(f"Cannot read .prepared marker ({e}), re-preparing")
            return False

        prev_variant = meta.get("cuda_variant", "")
        if not prev_variant:
            logger.info("Old-format .prepared marker (no cuda_variant), re-preparing")
            return False
        if meta.get("method") != "hf_dataset_image":
            logger.info("Prepared marker method is not valid for FARBench release")
            return False
        if meta.get("data_source") != "image:/rab_data":
            logger.info("Prepared marker data_source is stale; re-preparing")
            return False
        if meta.get("data_policy_version") != _DATA_POLICY_VERSION:
            logger.info("Prepared marker data policy version is stale; re-preparing")
            return False

        if cuda_variant and prev_variant != cuda_variant:
            logger.info(
                f"Prepared variant mismatch: {prev_variant} != {cuda_variant}, re-preparing"
            )
            return False

        if self.task_config.docker_image and not self._docker_image_exists(
            self.task_config.docker_image
        ):
            logger.info(
                f"Prepared marker exists but Docker image "
                f"{self.task_config.docker_image!r} is missing locally; re-preparing"
            )
            return False

        return True

    def verify(self) -> list[str]:
        """Verify task completeness and integrity."""
        errors = []

        if not self.task_config.name:
            errors.append("Missing: name")
        if not self.task_config.primary_metric:
            errors.append("Missing: primary_metric")

        # Check script/prepare.py exists
        script_dir = self.task_config.script_dir
        if script_dir:
            prepare_py = os.path.join(script_dir, "prepare.py")
            if not os.path.exists(prepare_py):
                logger.info(f"No prepare.py found at {prepare_py} (optional)")

        # Check evaluator class exists
        if self.task_config.evaluator_class:
            parts = self.task_config.evaluator_class.rsplit(".", 1)
            if len(parts) == 2:
                module_path = parts[0].replace(".", "/") + ".py"
                project_root = os.path.dirname(
                    os.path.dirname(self.task_config.task_dir)
                )
                evaluator_file = os.path.join(project_root, module_path)
                if not os.path.exists(evaluator_file):
                    errors.append(f"Evaluator file not found: {module_path}")

        return errors

    def mark_ready(self, cuda_variant: str) -> None:
        """Write the .prepared marker file.

        Args:
            cuda_variant: The CUDA variant this preparation is for (e.g. "cu118").
                Required — used by `check_status()` to detect variant switches.
        """
        prepared_path = os.path.join(self.task_config.task_dir, ".prepared")
        meta: dict = {
            "prepared_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "task_name": self.task_config.name,
            "method": "hf_dataset_image",
            "data_source": "image:/rab_data",
            "data_policy_version": _DATA_POLICY_VERSION,
            "cuda_variant": cuda_variant,
            "docker_image": self.task_config.docker_image,
            "hf_dataset_repo": _hf_image_dataset_repo(),
            "hf_image_archive": self._hf_image_filename(cuda_variant),
        }
        tmp_path = f"{prepared_path}.tmp"
        with open(tmp_path, "w") as f:
            json.dump(meta, f, indent=2)
        os.replace(tmp_path, prepared_path)
