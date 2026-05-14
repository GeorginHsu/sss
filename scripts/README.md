# FARBench Runtime Scripts

`scripts/run.sh` is the public experiment entry point. It runs one task at a
time, loads the selected task image from the FARBench Hugging Face dataset, and
then delegates to `farbench run` inside Docker Compose.
The wrapper loads `.env` before resolving CUDA and image-download settings.

Normal installation and experiment usage are documented in the root
`README.md`. This file records script-specific behavior and maintainer image
build steps.

## Image Loading

Published task images are stored as Docker archive files in the Hugging Face
dataset configured by `FARBENCH_HF_IMAGE_DATASET_REPO`.

Defaults:

```bash
FARBENCH_HF_IMAGE_DATASET_REPO=FARBenchAnonymous/FARBench
FARBENCH_HF_IMAGE_REPOSITORY=farbench/farbench
FARBENCH_HF_IMAGE_CACHE_DIR=farbench-images
FARBENCH_CUDA=cu118
```

For task `mnist_classification` and CUDA `cu118`, `run.sh` expects:

```text
mnist_classification-cu118.docker.tar.gz
farbench/farbench:mnist_classification-cu118
```

`cu118` and `cu128` are the only supported CUDA variants. Release runs do not
build Docker images locally; missing archives are treated as an error.

## Developer Image Builds

Local Docker builds are intentionally kept out of `scripts/run.sh`. Developers
who need to rebuild a task image can use the dev-only builder:

```bash
bash scripts/dev/build_task_image.sh mnist_classification --cuda cu118
bash scripts/dev/build_task_image.sh mnist_classification --cuda cu128 --archive
```

The builder creates:

```text
farbench/farbench:base-<cuda>
farbench/farbench:<task>-<cuda>-deps
farbench/farbench:<task>-<cuda>
```

It runs `benchmarks/<task>/script/prepare.py` into repo-relative host staging
directories under `farbench_data/`, then bakes those directories into the image at
`/rab_data/<task>/` and `/rab_data/<task>_test/`. With `--archive`, it exports
the same archive name expected by release runs:

```text
farbench-images/<task>-<cuda>.docker.tar.gz
```

`vlabench_manipulation` automatically uses the Python 3.12 base image
(`docker/Dockerfile.base.py312`). All other tasks use `docker/Dockerfile.base`.

## Supported Flags

```bash
bash scripts/run.sh --task TASK --gpus GPU_IDS [options]
```

Required:

```text
--task TASK      Benchmark task name.
--gpus GPU_IDS   Comma-separated host GPU IDs, for example 0 or 0,1.
```

Common options:

```text
--preset NAME    LLM provider preset, for example claude, gemini, gpt54.
--cuda TAG       CUDA variant: cu118 or cu128.
--prepare        Forwarded to farbench run. Enabled by default.
--mode MODE      Forwarded to farbench run. Default: api.
```

The release wrapper runs one task per command.
