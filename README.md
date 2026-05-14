# FARBench: How Far Are LLM Agents from Autonomous Research?

FARBench measures an agent's ability to iteratively explore, code, and optimize
solutions on real-world machine learning tasks across diverse domains.

The benchmark contains 29 frontier research tasks across computer vision,
robotics, natural language processing, audio/speech understanding, and AI for
science. Agents start from an empty workspace and complete an end-to-end
research loop under controlled compute and time budgets.

## Installation

Prerequisites:

- Docker Engine 24 or newer with Docker Compose V2.
- NVIDIA driver and NVIDIA Container Toolkit for GPU tasks.
- Python 3.10 or newer on the host.

Create a virtual environment and install FARBench from `pyproject.toml`:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dashboard]"
cp .env.example .env
```

Edit `.env` and set the API key for the preset you plan to run, for example
`CLAUDE_API_KEY`, `GPT54_API_KEY`, `GEMINI_API_KEY`, `KIMI_API_KEY`, or
`DEEPSEEK_API_KEY`. Both `scripts/run.sh` and local `farbench` commands load
this environment configuration.

## Quick Start

Run a single task:

```bash
bash scripts/run.sh --task mnist_classification --preset claude --gpus 0 --cuda cu118
```

Use `--gpus` to select host GPU IDs, for example `0` or `0,1`. Use
`--cuda cu118|cu128` to select the published task image variant. The first run
downloads and loads the required task image archive automatically.

Task GPU requirements vary. Before running a task, check
[`benchmarks/task.md`](benchmarks/task.md) for its recommended GPU class,
maximum GPU count, metric, and time budget. Use that table to choose both
`--task` and the GPU IDs passed to `--gpus`.

CUDA variant selection:

- `cu118`: default release image, built on CUDA 11.8. Use this for most NVIDIA
  GPUs and driver setups.
- `cu128`: CUDA 12.8 image for newer GPUs that require the CUDA 12 runtime, such
  as Blackwell-generation cards. Use this if `cu118` cannot run on your system or
  your deployment standard is CUDA 12.8.

The selected variant must exist in the FARBench Hugging Face dataset for the
task you run.

Resume an interrupted experiment:

```bash
farbench resume \
  --experiment-dir experiments/mnist_classification/<run_id> \
  --gpus 0 \
  --cuda cu118
```

Resume reads the original task and agent preset from the experiment
`config.json`, then continues from the last complete iteration.

Launch the dashboard:

```bash
farbench dashboard --port 8501
```

Open `http://localhost:8501/dashboard` to inspect trajectories, leaderboard
aggregates, and analysis views. Experiment records are written under
`experiments/<task>/<agent_id>_<timestamp>/`.

## Datasets

Release task data and task dependencies are packaged in pre-built Docker image
archives hosted in the Hugging Face dataset
[`FARBenchAnonymous/FARBench`][hf-images].

For example, the CUDA 11.8 MNIST task image is distributed as:

```text
mnist_classification-cu118.docker.tar.gz
farbench/farbench:mnist_classification-cu118
```

Downloaded archives are cached under `farbench-images/`. No separate dataset
download is required for normal benchmark runs.

During execution, the agent sees the training view of the task data. Evaluation
runs in a separate container with the held-out test view, and the evaluator only
consumes the submitted prediction file.

## How to Add a Task

Add a new benchmark under `benchmarks/<task>/` with:

- `task.yaml`: task description, metric definition, budgets, resource limits,
  and evaluation contract.
- `script/prepare.py`: data preparation logic used when building the release
  task image.
- `script/evaluator.py`: scoring logic for submitted predictions.
- `docker/Dockerfile`: task-specific dependencies on top of the FARBench base
  image.

Task metadata is loaded in strict mode for release: invalid or unknown
`task.yaml` fields should be fixed rather than ignored. After implementing the
task, use the maintainer workflow in `scripts/README.md` to build and archive
the task image.

## License

MIT License. See [LICENSE](LICENSE) for details.

[hf-images]: https://huggingface.co/datasets/FARBenchAnonymous/FARBench
