ARG BASE_TAG=cu118
FROM farbench/farbench:base-${BASE_TAG}
ARG HF_TOKEN=""
ENV HF_TOKEN=${HF_TOKEN}

# BigCodeBench code generation: SFT / LoRA fine-tuning task.
# Base image already has: torch, transformers, datasets, accelerate, huggingface_hub

# ──────────────────────────────────────────────────────────────────────
# Evaluation timing knobs (all read by script/evaluator.py).
#
# TIMEOUT_SECONDS   per-test-case subprocess wall clock (kills hangs).
#                   240s matches the upstream BigCodeBench reference harness.
# EVAL_WORKERS      thread-pool size for concurrent subprocess exec.
#                   16 keeps peak RSS ≤ ~5 GB on a 32-64 core host and takes
#                   wall-clock from 40+ min sequential down to ~5-15 min.
# DEADLINE_SECONDS  global wall clock for the entire exec phase. Any test
#                   cases still running past this are recorded as
#                   "deadline exceeded" so the evaluator always returns in
#                   bounded time. 1500s = 25 min caps total eval below 30 min.
#                   Set to 0 to disable.
# ──────────────────────────────────────────────────────────────────────
ENV BIGCODEBENCH_EVAL_TIMEOUT_SECONDS="240"
ENV BIGCODEBENCH_EVAL_WORKERS="16"
ENV BIGCODEBENCH_EVAL_DEADLINE_SECONDS="1500"

# ---------------------------------------------------------------------------
# Training / framework packages.
#
# Covers three SFT routes mentioned in task.yaml agent_hints:
#   * LoRA           — peft + trl                                   (≤ 30 GB/GPU)
#   * QLoRA (4-bit)  — peft + trl + bitsandbytes                    (≤ 18 GB/GPU)
#   * Full FT+FSDP   — accelerate (already in base) + bitsandbytes  (≤ 40 GB/GPU)
#                      8-bit AdamW from bitsandbytes halves optimizer state.
#
# We do NOT ship DeepSpeed: the cu118 base is `-runtime-` (no nvcc), and
# deepspeed 0.17.x does `subprocess(nvcc -V)` at import time → crash. Full FT
# should use torch FSDP via accelerate, which is already functional.
#
# NOTE: `trl==0.12.2` downgrades `transformers` from 5.5.4 (shipped by base)
# to 4.46.3 in this (training) env. vLLM lives in its own venv at /opt/vllm
# with transformers 5.7+, so this does not affect inference.
# ---------------------------------------------------------------------------
RUN pip install --no-cache-dir \
    peft==0.13.2 \
    trl==0.12.2 \
    bitsandbytes==0.45.5 \
    jsonlines==4.0.0

# ---------------------------------------------------------------------------
# BigCodeBench runtime third-party libraries.
#
# We do NOT install the upstream `bigcodebench` pip package. Two reasons:
#   1. Our evaluator.py / prepare.py never `import bigcodebench` -- we
#      reimplement the (very small) eval harness in evaluator.py.
#   2. `bigcodebench==0.2.4` declares `vllm` as a required dep. pip's
#      resolver then tries to satisfy `vllm` against the base image's torch,
#      backtracks through every vllm release down to 0.1.3 (a source-only
#      tarball needing nvcc), and the build dies because the base is
#      `nvidia/cuda:*-runtime-*` (no nvcc). vLLM is already installed,
#      properly isolated, in /opt/vllm below.
#
# This list is the superset of imports the 1,140 BigCodeBench-Full test
# cases use, derived from upstream `Requirements/runtime.txt` and pinned
# to widely-available versions so the resolver stays happy on both
# cu118 (torch 2.7.1) and cu128 (torch 2.11) bases.
#
# Heavy optional deps (tensorflow, full CV stack) are intentionally skipped
# -- they cover a small fraction of cases and would balloon the image.
# ---------------------------------------------------------------------------
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    pandas==2.2.2 \
    scipy==1.13.1 \
    scikit-learn==1.5.1 \
    matplotlib==3.9.2 \
    seaborn==0.13.2 \
    statsmodels==0.14.2 \
    sympy==1.13.2 \
    networkx==3.3 \
    requests==2.32.3 \
    beautifulsoup4==4.12.3 \
    lxml==5.3.0 \
    nltk==3.9.1 \
    Pillow==10.4.0 \
    openpyxl==3.1.5 \
    xlrd==2.0.1 \
    xmltodict==0.13.0 \
    PyYAML==6.0.2 \
    cryptography==43.0.1 \
    pycryptodome==3.20.0 \
    rsa==4.9 \
    python-docx==1.1.2 \
    PyPDF2==3.0.1 \
    regex==2024.7.24 \
    pytz==2024.1 \
    python-dateutil==2.9.0.post0 \
    psutil==6.0.0 \
    tqdm==4.66.5 \
    flask==3.0.3 \
    jinja2==3.1.4 \
    werkzeug==3.0.4 \
    sqlalchemy==2.0.32 \
    jsonschema==4.23.0 \
    validators==0.33.0 \
    wordcloud==1.9.3 \
    geopandas==1.0.1 \
    shapely==2.0.6 \
    folium==0.17.0 \
    textblob==0.18.0.post0 \
    fuzzywuzzy==0.18.0 \
    python-Levenshtein==0.25.1 \
    mechanize==0.4.10 \
    prettytable==3.11.0

# ---------------------------------------------------------------------------
# Pre-download Qwen2.5-Coder-3B into HF cache (no network at runtime).
# This is executed at image-build time so farbench_data stays lightweight and
# `farbench run` works offline.
# ---------------------------------------------------------------------------
RUN HF_HUB_DISABLE_PROGRESS_BARS=1 python -c "\
from transformers import AutoModelForCausalLM, AutoTokenizer; \
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Coder-3B'); \
AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-Coder-3B'); \
print('Qwen2.5-Coder-3B cached')"

# ---------------------------------------------------------------------------
# vLLM: high-throughput inference engine (used by agent predict.py).
#
# WHY A SEPARATE VENV.
#   vLLM's PyPI wheels are cu12-linked and, as of 0.19.x, require
#   torch==2.10.x and transformers>=5.7, which are incompatible with the
#   base image's training stack (torch 2.7.1+cu118, transformers 5.5.4).
#   We therefore install vLLM into an isolated venv at /opt/vllm so the
#   base training environment is left untouched; training still uses the
#   system `python` (torch cu118), and inference uses /opt/vllm.
#
# CUDA-12 RUNTIME RESOLUTION.
#   The nvidia-cuda-* cu12 pip packages pulled in by vllm ship
#   libcudart.so.12, libcublas.so.12, libnccl.so.2, etc. inside the venv.
#   They are NOT on the default loader path, so a thin wrapper
#   `/usr/local/bin/vllm-python` prepends them to LD_LIBRARY_PATH before
#   exec-ing the venv's python. That keeps the runtime libs scoped to the
#   vLLM process and avoids poisoning the host torch's CUDA resolution.
#
# AGENT USAGE.
#   vllm-python predict.py --data_path ... --checkpoint ... --output ...
#   vllm-python -c "from vllm import LLM, SamplingParams; ..."
#   vllm-python -m vllm.entrypoints.openai.api_server --model <ckpt>   # REST
# ---------------------------------------------------------------------------
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends \
        python3.10-venv && \
    rm -rf /var/lib/apt/lists/*

RUN python3.10 -m venv /opt/vllm && \
    /opt/vllm/bin/pip install --no-cache-dir --upgrade pip wheel

# Pin vllm for reproducibility. 0.19.1 is the first release that allows
# transformers>=5.5.1 while still building against torch 2.10+cu128, which
# works on sm_89 (RTX 6000 Ada) and sm_120 (RTX 5090) via driver forward
# compat. Upgrade-only deliberately -- bumping this must re-run smoke below.
RUN /opt/vllm/bin/pip install --no-cache-dir "vllm==0.19.1"

# Wrapper script: sets LD_LIBRARY_PATH to the cu12 libs shipped in the venv,
# then exec's the venv's python. Any additional args pass through (-c, -m,
# path/to/script.py, ...). Uses find(1) so new nvidia-* sub-packages get
# picked up automatically if the vllm pin is bumped.
RUN printf '%s\n' \
    '#!/bin/bash' \
    'set -e' \
    'NV_ROOT=/opt/vllm/lib/python3.10/site-packages/nvidia' \
    'if [ -d "$NV_ROOT" ]; then' \
    '  NV_LIBS=$(find "$NV_ROOT" -maxdepth 2 -type d -name lib 2>/dev/null | tr "\n" ":" | sed "s/:$//")' \
    '  export LD_LIBRARY_PATH="${NV_LIBS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"' \
    'fi' \
    'exec /opt/vllm/bin/python "$@"' \
    > /usr/local/bin/vllm-python && \
    chmod +x /usr/local/bin/vllm-python

ENV VLLM_PYTHON=/usr/local/bin/vllm-python

# Copy + run the smoke test. Build fails here if anything is off (wrong torch,
# missing libcudart.so.12, transformers API drift, etc.).
COPY script/vllm_smoke.py /opt/vllm/vllm_smoke.py
RUN /usr/local/bin/vllm-python /opt/vllm/vllm_smoke.py
