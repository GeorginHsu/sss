ARG BASE_TAG=cu118
FROM farbench/farbench:base-${BASE_TAG}
ARG HF_TOKEN=""
ENV HF_TOKEN=${HF_TOKEN}

# AIME Math RL — reinforcement-learning fine-tuning of Qwen3-4B on
# Skywork-OR1-RL-Data (math split), evaluated on AIME 2024 + 2025 (60 problems).
#
# The image ships the Python environment + a separate vLLM venv at /opt/vllm,
# and pre-downloads the Qwen/Qwen3-4B base model into the HF cache at build
# time. Task data (train / val / test jsonl) is materialized on the host by
# script/prepare.py during `farbench tasks prepare`, and baked into the hub image
# by scripts/dev/FARBench image archive build pipeline (step 3: COPY farbench_data/<task>/). The
# base model is NOT in farbench_data — agents access it via the canonical HF id
# `Qwen/Qwen3-4B`, which both the training env and the vLLM venv resolve from
# /root/.cache/huggingface/ (system-wide cache, shared across both venvs).

# ---------------------------------------------------------------------------
# Training / framework packages.
#
#   * LoRA / QLoRA via peft + trl      (+ bitsandbytes for 4-bit NF4)
#   * Full FT via accelerate FSDP      (already in base)
#
# We deliberately do NOT install deepspeed: the base image is `-runtime-`
# (no nvcc) so deepspeed 0.17.x crashes at import. flash-attn is also not
# installed (source-only build, needs nvcc); torch's sdpa backend is the
# substitute for fused attention here.
# ---------------------------------------------------------------------------
# NOTE on the transformers upper bound (<5.0):
#   transformers 5.x raised its quantizer's BITSANDBYTES_MIN_VERSION to 0.46.1.
#   bitsandbytes 0.46+ is cu12-only, which is incompatible with this cu118
#   base image. Pinning transformers <5 keeps the MIN at 0.45.x so the
#   bitsandbytes==0.45.5 we install here is accepted by the 4-bit validator
#   (`transformers.quantizers.quantizer_bnb_4bit`). Without this pin, agents
#   that call `BitsAndBytesConfig(load_in_4bit=True)` hit:
#     ImportError: Using `bitsandbytes` 4-bit quantization requires
#                  bitsandbytes: pip install -U bitsandbytes>=0.46.1
#   on a runtime that cannot actually upgrade bitsandbytes.
RUN pip install --no-cache-dir \
    "transformers>=4.51,<5.0" \
    "datasets>=2.19" \
    "huggingface_hub>=0.24" \
    "hf_transfer>=0.1.6" \
    "accelerate>=0.33" \
    "peft>=0.12" \
    "trl>=0.10" \
    "bitsandbytes==0.45.5" \
    "sympy" \
    "jsonlines"

# Faster HF dataset downloads during `prepare.py` (Skywork-OR1 + AIME sets).
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# ---------------------------------------------------------------------------
# vLLM: high-throughput inference engine. Used for both:
#   (a) RL rollouts during training (5-10x faster than HF model.generate),
#   (b) the agent's predict.py during evaluation.
#
# WHY A SEPARATE VENV.
#   vLLM's PyPI wheels are cu12-linked and, as of 0.19.x, require
#   torch==2.10.x and transformers>=5.7, which are incompatible with this
#   image's training stack (torch 2.7.1+cu118, transformers 4.51-ish that
#   trl/peft pin). Installing vLLM into the base would force the trainer
#   stack to upgrade transformers and break trl/peft pins. Solution: drop
#   vLLM into an isolated venv at /opt/vllm so the base env is left untouched.
#
# CUDA-12 RUNTIME RESOLUTION.
#   The nvidia-cuda-* cu12 pip packages pulled in by vllm ship
#   libcudart.so.12, libcublas.so.12, libnccl.so.2, etc. inside the venv.
#   They are NOT on the default loader path, so a thin wrapper
#   `/usr/local/bin/vllm-python` prepends them to LD_LIBRARY_PATH before
#   exec-ing the venv's python. That keeps the runtime libs scoped to vLLM.
#
# AGENT USAGE.
#   /usr/local/bin/vllm-python predict.py --data_path ... --checkpoint ... --output ...
#   /usr/local/bin/vllm-python -c "from vllm import LLM, SamplingParams; ..."
#   /usr/local/bin/vllm-python -m vllm.entrypoints.openai.api_server --model <ckpt>
#
# Host driver requirement: NVIDIA driver >= R535 (CUDA 12.0+ on host).
# ---------------------------------------------------------------------------
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends \
        python3.10-venv && \
    rm -rf /var/lib/apt/lists/*

RUN python3.10 -m venv /opt/vllm && \
    /opt/vllm/bin/pip install --no-cache-dir --upgrade pip wheel

# Pin vllm for reproducibility. 0.19.1 is the first release that allows
# transformers>=5.5.1 while still building against torch 2.10+cu128, which
# works on Ada (sm_89), Hopper (sm_90), and Blackwell (sm_120) via driver
# forward compat. Officially supports the full Qwen3 family (incl. Qwen3-4B).
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
# missing libcudart.so.12, transformers API drift, Qwen3 tokenizer missing,
# etc.). Smoke is CPU-only and does NOT load Qwen3-4B weights (we just verify
# the vLLM stack imports cleanly).
COPY script/vllm_smoke.py /opt/vllm/vllm_smoke.py
RUN /usr/local/bin/vllm-python /opt/vllm/vllm_smoke.py

# ---------------------------------------------------------------------------
# Pre-download Qwen/Qwen3-4B into the system HF cache (~8 GB, weights only).
#
# WHY at build time, not at agent runtime:
#   * FARBench sandboxes run with `network_access: false`. If the model isn't
#     in the image, the agent cannot fetch it.
#   * Baking it in via prepare.py + farbench_data path also works, but forces the
#     agent to remember a magic absolute path like "/data/base_model/Qwen3-4B".
#     Hard-coding the canonical HF id "Qwen/Qwen3-4B" instead lets agents use
#     the idiomatic `from_pretrained("Qwen/Qwen3-4B")` they have seen
#     thousands of times in their training data → far fewer FileNotFound
#     mistakes (this was the model.py error in the previous run).
#
# WHAT gets cached:
#   AutoTokenizer + AutoModelForCausalLM trigger HF Hub to mirror the repo's
#   bf16 *.safetensors, tokenizer files, config.json, and generation_config
#   into /root/.cache/huggingface/hub/. We keep `low_cpu_mem_usage=True` and
#   `torch_dtype="auto"` so model construction happens on meta tensors and
#   never allocates the 8 GB of GPU/CPU activations during build.
#
# WHO can see it at runtime:
#   The training env's `python` (system) and the inference venv's vllm-python
#   (/opt/vllm/bin/python) both default to HF_HOME=/root/.cache/huggingface,
#   so a single download serves both stacks. No symlink games needed.
# ---------------------------------------------------------------------------
RUN HF_HUB_DISABLE_PROGRESS_BARS=1 python -c "\
import torch; \
from transformers import AutoModelForCausalLM, AutoTokenizer; \
AutoTokenizer.from_pretrained('Qwen/Qwen3-4B'); \
AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-4B', torch_dtype='auto', low_cpu_mem_usage=True); \
print('Qwen/Qwen3-4B cached')"
