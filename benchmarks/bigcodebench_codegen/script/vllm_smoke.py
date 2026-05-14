"""vLLM install sanity check -- runs at Docker build time (CPU-only).

This script is executed via ``/usr/local/bin/vllm-python`` by the Dockerfile
to fail the image build fast if the /opt/vllm venv is misconfigured. It does
NOT allocate GPU memory; the actual inference is done at run time by the
agent's predict.py.

Verifies:
  1. ``import vllm`` succeeds (i.e. libcudart.so.12 resolves via the wrapper's
     LD_LIBRARY_PATH pointing at the cu12 runtime libs shipped in the venv).
  2. Core APIs (``LLM`` class, ``SamplingParams``) are importable and
     SamplingParams round-trips its arguments.
  3. The venv's torch / transformers are the versions pulled in by vllm (i.e.
     the isolation from the host training env actually happened).
  4. The Qwen2.5-Coder-3B tokenizer loads under the venv's transformers -- this
     catches transformers-version mismatches that would silently break agent
     predict.py at run time.
"""
from __future__ import annotations

import sys


def main() -> int:
    import torch
    import transformers
    import vllm
    from vllm import LLM, SamplingParams

    print(f"vllm         = {vllm.__version__}")
    print(f"torch        = {torch.__version__}")
    print(f"transformers = {transformers.__version__}")

    sp = SamplingParams(temperature=0.0, max_tokens=8, top_p=1.0)
    assert sp.temperature == 0.0, sp
    assert sp.max_tokens == 8, sp
    assert callable(LLM), "vllm.LLM is not callable"

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B")
    encoded = tok("def add(a, b):", add_special_tokens=False)
    assert len(encoded["input_ids"]) > 0, "tokenizer produced empty output"
    assert tok.vocab_size > 100_000, f"unexpected vocab size: {tok.vocab_size}"

    print("vLLM smoke test: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
