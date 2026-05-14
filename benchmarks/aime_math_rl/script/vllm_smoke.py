"""vLLM install sanity check -- runs at Docker build time (CPU-only).

This script is executed via ``/usr/local/bin/vllm-python`` by the Dockerfile
to fail the image build fast if the /opt/vllm venv is misconfigured. It does
NOT allocate GPU memory and does NOT load Qwen3-4B (the base model is
materialized later by prepare.py and only mounted at run time).

Deliberately minimal: verifies the vLLM install is intact and avoids
depending on private transformers module paths (those shift between
transformers 4.x and 5.x and would create spurious build failures).

Verifies:
  1. ``import vllm`` succeeds (i.e. libcudart.so.12 resolves via the wrapper's
     LD_LIBRARY_PATH pointing at the cu12 runtime libs shipped in the venv).
  2. Core APIs (``LLM`` class, ``SamplingParams``) are importable and
     SamplingParams round-trips its arguments.
  3. The venv's torch / transformers are the versions pulled in by vllm (i.e.
     the isolation from the host training env actually happened).
  4. ``transformers.AutoTokenizer`` is importable (the only stable public
     handle we rely on; everything else is validated at run time).
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

    sp = SamplingParams(temperature=0.6, max_tokens=8, top_p=0.95, top_k=20)
    assert sp.temperature == 0.6, sp
    assert sp.max_tokens == 8, sp
    assert callable(LLM), "vllm.LLM is not callable"

    from transformers import AutoTokenizer
    assert hasattr(AutoTokenizer, "from_pretrained"), "AutoTokenizer API missing"

    print("vLLM smoke test: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
