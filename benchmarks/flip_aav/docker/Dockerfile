ARG BASE_TAG=cu118
FROM farbench/farbench:base-${BASE_TAG}
ARG HF_TOKEN=""
ENV HF_TOKEN=${HF_TOKEN}

# FLIP AAV protein fitness prediction (sequence → fitness regression)
# Base provides: torch, numpy, scipy, pandas, scikit-learn, transformers (ESM-2 support)
# Task-specific:
#   - biopython: sequence utilities
#   - fair-esm: Meta's ESM protein language models (ESM-1b, ESM-2, native representation extraction)
#   - peft: parameter-efficient fine-tuning (LoRA etc.) for large protein models under GPU constraints

RUN pip install --no-cache-dir \
    biopython==1.84 \
    fair-esm \
    peft

# Pre-download ESM-2 protein language models: 8M, 35M, 150M
RUN HF_HUB_DISABLE_PROGRESS_BARS=1 python -c "\
import esm; \
esm.pretrained.esm2_t6_8M_UR50D(); \
esm.pretrained.esm2_t12_35M_UR50D(); \
esm.pretrained.esm2_t30_150M_UR50D(); \
print('ESM-2 models cached')"
