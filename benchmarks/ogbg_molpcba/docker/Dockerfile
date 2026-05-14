ARG BASE_TAG=cu118
FROM farbench/farbench:base-${BASE_TAG}
ARG HF_TOKEN=""
ENV HF_TOKEN=${HF_TOKEN}

# ogbg-molpcba: multi-task molecular graph property prediction (128 binary tasks)
# Requires: torch-geometric ecosystem for GNN models, ogb for data loading, rdkit for molecular features

# torch-geometric + extensions (must match torch 2.7.1 + CUDA 11.8)
RUN pip install --no-cache-dir \
    torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.7.0+cu118.html \
    && pip install --no-cache-dir torch-geometric==2.6.1

# OGB (dataset loading and official evaluator) + rdkit (molecular featurization)
# rdkit >=2024.9.1 supports NumPy 2.x natively.
RUN pip install --no-cache-dir \
    ogb==1.3.6 \
    rdkit==2024.9.6

# Suppress RDKit deprecation warnings that flood agent output
RUN echo 'import importlib; rdlogger=importlib.import_module("rdkit.RDLogger"); rdlogger.DisableLog("rdApp.*")' \
    > /usr/local/lib/python3.10/dist-packages/suppress_rdkit_warnings.pth
