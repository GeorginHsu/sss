ARG BASE_TAG=cu118
FROM farbench/farbench:base-${BASE_TAG}
ARG HF_TOKEN=""
ENV HF_TOKEN=${HF_TOKEN}

# QM9: molecular property regression (HOMO-LUMO gap)
# Requires: torch-geometric ecosystem for 3D geometric deep learning
#           (SchNet, DimeNet++, PaiNN, EGNN, etc.)

# torch-geometric + extensions (must match torch 2.7.1 + CUDA 11.8)
RUN pip install --no-cache-dir \
    torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.7.0+cu118.html \
    && pip install --no-cache-dir torch-geometric==2.6.1

# rdkit: SMILES parsing, molecular featurization, SDF parsing for 3D coordinates
# rdkit >=2024.9.1 supports NumPy 2.x natively.
RUN pip install --no-cache-dir \
    rdkit==2024.9.6

# Equivariant GNN building blocks + ready-made 3D models (PaiNN / ET / TensorNet)
RUN pip install --no-cache-dir \
    e3nn==0.5.6 \
    torchmd-net==2.6.2

# Mordred 2D molecular descriptors (classical-ML / fingerprint pipelines)
RUN pip install --no-cache-dir \
    mordredcommunity==2.0.6

# Suppress RDKit deprecation warnings that flood agent output
RUN echo 'import importlib; rdlogger=importlib.import_module("rdkit.RDLogger"); rdlogger.DisableLog("rdApp.*")' \
    > /usr/local/lib/python3.10/dist-packages/suppress_rdkit_warnings.pth
