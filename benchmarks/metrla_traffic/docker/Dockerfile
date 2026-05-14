ARG BASE_TAG=cu118
FROM farbench/farbench:base-${BASE_TAG}
ARG HF_TOKEN=""
ENV HF_TOKEN=${HF_TOKEN}

# METR-LA spatiotemporal traffic speed prediction (207 sensors, graph-based)
# Base provides: torch, numpy, scipy, pandas, scikit-learn, einops
# Task-specific: torch-geometric ecosystem (GNN), networkx, gdown (Google Drive), tables (HDF5)

# torch-geometric + extensions (must match torch + CUDA 11.8 in base image)
RUN pip install --no-cache-dir \
    torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.7.0+cu118.html \
    && pip install --no-cache-dir torch-geometric==2.6.1

# Graph utilities + data download tools
RUN pip install --no-cache-dir \
    networkx==3.2.1 \
    gdown==5.2.0 \
    tables==3.10.1
