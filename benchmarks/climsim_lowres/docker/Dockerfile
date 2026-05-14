ARG BASE_TAG=cu118
FROM farbench/farbench:base-${BASE_TAG}

# ClimSim low-res: subgrid-scale atmospheric physics emulation (tabular regression)
# Base provides: torch, numpy, scipy, pandas, scikit-learn, h5py, tqdm, matplotlib,
#   transformers, timm, einops, torchmetrics
# Task-specific: xarray/netCDF4 (in case agent wants to inspect original data formats),
#   huggingface_hub (data download in prepare.py)

RUN pip install --no-cache-dir \
    xarray==2024.10.0 \
    netCDF4==1.7.2 \
    huggingface-hub==0.27.1
