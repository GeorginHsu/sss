ARG BASE_TAG=cu118
FROM farbench/farbench:base-${BASE_TAG}
ARG HF_TOKEN=""
ENV HF_TOKEN=${HF_TOKEN}

# WeatherBench Z500+T850: global weather field forecasting (5.625deg, 32x64)
# xarray + netCDF4: required for data preparation (loading ERA5 NetCDF files)
#   Also useful if agent wants to explore raw climate data structure
# Base provides: torch, torchvision, numpy, scipy, pandas, scikit-learn,
#   timm, einops, matplotlib, h5py, tqdm

RUN pip install --no-cache-dir \
    xarray==2024.10.0 \
    netCDF4==1.7.2 \
    torch-harmonics==0.7.1 \
    neuraloperator==0.3.0
