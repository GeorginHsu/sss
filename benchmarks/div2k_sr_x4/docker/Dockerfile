ARG BASE_TAG=cu118
FROM farbench/farbench:base-${BASE_TAG}
ARG HF_TOKEN=""
ENV HF_TOKEN=${HF_TOKEN}

# DIV2K 4x image super-resolution (PSNR track, bicubic degradation)
#   basicsr:      dominant SR framework (SRResNet, EDSR, RCAN, SwinIR configs)
#   scikit-image: SSIM computation (structural_similarity)
#   lmdb:         fast data loading (BasicSR convention)

RUN pip install --no-cache-dir \
    basicsr==1.4.2 \
    scikit-image==0.24.0 \
    lmdb==1.5.1 \
    torchsr==1.0.4

# Pre-download models for DIV2K x4 classical SR (2018-2022):
#   - VGG-19 (143M) — perceptual loss, used by virtually all SR papers
#   - RCAN x4 (15.4M, 32.64 dB Set5) — classic channel-attention SR baseline (Zhang et al., ECCV 2018)
#   - SwinIR-M x4 (11.9M, 32.92 dB Set5) — Swin Transformer SR (Liang et al., ICCVW 2021)
#   - Swin2SR x4 (12.2M, ~32.9 dB Set5) — SwinV2-based SR (Conde et al., ECCVW 2022)

# VGG-19 for perceptual loss
RUN HF_HUB_DISABLE_PROGRESS_BARS=1 python -c "\
from torchvision.models import vgg19, VGG19_Weights; \
vgg19(weights=VGG19_Weights.IMAGENET1K_V1); \
print('VGG-19 cached')"

# RCAN x4 pretrained (torchsr, classic baseline)
RUN HF_HUB_DISABLE_PROGRESS_BARS=1 python -c "\
from torchsr.models import rcan; \
rcan(scale=4, pretrained=True); \
print('RCAN x4 cached')"

# SwinIR-M x4 pretrained weights (basicsr has the architecture)
RUN mkdir -p /root/.cache/swinir && \
    curl -sL -o /root/.cache/swinir/001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth \
    https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth

# EDSR x4 pretrained (torchsr, classic baseline)
RUN HF_HUB_DISABLE_PROGRESS_BARS=1 python -c "\
from torchsr.models import edsr; \
edsr(scale=4, pretrained=True); \
print('EDSR x4 cached')"

# Swin2SR x4 classical SR (transformers native)
RUN HF_HUB_DISABLE_PROGRESS_BARS=1 python -c "\
from transformers import Swin2SRForImageSuperResolution, AutoImageProcessor; \
Swin2SRForImageSuperResolution.from_pretrained('caidas/swin2SR-classical-sr-x4-64'); \
AutoImageProcessor.from_pretrained('caidas/swin2SR-classical-sr-x4-64'); \
print('Swin2SR x4 cached')"
