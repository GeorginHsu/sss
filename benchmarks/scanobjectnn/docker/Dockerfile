ARG BASE_TAG=cu118
FROM farbench/farbench:base-${BASE_TAG}
ARG HF_TOKEN=""
ENV HF_TOKEN=${HF_TOKEN}

# ScanObjectNN 3D point cloud classification (PB_T50_RS)
# Task forbids 3D pretrained models (ShapeNet/ModelNet) — only ImageNet-pretrained 2D backbones allowed.
#   gdown: download from Google Drive (official dataset host)

RUN pip install --no-cache-dir \
    gdown==5.2.0

# Pre-download 2D ImageNet backbones for Point-to-Pixel (P2P) style approaches.
# P2P (NeurIPS 2022) projects point clouds into 2D and uses frozen 2D backbones — achieves 89.3% OA.
# Also useful as feature extractors for hybrid 2D-3D approaches.
#   - ResNet-50 (25M) — classic P2P baseline
#   - ConvNeXt-Tiny (28M) — strong CNN backbone for P2P
#   - DINOv2 ViT-B/14 (86M) — self-supervised features, robust for cross-modal transfer
RUN HF_HUB_DISABLE_PROGRESS_BARS=1 python -c "\
from torchvision.models import resnet50, ResNet50_Weights; \
resnet50(weights=ResNet50_Weights.IMAGENET1K_V2); \
print('torchvision cached')" \
 && HF_HUB_DISABLE_PROGRESS_BARS=1 python -c "\
import timm; \
timm.create_model('convnext_tiny.fb_in22k_ft_in1k', pretrained=True); \
print('timm cached')" \
 && HF_HUB_DISABLE_PROGRESS_BARS=1 python -c "\
from transformers import AutoModel, AutoImageProcessor; \
AutoModel.from_pretrained('facebook/dinov2-base'); \
AutoImageProcessor.from_pretrained('facebook/dinov2-base'); \
print('DINOv2 model+processor cached')" \
 && HF_HUB_DISABLE_PROGRESS_BARS=1 python -c "\
import timm; \
timm.create_model('vit_base_patch16_clip_224.openai', pretrained=True); \
print('CLIP ViT-B/16 cached')"
