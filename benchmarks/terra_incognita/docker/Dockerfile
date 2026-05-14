ARG BASE_TAG=cu118
FROM farbench/farbench:base-${BASE_TAG}
ARG HF_TOKEN=""
ENV HF_TOKEN=${HF_TOKEN}

# TerraIncognita: wildlife camera trap domain generalization (10-class, 4 domains)
#   jsonlines: annotation parsing utility

RUN pip install --no-cache-dir \
    jsonlines==4.0.0

# Pre-download models that recent DG papers (2023-2025) actually use for TerraIncognita:
#   - CLIP ViT-B/16 (86M) — dominant in 2024 DG papers (CLIPCEIL NeurIPS'24, ERM++ WACV'25)
#   - DINOv2 ViT-B/14 (86M) — self-supervised, strong OOD features
#   - ResNet-50 (25M) — classic DomainBed baseline
RUN HF_HUB_DISABLE_PROGRESS_BARS=1 python -c "\
import timm; \
timm.create_model('vit_base_patch16_clip_224.openai', pretrained=True); \
print('CLIP ViT-B/16 cached via timm')" \
 && HF_HUB_DISABLE_PROGRESS_BARS=1 python -c "\
from transformers import AutoModel, AutoImageProcessor; \
AutoModel.from_pretrained('facebook/dinov2-base'); \
AutoImageProcessor.from_pretrained('facebook/dinov2-base'); \
print('DINOv2 ViT-B/14 model+processor cached')" \
 && HF_HUB_DISABLE_PROGRESS_BARS=1 python -c "\
from torchvision.models import resnet50, ResNet50_Weights; \
resnet50(weights=ResNet50_Weights.IMAGENET1K_V2); \
print('torchvision cached')"
