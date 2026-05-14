ARG BASE_TAG=cu118
FROM farbench/farbench:base-${BASE_TAG}
ARG HF_TOKEN=""
ENV HF_TOKEN=${HF_TOKEN}

# DomainNet Quickdraw: domain generalization (5 source domains → Quickdraw)
# No additional pip dependencies beyond base image.
# Base provides: albumentations, opencv, timm, torchmetrics, scikit-learn, pandas, einops

# Pre-download backbones that DomainNet DG papers (2021-2026) actually use.
# Selected to span both evaluation tracks of the task and minimize agent guesswork:
#
# Track A — ImageNet-1k supervised (DomainBed protocol):
#   - ResNet-50 IN-1k V1 (25M, torchvision) — DomainBed default; ERM/ERM++/MIRO/SWAD/DiWA
#     all report numbers from this checkpoint. V1 (NOT V2 / "ResNet strikes back") is the
#     correct match for the published 12-17% Quickdraw range.
#   - ViT-B/16 IN-1k AugReg (86M, timm) — ERM++ WACV 2025 Table 3(b) ablation; supervised-
#     IN-1k transformer baseline. `augreg_in1k` is purely IN-1k (no IN-21k contamination).
#
# Track B — Large-scale external pretraining:
#   - CLIP ViT-B/16 (86M, timm OpenAI weights) — used by every DomainNet CLIP-prompt-tuning
#     paper (CoOp IJCV'22, CoCoOp CVPR'22, MaPLe CVPR'23, SPG'24, DPSPG'25, CLIPCEIL NeurIPS'24).
#   - DINOv2 ViT-B/14 (86M, transformers HF) — self-supervised on LVD-142M; ERM++ Table 3(b)
#     reports DomainNet-avg 59.8% with this checkpoint; growing 2024-2025 SSL DG ablation.
RUN HF_HUB_DISABLE_PROGRESS_BARS=1 python -c "\
from torchvision.models import resnet50, ResNet50_Weights; \
resnet50(weights=ResNet50_Weights.IMAGENET1K_V1); \
print('ResNet-50 IN-1k V1 cached (torchvision)')" \
 && HF_HUB_DISABLE_PROGRESS_BARS=1 python -c "\
import timm; \
timm.create_model('vit_base_patch16_224.augreg_in1k', pretrained=True); \
print('ViT-B/16 IN-1k AugReg cached (timm)')" \
 && HF_HUB_DISABLE_PROGRESS_BARS=1 python -c "\
import timm; \
timm.create_model('vit_base_patch16_clip_224.openai', pretrained=True); \
print('CLIP ViT-B/16 OpenAI cached (timm)')" \
 && HF_HUB_DISABLE_PROGRESS_BARS=1 python -c "\
from transformers import AutoModel, AutoImageProcessor; \
AutoModel.from_pretrained('facebook/dinov2-base'); \
AutoImageProcessor.from_pretrained('facebook/dinov2-base'); \
print('DINOv2 ViT-B/14 model+processor cached (transformers)')"
