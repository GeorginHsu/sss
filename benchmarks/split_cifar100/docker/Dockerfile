ARG BASE_TAG=cu118
FROM farbench/farbench:base-${BASE_TAG}
ARG HF_TOKEN=""
ENV HF_TOKEN=${HF_TOKEN}

# Split-CIFAR-100 continual learning (Class-IL, ResNet-18 from scratch)
#   quadprog: QP solver for GEM/A-GEM gradient episodic memory methods
#   kornia:   differentiable data augmentation (common in CL pipelines)

RUN pip install --no-cache-dir \
    quadprog==0.1.12 \
    kornia==0.7.3

# Pre-download models that recent CL papers (2023-2025) use for Split-CIFAR-100:
#   - ResNet-18 (11.7M) — standard CL protocol backbone (EWC, GEM, ER, DER++)
#   - ViT-B/16 IN-21k (86M) — prompt-based CL: L2P (CVPR'22), DualPrompt (ECCV'22), CODA-Prompt (CVPR'23)
#   - CLIP ViT-B/16 (86M) — AttriCLIP (CVPR'23), ZSCL (NeurIPS'23) CL methods
RUN HF_HUB_DISABLE_PROGRESS_BARS=1 python -c "\
from torchvision.models import resnet18, ResNet18_Weights; \
resnet18(weights=ResNet18_Weights.IMAGENET1K_V1); \
print('torchvision cached')" \
 && HF_HUB_DISABLE_PROGRESS_BARS=1 python -c "\
import timm; \
timm.create_model('vit_base_patch16_224.augreg_in21k_ft_in1k', pretrained=True); \
timm.create_model('vit_base_patch16_clip_224.openai', pretrained=True); \
print('timm ViT+CLIP cached')"
