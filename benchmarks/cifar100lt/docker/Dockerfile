ARG BASE_TAG=cu118
FROM farbench/farbench:base-${BASE_TAG}
ARG HF_TOKEN=""
ENV HF_TOKEN=${HF_TOKEN}

# CIFAR-100-LT long-tailed image classification
#   lightly: contrastive learning framework (SimCLR, MoCo, BYOL projectors)

RUN pip install --no-cache-dir \
    lightly==1.5.14

# Pre-download models that recent long-tailed recognition papers (2023-2025) use:
#   - CLIP ViT-B/16 (150M total, 86M vision) — CANDLE (KDD'24) uses CLIP as frozen feature extractor
#   - ResNet-50 (25M) — standard ImageNet-pretrained backbone for LT fine-tuning
#   - ConvNeXt-Tiny (28M) — strong CNN backbone for transfer learning
RUN HF_HUB_DISABLE_PROGRESS_BARS=1 python -c "\
import timm; \
timm.create_model('vit_base_patch16_clip_224.openai', pretrained=True); \
print('CLIP ViT-B/16 cached via timm')" \
 && HF_HUB_DISABLE_PROGRESS_BARS=1 python -c "\
from torchvision.models import resnet50, ResNet50_Weights; \
resnet50(weights=ResNet50_Weights.IMAGENET1K_V2); \
print('torchvision cached')" \
 && HF_HUB_DISABLE_PROGRESS_BARS=1 python -c "\
import timm; \
timm.create_model('convnext_tiny.fb_in22k_ft_in1k', pretrained=True); \
print('timm convnext cached')"
