ARG BASE_TAG=cu118
FROM farbench/farbench:base-${BASE_TAG}
ARG HF_TOKEN=""
ENV HF_TOKEN=${HF_TOKEN}

# CIFAR-100N: learning with noisy labels
#   cleanlab: confident learning for noise detection and label cleaning

RUN pip install --no-cache-dir \
    cleanlab==2.7.0

# Pre-download models that recent noisy label papers (2023-2025) use:
#   - CLIP ViT-B/16 (86M vision) — DeFT (NeurIPS'24) uses CLIP for noisy label detection
#   - ResNet-50 (25M) — ImageNet-pretrained backbone for fine-tuning (IJCAI'24)
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
