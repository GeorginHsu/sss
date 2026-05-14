ARG BASE_TAG=cu118
FROM farbench/farbench:base-${BASE_TAG}

# CROHME 2019 Handwritten Mathematical Expression Recognition (HMER)
#   pytorch-lightning:  standard training loop used by CoMER, BTTR, TAMER reference code
#   python-Levenshtein: O(n) edit distance for ExpRate<=1 / ExpRate<=2 metrics
#   lxml:               robust InkML XML parsing (prepare.py also needs it if reparsing)

RUN pip install --no-cache-dir \
    pytorch-lightning==2.4.0 \
    python-Levenshtein==0.26.1 \
    lxml==5.3.0

# Pre-download backbones that recent HMER papers (2022-2025) actually use:
#   - DenseNet-121 — dominant backbone in CoMER (ECCV'22), CAN (ECCV'22),
#                    TAMER (AAAI'25), MFH (2025). Standard choice for HMER.
#   - ResNet-50    — occasionally used (SAN CVPR'22 uses ResNet-style encoder);
#                    kept for agent flexibility.
# NOTE: No pretrained HMER-specific checkpoints are published on HF/torchhub;
# agent must train the decoder + encoder-decoder integration from scratch.

RUN HF_HUB_DISABLE_PROGRESS_BARS=1 python -c "\
from torchvision.models import densenet121, DenseNet121_Weights; \
densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1); \
print('DenseNet-121 cached')" \
 && HF_HUB_DISABLE_PROGRESS_BARS=1 python -c "\
from torchvision.models import resnet50, ResNet50_Weights; \
resnet50(weights=ResNet50_Weights.IMAGENET1K_V2); \
print('ResNet-50 cached')"
