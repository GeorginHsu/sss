ARG BASE_TAG=cu118
FROM farbench/farbench:base-${BASE_TAG}
ARG HF_TOKEN=""
ENV HF_TOKEN=${HF_TOKEN}

# MNIST classification (smoke test task)
# MNIST is trivially solvable — pretrained models are overkill but included
# for agent flexibility. A simple CNN from scratch achieves >99%.

# Pre-download standard vision backbones (agent may attempt transfer learning)
RUN HF_HUB_DISABLE_PROGRESS_BARS=1 python -c "\
from torchvision.models import resnet50, ResNet50_Weights; \
resnet50(weights=ResNet50_Weights.IMAGENET1K_V2); \
print('torchvision cached')"
