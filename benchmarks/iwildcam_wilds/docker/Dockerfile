ARG BASE_TAG=cu118
FROM farbench/farbench:base-${BASE_TAG}
ARG HF_TOKEN=""
ENV HF_TOKEN=${HF_TOKEN}

# iWildCam-WILDS: wildlife camera trap OOD species classification (182 classes, 323 locations)
#   wilds: WILDS benchmark utilities (dataset loading, grouper, transforms)

RUN pip install --no-cache-dir \
    wilds==2.0.0

# Pre-download models aligned with iWildCam-WILDS 2023-2025 literature:
# Research (FLYP CVPR'23, VDPG ICLR'24, DRM NeurIPS'24, L2C ICLR'25, AutoFT 2024) confirms
# CLIP ViT-L/14 is the dominant SOTA backbone (46-52 Macro F1). ViT-B is kept as a lightweight
# prototyping option. DINOv2 / ResNet-50 are not in the 2023-2025 iWildCam literature and
# are omitted (ResNet-50 remains available via torchvision, no pre-download needed since
# it's small and ships with the base image).
#   - CLIP ViT-L/14 @224 (304M) — 2023-2025 dominant backbone
#   - CLIP ViT-B/16  @224 (150M) — lightweight prototyping
# Cache the model AND the matching processor: CLIPModel.from_pretrained downloads
# only config.json + weights, NOT preprocessor_config.json / tokenizer files. Agents
# that call CLIPImageProcessor / CLIPProcessor / CLIPTokenizer would otherwise hit
# OSError under the offline (HF_HUB_OFFLINE=1) sandbox.
RUN HF_HUB_DISABLE_PROGRESS_BARS=1 python -c "\
from transformers import CLIPModel, CLIPProcessor; \
[ (CLIPModel.from_pretrained(m), CLIPProcessor.from_pretrained(m)) for m in \
  ('openai/clip-vit-large-patch14', 'openai/clip-vit-base-patch16') ]; \
print('CLIP model+processor cached')"
