ARG BASE_TAG=cu118
FROM farbench/farbench:base-${BASE_TAG}
ARG HF_TOKEN=""
ENV HF_TOKEN=${HF_TOKEN}

# WILDS-FMoW: satellite image classification with temporal/geographic distribution shift
#   wilds: WILDS benchmark utilities (grouper, evaluator, data transforms)

RUN pip install --no-cache-dir \
    wilds==2.0.0

# Pre-download models used by top WILDS-FMoW methods (2022-2025).
# Research (9/10 recent papers): CLIP ViT-L/14 is the dominant backbone on this benchmark
# (Model Soups, FLYP, AutoFT 51.8%, DRM 53.1%, VDPG — all use CLIP ViT-L/14).
# DINOv2 / ConvNeXt / DenseNet: absent from the 2023-2025 WILDS-FMoW literature, not pre-installed.
#   CLIP ViT-L/14 @224 (304M) — dominant backbone; fast-iteration default at 224 (~46-48% worst-region)
#   CLIP ViT-B/16  @224 (150M) — lightweight prototyping option (FLYP ViT-B ~41%)
# Cache the model AND the matching processor: CLIPModel.from_pretrained downloads
# only config.json + weights, NOT preprocessor_config.json / tokenizer files. Agents
# that call CLIPImageProcessor / CLIPProcessor / CLIPTokenizer would otherwise hit
# OSError under the offline (HF_HUB_OFFLINE=1) sandbox.
RUN HF_HUB_DISABLE_PROGRESS_BARS=1 python -c "\
from transformers import CLIPModel, CLIPProcessor; \
[ (CLIPModel.from_pretrained(m), CLIPProcessor.from_pretrained(m)) for m in \
  ('openai/clip-vit-large-patch14', 'openai/clip-vit-base-patch16') ]; \
print('CLIP model+processor cached')"
