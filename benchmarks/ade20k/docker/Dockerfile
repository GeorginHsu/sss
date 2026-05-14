ARG BASE_TAG=cu118
FROM farbench/farbench:base-${BASE_TAG}
ARG HF_TOKEN=""
ENV HF_TOKEN=${HF_TOKEN}

# ADE20K semantic segmentation (150 classes, 20K training images)
#   mmsegmentation: dominant segmentation framework (FCN, DeepLab, SegFormer, UperNet, etc.)
#   mmcv/mmengine:  OpenMMLab foundation (required by mmseg)
#   ftfy, regex:    text processing (needed by some vision backbones, e.g. CLIP)

# OpenMMLab ecosystem (mmsegmentation + dependencies)
# Pin setuptools<70 so pkg_resources stays available for mmcv source build
# (no pre-built mmcv wheel exists for torch 2.7.0)
RUN pip install --no-cache-dir \
    openmim==0.3.9 \
    "setuptools<70"

RUN mim install mmengine==0.10.5 && \
    pip install --no-cache-dir --no-build-isolation "mmcv==2.1.0" \
        -f https://download.openmmlab.com/mmcv/dist/cu118/index.html && \
    mim install mmsegmentation==1.2.2

RUN pip install --no-cache-dir \
    ftfy==6.3.1 \
    regex==2025.10.22

# Pre-download initialization backbones used by common/recent ADE20K research.
# Do not cache ADE20K-finetuned segmentation checkpoints here; those would turn
# the benchmark into direct inference instead of model building.
#   - ResNet-50 (25M)                 — FCN / DeepLabv3+ / classic UPerNet baseline
#   - MiT-B5 (81M)                    — SegFormer-B5 family
#   - Swin-L IN22K (197M)             — UPerNet / Mask2Former / OneFormer family
#   - ConvNeXt-L IN22K (198M)         — modern CNN backbone used in dense prediction
#   - BEiTv2-L (304M)                 — ViT-Adapter / Mask2Former family
#   - EVA-02-L (305M)                 — strong 2023 ViT-L backbone for ADE20K
#   - DINOv2 ViT-L/14 (304M)          — 2023-2025 foundation backbone, EoMT/ViT-Adapter
RUN HF_HUB_DISABLE_PROGRESS_BARS=1 python -c "\
from torchvision.models import resnet50, ResNet50_Weights; \
resnet50(weights=ResNet50_Weights.IMAGENET1K_V2); \
print('torchvision cached')"

RUN HF_HUB_DISABLE_PROGRESS_BARS=1 python -c "\
import timm; \
timm.create_model('swin_large_patch4_window12_384.ms_in22k_ft_in1k', pretrained=True); \
timm.create_model('convnext_large.fb_in22k_ft_in1k', pretrained=True); \
timm.create_model('beitv2_large_patch16_224.in1k_ft_in22k_in1k', pretrained=True); \
timm.create_model('eva02_large_patch14_448.mim_m38m_ft_in22k_in1k', pretrained=True); \
print('timm cached')"

RUN HF_HUB_DISABLE_PROGRESS_BARS=1 python -c "\
from transformers import AutoModel, AutoImageProcessor; \
AutoModel.from_pretrained('nvidia/mit-b5'); \
AutoImageProcessor.from_pretrained('nvidia/mit-b5'); \
print('MiT-B5 model+processor cached')"

RUN HF_HUB_DISABLE_PROGRESS_BARS=1 python -c "\
from transformers import AutoModel, AutoImageProcessor; \
AutoModel.from_pretrained('facebook/dinov2-large'); \
AutoImageProcessor.from_pretrained('facebook/dinov2-large'); \
print('DINOv2-L model+processor cached')"

# openmim is only needed during image build. Its openxlab dependency pins older
# requests/tqdm versions, so remove the build helper and restore base-compatible
# runtime packages after all OpenMMLab installs are complete.
RUN pip uninstall -y openmim opendatalab openxlab && \
    pip install --no-cache-dir \
        requests==2.33.1 \
        tqdm==4.67.3
