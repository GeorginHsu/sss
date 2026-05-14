ARG BASE_TAG=cu118
FROM farbench/farbench:base-${BASE_TAG}

# Objaverse 3D Generation (Novel View Synthesis)
# Data: Arb-Objaverse pre-rendered multi-view images
# Metrics: LPIPS (primary), PSNR, SSIM

# Install CUDA development tools for compiling torchmcubes
RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-toolkit-11-8 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    lpips==0.1.4 \
    scikit-image==0.24.0 \
    kornia==0.7.3 \
    && pip install --no-cache-dir git+https://github.com/tatsy/torchmcubes.git

# TripoSR and NVS dependencies
RUN pip install --no-cache-dir \
    huggingface_hub \
    transformers \
    einops \
    omegaconf \
    trimesh \
    diffusers \
    accelerate \
    imageio

RUN git clone https://github.com/VAST-AI-Research/TripoSR.git /opt/triposr && \
    sed -i 's/import rembg/try:\n    import rembg\nexcept ImportError:\n    rembg = None  # Optional for inference/' /opt/triposr/tsr/utils.py

# Install remaining TripoSR dependencies
RUN pip install --no-cache-dir \
    omegaconf==2.3.0 \
    Pillow==10.1.0 \
    einops==0.7.0 \
    transformers==4.35.0 \
    trimesh==4.0.5 \
    huggingface-hub \
    imageio[ffmpeg] \
    PyMCubes

# Set PYTHONPATH for TripoSR in multiple ways
ENV PYTHONPATH="/opt/triposr:${PYTHONPATH}"
RUN echo 'export PYTHONPATH="/opt/triposr:$PYTHONPATH"' >> /etc/profile.d/triposr.sh && \
    chmod +x /etc/profile.d/triposr.sh
RUN pip install --no-cache-dir huggingface-hub && \
    python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download( \
    repo_id='stabilityai/TripoSR', \
    local_dir='/models/triposr', \
    ignore_patterns=['examples*', '*.md', '*.txt', '.gitattributes'] \
)" && \
    # Resolve symlinks to ensure model files are actually present in the layer
    if [ -L /models/triposr/model.ckpt ]; then \
        cp -L /models/triposr/model.ckpt /models/triposr/model.ckpt.tmp && \
        mv /models/triposr/model.ckpt.tmp /models/triposr/model.ckpt; \
    fi && \
    echo "TripoSR model cached and resolved"

# Pre-download DINO ViT-B/16 config.json (needed by TripoSR's image tokenizer)
RUN python -c "\
from huggingface_hub import hf_hub_download; \
hf_hub_download(repo_id='facebook/dino-vitb16', filename='config.json'); \
print('DINO ViT-B/16 config cached')"

# Pre-download VGG-16 for LPIPS evaluation (~60MB, no auth needed)
RUN python -c "\
import lpips; \
model = lpips.LPIPS(net='vgg'); \
print('LPIPS VGG cached')"

# Pre-download ResNet-50 (~98MB, no auth needed)
RUN python -c "\
import torchvision.models as models; \
m = models.resnet50(weights='IMAGENET1K_V1'); \
print('ResNet-50 cached')"
