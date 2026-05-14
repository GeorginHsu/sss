ARG BASE_TAG=cu118
FROM farbench/farbench:base-${BASE_TAG}
ARG HF_TOKEN=""
ENV HF_TOKEN=${HF_TOKEN}

# Clotho audio captioning: environmental audio → text caption
#   aac-metrics: SPIDEr, CIDEr, SPICE, METEOR evaluation (needs Java for SPICE)
#   py7zr: extract .7z audio archives from Zenodo
#   sentence-transformers: FENSE/SBERT metrics (aac-metrics dependency)
#   panns-inference: load/use pre-downloaded CNN14 AudioSet encoder

# System: Java for SPICE/METEOR (Stanford CoreNLP)
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-11-jre-headless \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64 \
    TRANSFORMERS_NO_TF=1 \
    TRANSFORMERS_NO_FLAX=1 \
    USE_TF=0 \
    USE_FLAX=0

# Task-specific Python packages
RUN pip install --no-cache-dir \
    aac-metrics==0.5.5 \
    py7zr==0.22.0 \
    sentence-transformers==3.3.1 \
    panns-inference==0.1.1 \
    torchlibrosa==0.1.0

# Pre-download aac-metrics external data/models.
# Required offline for SPICE, METEOR, SPIDEr-FL, and FENSE computation.
RUN aac-metrics-download

# Pre-download PANNs CNN14 encoder (classic AudioSet CNN, 80M params).
# panns-inference defaults to /root/panns_data/Cnn14_mAP=0.431.pth;
# keep a stable /opt/pretrained_models symlink for agent-visible hints too.
RUN mkdir -p /opt/pretrained_models /root/panns_data && \
    curl -fSL -o /root/panns_data/Cnn14_mAP=0.431.pth \
    "https://zenodo.org/records/3987831/files/Cnn14_mAP%3D0.431.pth?download=1" && \
    ln -sf /root/panns_data/Cnn14_mAP=0.431.pth \
    /opt/pretrained_models/Cnn14_mAP_0.431.pth

# Pre-download BART-base text decoder (dominant decoder in DCASE winning systems)
RUN HF_HUB_DISABLE_PROGRESS_BARS=1 python -c "\
from transformers import BartForConditionalGeneration, BartTokenizer; \
BartForConditionalGeneration.from_pretrained('facebook/bart-base'); \
BartTokenizer.from_pretrained('facebook/bart-base'); \
print('BART-base cached')"

# Pre-download CED-base (Xiaomi, 85M, AudioSet mAP 50.0%)
# Referenced by LOAE (DCASE 2024 top, SPIDEr ~0.330) as audio encoder
RUN HF_HUB_DISABLE_PROGRESS_BARS=1 python -c "\
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor; \
AutoModelForAudioClassification.from_pretrained('mispeech/ced-base', trust_remote_code=True); \
AutoFeatureExtractor.from_pretrained('mispeech/ced-base', trust_remote_code=True); \
print('CED-base cached')"

# Pre-download LAION-CLAP (193M, audio-text joint embedding)
# Used by EnCLAP (ICASSP 2024) and SLAM-AAC's CLAP-Refine reranker
RUN HF_HUB_DISABLE_PROGRESS_BARS=1 python -c "\
from transformers import ClapModel, ClapProcessor; \
ClapModel.from_pretrained('laion/clap-htsat-unfused'); \
ClapProcessor.from_pretrained('laion/clap-htsat-unfused'); \
print('LAION-CLAP cached')"
