ARG BASE_TAG=cu118
FROM farbench/farbench:base-${BASE_TAG}
ARG BASE_TAG
ARG HF_TOKEN=""
ENV HF_TOKEN=${HF_TOKEN}

# ASVspoof 2021 LA: audio deepfake detection (bonafide vs spoof)
# Base provides: torchaudio, librosa, soundfile, scikit-learn, tqdm, libsndfile1, ffmpeg

RUN pip install --no-cache-dir \
    speechbrain==1.1.0 \
    pyloudnorm==0.1

# TorchAudio 2.9+ routes torchaudio.load()/save() through TorchCodec.  Pin the
# ABI-compatible TorchCodec release for each supported base image.
RUN if [ "$BASE_TAG" = "cu118" ]; then \
        pip install --no-cache-dir torchcodec==0.5; \
    elif [ "$BASE_TAG" = "cu128" ]; then \
        pip install --no-cache-dir torchcodec==0.11.1; \
    else \
        echo "Unsupported BASE_TAG=$BASE_TAG" >&2; exit 1; \
    fi

# Pre-download speech encoders for anti-spoofing:
#   - Wav2Vec2-Base (95M) — lightweight baseline for fast prototyping
#   - XLS-R 300M (300M) — multilingual wav2vec2, used in SOTA 0.82% EER system (Tak et al.)
#   - WavLM-Large (317M) — top WavLM variant, ASVspoof 5 winning approach
# Cache the model AND the matching feature extractor: AutoModel.from_pretrained
# downloads only config.json + weights, NOT preprocessor_config.json. Agents that
# call AutoFeatureExtractor / Wav2Vec2FeatureExtractor would otherwise hit OSError
# under the offline (HF_HUB_OFFLINE=1) sandbox.
RUN HF_HUB_DISABLE_PROGRESS_BARS=1 python -c "\
from transformers import AutoModel, AutoFeatureExtractor; \
[ (AutoModel.from_pretrained(m), AutoFeatureExtractor.from_pretrained(m)) for m in \
  ('facebook/wav2vec2-base', 'facebook/wav2vec2-xls-r-300m', 'microsoft/wavlm-large') ]; \
print('speech encoders model+feature_extractor cached')"
