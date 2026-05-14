ARG BASE_TAG=cu118
FROM farbench/farbench:base-${BASE_TAG}
ARG HF_TOKEN=""
ENV HF_TOKEN=${HF_TOKEN}

# VoiceBank+DEMAND speech enhancement (noisy → clean waveform)
#   pesq:  ITU-T P.862 Perceptual Evaluation of Speech Quality
#   pystoi: Short-Time Objective Intelligibility
#   audiomentations: audio data augmentation (noise, gain, pitch shift, etc.)

RUN pip install --no-cache-dir \
    pesq==0.0.4 \
    pystoi==0.4.1 \
    audiomentations==0.39.0 \
    speechbrain \
    asteroid \
    pyloudnorm \
    auraloss==0.4.0

# Pre-download SpeechBrain MetricGAN+ pretrained on VoiceBank+DEMAND (PESQ 3.15)
#   Agent can fine-tune with different hyperparameters, augmentation, loss, etc.
RUN HF_HUB_DISABLE_PROGRESS_BARS=1 python -c "\
from speechbrain.inference.enhancement import SpectralMaskEnhancement; \
SpectralMaskEnhancement.from_hparams(source='speechbrain/metricgan-plus-voicebank', savedir='/opt/pretrained_models/metricgan-plus-voicebank'); \
print('MetricGAN+ cached')"
