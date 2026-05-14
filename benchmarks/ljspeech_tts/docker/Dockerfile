ARG BASE_TAG=cu118
FROM farbench/farbench:base-${BASE_TAG}
ARG HF_TOKEN=""
ENV HF_TOKEN=${HF_TOKEN}

# LJSpeech TTS: single-speaker text-to-speech synthesis
#   speechbrain: TTS inference (FastSpeech2, HiFi-GAN vocoder)
#   g2p-en: grapheme-to-phoneme conversion (CMUDict + neural fallback)
#   phonemizer: text-to-IPA via espeak-ng backend
#   pymcd: mel cepstral distortion (evaluator + agent self-eval)
#   openai-whisper + jiwer: ASR-based intelligibility checking
#   UTMOS metric uses the pre-cached SpeechMOS torch.hub repo, not PyPI `utmos`

# System: espeak-ng for phonemizer backend
RUN apt-get update && apt-get install -y --no-install-recommends \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

# Task-specific Python packages
# Pin transformers<4.47 to avoid BertTokenizer.encode_plus removal that breaks
# SpeechBrain's soundchoice-g2p word embedding pipeline.
RUN echo "setuptools<81" > /tmp/pip-constraints.txt && \
    PIP_CONSTRAINT=/tmp/pip-constraints.txt pip install --no-cache-dir \
    "transformers>=4.40,<4.47" \
    speechbrain==1.1.0 \
    g2p-en==2.1.0 \
    phonemizer==3.3.0 \
    pymcd==0.2.1 \
    openai-whisper==20240930 \
    jiwer==3.0.4

# Pre-download NLTK data for g2p_en (offline fallback G2P)
RUN python -c "import nltk; nltk.download('cmudict'); nltk.download('averaged_perceptron_tagger'); nltk.download('averaged_perceptron_tagger_eng')"

# Pre-download SpeechBrain HiFi-GAN vocoder (mel → waveform, trained on LJSpeech)
RUN HF_HUB_DISABLE_PROGRESS_BARS=1 python -c "\
from speechbrain.inference.vocoders import HIFIGAN; \
HIFIGAN.from_hparams(source='speechbrain/tts-hifigan-ljspeech', savedir='/opt/pretrained_models/tts-hifigan-ljspeech'); \
print('HiFi-GAN cached')"

# Pre-download SpeechBrain FastSpeech2 acoustic model (text → mel, trained on LJSpeech)
RUN HF_HUB_DISABLE_PROGRESS_BARS=1 python -c "\
from speechbrain.inference.TTS import FastSpeech2; \
FastSpeech2.from_hparams(source='speechbrain/tts-fastspeech2-ljspeech', savedir='/opt/pretrained_models/tts-fastspeech2-ljspeech'); \
print('FastSpeech2 cached')"

# BigVGAN inference wrapper (--no-deps avoids pulling an older huggingface-hub)
# Adds a modern vocoder option beyond HiFi-GAN (ICLR 2023, anti-aliased generator)
RUN pip install --no-cache-dir --no-deps \
    bigvganinference==0.0.3 \
    ninja==1.13.0

# Pre-download BigVGAN v2 22kHz 80-band checkpoint (~112M params, mel→waveform)
# Note: bigvganinference 0.0.3's from_pretrained is incompatible with huggingface-hub 1.x;
# we fetch config + generator.pt via hf_hub_download and instantiate manually.
RUN HF_HUB_DISABLE_PROGRESS_BARS=1 python -c "\
from huggingface_hub import hf_hub_download; \
hf_hub_download(repo_id='nvidia/bigvgan_v2_22khz_80band_256x', filename='config.json'); \
hf_hub_download(repo_id='nvidia/bigvgan_v2_22khz_80band_256x', filename='bigvgan_generator.pt'); \
print('BigVGAN v2 cached')"

# Pre-download UTMOS (automated MOS predictor) — used by evaluator.
# Caches the SpeechMOS torch.hub repo + wav2vec2 weights in TORCH_HOME.
ENV TORCH_HOME=/root/.cache/torch
RUN python -c "\
import torch; \
predictor = torch.hub.load('tarepan/SpeechMOS:v1.2.0', 'utmos22_strong', trust_repo=True); \
dummy = torch.randn(1, 16000); \
score = predictor(dummy, sr=16000); \
print(f'UTMOS test score: {score.item():.2f}'); \
print('UTMOS cached')"

# Pre-download Whisper small.en (244M, for WER-based intelligibility self-eval)
# small.en is sufficient for clean TTS output; medium.en added ~1GB for negligible WER gain
RUN python -c "\
import whisper; \
whisper.load_model('small.en'); \
print('Whisper small.en cached')"

# End-to-end smoke test: verify FastSpeech2 + HiFi-GAN pipeline works
RUN python -c "\
from speechbrain.inference.TTS import FastSpeech2; \
from speechbrain.inference.vocoders import HIFIGAN; \
fs2 = FastSpeech2.from_hparams(source='speechbrain/tts-fastspeech2-ljspeech', savedir='/opt/pretrained_models/tts-fastspeech2-ljspeech'); \
hfg = HIFIGAN.from_hparams(source='speechbrain/tts-hifigan-ljspeech', savedir='/opt/pretrained_models/tts-hifigan-ljspeech'); \
mel, dur, pitch, energy = fs2.encode_text(['Hello world, this is a test.']); \
wav = hfg.decode_batch(mel); \
assert wav.shape[-1] > 1000, f'Synthesis failed: wav shape {wav.shape}'; \
print(f'TTS smoke test PASSED: wav shape {wav.shape}')"

# Smoke test g2p_en fallback (offline, uses pre-downloaded NLTK data)
RUN python -c "\
from g2p_en import G2p; \
g2p = G2p(); \
ph = g2p('Hello world'); \
assert len(ph) > 3, f'g2p_en failed: {ph}'; \
print(f'g2p_en smoke test PASSED: {ph}')"
