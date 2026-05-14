ARG BASE_TAG=cu118
FROM farbench/farbench:base-${BASE_TAG}
ARG HF_TOKEN=""
ENV HF_TOKEN=${HF_TOKEN}

# ETTh1 multivariate time series forecasting
#   statsmodels:     ARIMA, exponential smoothing, seasonal decomposition, ADF test
#   reformer-pytorch: efficient attention (Reformer-style sparse attention)
#   local-attention:  local/sliding-window attention patterns
#   sympy:           symbolic math (used by some learned positional encodings)
#   chronos-forecasting: Amazon Chronos time series foundation models (ICML'24)

RUN pip install --no-cache-dir \
    statsmodels==0.14.4 \
    reformer-pytorch==1.4.4 \
    local-attention==1.11.2 \
    sympy==1.13.3 \
    chronos-forecasting==1.4.1 \
 && pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Pre-download time series foundation models (recent papers use these for ETTh1):
#   - Chronos-Bolt-Small (48M) — Amazon, ICML'24, fast zero-shot/fine-tune forecasting
#   - Chronos-T5-Small (46M) — original Chronos, seq2seq architecture
RUN HF_HUB_DISABLE_PROGRESS_BARS=1 python -c "\
from chronos import ChronosBoltPipeline, ChronosPipeline; \
ChronosBoltPipeline.from_pretrained('amazon/chronos-bolt-small'); \
ChronosPipeline.from_pretrained('amazon/chronos-t5-small'); \
print('Chronos models cached')"
