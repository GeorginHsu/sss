ARG BASE_TAG=cu118
FROM farbench/farbench:base-${BASE_TAG}

# CogniPlan — Autonomous Exploration
# Metrics: travel_distance (primary, lower is better), explored_rate, success_rate

RUN pip install --no-cache-dir \
    ray[default]==2.35.0

# Install cogniplan_env (exploration environment).
# FARBench image archive build pipeline builds with benchmarks/cogniplan as the context.
COPY script/cogniplan_env.py /usr/local/lib/python3.10/dist-packages/cogniplan_env.py

# Pre-download ResNet-50 (~98MB, no auth needed)
RUN python -c "\
import torchvision.models as models; \
m = models.resnet50(weights='IMAGENET1K_V1'); \
print('ResNet-50 cached')"
