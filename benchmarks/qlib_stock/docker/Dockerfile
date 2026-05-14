ARG BASE_TAG=cu118
FROM farbench/farbench:base-${BASE_TAG}

# Qlib CSI300 Stock Prediction
# Metrics: IC (primary), ICIR, Rank IC

RUN pip install --no-cache-dir \
    pyqlib==0.9.6 \
    lightgbm==4.5.0 \
    catboost==1.2.7 \
    xgboost==2.1.1 \
    scikit-learn==1.5.1 \
    pandas==2.2.2 \
    joblib==1.3.2
