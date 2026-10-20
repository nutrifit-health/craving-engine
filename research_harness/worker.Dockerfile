# Operator resolves BASE to an official digest before this build. No paid access.
ARG BASE
FROM ${BASE}
ARG OPENCODE_SHA256=8b743b66286b6170dadff3a4383966c6dec312e1b8a0f0f1c496006ccbbfa959
LABEL org.nutrifit.opencode-commit="23fb5e0516c99ac04a1aa46c193efda2e1b9bb24"
LABEL org.nutrifit.research-worker="v1"
LABEL org.nutrifit.opencode-version="1.14.18"
LABEL org.nutrifit.opencode-archive-sha256="8b743b66286b6170dadff3a4383966c6dec312e1b8a0f0f1c496006ccbbfa959"
COPY opencode-linux-arm64.tar.gz /tmp/opencode.tar.gz
RUN echo "${OPENCODE_SHA256}  /tmp/opencode.tar.gz" | sha256sum -c - \
 && mkdir -p /opt/opencode/bin \
 && tar -xzf /tmp/opencode.tar.gz -C /opt/opencode/bin \
 && rm /tmp/opencode.tar.gz \
 && /opt/opencode/bin/opencode --version | grep -Fx '1.14.18'
RUN pip install --no-cache-dir numpy==2.5.3 scipy==1.18.1 scikit-learn==1.9.1 joblib==1.6.0 threadpoolctl==3.7.0 pydantic==2.12.5
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
USER 65534:65534
