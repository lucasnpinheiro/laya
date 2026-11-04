# Sempre a versão mais recente: Python 3 (maior minor disponível) sobre Debian slim (tag atual).
# Se torch CPU ou laya não tiverem wheel para essa versão de Python, fixar a maior minor que
# funciona e documentar o motivo AQUI (não pinar em silêncio).
FROM python:3-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    USE_TF=0 \
    USE_TORCH=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=6 \
    HF_HOME=/home/laya/.cache/huggingface

# Atualiza a base do SO a cada build (CVEs) e instala curl (healthcheck do docker-compose)
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 10001 laya \
    && useradd --uid 10001 --gid laya --create-home laya \
    && mkdir -p /home/laya/.cache/huggingface \
    && chown -R laya:laya /home/laya

WORKDIR /app

# Sempre a última release de laya, torch, fastapi e uvicorn (decisão do usuário). Regressão conhecida
# de laya >= 0.3.6 em ambiguidade PT: ver PR #1; a triagem PT é mitigada por LAYA_DEFAULT_MODEL=english (app.py).
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir --upgrade torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir --upgrade laya fastapi "uvicorn[standard]"

# Remove o pip da imagem final: o runtime não usa pip. O pip vendoriza setuptools 70.3.0
# (pip/_vendor/pkg_resources, CVE-2025-47273 no scout) e nenhuma release atual do pip corrige isso.
RUN python -m pip uninstall -y pip

COPY --chown=laya:laya app.py lowmem.py /app/

USER laya

VOLUME /home/laya/.cache/huggingface

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
