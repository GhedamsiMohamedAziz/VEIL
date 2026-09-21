# VEIL API + worker. One image: the worker is a thread in the API process
# (ADR-005), so there is nothing else to build.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# libgl/libglib are OpenCV's runtime deps; they are cheap and make the
# [camera] extra installable in the same image.
RUN apt-get update && apt-get install -y --no-install-recommends \
      git libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY veil ./veil
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision \
 && pip install .

ENV VEIL_DATABASE_URL=sqlite:////data/veil.db \
    VEIL_ARTIFACT_ROOT=/data/artifacts
VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "veil.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
