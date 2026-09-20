# Moirai forecasting connector, containerised.
#
# Why Docker rather than another venv: uni2ts pins `scipy~=1.11.3`, which has no wheel for
# Python 3.13, so pip tries to compile scipy from source on Windows and fails. Pinning the
# whole connector to Python 3.11 inside a container is cleaner than maintaining a third
# interpreter on the host, and it keeps a heavy scientific stack out of the engine venv
# entirely.
#
# MCP speaks over stdio, and `docker run -i` connects the container's stdin/stdout to the
# caller -- so a containerised connector needs no special support in the registry. It is
# just a different `command`.
FROM python:3.11-slim

# libgomp: torch's CPU kernels need OpenMP at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# CPU torch first, from the CPU index: the default wheel drags in ~2 GB of CUDA libraries
# this connector will never use.
RUN pip install --no-cache-dir "torch>=2.2,<2.6" \
      --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir uni2ts "mcp>=2"

# Weights live on a mounted volume, so the image stays small and a model is downloaded
# once rather than per container.
ENV HF_HOME=/cache/huggingface
ENV PYTHONUNBUFFERED=1

WORKDIR /app
COPY moirai_forecast.py /app/moirai_forecast.py

# stdio transport: no ports, no network service.
ENTRYPOINT ["python", "/app/moirai_forecast.py"]
