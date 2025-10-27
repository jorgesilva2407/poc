FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev git && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

RUN pip install uv

COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-dev --no-cache-dir

COPY main.py ./
COPY src ./src

ENV PYTHONBUFFERED=1

RUN uv run python --version

CMD ["uv", "run", "main.py"]
