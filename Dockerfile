FROM python:3.11-slim
# FROM llamaedge/gemma-2b:latest

WORKDIR /app

# Install curl to fetch uv
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# ---- Install uv ----
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:/root/.local/bin:$PATH"

WORKDIR /app

COPY . .
RUN uv sync --frozen

ENV OLLAMA_URL=http://host.docker.internal:11434
ENV GEMMA_MODEL=gemma:2b
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["uv", "run", "python", "cli.py"]
# CMD ["bash", "-c", "ollama run gemma:2b && uv run python cli.py"]