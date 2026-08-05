FROM python:3.11-slim

# Install OpenMP runtime required by LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

# Install uv package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uv/bin/uv
ENV PATH="/uv/bin:${PATH}"

WORKDIR /app

# Copy dependency files and source code
COPY pyproject.toml uv.lock ./
COPY src/ ./src/

# Install exact locked dependencies
RUN uv sync --frozen

# Expose default Cloud Run port
EXPOSE 8080

# Launch Gunicorn server for Cloud Run
CMD ["uv", "run", "gunicorn", "--bind", "0.0.0.0:8080", "src.app:app"]
