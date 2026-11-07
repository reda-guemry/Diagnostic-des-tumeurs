# syntax=docker/dockerfile:1.7

ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

# OpenCV (Ultralytics) and TensorFlow runtime libraries.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /uvx /bin/

# Install Python dependencies first so source-only changes reuse this layer.
# The datasets and model weights are deliberately not copied: Compose mounts
# them at runtime (see docker-compose.yml).
FROM base AS runtime

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY app ./app
COPY airflow ./airflow
COPY .streamlit ./.streamlit
RUN uv sync --frozen --no-dev --no-editable

ENV PATH="/app/.venv/bin:${PATH}" \
    BTD_PROJECT_ROOT=/app

EXPOSE 5000 8000 8501

CMD ["uvicorn", "diagnostic_des_tumeurs.api:app", "--host", "0.0.0.0", "--port", "8000"]


# Optional Airflow image. Build it only with the Compose "orchestration"
# profile. Airflow remains an optional dependency, exactly as in the local
# setup, so the default runtime image stays free of orchestration packages.
FROM runtime AS airflow

RUN uv pip install --python /app/.venv/bin/python -r airflow/requirements.txt

COPY docker/airflow-entrypoint.sh /usr/local/bin/airflow-entrypoint
RUN chmod +x /usr/local/bin/airflow-entrypoint

ENV AIRFLOW_HOME=/app/.airflow \
    AIRFLOW__CORE__LOAD_EXAMPLES=False

EXPOSE 8080

ENTRYPOINT ["/usr/local/bin/airflow-entrypoint"]
CMD ["airflow", "standalone"]
