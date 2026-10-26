FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app

COPY pyproject.toml uv.lock README.md LICENSE ATTRIBUTION.md ./
COPY src/ src/
RUN pip install --no-cache-dir uv==0.11.28 \
    && uv sync --locked --no-dev --no-editable \
    && groupadd --system app && useradd --system --gid app app

ENV PATH="/app/.venv/bin:$PATH"
USER app
EXPOSE 8000
CMD ["uvicorn", "medical_multi_agent.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
