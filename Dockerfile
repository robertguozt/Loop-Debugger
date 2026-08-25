FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

WORKDIR /app
COPY pyproject.toml README.md ./
COPY loopdbg ./loopdbg
RUN pip install --no-cache-dir . && useradd --uid 1000 --create-home runner

USER 1000
ENTRYPOINT ["loopdbg"]
