# ---- build stage: install deps into a venv so the runtime image stays slim
FROM python:3.12-slim AS build
WORKDIR /app
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY pyproject.toml README.md ./
COPY expforge ./expforge
RUN pip install --no-cache-dir .

# ---- runtime stage
FROM python:3.12-slim
RUN useradd --create-home --uid 1000 expforge
COPY --from=build /opt/venv /opt/venv
COPY configs /app/configs
WORKDIR /app
ENV PATH="/opt/venv/bin:$PATH" \
    EXPFORGE_HOME=/data \
    PYTHONUNBUFFERED=1
RUN mkdir -p /data && chown expforge /data
USER expforge
VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/api/health')"
CMD ["expforge", "server", "--host", "0.0.0.0", "--port", "8000"]
