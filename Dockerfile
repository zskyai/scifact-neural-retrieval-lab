FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY retrieval_lab ./retrieval_lab
COPY scripts ./scripts
RUN pip install --no-cache-dir .

EXPOSE 8000
CMD ["python", "scripts/serve_rag.py", "--data-root", "/data/scifact", "--qdrant-url", "http://qdrant:6333"]
