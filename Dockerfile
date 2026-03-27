# ── Base image ──────────────────────────────────────────────
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# ── Install dependencies first (Docker layer caching) ───────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy all project files ───────────────────────────────────
COPY . .

# ── Expose port 7860 (required by Hugging Face Spaces) ──────
EXPOSE 7860

# ── Health check ─────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/')"

# ── Start the FastAPI server ─────────────────────────────────
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
