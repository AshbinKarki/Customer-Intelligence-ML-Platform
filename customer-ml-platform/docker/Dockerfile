# Multi-stage Dockerfile for ML Platform

FROM python:3.11-slim as base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# ==================== API Service ====================

FROM base as api

COPY . .

ENV PYTHONUNBUFFERED=1
ENV API_HOST=0.0.0.0
ENV API_PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ==================== Dashboard Service ====================

FROM base as dashboard

COPY . .

ENV PYTHONUNBUFFERED=1

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# ==================== Batch Processing Service ====================

FROM base as batch

COPY . .

ENV PYTHONUNBUFFERED=1

VOLUME ["/app/data"]

CMD ["python", "src/data/batch_processor.py"]

# ==================== Development Image ====================

FROM base as dev

COPY . .

RUN pip install --no-cache-dir ipython jupyter notebook

ENV PYTHONUNBUFFERED=1

EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
