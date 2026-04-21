# Hugging Face Spaces Dockerfile for the market-terminal app.
# HF Spaces expects the container to listen on port 7860 by default.

FROM python:3.12-slim

# System dependency: libgomp (OpenMP) is required by both XGBoost and
# some sklearn internals on Linux. The workaround that was needed on
# macOS (a separate libomp.dylib) is not needed here since both libs
# link against the same system libgomp.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces runs the container as a non-root user with UID 1000.
# Set up a matching user and a writable home.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

WORKDIR $HOME/app

# Install dependencies first (better layer caching).
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application, including pre-trained model
# artifacts under models/. Training from scratch inside the container
# would exceed the free tier's build-time limits and is unnecessary --
# we ship the trained models directly.
COPY --chown=user . .

# Hugging Face Spaces expects the web server on port 7860.
EXPOSE 7860

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
