FROM python:3.10-slim

# Install system audio libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set up non-root user
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Pin pip to avoid index name inconsistency bug in pip 24.1+
RUN pip install --no-cache-dir "pip<24.1"

# Pre-install Jinja2 and typing-extensions from PyPI to avoid pytorch wheel case mismatch
RUN pip install --no-cache-dir "Jinja2>=3.1.2" "typing-extensions>=4.10.0"

# Install CPU PyTorch from official CPU wheel repo
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install application dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files and voiceprints
COPY --chown=user . .

# Render exposes $PORT (default 10000), HuggingFace uses 7860
EXPOSE 10000

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000}"]
