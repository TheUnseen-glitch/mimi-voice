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

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install CPU PyTorch cleanly using extra-index-url (prevents Jinja2/typing_extensions metadata conflict)
RUN pip install --no-cache-dir torch torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

# Install application dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files and voiceprints
COPY --chown=user . .

# Render exposes $PORT (default 10000), HuggingFace uses 7860
EXPOSE 10000

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000}"]
