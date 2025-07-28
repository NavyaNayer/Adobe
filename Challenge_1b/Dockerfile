# Dockerfile for Challenge 1B - Persona-Driven Document Intelligence
# Compatible with AMD64 architecture, offline processing
# Optimized for batch processing of document collections

FROM --platform=linux/amd64 python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for PyMuPDF and ML libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libffi-dev \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first (for better Docker layer caching)
COPY requirements_docker.txt ./requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip cache purge

# Copy the Challenge_1b application code (includes all PDF outline extractors)
COPY run_challenge1b.py ./
COPY selector.py ./
COPY docker_runner_1b.py ./
COPY pdf_outline_extractor.py ./
COPY pdf_outline_multilang.py ./
COPY pdf_outline_nonlatin.py ./

# Create input and output directories with proper permissions
RUN mkdir -p /app/input /app/output && \
    chmod 755 /app/input /app/output

# Set the entrypoint
ENTRYPOINT ["python", "docker_runner_1b.py"]
