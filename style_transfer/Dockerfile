FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -U pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Set up a non-root user for better security
ARG USER_ID=1000
RUN useradd -m -u $USER_ID -s /bin/bash user
USER user

# Set environment variables
ENV PYTHONPATH=/workspace:$PYTHONPATH
ENV PATH=/workspace:$PATH

# Default command
CMD ["bash"]
