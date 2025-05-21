FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    htop \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Set up a non-root user for better security
ARG USERNAME=user
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Install code-server for VS Code remote development
RUN curl -fsSL https://code-server.dev/install.sh | sh

# Install Python and dev tools
RUN pip install --no-cache-dir -U pip setuptools wheel black pylint autopep8 ipython

# Set working directory
WORKDIR /workspace

# Set environment variables
ENV PYTHONPATH=/workspace:$PYTHONPATH
ENV PATH=/workspace:$PATH

# Switch to the non-root user
USER $USERNAME
