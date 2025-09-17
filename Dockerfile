FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libssl-dev \
    libsqlite3-dev \
    libjsoncpp-dev \
    python3-pip \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install additional crypto libraries
RUN apt-get update && apt-get install -y \
    libscrypt-dev \
    libsodium-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy source files
COPY . .

# Create directories
RUN mkdir -p build checkpoints wallets logs

# Build the application
RUN cd build && \
    cmake .. && \
    make -j$(nproc)

# Set environment variables
ENV CUDA_VISIBLE_DEVICES=0
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Default command
CMD ["./build/cryptocracker", "--help"]