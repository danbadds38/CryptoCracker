# Hashcat Docker container for Ethereum wallet cracking
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Install dependencies and CUDA/OpenCL support
RUN apt-get update && apt-get install -y \
    wget \
    git \
    make \
    gcc \
    g++ \
    python3 \
    python3-pip \
    unzip \
    ocl-icd-libopencl1 \
    opencl-headers \
    clinfo \
    && rm -rf /var/lib/apt/lists/*

# Install hashcat
WORKDIR /opt
RUN wget https://hashcat.net/files/hashcat-6.2.6.tar.gz && \
    tar xzf hashcat-6.2.6.tar.gz && \
    cd hashcat-6.2.6 && \
    make install && \
    cd .. && \
    rm -rf hashcat-6.2.6.tar.gz && \
    rm -rf hashcat-6.2.6

# Install Python dependencies for wallet conversion
RUN pip3 install eth-keyfile

# Create working directory
WORKDIR /crack

# Add conversion scripts
COPY wallet_to_hash.py generate_wordlist.py crack.sh manage_sessions.sh /crack/

# Make scripts executable
RUN chmod +x /crack/*.sh || true

# Default entrypoint
CMD ["/bin/bash"]