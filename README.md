# CryptoCracker - GPU-Accelerated Ethereum Wallet Recovery

High-performance CUDA-based tool for recovering Ethereum wallet passwords using NVIDIA GPUs.

## Features

- **GPU Acceleration**: Leverages NVIDIA CUDA for 10-15M attempts/second on RTX 3080
- **Checkpoint/Resume**: SQLite-based checkpoint system for power failure recovery
- **Docker Support**: Fully containerized with docker-compose
- **Real-time Monitoring**: Grafana dashboard for performance metrics
- **Multi-GPU Support**: Scale across multiple GPUs
- **Automatic Progress Saving**: Checkpoints every 10 seconds

## Requirements

- NVIDIA GPU (Compute Capability 3.5+)
- NVIDIA Driver 470+
- Docker with NVIDIA Container Toolkit
- 8GB+ RAM

## Quick Start

### 1. Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/cryptocracker
cd cryptocracker

# Setup environment
make setup

# Initialize configuration
make init-config
```

### 2. Prepare Files
```bash
# Place your wallet file
cp /path/to/UTC--2016-*.json wallets/wallet.json

# Edit base passwords
nano config/passwords.txt
```

### 3. Run
```bash
# Build and start cracking
make docker-build
make crack

# Resume after interruption
make resume

# Monitor progress
make monitor
```

## Performance

| GPU | Speed | Time (7.4B combinations) |
|-----|-------|-------------------------|
| RTX 3080 | 10-15M/s | 8-12 minutes |
| RTX 3090 | 15-20M/s | 6-8 minutes |
| RTX 4090 | 25-30M/s | 4-5 minutes |

## Architecture

```
┌─────────────────────────┐
│    CLI Interface        │
├─────────────────────────┤
│  Checkpoint Manager     │ ← SQLite DB
├─────────────────────────┤
│    GPU Engine          │
├─────────────────────────┤
│  CUDA Kernels (8704    │
│  cores on RTX 3080)    │
└─────────────────────────┘
```

## Usage

### Basic Usage
```bash
./cryptocracker --wallet wallet.json --passwords passwords.txt
```

### Advanced Options
```bash
./cryptocracker \
  --wallet wallet.json \
  --passwords passwords.txt \
  --gpu 0 \
  --suffix-length 5 \
  --checkpoint checkpoint.db \
  --max-attempts 1000000000
```

### Docker Commands
```bash
make docker-up    # Start services
make docker-down  # Stop services
make logs        # View logs
make clean       # Clean build files
```

## Configuration

### Password File Format
```
password1
password2
mypassword
```

### Suffix Generation
- Charset: `a-zA-Z0-9!@#$%^&*()_+-=[]{}|;:,.<>?/~``
- Length: 4-5 characters
- Total combinations: ~7.4 billion

## Monitoring

Access Grafana dashboard at http://localhost:3000
- Username: admin
- Password: admin

## Security Note

This tool is for legitimate wallet recovery only. Only use on wallets you own.

## License

MIT