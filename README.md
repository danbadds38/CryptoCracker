# CryptoCracker - GPU-Accelerated Ethereum Wallet Recovery

High-performance CUDA-based tool for recovering Ethereum (Mist) wallet passwords using NVIDIA GPUs with real cryptographic implementations.

## ⚠️ IMPORTANT: Performance Reality

This tool implements **real Ethereum wallet cryptography** including scrypt with N=262144, which is designed to be extremely slow. Unlike theoretical speeds, actual performance is limited by the memory-hard scrypt function.

### Real-World Performance on RTX 3080

| Operation | Speed |
|-----------|-------|
| **Theoretical (no crypto)** | 1+ billion/sec |
| **Real scrypt (N=262144)** | **~10-100/sec** |
| **Real PBKDF2 (c=4096)** | ~10,000/sec |

## Features

- **GPU Acceleration**: CUDA-based implementation for NVIDIA GPUs
- **Real Cryptography**: Actual Keccak-256, scrypt, PBKDF2, and AES-128-CTR
- **Ethereum Compatible**: Reads actual wallet.json KDF parameters
- **Checkpoint/Resume**: SQLite-based system for power failure recovery
- **Docker Support**: Fully containerized with NVIDIA GPU support
- **Variable Suffix Length**: Support for 1-12 character suffixes
- **Memory Efficient**: Optimized for scrypt's memory requirements

## Requirements

- NVIDIA GPU (RTX 3080 or better recommended)
- NVIDIA Driver 470+
- Docker with NVIDIA Container Toolkit
- 10GB+ GPU memory for scrypt operations
- Linux or WSL2

## Quick Start

### 1. Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/cryptocracker
cd CryptoCracker

# Build the Docker container
make docker-build

# Start the container
make docker-up
```

### 2. Prepare Your Wallet and Passwords
```bash
# Place your Ethereum wallet file
cp /path/to/UTC--2016-*.json wallets/wallet.json

# Edit base passwords (one per line)
nano config/passwords.txt
```

### 3. Run Password Recovery
```bash
# Test with 3-character suffixes (fastest)
make crack-3

# Test with 4-character suffixes
make crack-4

# Test with 5-character suffixes
make crack-5

# Custom suffix length
docker exec cryptocracker ./build/cryptocracker \
  --wallet /app/wallets/wallet.json \
  --passwords /app/config/passwords.txt \
  --suffix-length 6
```

## Time Estimates for Real Ethereum Wallets

### With scrypt (N=262144, r=8, p=1) - Mist Wallets

Based on **real performance of ~10-100 attempts/second** on RTX 3080:

| Suffix Length | Combinations (13 passwords) | Time @ 10/sec | Time @ 100/sec |
|--------------|----------------------------|---------------|----------------|
| 1 char | 1,222 | 2 minutes | 12 seconds |
| 2 chars | 114,868 | 3.2 hours | 19 minutes |
| 3 chars | 10.8 million | **12.5 days** | **30 hours** |
| 4 chars | 1 billion | **3.2 years** | **116 days** |
| 5 chars | 95 billion | **301 years** | **30 years** |
| 6 chars | 8.9 trillion | **28,000+ years** | **2,800+ years** |

### With PBKDF2 (c=4096) - Older Wallets

Based on ~10,000 attempts/second:

| Suffix Length | Combinations (13 passwords) | Time Estimate |
|--------------|----------------------------|---------------|
| 1 char | 1,222 | < 1 second |
| 2 chars | 114,868 | 11 seconds |
| 3 chars | 10.8 million | 18 minutes |
| 4 chars | 1 billion | 28 hours |
| 5 chars | 95 billion | 110 days |
| 6 chars | 8.9 trillion | 28 years |

## Understanding the Challenge

Ethereum wallets use **scrypt** specifically to make brute-force attacks impractical:
- **N=262144**: Requires 256MB of memory per attempt
- **r=8, p=1**: Additional mixing rounds
- **Result**: Each password attempt takes ~10-100ms instead of microseconds

This is **by design** - the wallet format prioritizes security over convenience.

## Realistic Usage Scenarios

### ✅ Practical (might succeed)
- You know the exact base password and 1-2 unknown characters
- You have a short list of passwords with minor variations
- The wallet uses PBKDF2 instead of scrypt

### ⚠️ Challenging (weeks to months)
- You know the base password but missing 3 characters
- You have 10-20 base passwords to try with 2-character suffixes

### ❌ Impractical (years to centuries)
- Missing 4+ characters with scrypt wallets
- Large password lists with unknown suffixes
- No knowledge of the base password

## Configuration

### Password File Format (config/passwords.txt)
```
cashfl0w
K0sk3sh
Rashiddy
Freedom
mypassword123
```

### Character Set for Suffixes
- Lowercase: a-z (26)
- Uppercase: A-Z (26)
- Numbers: 0-9 (10)
- Symbols: !@#$%^&*()_+-=[]{}|;:,.<>?/~` (32)
- **Total**: 94 characters

## Docker Commands

```bash
make docker-build   # Build container
make docker-up      # Start services
make docker-down    # Stop services
make clean          # Clean build artifacts
make logs           # View logs
```

## Advanced Options

```bash
./cryptocracker \
  --wallet wallet.json \
  --passwords passwords.txt \
  --suffix-length 3 \
  --checkpoint checkpoint.db \
  --max-attempts 1000000 \
  --gpu 0
```

## Optimization Tips

1. **Start with shortest suffix lengths** - Each additional character multiplies time by 94x
2. **Minimize base passwords** - Each password multiplies total attempts
3. **Check wallet type first** - PBKDF2 is 100x faster than scrypt
4. **Use checkpoint system** - Allows resuming after interruptions
5. **Monitor GPU temperature** - Sustained loads can cause thermal throttling

## Technical Details

### Cryptographic Pipeline
1. Generate password candidate (base + suffix)
2. Derive key using scrypt/PBKDF2 with wallet's parameters
3. Calculate MAC using Keccak-256
4. Compare with wallet's stored MAC
5. If match, decrypt private key with AES-128-CTR

### GPU Memory Requirements
- Each scrypt attempt needs 256KB scratch memory
- Batch size limited to 32 passwords (8MB total)
- RTX 3080 with 10GB can handle this comfortably

## Security Note

This tool is for **legitimate wallet recovery only**. Only use on wallets you own. The computational requirements make unauthorized use impractical anyway.

## Troubleshooting

### "Out of memory" errors
- Reduce batch size in gpu_engine.cu
- Close other GPU applications

### Very slow performance
- This is normal for scrypt wallets
- Check if wallet uses PBKDF2 instead (100x faster)
- Ensure GPU is not thermal throttling

### Checkpoint not saving
- Ensure /data directory is writable
- Check disk space

## License

MIT - See LICENSE file for details