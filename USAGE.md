# CryptoCracker Usage Guide

## Quick Start

### 1. Place Your Wallet File
```bash
# Copy your actual wallet file to the wallets directory
cp /path/to/UTC--2016-05-01T18-11-51.988445605Z--579f2f10d38787ffb573f0ce3370f196f357fa69 wallets/wallet.json
```

### 2. Add Your Base Passwords
Edit `config/passwords.txt` and add your 10 known passwords, one per line:
```
password1
password2
mypassword
# ... up to 10 passwords
```

### 3. Run the Cracker
```bash
# Start cracking with GPU
docker run --rm --gpus all \
  -v $(pwd)/wallets:/app/wallets \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/checkpoints:/app/checkpoints \
  cryptocracker:latest \
  ./build/cryptocracker \
    --wallet /app/wallets/wallet.json \
    --passwords /app/config/passwords.txt \
    --gpu 0
```

### 4. Resume After Interruption
```bash
# If power fails or you stop, resume from checkpoint
docker run --rm --gpus all \
  -v $(pwd)/wallets:/app/wallets \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/checkpoints:/app/checkpoints \
  cryptocracker:latest \
  ./build/cryptocracker \
    --resume \
    --checkpoint /app/checkpoints/session.db
```

## Using Make Commands (Easier)

```bash
# Build the container
make docker-build

# Start cracking
make crack

# Resume from checkpoint
make resume

# Monitor logs
make logs
```

## Performance Expectations

With RTX 3080:
- **Speed**: ~10-15 million attempts/second
- **4-char suffix**: ~78 million combinations = ~5-8 seconds per password
- **5-char suffix**: ~7.3 billion combinations = ~8-12 minutes per password
- **Total time**: 8-12 minutes for all 10 passwords with 4-5 char suffixes

## Checkpoint System

- Automatically saves progress every 10 seconds
- Checkpoint stored in `checkpoints/session.db`
- Resume is instant - picks up exactly where it left off
- Power failure safe - no progress lost

## What Happens When Password is Found

1. The program will stop immediately
2. Display the found password
3. Show the decrypted private key in hex format
4. Clear the checkpoint (successful completion)

Example output:
```
========================================
[SUCCESS] PASSWORD FOUND!
Password: mypassword#8xZ!
Private Key: 0x3a1b2c3d4e5f...
========================================
```

## Troubleshooting

### GPU Not Detected
```bash
# Check GPU status
make gpu-status

# Ensure NVIDIA Container Toolkit is installed
sudo apt-get install nvidia-container-toolkit
sudo systemctl restart docker
```

### Out of Memory
Reduce batch size:
```bash
--batch-size 524288  # Use 512K instead of 1M
```

### Verify Wallet File
```bash
# Check if wallet file is valid JSON
cat wallets/wallet.json | python3 -m json.tool
```

## Security Notes

1. **Never share your private key** once recovered
2. **Delete checkpoint file** after successful recovery
3. **Keep wallet file secure** during the process
4. This tool only works on wallets you legally own