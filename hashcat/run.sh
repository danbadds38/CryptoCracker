#!/bin/bash
# Quick runner for hashcat

SUFFIX=${1:-1}
echo "Running hashcat with $SUFFIX-character suffix..."

docker run --rm \
  --runtime=nvidia \
  --gpus all \
  -v $(pwd)/../wallets:/data/wallets:ro \
  -v $(pwd)/../config:/data/config:ro \
  -v $(pwd)/results:/results \
  -e SUFFIX_LENGTH=$SUFFIX \
  hashcat-hashcat:latest \
  bash -c "cp /data/wallets/wallet.json /data/wallet.json && cp /data/config/passwords.txt /data/passwords.txt && /crack/crack.sh"