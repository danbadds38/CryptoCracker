#!/usr/bin/env python3
import json
import sys
import time
from eth_keyfile import decode_keyfile_json

# Load wallet
with open('wallets/wallet.json', 'r') as f:
    wallet = json.load(f)

base_password = "k0sk3sh$1"
charset = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~"

print(f"Testing {len(charset)} passwords...")
print(f"Base: {base_password}")

start_time = time.time()

for i, char in enumerate(charset):
    password = base_password + char
    
    if i % 10 == 0:
        print(f"Testing position {i}: {password}")
    
    try:
        # This will throw an exception if password is wrong
        private_key = decode_keyfile_json(wallet, password.encode('utf-8'))
        
        print(f"\n✅ PASSWORD FOUND: {password}")
        print(f"Private key: {private_key.hex()}")
        print(f"Time taken: {time.time() - start_time:.2f} seconds")
        sys.exit(0)
        
    except ValueError:
        # Wrong password, continue
        continue

print(f"\n❌ Password not found")
print(f"Time taken: {time.time() - start_time:.2f} seconds")