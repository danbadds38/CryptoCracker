#!/usr/bin/env python3
import json
import hashlib
from Crypto.Protocol.KDF import scrypt
from Crypto.Hash import keccak

# Load wallet
with open('wallets/wallet.json', 'r') as f:
    wallet = json.load(f)

crypto = wallet.get('Crypto') or wallet.get('crypto')
kdf_params = crypto['kdfparams']

# Test password
password = 'k0sk3sh$12'
print(f"Testing password: {password}")

# Derive key using scrypt
derived = scrypt(
    password.encode('utf-8'),
    bytes.fromhex(kdf_params['salt']),
    key_len=kdf_params['dklen'],
    N=kdf_params['n'],
    r=kdf_params['r'],
    p=kdf_params['p']
)

print(f"Derived key: {derived.hex()}")

# Verify MAC
ciphertext = bytes.fromhex(crypto['ciphertext'])
mac_body = derived[16:32] + ciphertext

keccak_hash = keccak.new(digest_bits=256)
keccak_hash.update(mac_body)
calculated_mac = keccak_hash.hexdigest()

expected_mac = crypto['mac']
print(f"Expected MAC: {expected_mac}")
print(f"Calculated MAC: {calculated_mac}")

if calculated_mac.lower() == expected_mac.lower():
    print("PASSWORD IS CORRECT!")
else:
    print("Password is incorrect")