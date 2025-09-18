#!/usr/bin/env python3
"""
Convert Ethereum wallet.json to hashcat format
Hashcat mode 15700 for Ethereum Wallet, scrypt
"""
import json
import sys
import os

def wallet_to_hashcat(wallet_path):
    """Convert Ethereum wallet to hashcat format"""
    
    with open(wallet_path, 'r') as f:
        wallet = json.load(f)
    
    # Get crypto section (handle both 'Crypto' and 'crypto')
    crypto = wallet.get('Crypto', wallet.get('crypto'))
    if not crypto:
        print("Error: No crypto section found in wallet")
        sys.exit(1)
    
    # Extract required fields
    kdf = crypto.get('kdf')
    kdfparams = crypto.get('kdfparams', {})
    
    if kdf == 'scrypt':
        # Format for hashcat mode 15700 (Ethereum Wallet, scrypt)
        # $ethereum$s*N*r*p*salt*ciphertext*mac
        n = kdfparams.get('n', 262144)
        r = kdfparams.get('r', 8)
        p = kdfparams.get('p', 1)
        salt = kdfparams.get('salt')
        ciphertext = crypto.get('ciphertext')
        mac = crypto.get('mac')
        
        # Hashcat expects format: $ethereum$s*n*r*p*salt*ciphertext*mac
        hash_line = f"$ethereum$s*{n}*{r}*{p}*{salt}*{ciphertext}*{mac}"
        
        print(f"Wallet converted to hashcat format (mode 15700):")
        print(f"KDF: scrypt (N={n}, r={r}, p={p})")
        
    elif kdf == 'pbkdf2':
        # Format for hashcat mode 15600 (Ethereum Wallet, PBKDF2-HMAC-SHA256)
        # $ethereum$p*iterations*salt*ciphertext*mac
        iterations = kdfparams.get('c', 262144)
        salt = kdfparams.get('salt')
        ciphertext = crypto.get('ciphertext')
        mac = crypto.get('mac')
        
        hash_line = f"$ethereum$p*{iterations}*{salt}*{ciphertext}*{mac}"
        
        print(f"Wallet converted to hashcat format (mode 15600):")
        print(f"KDF: pbkdf2 (iterations={iterations})")
    
    else:
        print(f"Error: Unsupported KDF type: {kdf}")
        sys.exit(1)
    
    return hash_line

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 wallet_to_hash.py <wallet.json>")
        sys.exit(1)
    
    wallet_path = sys.argv[1]
    if not os.path.exists(wallet_path):
        print(f"Error: Wallet file not found: {wallet_path}")
        sys.exit(1)
    
    hash_line = wallet_to_hashcat(wallet_path)
    
    # Save to file
    with open('/crack/wallet.hash', 'w') as f:
        f.write(hash_line + '\n')
    
    print(f"\nHash saved to: wallet.hash")
    print("\nHash content:")
    print(hash_line)