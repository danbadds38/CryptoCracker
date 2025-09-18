#!/usr/bin/env python3
import json
import itertools
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import sys

def load_wallet(wallet_path):
    with open(wallet_path, 'r') as f:
        return json.load(f)

def test_password_batch(args):
    """Test a batch of passwords"""
    wallet, passwords = args
    from eth_keyfile import decode_keyfile_json
    
    for password in passwords:
        try:
            private_key = decode_keyfile_json(wallet, password.encode('utf-8'))
            return (True, password, private_key.hex())
        except:
            pass
    return (False, None, None)

def generate_passwords(base_passwords, suffix_length):
    """Generate all password combinations"""
    charset = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~"
    
    for base in base_passwords:
        for suffix in itertools.product(charset, repeat=suffix_length):
            yield base + ''.join(suffix)

def main():
    # Configuration
    wallet_path = 'wallets/wallet.json'
    password_file = 'config/passwords.txt'
    suffix_length = 3  # Start with 3
    batch_size = 100
    num_workers = 36  # Your i9-10980XE has 36 threads
    
    # Load wallet
    wallet = load_wallet(wallet_path)
    
    # Load base passwords
    with open(password_file, 'r') as f:
        base_passwords = [line.strip() for line in f if line.strip()]
    
    print(f"Testing {len(base_passwords)} base passwords with {suffix_length}-char suffixes")
    print(f"Total combinations: {len(base_passwords) * (95 ** suffix_length):,}")
    print(f"Using {num_workers} CPU threads")
    
    # Generate passwords in batches
    password_gen = generate_passwords(base_passwords, suffix_length)
    
    start_time = time.time()
    tested = 0
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        
        while True:
            # Create batch
            batch = list(itertools.islice(password_gen, batch_size))
            if not batch:
                break
                
            # Submit batch to worker
            future = executor.submit(test_password_batch, (wallet, batch))
            futures.append(future)
            
            # Process completed futures
            while futures and futures[0].done():
                future = futures.pop(0)
                found, password, key = future.result()
                
                if found:
                    elapsed = time.time() - start_time
                    print(f"\n✅ PASSWORD FOUND: {password}")
                    print(f"Private key: {key}")
                    print(f"Time: {elapsed:.2f} seconds")
                    print(f"Speed: {tested/elapsed:.1f} H/s")
                    executor.shutdown(wait=False)
                    sys.exit(0)
                
                tested += batch_size
                if tested % 10000 == 0:
                    elapsed = time.time() - start_time
                    speed = tested / elapsed
                    print(f"Tested: {tested:,} | Speed: {speed:.1f} H/s | Time: {elapsed:.1f}s")
    
    print(f"\n❌ Password not found")
    print(f"Total tested: {tested:,}")
    print(f"Time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()