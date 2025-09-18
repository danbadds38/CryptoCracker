#!/usr/bin/env python3
"""
Ethereum wallet password cracker using Python
Supports dynamic suffix lengths and multiple base passwords
"""
import json
import itertools
import sys
import time
import argparse
from multiprocessing import Pool, cpu_count
from eth_keyfile import decode_keyfile_json

def load_wallet(wallet_path):
    """Load wallet from JSON file"""
    with open(wallet_path, 'r') as f:
        return json.load(f)

def test_password(args):
    """Test a single password"""
    wallet, password = args
    try:
        private_key = decode_keyfile_json(wallet, password.encode('utf-8'))
        return (True, password, private_key.hex())
    except:
        return (False, None, None)

def test_password_batch(args):
    """Test a batch of passwords"""
    wallet, passwords = args
    for password in passwords:
        try:
            private_key = decode_keyfile_json(wallet, password.encode('utf-8'))
            return (True, password, private_key.hex())
        except:
            pass
    return (False, None, None)

def generate_passwords(base_passwords, suffix_length):
    """Generate all password combinations"""
    # Character set: 95 printable ASCII characters
    charset = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~"
    
    for base in base_passwords:
        for suffix in itertools.product(charset, repeat=suffix_length):
            yield base + ''.join(suffix)

def main():
    parser = argparse.ArgumentParser(description='Ethereum Wallet Password Cracker')
    parser.add_argument('--wallet', default='wallets/wallet.json', help='Path to wallet.json')
    parser.add_argument('--passwords', default='config/passwords.txt', help='Path to base passwords')
    parser.add_argument('--suffix-length', type=int, default=1, help='Suffix length (1-5)')
    parser.add_argument('--workers', type=int, default=0, help='Number of workers (0=auto)')
    parser.add_argument('--batch-size', type=int, default=100, help='Passwords per batch')
    parser.add_argument('--max-attempts', type=int, default=0, help='Max attempts (0=all)')
    
    args = parser.parse_args()
    
    # Load wallet
    print(f"Loading wallet: {args.wallet}")
    wallet = load_wallet(args.wallet)
    
    # Load base passwords
    print(f"Loading base passwords: {args.passwords}")
    with open(args.passwords, 'r') as f:
        base_passwords = [line.strip() for line in f if line.strip()]
    
    print(f"Base passwords: {len(base_passwords)}")
    print(f"Suffix length: {args.suffix_length}")
    print(f"Character set size: 95")
    
    total_combinations = len(base_passwords) * (95 ** args.suffix_length)
    print(f"Total combinations: {total_combinations:,}")
    
    if total_combinations > 1000000:
        print("\n⚠️  WARNING: Large search space!")
        print(f"Estimated time at 100 H/s: {total_combinations / 100 / 3600:.1f} hours")
        print(f"Estimated time at 10 H/s: {total_combinations / 10 / 3600:.1f} hours")
        
        if input("\nContinue? (y/n): ").lower() != 'y':
            sys.exit(0)
    
    # Determine number of workers
    num_workers = args.workers if args.workers > 0 else cpu_count()
    print(f"\nUsing {num_workers} CPU workers")
    
    # Start cracking
    start_time = time.time()
    tested = 0
    found = False
    
    print("\nStarting password cracking...")
    print("-" * 40)
    
    with Pool(num_workers) as pool:
        # Generate passwords in batches
        password_gen = generate_passwords(base_passwords, args.suffix_length)
        
        while True:
            # Create batch
            batch = list(itertools.islice(password_gen, args.batch_size))
            if not batch:
                break
            
            # Check max attempts
            if args.max_attempts > 0 and tested >= args.max_attempts:
                print(f"\n⏸️  Max attempts reached ({args.max_attempts})")
                break
            
            # Test batch in parallel
            # Create args for each worker (divide batch among workers)
            batch_per_worker = len(batch) // num_workers + 1
            worker_args = []
            for i in range(0, len(batch), batch_per_worker):
                worker_batch = batch[i:i+batch_per_worker]
                if worker_batch:
                    worker_args.append((wallet, worker_batch))
            
            # Submit to workers
            results = pool.map(test_password_batch, worker_args)
            
            # Check results
            for success, password, key in results:
                if success:
                    elapsed = time.time() - start_time
                    print(f"\n✅ PASSWORD FOUND: {password}")
                    print(f"🔑 Private key: {key}")
                    print(f"⏱️  Time: {elapsed:.2f} seconds")
                    print(f"🚀 Speed: {tested/elapsed:.1f} H/s")
                    
                    # Save to file
                    with open('FOUND_PASSWORD.txt', 'w') as f:
                        f.write(f"Password: {password}\n")
                        f.write(f"Private key: {key}\n")
                        f.write(f"Time: {elapsed:.2f} seconds\n")
                    
                    print(f"\n💾 Saved to FOUND_PASSWORD.txt")
                    pool.terminate()
                    return
            
            tested += len(batch)
            
            # Progress update
            if tested % (args.batch_size * 10) == 0:
                elapsed = time.time() - start_time
                speed = tested / elapsed if elapsed > 0 else 0
                progress = (tested / total_combinations) * 100 if total_combinations > 0 else 0
                eta = (total_combinations - tested) / speed if speed > 0 else 0
                
                print(f"Progress: {progress:.2f}% | Tested: {tested:,} | Speed: {speed:.1f} H/s | ETA: {eta/3600:.1f} hours", end='\r')
    
    # Not found
    elapsed = time.time() - start_time
    print(f"\n\n❌ Password not found")
    print(f"Total tested: {tested:,}")
    print(f"Time: {elapsed:.2f} seconds")
    print(f"Average speed: {tested/elapsed:.1f} H/s")

if __name__ == "__main__":
    main()