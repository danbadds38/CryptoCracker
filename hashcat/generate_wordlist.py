#!/usr/bin/env python3
"""
Generate wordlist with suffixes for hashcat
"""
import sys
import itertools
import os

def generate_wordlist_with_suffix(passwords_file, suffix_length, output_file):
    """Generate wordlist with all suffix combinations"""
    
    # Character set for suffixes (95 printable ASCII characters)
    charset = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~"
    
    # Read base passwords
    with open(passwords_file, 'r') as f:
        base_passwords = [line.strip() for line in f if line.strip()]
    
    print(f"Generating wordlist with {len(base_passwords)} base passwords")
    print(f"Suffix length: {suffix_length}")
    print(f"Character set size: {len(charset)}")
    
    total_combinations = len(base_passwords) * (len(charset) ** suffix_length)
    print(f"Total combinations: {total_combinations:,}")
    
    if total_combinations > 10000000:  # 10 million
        print("\nWARNING: Large wordlist! Consider using hashcat's mask attack instead.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Aborted. Use mask attack mode instead (see crack.sh)")
            sys.exit(0)
    
    # Generate combinations
    count = 0
    with open(output_file, 'w') as out:
        for base_password in base_passwords:
            for suffix in itertools.product(charset, repeat=suffix_length):
                password = base_password + ''.join(suffix)
                out.write(password + '\n')
                count += 1
                if count % 100000 == 0:
                    print(f"Generated {count:,} passwords...")
    
    print(f"\nGenerated {count:,} passwords to {output_file}")
    return count

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 generate_wordlist.py <passwords.txt> <suffix_length> <output.txt>")
        print("Example: python3 generate_wordlist.py passwords.txt 2 wordlist.txt")
        sys.exit(1)
    
    passwords_file = sys.argv[1]
    suffix_length = int(sys.argv[2])
    output_file = sys.argv[3]
    
    if not os.path.exists(passwords_file):
        print(f"Error: Passwords file not found: {passwords_file}")
        sys.exit(1)
    
    generate_wordlist_with_suffix(passwords_file, suffix_length, output_file)