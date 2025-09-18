#!/bin/bash
# Hashcat cracking script for Ethereum wallets

set -e

# Default values
WALLET_FILE="/data/wallet.json"
PASSWORDS_FILE="/data/passwords.txt"
SUFFIX_LENGTH=${SUFFIX_LENGTH:-1}
ATTACK_MODE=${ATTACK_MODE:-mask}  # 'mask' or 'wordlist'
RESUME=${RESUME:-0}

echo "=========================================="
echo "Hashcat Ethereum Wallet Cracker"
echo "=========================================="

# Convert wallet to hashcat format
echo "[1/4] Converting wallet to hashcat format..."
python3 /crack/wallet_to_hash.py "$WALLET_FILE"

# Extract wallet address for session naming
WALLET_ADDRESS=$(python3 -c "import json; print(json.load(open('$WALLET_FILE'))['address'])" 2>/dev/null || echo "unknown")
echo "Wallet address: $WALLET_ADDRESS"

# Detect hash mode from the hash file
if grep -q '\$ethereum\$s\*' /crack/wallet.hash; then
    HASH_MODE=15700
    echo "Detected: Ethereum Wallet with scrypt (mode 15700)"
elif grep -q '\$ethereum\$p\*' /crack/wallet.hash; then
    HASH_MODE=15600
    echo "Detected: Ethereum Wallet with PBKDF2 (mode 15600)"
else
    echo "Error: Could not detect wallet type"
    exit 1
fi

# Show GPU info
echo ""
echo "[2/4] GPU Information:"
hashcat -I 2>/dev/null | grep -A 2 "Backend Device ID" || echo "No GPU detected, using CPU"

# Prepare attack based on mode
echo ""
echo "[3/4] Preparing attack..."
echo "Base passwords file: $PASSWORDS_FILE"
echo "Suffix length: $SUFFIX_LENGTH"
echo "Attack mode: $ATTACK_MODE"

if [ "$ATTACK_MODE" = "wordlist" ]; then
    # Generate full wordlist (good for small suffix lengths)
    echo "Generating wordlist with ${SUFFIX_LENGTH}-character suffixes..."
    python3 /crack/generate_wordlist.py "$PASSWORDS_FILE" "$SUFFIX_LENGTH" /crack/wordlist.txt
    ATTACK_CMD="-a 0 /crack/wordlist.txt"
else
    # Use mask attack (more efficient for larger suffix lengths)
    echo "Using mask attack with ${SUFFIX_LENGTH}-character suffixes..."
    
    # Create custom charset file for all printable ASCII
    echo ' !"#$%&'"'"'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~' > /crack/charset.hcchr
    
    # Build mask (?1 = custom charset)
    MASK=""
    for i in $(seq 1 $SUFFIX_LENGTH); do
        MASK="${MASK}?1"
    done
    
    # For mask attack with base passwords
    # We'll use combinator attack (mode 1) or use stdin
    # But hashcat doesn't directly support base+mask, so we'll use a hybrid approach
    
    # Create rules file for appending characters
    echo "Preparing hybrid attack..."
    
    # Use wordlist + mask (hybrid attack mode 6)
    ATTACK_CMD="-a 6 $PASSWORDS_FILE $MASK --custom-charset1=/crack/charset.hcchr"
fi

# Prepare hashcat command
HASHCAT_CMD="hashcat -m $HASH_MODE"

# Add optimization flags for RTX 3080
# -w 4: Maximum workload profile (lets hashcat auto-tune kernel parameters)
# --force: Override warnings about manual kernel settings
HASHCAT_CMD="$HASHCAT_CMD -w 4"
#HASHCAT_CMD="$HASHCAT_CMD -O"  # -O optimization disabled for scrypt

# Create composite session name: wallet_address + suffix_length + attack_mode
# This ensures different wallets don't share sessions
SESSION_NAME="eth_${WALLET_ADDRESS}_s${SUFFIX_LENGTH}_${ATTACK_MODE}"
HASHCAT_CMD="$HASHCAT_CMD --session=$SESSION_NAME"

# Session files will be stored in a persistent volume
SESSION_DIR="/crack-session"
mkdir -p "$SESSION_DIR"
cd "$SESSION_DIR"

# Check if we should resume
if [ "$RESUME" = "1" ]; then
    if [ -f "${SESSION_NAME}.restore" ]; then
        echo "✅ Found existing session for wallet ${WALLET_ADDRESS} with ${SUFFIX_LENGTH}-char suffix"
        echo "Resuming from checkpoint..."
        HASHCAT_CMD="$HASHCAT_CMD --restore"
    else
        echo "⚠️  No previous session found for wallet ${WALLET_ADDRESS} with ${SUFFIX_LENGTH}-char suffix"
        echo "Starting fresh attack..."
        HASHCAT_CMD="$HASHCAT_CMD /crack/wallet.hash $ATTACK_CMD"
    fi
else
    if [ -f "${SESSION_NAME}.restore" ]; then
        echo "⚠️  Found existing session but RESUME=0, overwriting..."
    fi
    # Remove old session file if exists
    rm -f "${SESSION_NAME}.restore" "${SESSION_NAME}.log" "${SESSION_NAME}.outfiles"
    HASHCAT_CMD="$HASHCAT_CMD /crack/wallet.hash $ATTACK_CMD"
fi

# Add output file
HASHCAT_CMD="$HASHCAT_CMD --outfile=/crack/cracked.txt --outfile-format=2"

# Add status timer
HASHCAT_CMD="$HASHCAT_CMD --status --status-timer=10"

# Run hashcat
echo ""
echo "[4/4] Starting hashcat..."
echo "Command: $HASHCAT_CMD"
echo "=========================================="
echo ""

# Run and capture exit code
set +e
$HASHCAT_CMD
EXIT_CODE=$?
set -e

# Check results
echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    if [ -f /crack/cracked.txt ]; then
        echo "✅ PASSWORD FOUND!"
        echo "Result:"
        cat /crack/cracked.txt
        
        # Extract just the password
        FOUND_PASSWORD=$(cat /crack/cracked.txt)
        echo ""
        echo "Password: $FOUND_PASSWORD"
    else
        echo "❌ Password not found in the given search space"
    fi
elif [ $EXIT_CODE -eq 255 ]; then
    echo "⏸️  Session paused. Run with RESUME=1 to continue."
else
    echo "⚠️  Hashcat exited with code: $EXIT_CODE"
fi

# Show statistics
if [ -f /crack/hashcat.potfile ]; then
    echo ""
    echo "Cracked passwords in potfile:"
    cat /crack/hashcat.potfile
fi

echo "=========================================="