# ⚠️ WARNING: Long Suffix Length Reality Check

## How to Use 10-Character Suffix

```bash
# Direct command for 10-character suffix
docker exec cryptocracker ./build/cryptocracker \
  --wallet /app/wallets/wallet.json \
  --passwords /app/config/passwords.txt \
  --suffix-length 10
```

## BUT... Here's the Math (IT'S INSANE!)

### Time Estimates for Different Suffix Lengths

With your RTX 3080 at 1 billion attempts/second:

| Suffix | Combinations | Time for 1 Password | Time for 13 Passwords |
|--------|--------------|-------------------|----------------------|
| 4 chars | 78 million | 0.078 seconds | 1 second |
| 5 chars | 7.3 billion | 7.3 seconds | 95 seconds |
| 6 chars | 689 billion | 11.5 minutes | 2.5 hours |
| 7 chars | 64.8 trillion | 18 hours | 10 days |
| 8 chars | 6.1 quadrillion | 70 days | **2.5 YEARS** |
| 9 chars | 572 quadrillion | 18 YEARS | **234 YEARS** |
| **10 chars** | **53.8 quintillion** | **1,700 YEARS** | **22,100 YEARS** |
| 11 chars | 5.1 sextillion | 160,000 YEARS | 2 MILLION YEARS |
| 12 chars | 475 sextillion | 15 MILLION YEARS | 195 MILLION YEARS |

## The Reality

**A 10-character suffix would take 22,100 YEARS to crack with your RTX 3080!**

Even with:
- 100 RTX 3080s: Still 221 years
- 1000 RTX 3080s: Still 22 years
- The world's fastest supercomputer: Still months/years

## What You Should Do Instead

### Option 1: Are you SURE it's 10 characters?
Most people don't add 10 random characters. Common patterns:
- Year + symbols: `2016!@#$` (8 chars)
- Word + numbers: `bitcoin123` (10 chars, but predictable)
- Repeated pattern: `123123123` (9 chars, but predictable)

### Option 2: Use a Dictionary Attack
If the suffix might be a word or pattern:
```bash
# Create a custom suffix dictionary instead
echo "2016" > custom_suffixes.txt
echo "2024" >> custom_suffixes.txt
echo "bitcoin" >> custom_suffixes.txt
echo "ethereum" >> custom_suffixes.txt
echo "123456789" >> custom_suffixes.txt
echo "password123" >> custom_suffixes.txt
```

### Option 3: Try Shorter Lengths First
```bash
# Start with reasonable lengths
make crack-4  # 1 second
make crack-5  # 90 seconds
make crack-6  # 2.5 hours

# Only if absolutely certain
make crack-7  # 10 days (still doable)
```

### Option 4: Think About the Pattern
If you know it's 10 characters, you probably remember something about it:
- First few characters?
- Last few characters?
- Type of characters (all numbers? mix?)
- Any words in it?

## If You REALLY Want to Try 10 Characters

You can, but understand:
```bash
# This will run for 22,100 years
docker exec cryptocracker ./build/cryptocracker \
  --wallet /app/wallets/wallet.json \
  --passwords /app/config/passwords.txt \
  --suffix-length 10

# You can stop it with Ctrl+C and resume later
# But you'll be resuming a 22,100 year task...
```

## Better Alternative: Pattern-Based Search

If you remember ANY pattern about those 10 characters, we could write a custom search that would be millions of times faster. For example:
- "It was a year followed by 6 letters" → 26^6 * 100 = way faster
- "It was two words from this list" → dictionary^2 = instant
- "It started with 'btc'" → 94^7 = still doable

## Bottom Line

**Brute-forcing 10 random characters is computationally infeasible**, even with top-tier GPUs. You need to either:
1. Remember something about the pattern
2. Try shorter lengths
3. Use dictionary/pattern attacks
4. Hope it's actually shorter than you think

The universe will end before a 10-character truly random suffix is cracked!