# Suffix Length Guide

## Available Commands

```bash
# Default (tries both 4 and 5 character suffixes)
make crack

# Specific suffix lengths
make crack-3   # 3 characters only
make crack-4   # 4 characters only  
make crack-5   # 5 characters only
make crack-6   # 6 characters only

# Custom length (1-8 characters)
docker exec cryptocracker ./build/cryptocracker \
  --wallet /app/wallets/wallet.json \
  --passwords /app/config/passwords.txt \
  --suffix-length 7
```

## Time Estimates for 13 Passwords

With RTX 3080 at 1+ billion attempts/second:

| Suffix Length | Combinations per Password | Total (13 passwords) | Time Estimate |
|--------------|--------------------------|---------------------|---------------|
| 1 char | 94 | 1,222 | < 0.001 seconds |
| 2 chars | 94² = 8,836 | 114,868 | < 0.001 seconds |
| 3 chars | 94³ = 830,584 | 10.8 million | 0.01 seconds |
| **4 chars** | 94⁴ = 78 million | 1 billion | **1 second** |
| **5 chars** | 94⁵ = 7.3 billion | 95 billion | **95 seconds** |
| 6 chars | 94⁶ = 689 billion | 8.9 trillion | ~2.5 hours |
| 7 chars | 94⁷ = 64.8 trillion | 842 trillion | ~10 days |
| 8 chars | 94⁸ = 6 quadrillion | 79 quadrillion | ~2.5 years |

## Recommendations

### Quick Test (seconds)
```bash
make crack-3   # Tests up to 3 chars (instant)
make crack-4   # Tests up to 4 chars (1 second)
```

### Normal Search (minutes)
```bash
make crack-5   # Tests up to 5 chars (1-2 minutes)
make crack     # Tests both 4 and 5 (default, ~2 minutes)
```

### Extended Search (hours)
```bash
make crack-6   # Tests up to 6 chars (2-3 hours)
```

### Very Long Search (days+)
```bash
# Only if you're certain it's 7+ characters
docker exec cryptocracker ./build/cryptocracker \
  --wallet /app/wallets/wallet.json \
  --passwords /app/config/passwords.txt \
  --suffix-length 7
```

## Character Set (94 total)

The suffix uses all these characters:
- Lowercase: a-z (26)
- Uppercase: A-Z (26)  
- Numbers: 0-9 (10)
- Symbols: !@#$%^&*()_+-=[]{}|;:,.<>?/~` (32)

## Examples

If your missing suffix might be:
- `123` → use `make crack-3`
- `2024` → use `make crack-4`
- `!abc$` → use `make crack-5`
- `Secret` → use `make crack-6`

## Pro Tip

Start with shorter lengths and work up:
```bash
make crack-3  # Try 3 first (instant)
make crack-4  # Then 4 (1 second)
make crack-5  # Then 5 (90 seconds)
make crack-6  # Only if needed (2+ hours)
```