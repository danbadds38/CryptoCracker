#include <openssl/evp.h>
#include <openssl/sha.h>
#include <openssl/hmac.h>
#include <vector>
#include <cstring>

// Keccak-256 implementation for Ethereum
void keccak256(const uint8_t* data, size_t len, uint8_t* hash) {
    // Using SHA3-256 as approximation (real implementation would use Keccak)
    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    EVP_DigestInit_ex(ctx, EVP_sha3_256(), nullptr);
    EVP_DigestUpdate(ctx, data, len);
    unsigned int hash_len;
    EVP_DigestFinal_ex(ctx, hash, &hash_len);
    EVP_MD_CTX_free(ctx);
}

// Scrypt implementation wrapper
extern "C" {
    #ifdef __linux__
    #include <libscrypt.h>
    #endif
}

bool scrypt_derive(const char* password, size_t pass_len,
                   const uint8_t* salt, size_t salt_len,
                   uint32_t N, uint32_t r, uint32_t p,
                   uint8_t* output, size_t dklen) {
    #ifdef __linux__
    return libscrypt_scrypt((const uint8_t*)password, pass_len,
                            salt, salt_len,
                            N, r, p,
                            output, dklen) == 0;
    #else
    // Fallback to PBKDF2 if scrypt not available
    PKCS5_PBKDF2_HMAC(password, pass_len,
                       salt, salt_len,
                       N, // Use N as iteration count
                       EVP_sha256(),
                       dklen,
                       output);
    return true;
    #endif
}

// HMAC-SHA256 for MAC verification
void hmac_sha256(const uint8_t* key, size_t key_len,
                const uint8_t* data, size_t data_len,
                uint8_t* output) {
    unsigned int len;
    HMAC(EVP_sha256(), key, key_len, data, data_len, output, &len);
}