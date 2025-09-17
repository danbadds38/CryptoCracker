#include "wallet.h"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <openssl/evp.h>
#include <openssl/sha.h>
#include <openssl/hmac.h>
#include <openssl/aes.h>
#include <cstring>

// Helper function to convert hex string to bytes
std::vector<uint8_t> hexToBytes(const std::string& hex) {
    std::vector<uint8_t> bytes;
    for (size_t i = 0; i < hex.length(); i += 2) {
        std::string byteString = hex.substr(i, 2);
        uint8_t byte = static_cast<uint8_t>(strtol(byteString.c_str(), nullptr, 16));
        bytes.push_back(byte);
    }
    return bytes;
}

// Helper function to convert bytes to hex string
std::string bytesToHex(const std::vector<uint8_t>& bytes) {
    std::stringstream ss;
    for (uint8_t byte : bytes) {
        ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(byte);
    }
    return ss.str();
}

bool WalletLoader::loadFromFile(const std::string& filepath, WalletData& wallet) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Failed to open wallet file: " << filepath << std::endl;
        return false;
    }
    
    Json::Value root;
    Json::CharReaderBuilder builder;
    std::string errs;
    
    if (!Json::parseFromStream(builder, file, &root, &errs)) {
        std::cerr << "Failed to parse wallet JSON: " << errs << std::endl;
        return false;
    }
    
    file.close();
    
    wallet.filepath = filepath;
    
    // Check version
    if (root.isMember("version") && root["version"].asInt() == 3) {
        return parseKeystoreV3(root, wallet);
    }
    
    std::cerr << "Unsupported wallet version" << std::endl;
    return false;
}

bool WalletLoader::parseKeystoreV3(const Json::Value& root, WalletData& wallet) {
    try {
        // Extract address
        if (root.isMember("address")) {
            wallet.address = root["address"].asString();
        }
        
        // Extract crypto parameters (handle both "crypto" and "Crypto")
        const Json::Value& crypto = root.isMember("crypto") ? root["crypto"] : root["Crypto"];
        
        // Ciphertext
        std::string ciphertext_hex = crypto["ciphertext"].asString();
        wallet.ciphertext = hexToBytes(ciphertext_hex);
        
        // Cipher parameters
        const Json::Value& cipher_params = crypto["cipherparams"];
        std::string iv_hex = cipher_params["iv"].asString();
        wallet.iv = hexToBytes(iv_hex);
        
        // KDF parameters
        const Json::Value& kdf_params = crypto["kdfparams"];
        
        if (crypto["kdf"].asString() == "scrypt") {
            wallet.kdf_params_n = kdf_params["n"].asInt();
            wallet.kdf_params_r = kdf_params["r"].asInt();
            wallet.kdf_params_p = kdf_params["p"].asInt();
            wallet.kdf_params_dklen = kdf_params["dklen"].asInt();
            
            std::string salt_hex = kdf_params["salt"].asString();
            wallet.salt = hexToBytes(salt_hex);
        } else if (crypto["kdf"].asString() == "pbkdf2") {
            wallet.kdf_params_n = kdf_params["c"].asInt();  // iterations
            wallet.kdf_params_dklen = kdf_params["dklen"].asInt();
            
            std::string salt_hex = kdf_params["salt"].asString();
            wallet.salt = hexToBytes(salt_hex);
            
            // For PBKDF2, set r and p to 0 as indicators
            wallet.kdf_params_r = 0;
            wallet.kdf_params_p = 0;
        }
        
        // MAC
        std::string mac_hex = crypto["mac"].asString();
        wallet.mac = hexToBytes(mac_hex);
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error parsing keystore: " << e.what() << std::endl;
        return false;
    }
}

bool WalletLoader::deriveKey(const std::string& password, const WalletData& wallet,
                             std::vector<uint8_t>& derived_key) {
    
    derived_key.resize(wallet.kdf_params_dklen);
    
    // Check if using scrypt or pbkdf2
    if (wallet.kdf_params_r > 0) {
        // Scrypt
        // Note: In production, use libscrypt or similar
        // For now, we'll use a simplified version
        
        // This would normally call scrypt function
        // scrypt(password, salt, N, r, p, dklen, derived_key)
        
        // Placeholder implementation
        EVP_MD_CTX* ctx = EVP_MD_CTX_new();
        EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr);
        EVP_DigestUpdate(ctx, password.c_str(), password.length());
        EVP_DigestUpdate(ctx, wallet.salt.data(), wallet.salt.size());
        
        unsigned char temp[32];
        unsigned int len;
        EVP_DigestFinal_ex(ctx, temp, &len);
        
        memcpy(derived_key.data(), temp, std::min(wallet.kdf_params_dklen, 32u));
        EVP_MD_CTX_free(ctx);
        
    } else {
        // PBKDF2
        PKCS5_PBKDF2_HMAC(
            password.c_str(),
            password.length(),
            wallet.salt.data(),
            wallet.salt.size(),
            wallet.kdf_params_n,  // iterations
            EVP_sha256(),
            wallet.kdf_params_dklen,
            derived_key.data()
        );
    }
    
    return true;
}

bool WalletLoader::validatePassword(const WalletData& wallet, const std::string& password,
                                   std::vector<uint8_t>& private_key) {
    
    // Derive key from password
    std::vector<uint8_t> derived_key;
    if (!deriveKey(password, wallet, derived_key)) {
        return false;
    }
    
    // Verify MAC
    // MAC = SHA3(derived_key[16:32] || ciphertext)
    std::vector<uint8_t> mac_data;
    mac_data.insert(mac_data.end(), derived_key.begin() + 16, derived_key.begin() + 32);
    mac_data.insert(mac_data.end(), wallet.ciphertext.begin(), wallet.ciphertext.end());
    
    unsigned char calculated_mac[32];
    unsigned int mac_len;
    
    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr);  // Should be Keccak256
    EVP_DigestUpdate(ctx, mac_data.data(), mac_data.size());
    EVP_DigestFinal_ex(ctx, calculated_mac, &mac_len);
    EVP_MD_CTX_free(ctx);
    
    // Compare MACs
    if (memcmp(calculated_mac, wallet.mac.data(), 32) != 0) {
        return false;  // Invalid password
    }
    
    // Decrypt private key using AES-128-CTR
    private_key.resize(wallet.ciphertext.size());
    
    EVP_CIPHER_CTX* cipher_ctx = EVP_CIPHER_CTX_new();
    EVP_DecryptInit_ex(cipher_ctx, EVP_aes_128_ctr(), nullptr, 
                       derived_key.data(), wallet.iv.data());
    
    int len;
    int plaintext_len;
    
    EVP_DecryptUpdate(cipher_ctx, private_key.data(), &len, 
                      wallet.ciphertext.data(), wallet.ciphertext.size());
    plaintext_len = len;
    
    EVP_DecryptFinal_ex(cipher_ctx, private_key.data() + len, &len);
    plaintext_len += len;
    
    EVP_CIPHER_CTX_free(cipher_ctx);
    
    private_key.resize(plaintext_len);
    
    return true;
}