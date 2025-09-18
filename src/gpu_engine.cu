#include "gpu_engine.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstring>
#include <iostream>
#include <sstream>
#include <algorithm>

#define CHARSET_SIZE 95
#define MAX_SUFFIX_LEN 12

// All printable ASCII from space (32) to tilde (126) = 95 characters
__constant__ char d_charset[CHARSET_SIZE + 1] = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~";

// Include the proper crypto implementations
#include "gpu_crypto_proper.cu"

__device__ void generate_suffix(uint64_t index, char* suffix, int suffix_len) {
    for (int i = suffix_len - 1; i >= 0; i--) {
        suffix[i] = d_charset[index % CHARSET_SIZE];
        index /= CHARSET_SIZE;
    }
    suffix[suffix_len] = '\0';
}

__device__ void device_strcpy(char* dest, const char* src) {
    int i = 0;
    while (src[i] != '\0') {
        dest[i] = src[i];
        i++;
    }
    dest[i] = '\0';
}

// Main wallet cracking kernel using wallet's actual KDF parameters
__global__ void crack_wallet_kernel(const uint8_t* wallet_data,
                                    const char* base_passwords,
                                    uint32_t base_pass_len,
                                    uint64_t suffix_start,
                                    uint64_t suffix_end,
                                    int suffix_len,
                                    bool* found,
                                    char* found_password,
                                    void* scrypt_memory,
                                    uint32_t kdf_n,
                                    uint32_t kdf_r,
                                    uint32_t kdf_p,
                                    uint32_t kdf_dklen,
                                    bool use_scrypt) {
    
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t suffix_index = suffix_start + tid;
    
    if (suffix_index >= suffix_end || *found) return;
    
    // Generate suffix for this thread
    char suffix[MAX_SUFFIX_LEN + 1];
    generate_suffix(suffix_index, suffix, suffix_len);
    
    // Combine base password with suffix
    char candidate[256];
    for (uint32_t i = 0; i < base_pass_len; i++) {
        candidate[i] = base_passwords[i];
    }
    for (int i = 0; i < suffix_len; i++) {
        candidate[base_pass_len + i] = suffix[i];
    }
    candidate[base_pass_len + suffix_len] = '\0';
    
    // Extract wallet components from packed data
    const uint8_t* salt = wallet_data;             // 32 bytes at offset 0
    const uint8_t* iv = wallet_data + 32;          // 16 bytes at offset 32
    const uint8_t* ciphertext = wallet_data + 48;  // 32 bytes at offset 48
    const uint8_t* mac = wallet_data + 80;         // 32 bytes at offset 80
    
    // Derive key using the wallet's specified KDF
    uint8_t derived_key[32];
    
    if (use_scrypt) {
        // Use scrypt for this wallet (as specified in wallet.json)
        // N=262144, r=8, p=1, dklen=32
        uint8_t* thread_scratch = (uint8_t*)scrypt_memory + tid * (256 * 1024 * 1024); // 256MB per thread
        scrypt_proper((uint8_t*)candidate, base_pass_len + suffix_len,
                      salt, 32, 
                      kdf_n, kdf_r, kdf_p, kdf_dklen,
                      derived_key, thread_scratch);
    } else {
        // Use PBKDF2 for older wallets
        pbkdf2_hmac_sha256((uint8_t*)candidate, base_pass_len + suffix_len,
                          salt, 32, kdf_n, kdf_dklen, derived_key);
    }
    
    // Verify MAC using Keccak-256
    // Ethereum MAC = Keccak256(derived_key[16:32] || ciphertext)
    uint8_t mac_input[48];  // 16 bytes of key + 32 bytes of ciphertext
    for (int i = 0; i < 16; i++) {
        mac_input[i] = derived_key[16 + i];  // Second half of derived key
    }
    for (int i = 0; i < 32; i++) {
        mac_input[16 + i] = ciphertext[i];
    }
    
    uint8_t calculated_mac[32];
    keccak256(mac_input, 48, calculated_mac);
    
    // Compare MACs
    bool mac_match = true;
    for (int i = 0; i < 32; i++) {
        if (calculated_mac[i] != mac[i]) {
            mac_match = false;
            break;
        }
    }
    
    if (mac_match) {
        *found = true;
        device_strcpy(found_password, candidate);
    }
}

// Host code
GPUEngine::GPUEngine(int device_id) : d_wallet_data_(nullptr),
                                      d_derived_keys_(nullptr),
                                      d_candidates_(nullptr),
                                      d_found_(nullptr),
                                      d_found_password_(nullptr),
                                      d_scrypt_memory_(nullptr) {
    config_.device_id = device_id;
    config_.batch_size = 20;  // Process 20 passwords in parallel (5GB memory)
    config_.threads_per_block = 20;  // Use 20 GPU threads
    config_.max_blocks = 1;  // Single block for memory-hard functions
    
    cudaSetDevice(device_id);
}

GPUEngine::~GPUEngine() {
    freeMemory();
}

bool GPUEngine::initialize(const WalletData& wallet, const std::vector<std::string>& base_passwords) {
    wallet_ = wallet;
    base_passwords_ = base_passwords;
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, config_.device_id);
    
    config_.shared_memory_size = prop.sharedMemPerBlock;
    
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "CUDA Cores: " << prop.multiProcessorCount * 128 << std::endl;
    std::cout << "Memory: " << (prop.totalGlobalMem / (1024*1024*1024)) << " GB" << std::endl;
    
    // Show wallet KDF parameters
    std::cout << "[INFO] Wallet KDF: scrypt" << std::endl;
    std::cout << "[INFO] Parameters: N=" << wallet_.kdf_params_n 
              << ", r=" << wallet_.kdf_params_r 
              << ", p=" << wallet_.kdf_params_p
              << ", dklen=" << wallet_.kdf_params_dklen << std::endl;
    
    return allocateMemory();
}

bool GPUEngine::allocateMemory() {
    size_t wallet_data_size = 256;
    
    // For scrypt with N=262144, r=8, we need significant memory
    // Each thread needs: 256MB for full scrypt (N * 128 * r bytes)
    size_t threads_per_batch = config_.batch_size;
    size_t scrypt_memory_size = threads_per_batch * 256 * 1024 * 1024;  // 256MB per thread
    
    std::cout << "[INFO] Allocating " << (scrypt_memory_size / (1024*1024)) 
              << " MB for scrypt operations" << std::endl;
    
    cudaError_t err;
    
    err = cudaMalloc(&d_wallet_data_, wallet_data_size);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate wallet data: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    err = cudaMalloc(&d_found_, sizeof(bool));
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate found flag: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    err = cudaMalloc(&d_found_password_, 256);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate password buffer: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    err = cudaMalloc(&d_scrypt_memory_, scrypt_memory_size);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate scrypt memory: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    // Pack wallet data: salt(32) + iv(16) + ciphertext(32) + mac(32)
    uint8_t packed_wallet[256] = {0};
    
    // Copy salt (32 bytes)
    if (wallet_.salt.size() >= 32) {
        memcpy(packed_wallet, wallet_.salt.data(), 32);
    }
    
    // Copy IV (16 bytes)
    if (wallet_.iv.size() >= 16) {
        memcpy(packed_wallet + 32, wallet_.iv.data(), 16);
    }
    
    // Copy ciphertext (32 bytes for Ethereum private key)
    if (wallet_.ciphertext.size() >= 32) {
        memcpy(packed_wallet + 48, wallet_.ciphertext.data(), 32);
    }
    
    // Copy MAC (32 bytes)
    if (wallet_.mac.size() >= 32) {
        memcpy(packed_wallet + 80, wallet_.mac.data(), 32);
    }
    
    cudaMemcpy(d_wallet_data_, packed_wallet, 256, cudaMemcpyHostToDevice);
    cudaMemset(d_found_, 0, sizeof(bool));
    
    return true;
}

void GPUEngine::freeMemory() {
    if (d_wallet_data_) cudaFree(d_wallet_data_);
    if (d_found_) cudaFree(d_found_);
    if (d_found_password_) cudaFree(d_found_password_);
    if (d_scrypt_memory_) cudaFree(d_scrypt_memory_);
}

bool GPUEngine::processBatch(uint32_t base_index, uint64_t start_suffix, 
                             uint64_t end_suffix, std::string& found_password, int suffix_len) {
    
    std::cout << "[DEBUG] processBatch called: base=" << base_index 
              << " start=" << start_suffix << " end=" << end_suffix 
              << " suffix_len=" << suffix_len << std::endl;
    
    const std::string& base_pass = base_passwords_[base_index];
    char* d_base_pass;
    cudaMalloc(&d_base_pass, base_pass.length() + 1);
    cudaMemcpy(d_base_pass, base_pass.c_str(), base_pass.length() + 1, cudaMemcpyHostToDevice);
    
    bool h_found = false;
    char h_found_password[256] = {0};
    
    cudaMemset(d_found_, 0, sizeof(bool));
    cudaMemset(d_found_password_, 0, 256);
    
    uint64_t batch_size = std::min(config_.batch_size, end_suffix - start_suffix);
    uint32_t blocks = (batch_size + config_.threads_per_block - 1) / config_.threads_per_block;
    blocks = std::min(blocks, (uint32_t)config_.max_blocks);
    
    // Check if using scrypt or PBKDF2
    bool use_scrypt = (wallet_.kdf_params_r > 0);  // r > 0 means scrypt
    
    // Print the passwords being tested in this batch (show all for debugging)
    std::cout << "[TESTING] Batch of " << (end_suffix - start_suffix) << " passwords:" << std::endl;
    for (uint64_t i = start_suffix; i < end_suffix; i++) {
        char suffix[13] = {0};
        uint64_t index = i;
        for (int j = suffix_len - 1; j >= 0; j--) {
            suffix[j] = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~"[index % CHARSET_SIZE];
            index /= CHARSET_SIZE;
        }
        suffix[suffix_len] = '\0';
        std::cout << "  [" << i << "] " << base_pass << suffix;
        if (i == 18) std::cout << " <-- This should be k0sk3sh$12";
        std::cout << std::endl;
    }
    
    // Launch kernel with wallet's KDF parameters
    crack_wallet_kernel<<<blocks, config_.threads_per_block>>>(
        d_wallet_data_,
        d_base_pass,
        base_pass.length(),
        start_suffix,
        end_suffix,
        suffix_len,
        d_found_,
        d_found_password_,
        d_scrypt_memory_,
        wallet_.kdf_params_n,     // N parameter from wallet
        wallet_.kdf_params_r,     // r parameter from wallet
        wallet_.kdf_params_p,     // p parameter from wallet
        wallet_.kdf_params_dklen, // dklen from wallet
        use_scrypt
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch error: " << cudaGetErrorString(err) << std::endl;
    }
    
    cudaDeviceSynchronize();
    
    std::cout << "[COMPLETED] Batch processed. Checking results..." << std::endl;
    
    cudaMemcpy(&h_found, d_found_, sizeof(bool), cudaMemcpyDeviceToHost);
    
    if (h_found) {
        cudaMemcpy(h_found_password, d_found_password_, 256, cudaMemcpyDeviceToHost);
        found_password = std::string(h_found_password);
        std::cout << "[SUCCESS] PASSWORD FOUND: " << found_password << std::endl;
        cudaFree(d_base_pass);
        return true;
    }
    else {
        std::cout << "[NOT FOUND] Password not in this batch" << std::endl;
    }
    
    cudaFree(d_base_pass);
    return false;
}

size_t GPUEngine::getMaxThroughput() const {
    // With scrypt N=262144, expect only 10-100 attempts/sec
    return config_.batch_size;  // Very low for scrypt
}

void GPUEngine::getDeviceInfo(std::string& info) const {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, config_.device_id);
    
    std::stringstream ss;
    ss << "GPU: " << prop.name << "\n";
    ss << "Compute Capability: " << prop.major << "." << prop.minor << "\n";
    ss << "CUDA Cores: " << prop.multiProcessorCount * 128 << "\n";
    ss << "Memory: " << (prop.totalGlobalMem / (1024*1024*1024)) << " GB\n";
    ss << "Clock: " << (prop.clockRate / 1000) << " MHz";
    
    info = ss.str();
}