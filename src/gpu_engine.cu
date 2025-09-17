#include "gpu_engine.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstring>
#include <iostream>
#include <sstream>

#define CHARSET_SIZE 94
#define MAX_SUFFIX_LEN 5
#define MIN_SUFFIX_LEN 4
#define SCRYPT_N 262144
#define SCRYPT_R 8
#define SCRYPT_P 1
#define SCRYPT_DKLEN 32

__constant__ char d_charset[CHARSET_SIZE + 1] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()_+-=[]{}|;:,.<>?/~`";

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

__device__ void pbkdf2_sha256(const uint8_t* password, size_t pass_len,
                              const uint8_t* salt, size_t salt_len,
                              uint32_t iterations, uint8_t* output, size_t dklen) {
    // Simplified PBKDF2-SHA256 implementation
    // In production, use optimized crypto library
    
    uint8_t U[32], T[32];
    uint32_t block_count = (dklen + 31) / 32;
    
    for (uint32_t block = 1; block <= block_count; block++) {
        // First iteration
        // HMAC-SHA256(password, salt || block_index)
        // Simplified - would need full SHA256 implementation
        memset(U, block, 32);
        memcpy(T, U, 32);
        
        // Remaining iterations
        for (uint32_t iter = 1; iter < iterations; iter++) {
            // HMAC-SHA256(password, U)
            // XOR with T
            for (int i = 0; i < 32; i++) {
                T[i] ^= U[i];
            }
        }
        
        size_t copy_len = (block == block_count && dklen % 32) ? (dklen % 32) : 32;
        memcpy(output + (block - 1) * 32, T, copy_len);
    }
}

__device__ void scrypt_kdf(const uint8_t* password, size_t pass_len,
                           const uint8_t* salt, size_t salt_len,
                           uint8_t* output, void* scratch_memory) {
    // Scrypt KDF implementation
    // This is a simplified version - production would need full scrypt
    pbkdf2_sha256(password, pass_len, salt, salt_len, 1, output, 32);
    
    // Scrypt core operations would go here
    // Using scratch_memory for the large memory requirement
}

__device__ bool verify_mac(const uint8_t* derived_key, const uint8_t* ciphertext,
                           size_t ct_len, const uint8_t* expected_mac) {
    // Verify MAC using derived key
    // Simplified - would need full HMAC-SHA256
    uint8_t calculated_mac[32];
    
    // Calculate HMAC-SHA256(derived_key[16:32], ciphertext)
    // For now, simplified comparison
    for (int i = 0; i < 32; i++) {
        calculated_mac[i] = derived_key[i % 32] ^ ciphertext[i % ct_len];
    }
    
    // Compare MACs
    for (int i = 0; i < 32; i++) {
        if (calculated_mac[i] != expected_mac[i]) {
            return false;
        }
    }
    return true;
}

__global__ void crack_wallet_kernel(const uint8_t* wallet_data,
                                    const char* base_passwords,
                                    uint32_t base_pass_len,
                                    uint64_t suffix_start,
                                    uint64_t suffix_end,
                                    int suffix_len,
                                    bool* found,
                                    char* found_password,
                                    void* scrypt_memory) {
    
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t suffix_index = suffix_start + tid;
    
    if (suffix_index >= suffix_end || *found) return;
    
    // Generate suffix for this thread
    char suffix[MAX_SUFFIX_LEN + 1];
    generate_suffix(suffix_index, suffix, suffix_len);
    
    // Combine base password with suffix
    char candidate[256];
    memcpy(candidate, base_passwords, base_pass_len);
    memcpy(candidate + base_pass_len, suffix, suffix_len);
    candidate[base_pass_len + suffix_len] = '\0';
    
    // Extract wallet components from packed data
    const uint8_t* salt = wallet_data;
    const uint8_t* ciphertext = wallet_data + 48;
    const uint8_t* mac = wallet_data + 176;
    
    // Derive key using scrypt
    uint8_t derived_key[64];
    size_t scratch_offset = tid * (128 * SCRYPT_R * SCRYPT_N);
    void* thread_scratch = (uint8_t*)scrypt_memory + scratch_offset;
    
    scrypt_kdf((uint8_t*)candidate, base_pass_len + suffix_len,
               salt, 32, derived_key, thread_scratch);
    
    // Verify MAC
    if (verify_mac(derived_key, ciphertext, 128, mac)) {
        *found = true;
        device_strcpy(found_password, candidate);
    }
}

GPUEngine::GPUEngine(int device_id) : d_wallet_data_(nullptr),
                                      d_derived_keys_(nullptr),
                                      d_candidates_(nullptr),
                                      d_found_(nullptr),
                                      d_found_password_(nullptr),
                                      d_scrypt_memory_(nullptr) {
    config_.device_id = device_id;
    config_.batch_size = 1024 * 1024;  // 1M passwords per batch
    config_.threads_per_block = 256;
    config_.max_blocks = 4096;
    
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
    
    return allocateMemory();
}

bool GPUEngine::allocateMemory() {
    size_t wallet_data_size = 256;  // Packed wallet data
    // Reduce memory allocation - scrypt needs about 128MB per thread for N=262144
    // For batch processing, we'll use a smaller subset
    size_t threads_per_batch = 1024;  // Process 1024 passwords at a time on GPU
    size_t scrypt_memory_size = threads_per_batch * 128 * SCRYPT_R * 1024;  // ~1GB total
    
    cudaError_t err;
    
    err = cudaMalloc(&d_wallet_data_, wallet_data_size);
    if (err != cudaSuccess) return false;
    
    err = cudaMalloc(&d_found_, sizeof(bool));
    if (err != cudaSuccess) return false;
    
    err = cudaMalloc(&d_found_password_, 256);
    if (err != cudaSuccess) return false;
    
    // Allocate large memory for scrypt operations
    err = cudaMalloc(&d_scrypt_memory_, scrypt_memory_size);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate " << (scrypt_memory_size / (1024*1024)) 
                  << " MB for scrypt memory" << std::endl;
        return false;
    }
    
    // Pack and upload wallet data
    uint8_t packed_wallet[256] = {0};
    memcpy(packed_wallet, wallet_.salt.data(), 32);
    memcpy(packed_wallet + 32, wallet_.iv.data(), 16);
    memcpy(packed_wallet + 48, wallet_.ciphertext.data(), 128);
    memcpy(packed_wallet + 176, wallet_.mac.data(), 32);
    
    cudaMemcpy(d_wallet_data_, packed_wallet, 256, cudaMemcpyHostToDevice);
    
    return true;
}

void GPUEngine::freeMemory() {
    if (d_wallet_data_) cudaFree(d_wallet_data_);
    if (d_found_) cudaFree(d_found_);
    if (d_found_password_) cudaFree(d_found_password_);
    if (d_scrypt_memory_) cudaFree(d_scrypt_memory_);
}

bool GPUEngine::processBatch(uint32_t base_index, uint64_t start_suffix, 
                             uint64_t end_suffix, std::string& found_password) {
    
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
    
    // Try suffix lengths 4 and 5
    for (int suffix_len = MIN_SUFFIX_LEN; suffix_len <= MAX_SUFFIX_LEN; suffix_len++) {
        crack_wallet_kernel<<<blocks, config_.threads_per_block>>>(
            d_wallet_data_,
            d_base_pass,
            base_pass.length(),
            start_suffix,
            end_suffix,
            suffix_len,
            d_found_,
            d_found_password_,
            d_scrypt_memory_
        );
        
        cudaDeviceSynchronize();
        
        cudaMemcpy(&h_found, d_found_, sizeof(bool), cudaMemcpyDeviceToHost);
        
        if (h_found) {
            cudaMemcpy(h_found_password, d_found_password_, 256, cudaMemcpyDeviceToHost);
            found_password = std::string(h_found_password);
            cudaFree(d_base_pass);
            return true;
        }
    }
    
    cudaFree(d_base_pass);
    return false;
}

size_t GPUEngine::getMaxThroughput() const {
    return config_.batch_size * 10;  // Estimated 10 batches per second
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