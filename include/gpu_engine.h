#ifndef GPU_ENGINE_H
#define GPU_ENGINE_H

#include "types.h"
#include <cuda_runtime.h>
#include <memory>

class GPUEngine {
public:
    GPUEngine(int device_id = 0);
    ~GPUEngine();
    
    bool initialize(const WalletData& wallet, const std::vector<std::string>& base_passwords);
    
    bool processBatch(uint32_t base_index, uint64_t start_suffix, uint64_t end_suffix,
                      std::string& found_password);
    
    size_t getBatchSize() const { return config_.batch_size; }
    size_t getMaxThroughput() const;
    
    void getDeviceInfo(std::string& info) const;
    
private:
    GPUConfig config_;
    WalletData wallet_;
    std::vector<std::string> base_passwords_;
    
    uint8_t* d_wallet_data_;
    uint8_t* d_derived_keys_;
    PasswordCandidate* d_candidates_;
    bool* d_found_;
    char* d_found_password_;
    
    void* d_scrypt_memory_;
    
    bool allocateMemory();
    void freeMemory();
};

#endif