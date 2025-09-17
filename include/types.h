#ifndef TYPES_H
#define TYPES_H

#include <string>
#include <vector>
#include <cstdint>

struct WalletData {
    std::string filepath;
    std::string address;
    std::vector<uint8_t> ciphertext;
    std::vector<uint8_t> mac;
    std::vector<uint8_t> iv;
    std::vector<uint8_t> salt;
    uint32_t kdf_params_n;
    uint32_t kdf_params_r;
    uint32_t kdf_params_p;
    uint32_t kdf_params_dklen;
};

struct CheckpointData {
    uint64_t total_attempts;
    uint32_t base_password_index;
    uint64_t suffix_position;
    std::string last_attempt;
    time_t last_save_time;
};

struct PasswordCandidate {
    char password[256];
    uint32_t length;
    uint32_t base_index;
    uint64_t suffix_index;
};

struct GPUConfig {
    int device_id;
    size_t batch_size;
    size_t threads_per_block;
    size_t max_blocks;
    size_t shared_memory_size;
};

#endif