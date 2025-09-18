// Simple test of specific password
#include <cuda_runtime.h>
#include <iostream>
#include <cstring>
#include <cstdio>

#include "src/gpu_crypto_proper.cu"

__global__ void test_password_kernel(const char* password, uint32_t passlen,
                                     const uint8_t* salt, const uint8_t* ciphertext,
                                     const uint8_t* expected_mac,
                                     uint8_t* derived_key_out, uint8_t* calculated_mac_out,
                                     uint8_t* scratch, bool* match_out) {
    uint8_t derived_key[32];
    
    // Initialize derived_key to non-zero for debugging
    for (int i = 0; i < 32; i++) {
        derived_key[i] = 0xAA;
    }
    
    printf("GPU: Starting scrypt with password '%s' (len %d)\n", password, passlen);
    printf("GPU: Salt[0-3]: %02x %02x %02x %02x\n", salt[0], salt[1], salt[2], salt[3]);
    
    // Test scrypt with known password
    scrypt_proper((const uint8_t*)password, passlen, salt, 32, 
                 262144, 8, 1, 32, 
                 derived_key, scratch);
    
    printf("GPU: Scrypt complete. Key[0-3]: %02x %02x %02x %02x\n", 
           derived_key[0], derived_key[1], derived_key[2], derived_key[3]);
    
    // Copy derived key to output for debugging
    for (int i = 0; i < 32; i++) {
        derived_key_out[i] = derived_key[i];
    }
    
    // Calculate MAC
    uint8_t mac_input[48];
    for (int i = 0; i < 16; i++) {
        mac_input[i] = derived_key[16 + i];
    }
    for (int i = 0; i < 32; i++) {
        mac_input[16 + i] = ciphertext[i];
    }
    
    keccak256(mac_input, 48, calculated_mac_out);
    
    // Check if MACs match
    bool match = true;
    for (int i = 0; i < 32; i++) {
        if (calculated_mac_out[i] != expected_mac[i]) {
            match = false;
            break;
        }
    }
    *match_out = match;
}

int main() {
    // The known correct password
    const char* password = "k0sk3sh$12";
    uint32_t passlen = strlen(password);
    
    std::cout << "Testing password: " << password << " (length " << passlen << ")" << std::endl;
    
    // Wallet data from wallet.json (hex decoded)
    uint8_t h_salt[32] = {
        0xc4, 0xca, 0xe4, 0xd7, 0x5f, 0x16, 0xd8, 0x07,
        0x89, 0x83, 0x4d, 0xd2, 0x04, 0x1c, 0x5d, 0xe7,
        0xd6, 0x80, 0x97, 0xf6, 0x68, 0x48, 0x8c, 0x49,
        0xd3, 0xe1, 0xdc, 0x7d, 0x54, 0x25, 0x24, 0xb2
    };
    
    uint8_t h_ciphertext[32] = {
        0x49, 0xdb, 0xf7, 0xd5, 0xfa, 0xc4, 0xc0, 0xb4,
        0xb8, 0x35, 0x21, 0xb9, 0xac, 0xbf, 0x2c, 0xaa,
        0xaf, 0xce, 0x81, 0xc5, 0x35, 0xf0, 0xbd, 0x4f,
        0x71, 0xfc, 0x28, 0x60, 0xa8, 0x01, 0xb3, 0xe1
    };
    
    uint8_t h_mac[32] = {
        0xc5, 0xcd, 0xa1, 0xac, 0xa0, 0xdc, 0x45, 0x30,
        0x79, 0xdc, 0xad, 0x01, 0x4a, 0xb4, 0x27, 0x29,
        0xcd, 0x73, 0xe7, 0xaf, 0xa8, 0x64, 0xb4, 0xbe,
        0x40, 0xb1, 0xb3, 0xbe, 0xef, 0xf9, 0x48, 0x42
    };
    
    // Allocate device memory
    char* d_password;
    uint8_t *d_salt, *d_ciphertext, *d_mac;
    uint8_t *d_derived_key, *d_calculated_mac, *d_scratch;
    bool *d_match;
    
    cudaMalloc(&d_password, passlen + 1);
    cudaMalloc(&d_salt, 32);
    cudaMalloc(&d_ciphertext, 32);
    cudaMalloc(&d_mac, 32);
    cudaMalloc(&d_derived_key, 32);
    cudaMalloc(&d_calculated_mac, 32);
    cudaMalloc(&d_scratch, 256 * 1024 * 1024 + 1024); // 256MB+ for scrypt
    cudaMalloc(&d_match, sizeof(bool));
    
    // Copy data to device
    cudaMemcpy(d_password, password, passlen + 1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_salt, h_salt, 32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ciphertext, h_ciphertext, 32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mac, h_mac, 32, cudaMemcpyHostToDevice);
    
    // Run kernel
    std::cout << "Running scrypt on GPU..." << std::endl;
    test_password_kernel<<<1, 1>>>(d_password, passlen, d_salt, d_ciphertext, d_mac,
                                   d_derived_key, d_calculated_mac, d_scratch, d_match);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    
    cudaDeviceSynchronize();
    
    // Get results
    uint8_t h_derived_key[32];
    uint8_t h_calculated_mac[32];
    bool h_match;
    
    cudaMemcpy(h_derived_key, d_derived_key, 32, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_calculated_mac, d_calculated_mac, 32, cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_match, d_match, sizeof(bool), cudaMemcpyDeviceToHost);
    
    // Print results
    std::cout << "\nDerived key: ";
    for (int i = 0; i < 32; i++) {
        printf("%02x", h_derived_key[i]);
    }
    std::cout << std::endl;
    
    std::cout << "\nCalculated MAC: ";
    for (int i = 0; i < 32; i++) {
        printf("%02x", h_calculated_mac[i]);
    }
    std::cout << std::endl;
    
    std::cout << "Expected MAC:   ";
    for (int i = 0; i < 32; i++) {
        printf("%02x", h_mac[i]);
    }
    std::cout << std::endl;
    
    std::cout << "\nMAC match: " << (h_match ? "✅ YES - PASSWORD CORRECT!" : "❌ NO - PASSWORD WRONG") << std::endl;
    
    // Cleanup
    cudaFree(d_password);
    cudaFree(d_salt);
    cudaFree(d_ciphertext);
    cudaFree(d_mac);
    cudaFree(d_derived_key);
    cudaFree(d_calculated_mac);
    cudaFree(d_scratch);
    cudaFree(d_match);
    
    return h_match ? 0 : 1;
}