// Test specific password position 18 which should be k0sk3sh$12
#include <cuda_runtime.h>
#include <iostream>
#include <cstring>

#include "src/gpu_crypto_proper.cu"

int main() {
    // The known correct password
    const char* password = "k0sk3sh$12";
    
    // Wallet data from wallet.json (hex decoded)
    uint8_t salt[32] = {
        0xc4, 0xca, 0xe4, 0xd7, 0x5f, 0x16, 0xd8, 0x07,
        0x89, 0x83, 0x4d, 0xd2, 0x04, 0x1c, 0x5d, 0xe7,
        0xd6, 0x80, 0x97, 0xf6, 0x68, 0x48, 0x8c, 0x49,
        0xd3, 0xe1, 0xdc, 0x7d, 0x54, 0x25, 0x24, 0xb2
    };
    
    uint8_t ciphertext[32] = {
        0x49, 0xdb, 0xf7, 0xd5, 0xfa, 0xc4, 0xc0, 0xb4,
        0xb8, 0x35, 0x21, 0xb9, 0xac, 0xbf, 0x2c, 0xaa,
        0xaf, 0xce, 0x81, 0xc5, 0x35, 0xf0, 0xbd, 0x4f,
        0x71, 0xfc, 0x28, 0x60, 0xa8, 0x01, 0xb3, 0xe1
    };
    
    uint8_t mac[32] = {
        0xc5, 0xcd, 0xa1, 0xac, 0xa0, 0xdc, 0x45, 0x30,
        0x79, 0xdc, 0xad, 0x01, 0x4a, 0xb4, 0x27, 0x29,
        0xcd, 0x73, 0xe7, 0xaf, 0xa8, 0x64, 0xb4, 0xbe,
        0x40, 0xb1, 0xb3, 0xbe, 0xef, 0xf9, 0x48, 0x42
    };
    
    // Allocate device memory
    uint8_t* d_derived_key;
    uint8_t* d_scratch;
    uint8_t* d_calculated_mac;
    
    cudaMalloc(&d_derived_key, 32);
    cudaMalloc(&d_scratch, 33 * 1024 * 1024); // 33MB for scrypt
    cudaMalloc(&d_calculated_mac, 32);
    
    // Copy data to device
    uint8_t *d_password, *d_salt, *d_ciphertext, *d_mac;
    cudaMalloc(&d_password, 11);
    cudaMalloc(&d_salt, 32);
    cudaMalloc(&d_ciphertext, 32);
    cudaMalloc(&d_mac, 32);
    
    cudaMemcpy(d_password, password, 10, cudaMemcpyHostToDevice);
    cudaMemcpy(d_salt, salt, 32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ciphertext, ciphertext, 32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mac, mac, 32, cudaMemcpyHostToDevice);
    
    // Run test on device
    dim3 grid(1);
    dim3 block(1);
    
    // Simple test kernel
    auto kernel = [] __global__ (uint8_t* password, uint8_t* salt, uint8_t* ciphertext,
                                  uint8_t* expected_mac, uint8_t* derived_key_out,
                                  uint8_t* calculated_mac_out, uint8_t* scratch) {
        uint8_t derived_key[32];
        
        // Test scrypt with known password
        scrypt_proper(password, 10, salt, 32, 
                     262144, 8, 1, 32, 
                     derived_key, scratch);
        
        // Copy derived key to output
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
    };
    
    // Run kernel
    cudaError_t err;
    void (*kernel_ptr)(uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*) = kernel;
    kernel_ptr<<<grid, block>>>(d_password, d_salt, d_ciphertext, d_mac, 
                                d_derived_key, d_calculated_mac, d_scratch);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    
    cudaDeviceSynchronize();
    
    // Get results
    uint8_t h_derived_key[32];
    uint8_t h_calculated_mac[32];
    
    cudaMemcpy(h_derived_key, d_derived_key, 32, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_calculated_mac, d_calculated_mac, 32, cudaMemcpyDeviceToHost);
    
    // Print results
    std::cout << "Password: " << password << std::endl;
    
    std::cout << "Derived key: ";
    for (int i = 0; i < 32; i++) {
        printf("%02x", h_derived_key[i]);
    }
    std::cout << std::endl;
    
    std::cout << "Calculated MAC: ";
    for (int i = 0; i < 32; i++) {
        printf("%02x", h_calculated_mac[i]);
    }
    std::cout << std::endl;
    
    std::cout << "Expected MAC:   ";
    for (int i = 0; i < 32; i++) {
        printf("%02x", mac[i]);
    }
    std::cout << std::endl;
    
    // Check if MACs match
    bool match = true;
    for (int i = 0; i < 32; i++) {
        if (h_calculated_mac[i] != mac[i]) {
            match = false;
            break;
        }
    }
    
    std::cout << "MAC match: " << (match ? "YES - PASSWORD CORRECT!" : "NO - PASSWORD WRONG") << std::endl;
    
    // Cleanup
    cudaFree(d_derived_key);
    cudaFree(d_scratch);
    cudaFree(d_calculated_mac);
    
    return 0;
}