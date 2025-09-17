#include <cuda_runtime.h>
#include <iostream>
#include <cstring>
#include <iomanip>

// Include the crypto implementations
#include "src/gpu_crypto.cu"

__global__ void test_keccak_kernel(uint8_t* output) {
    // Test Keccak-256 with known test vector
    const char input[8] = "testing";
    keccak256((uint8_t*)input, 7, output);
}

int main() {
    uint8_t* d_output;
    uint8_t h_output[32];
    
    cudaMalloc(&d_output, 32);
    
    // Test Keccak-256
    test_keccak_kernel<<<1, 1>>>(d_output);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output, d_output, 32, cudaMemcpyDeviceToHost);
    
    std::cout << "Keccak-256('testing') = ";
    for (int i = 0; i < 32; i++) {
        std::cout << std::hex << std::setw(2) << std::setfill('0') 
                  << (int)h_output[i];
    }
    std::cout << std::endl;
    
    // Expected: 5f16f4c7f149ac4f9510d9cf8cf384038ad348b3bcdc01915f95de12df9d1b02
    
    cudaFree(d_output);
    return 0;
}