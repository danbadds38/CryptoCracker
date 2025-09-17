cmake_minimum_required(VERSION 3.18)
project(CryptoCracker CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)

find_package(CUDA REQUIRED)
find_package(SQLite3 REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(Threads REQUIRED)

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
    ${OPENSSL_INCLUDE_DIR}
)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch=sm_86 -O3)

cuda_add_executable(cryptocracker
    src/main.cpp
    src/gpu_engine.cu
    src/checkpoint.cpp
    src/wallet.cpp
    src/crypto_utils.cpp
)

target_link_libraries(cryptocracker
    ${CUDA_LIBRARIES}
    SQLite::SQLite3
    OpenSSL::SSL
    OpenSSL::Crypto
    Threads::Threads
)

set_target_properties(cryptocracker PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "86"
)