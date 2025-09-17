#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <signal.h>
#include <getopt.h>
#include <thread>

#include "gpu_engine.h"
#include "checkpoint.h"
#include "wallet.h"
#include "types.h"

// Global flag for graceful shutdown
volatile sig_atomic_t g_shutdown = 0;

void signal_handler(int sig) {
    g_shutdown = 1;
    std::cout << "\n[INFO] Shutdown requested. Saving checkpoint..." << std::endl;
}

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --wallet PATH        Path to wallet JSON file (required)\n";
    std::cout << "  --passwords PATH     Path to base passwords file (required)\n";
    std::cout << "  --gpu ID            GPU device ID (default: 0)\n";
    std::cout << "  --checkpoint PATH    Checkpoint database path (default: checkpoints/session.db)\n";
    std::cout << "  --resume            Resume from last checkpoint\n";
    std::cout << "  --suffix-length N   Suffix length to test (4 or 5, default: both)\n";
    std::cout << "  --max-attempts N    Maximum attempts before stopping\n";
    std::cout << "  --batch-size N      GPU batch size (default: 1048576)\n";
    std::cout << "  --log PATH          Log file path\n";
    std::cout << "  --benchmark         Run benchmark mode\n";
    std::cout << "  --test              Run test mode\n";
    std::cout << "  --help              Show this help message\n";
}

std::vector<std::string> load_passwords(const std::string& filepath) {
    std::vector<std::string> passwords;
    std::ifstream file(filepath);
    
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open passwords file: " + filepath);
    }
    
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            passwords.push_back(line);
        }
    }
    
    file.close();
    return passwords;
}

void print_progress(uint64_t current, uint64_t total, double speed, double eta_seconds) {
    double percentage = (double)current / total * 100;
    int bar_width = 50;
    int filled = (int)(bar_width * percentage / 100);
    
    std::cout << "\r[";
    for (int i = 0; i < bar_width; i++) {
        if (i < filled) std::cout << "=";
        else if (i == filled) std::cout << ">";
        else std::cout << " ";
    }
    
    std::cout << "] " << std::fixed << std::setprecision(1) << percentage << "% ";
    std::cout << "| Speed: " << std::fixed << std::setprecision(2) << (speed / 1000000) << "M/s ";
    
    if (eta_seconds > 0) {
        int hours = eta_seconds / 3600;
        int minutes = (eta_seconds - hours * 3600) / 60;
        int seconds = eta_seconds - hours * 3600 - minutes * 60;
        std::cout << "| ETA: " << hours << "h " << minutes << "m " << seconds << "s ";
    }
    
    std::cout << std::flush;
}

int run_cracker(const std::string& wallet_path, const std::string& passwords_path,
                int gpu_id, const std::string& checkpoint_path, bool resume_mode,
                int suffix_length, uint64_t max_attempts) {
    
    // Setup signal handlers
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    // Load wallet
    WalletData wallet;
    if (!WalletLoader::loadFromFile(wallet_path, wallet)) {
        std::cerr << "Failed to load wallet file" << std::endl;
        return 1;
    }
    
    std::cout << "[INFO] Wallet loaded: " << wallet.address << std::endl;
    
    // Load base passwords
    std::vector<std::string> base_passwords = load_passwords(passwords_path);
    std::cout << "[INFO] Loaded " << base_passwords.size() << " base passwords" << std::endl;
    
    // Initialize GPU engine
    GPUEngine gpu_engine(gpu_id);
    if (!gpu_engine.initialize(wallet, base_passwords)) {
        std::cerr << "Failed to initialize GPU engine" << std::endl;
        return 1;
    }
    
    std::string gpu_info;
    gpu_engine.getDeviceInfo(gpu_info);
    std::cout << "[INFO] GPU initialized:\n" << gpu_info << std::endl;
    
    // Initialize checkpoint manager
    CheckpointManager checkpoint_mgr(checkpoint_path);
    if (!checkpoint_mgr.initialize()) {
        std::cerr << "Failed to initialize checkpoint manager" << std::endl;
        return 1;
    }
    
    // Load checkpoint if resuming
    CheckpointData checkpoint_data;
    if (resume_mode && checkpoint_mgr.hasCheckpoint()) {
        if (checkpoint_mgr.loadCheckpoint(checkpoint_data)) {
            std::cout << "[INFO] Resuming from checkpoint:\n";
            std::cout << "  - Total attempts: " << checkpoint_data.total_attempts << "\n";
            std::cout << "  - Base password index: " << checkpoint_data.base_password_index << "\n";
            std::cout << "  - Suffix position: " << checkpoint_data.suffix_position << std::endl;
        }
    } else {
        checkpoint_data.total_attempts = 0;
        checkpoint_data.base_password_index = 0;
        checkpoint_data.suffix_position = 0;
    }
    
    // Enable auto-save (disabled for debugging)
    // checkpoint_mgr.enableAutoSave(10);  // Save every 10 seconds
    
    // Calculate total search space
    uint64_t charset_size = 94;
    uint64_t total_combinations = 0;
    
    if (suffix_length == 0) {
        // Default: try both 4 and 5
        total_combinations += charset_size * charset_size * charset_size * charset_size;
        total_combinations += charset_size * charset_size * charset_size * charset_size * charset_size;
    } else {
        // Specific length: calculate for that length
        uint64_t single_combo = 1;
        for (int i = 0; i < suffix_length; i++) {
            single_combo *= charset_size;
        }
        total_combinations = single_combo;
    }
    
    total_combinations *= base_passwords.size();
    
    std::cout << "[INFO] Total search space: " << total_combinations << " combinations" << std::endl;
    
    // Main cracking loop
    auto start_time = std::chrono::steady_clock::now();
    uint64_t batch_size = gpu_engine.getBatchSize();
    bool found = false;
    std::string found_password;
    
    std::cout << "[DEBUG] Starting main loop, batch_size=" << batch_size << std::endl;
    
    for (uint32_t base_idx = checkpoint_data.base_password_index; 
         base_idx < base_passwords.size() && !g_shutdown && !found; 
         base_idx++) {
        
        std::cout << "[DEBUG] Processing base password index " << base_idx << std::endl;
        
        uint64_t suffix_start = (base_idx == checkpoint_data.base_password_index) 
                               ? checkpoint_data.suffix_position : 0;
        
        // Try different suffix lengths
        std::vector<int> lengths_to_try;
        if (suffix_length == 0) {
            lengths_to_try = {4, 5};
        } else {
            lengths_to_try = {suffix_length};
        }
        
        for (int len : lengths_to_try) {
            uint64_t max_suffix = 1;
            for (int i = 0; i < len; i++) {
                max_suffix *= charset_size;
            }
            
            std::cout << "[DEBUG] Testing suffix length " << len 
                      << ", max_suffix=" << max_suffix << std::endl;
            
            for (uint64_t suffix_pos = suffix_start; 
                 suffix_pos < max_suffix && !g_shutdown && !found; 
                 suffix_pos += batch_size) {
                
                uint64_t end_pos = std::min(suffix_pos + batch_size, max_suffix);
                
                std::cout << "[DEBUG] About to call processBatch" << std::endl;
                
                // Process batch on GPU
                if (gpu_engine.processBatch(base_idx, suffix_pos, end_pos, found_password, len)) {
                    found = true;
                    break;
                }
                
                // Update checkpoint data
                checkpoint_data.base_password_index = base_idx;
                checkpoint_data.suffix_position = suffix_pos;
                checkpoint_data.total_attempts += (end_pos - suffix_pos);
                checkpoint_data.last_attempt = base_passwords[base_idx] + " + suffix";
                
                // Debug output every 100 attempts
                if (checkpoint_data.total_attempts % 100 == 0) {
                    std::cout << "[DEBUG] Processed " << checkpoint_data.total_attempts 
                              << " attempts, current: " << base_passwords[base_idx] 
                              << " suffix_pos: " << suffix_pos << std::endl;
                }
                
                // Calculate and display progress
                auto current_time = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                    current_time - start_time).count();
                
                double speed = checkpoint_data.total_attempts / (elapsed + 0.001);
                double remaining = total_combinations - checkpoint_data.total_attempts;
                double eta = remaining / speed;
                
                print_progress(checkpoint_data.total_attempts, total_combinations, speed, eta);
                
                // Check max attempts
                if (max_attempts > 0 && checkpoint_data.total_attempts >= max_attempts) {
                    std::cout << "\n[INFO] Maximum attempts reached" << std::endl;
                    g_shutdown = 1;
                    break;
                }
            }
            
            suffix_start = 0;  // Reset for next length
        }
    }
    
    std::cout << std::endl;
    
    // Save final checkpoint
    checkpoint_data.last_save_time = std::time(nullptr);
    checkpoint_mgr.saveCheckpoint(checkpoint_data);
    checkpoint_mgr.disableAutoSave();
    
    // Report results
    if (found) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "[SUCCESS] PASSWORD FOUND!" << std::endl;
        std::cout << "Password: " << found_password << std::endl;
        
        // Validate and decrypt
        std::vector<uint8_t> private_key;
        if (WalletLoader::validatePassword(wallet, found_password, private_key)) {
            std::cout << "Private Key: 0x";
            for (uint8_t byte : private_key) {
                std::cout << std::hex << std::setw(2) << std::setfill('0') 
                         << static_cast<int>(byte);
            }
            std::cout << std::endl;
        }
        std::cout << "========================================" << std::endl;
        
        // Clear checkpoint on success
        checkpoint_mgr.clearCheckpoint();
        
        return 0;
    } else if (g_shutdown) {
        std::cout << "[INFO] Cracking interrupted. Checkpoint saved." << std::endl;
        std::cout << "[INFO] Run with --resume to continue" << std::endl;
        return 2;
    } else {
        std::cout << "[INFO] Password not found in search space" << std::endl;
        return 1;
    }
}

int main(int argc, char* argv[]) {
    std::string wallet_path;
    std::string passwords_path;
    std::string checkpoint_path = "checkpoints/session.db";
    std::string log_path;
    int gpu_id = 0;
    bool resume_mode = false;
    bool benchmark_mode = false;
    bool test_mode = false;
    int suffix_length = 0;  // 0 means try both 4 and 5
    uint64_t max_attempts = 0;
    size_t batch_size = 1048576;
    
    static struct option long_options[] = {
        {"wallet", required_argument, 0, 'w'},
        {"passwords", required_argument, 0, 'p'},
        {"gpu", required_argument, 0, 'g'},
        {"checkpoint", required_argument, 0, 'c'},
        {"resume", no_argument, 0, 'r'},
        {"suffix-length", required_argument, 0, 's'},
        {"max-attempts", required_argument, 0, 'm'},
        {"batch-size", required_argument, 0, 'b'},
        {"log", required_argument, 0, 'l'},
        {"benchmark", no_argument, 0, 'B'},
        {"test", no_argument, 0, 'T'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };
    
    int option_index = 0;
    int c;
    
    while ((c = getopt_long(argc, argv, "w:p:g:c:rs:m:b:l:BTh", 
                            long_options, &option_index)) != -1) {
        switch (c) {
            case 'w':
                wallet_path = optarg;
                break;
            case 'p':
                passwords_path = optarg;
                break;
            case 'g':
                gpu_id = std::stoi(optarg);
                break;
            case 'c':
                checkpoint_path = optarg;
                break;
            case 'r':
                resume_mode = true;
                break;
            case 's':
                suffix_length = std::stoi(optarg);
                if (suffix_length < 1 || suffix_length > 12) {
                    std::cerr << "Suffix length must be between 1 and 12" << std::endl;
                    std::cerr << "WARNING: Lengths above 6 will take extremely long!" << std::endl;
                    return 1;
                }
                break;
            case 'm':
                max_attempts = std::stoull(optarg);
                break;
            case 'b':
                batch_size = std::stoull(optarg);
                break;
            case 'l':
                log_path = optarg;
                break;
            case 'B':
                benchmark_mode = true;
                break;
            case 'T':
                test_mode = true;
                break;
            case 'h':
                print_usage(argv[0]);
                return 0;
            default:
                print_usage(argv[0]);
                return 1;
        }
    }
    
    // Benchmark mode
    if (benchmark_mode) {
        std::cout << "Running GPU benchmark..." << std::endl;
        // Benchmark implementation
        return 0;
    }
    
    // Test mode
    if (test_mode) {
        std::cout << "Running tests..." << std::endl;
        // Test implementation
        return 0;
    }
    
    // Validate required arguments
    if (wallet_path.empty() || passwords_path.empty()) {
        if (!resume_mode) {
            std::cerr << "Error: --wallet and --passwords are required" << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }
    
    // Run the cracker
    return run_cracker(wallet_path, passwords_path, gpu_id, checkpoint_path, 
                      resume_mode, suffix_length, max_attempts);
}