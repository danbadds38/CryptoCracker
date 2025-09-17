#include "checkpoint.h"
#include <iostream>
#include <chrono>
#include <thread>

CheckpointManager::CheckpointManager(const std::string& checkpoint_path)
    : db_path_(checkpoint_path), db_(nullptr), auto_save_enabled_(false), current_data_(nullptr) {
}

CheckpointManager::~CheckpointManager() {
    disableAutoSave();
    if (db_) {
        sqlite3_close(db_);
    }
    if (current_data_) {
        delete current_data_;
    }
}

bool CheckpointManager::initialize() {
    int rc = sqlite3_open(db_path_.c_str(), &db_);
    if (rc != SQLITE_OK) {
        std::cerr << "Cannot open checkpoint database: " << sqlite3_errmsg(db_) << std::endl;
        return false;
    }
    
    // Enable WAL mode for better concurrency
    const char* wal_sql = "PRAGMA journal_mode=WAL;";
    char* err_msg = nullptr;
    rc = sqlite3_exec(db_, wal_sql, nullptr, nullptr, &err_msg);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to set WAL mode: " << err_msg << std::endl;
        sqlite3_free(err_msg);
    }
    
    return createTables();
}

bool CheckpointManager::createTables() {
    const char* create_sql = R"(
        CREATE TABLE IF NOT EXISTS checkpoint (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            total_attempts INTEGER NOT NULL,
            base_password_index INTEGER NOT NULL,
            suffix_position INTEGER NOT NULL,
            last_attempt TEXT,
            last_save_time INTEGER NOT NULL,
            created_at INTEGER DEFAULT (strftime('%s', 'now')),
            updated_at INTEGER DEFAULT (strftime('%s', 'now'))
        );
        
        CREATE TABLE IF NOT EXISTS performance_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp INTEGER DEFAULT (strftime('%s', 'now')),
            attempts_per_second REAL,
            gpu_temperature INTEGER,
            gpu_utilization INTEGER,
            memory_usage INTEGER
        );
        
        CREATE INDEX IF NOT EXISTS idx_performance_timestamp ON performance_log(timestamp);
    )";
    
    char* err_msg = nullptr;
    int rc = sqlite3_exec(db_, create_sql, nullptr, nullptr, &err_msg);
    
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to create tables: " << err_msg << std::endl;
        sqlite3_free(err_msg);
        return false;
    }
    
    return true;
}

bool CheckpointManager::saveCheckpoint(const CheckpointData& data) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Delete any existing checkpoint (we only keep one)
    const char* delete_sql = "DELETE FROM checkpoint;";
    char* err_msg = nullptr;
    sqlite3_exec(db_, delete_sql, nullptr, nullptr, &err_msg);
    
    // Insert new checkpoint
    const char* insert_sql = R"(
        INSERT INTO checkpoint (id, total_attempts, base_password_index, 
                               suffix_position, last_attempt, last_save_time, updated_at)
        VALUES (1, ?, ?, ?, ?, ?, strftime('%s', 'now'));
    )";
    
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, insert_sql, -1, &stmt, nullptr);
    
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to prepare checkpoint insert: " << sqlite3_errmsg(db_) << std::endl;
        return false;
    }
    
    sqlite3_bind_int64(stmt, 1, data.total_attempts);
    sqlite3_bind_int(stmt, 2, data.base_password_index);
    sqlite3_bind_int64(stmt, 3, data.suffix_position);
    sqlite3_bind_text(stmt, 4, data.last_attempt.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_int64(stmt, 5, data.last_save_time);
    
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    
    if (rc != SQLITE_DONE) {
        std::cerr << "Failed to save checkpoint: " << sqlite3_errmsg(db_) << std::endl;
        return false;
    }
    
    // Update current_data if auto-save is enabled
    if (current_data_) {
        *current_data_ = data;
    }
    
    return true;
}

bool CheckpointManager::loadCheckpoint(CheckpointData& data) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    const char* select_sql = R"(
        SELECT total_attempts, base_password_index, suffix_position, 
               last_attempt, last_save_time
        FROM checkpoint
        WHERE id = 1;
    )";
    
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, select_sql, -1, &stmt, nullptr);
    
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to prepare checkpoint query: " << sqlite3_errmsg(db_) << std::endl;
        return false;
    }
    
    rc = sqlite3_step(stmt);
    
    if (rc == SQLITE_ROW) {
        data.total_attempts = sqlite3_column_int64(stmt, 0);
        data.base_password_index = sqlite3_column_int(stmt, 1);
        data.suffix_position = sqlite3_column_int64(stmt, 2);
        
        const char* last_attempt = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
        if (last_attempt) {
            data.last_attempt = last_attempt;
        }
        
        data.last_save_time = sqlite3_column_int64(stmt, 4);
        
        sqlite3_finalize(stmt);
        return true;
    }
    
    sqlite3_finalize(stmt);
    return false;
}

bool CheckpointManager::hasCheckpoint() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    const char* count_sql = "SELECT COUNT(*) FROM checkpoint;";
    sqlite3_stmt* stmt;
    
    int rc = sqlite3_prepare_v2(db_, count_sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        return false;
    }
    
    rc = sqlite3_step(stmt);
    int count = 0;
    
    if (rc == SQLITE_ROW) {
        count = sqlite3_column_int(stmt, 0);
    }
    
    sqlite3_finalize(stmt);
    return count > 0;
}

void CheckpointManager::clearCheckpoint() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    const char* delete_sql = "DELETE FROM checkpoint;";
    char* err_msg = nullptr;
    sqlite3_exec(db_, delete_sql, nullptr, nullptr, &err_msg);
    
    if (err_msg) {
        std::cerr << "Failed to clear checkpoint: " << err_msg << std::endl;
        sqlite3_free(err_msg);
    }
}

void CheckpointManager::enableAutoSave(int interval_seconds) {
    if (auto_save_enabled_) {
        return;
    }
    
    auto_save_enabled_ = true;
    current_data_ = new CheckpointData();
    
    auto_save_thread_ = std::thread(&CheckpointManager::autoSaveLoop, this, interval_seconds);
}

void CheckpointManager::disableAutoSave() {
    if (!auto_save_enabled_) {
        return;
    }
    
    auto_save_enabled_ = false;
    
    if (auto_save_thread_.joinable()) {
        auto_save_thread_.join();
    }
    
    if (current_data_) {
        delete current_data_;
        current_data_ = nullptr;
    }
}

void CheckpointManager::autoSaveLoop(int interval_seconds) {
    while (auto_save_enabled_) {
        std::this_thread::sleep_for(std::chrono::seconds(interval_seconds));
        
        if (auto_save_enabled_ && current_data_) {
            current_data_->last_save_time = std::time(nullptr);
            saveCheckpoint(*current_data_);
            
            std::cout << "\r[CHECKPOINT] Saved at " << current_data_->total_attempts 
                      << " attempts" << std::flush;
        }
    }
}