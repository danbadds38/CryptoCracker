#ifndef CHECKPOINT_H
#define CHECKPOINT_H

#include "types.h"
#include <sqlite3.h>
#include <memory>
#include <mutex>
#include <thread>

class CheckpointManager {
public:
    CheckpointManager(const std::string& checkpoint_path);
    ~CheckpointManager();
    
    bool initialize();
    bool saveCheckpoint(const CheckpointData& data);
    bool loadCheckpoint(CheckpointData& data);
    bool hasCheckpoint();
    void clearCheckpoint();
    
    void enableAutoSave(int interval_seconds = 10);
    void disableAutoSave();
    
private:
    std::string db_path_;
    sqlite3* db_;
    std::mutex mutex_;
    
    bool auto_save_enabled_;
    std::thread auto_save_thread_;
    CheckpointData* current_data_;
    
    bool createTables();
    void autoSaveLoop(int interval_seconds);
};

#endif