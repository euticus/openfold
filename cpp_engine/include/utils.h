/**
 * @file utils.h
 * @brief Utility functions and helpers for FoldEngine
 */

#pragma once

#include "fold_engine.h"
#include <string>
#include <vector>
#include <chrono>

namespace fold_engine {
namespace utils {

/**
 * @brief Load protein sequence from FASTA file
 * @param filename Path to FASTA file
 * @return Protein sequence string
 */
std::string load_fasta(const std::string& filename);

/**
 * @brief Save folding result to PDB file
 * @param result Folding result to save
 * @param filename Output PDB filename
 * @param include_confidence Whether to include confidence scores in B-factors
 */
void save_pdb(const FoldingResult& result, const std::string& filename, 
              bool include_confidence = true);

/**
 * @brief Calculate RMSD between two coordinate sets
 * @param coords1 First coordinate tensor [N, 3]
 * @param coords2 Second coordinate tensor [N, 3]
 * @return RMSD value in Angstroms
 */
float calculate_rmsd(const torch::Tensor& coords1, const torch::Tensor& coords2);

/**
 * @brief Calculate TM-score between two structures
 * @param coords1 First coordinate tensor [N, 3]
 * @param coords2 Second coordinate tensor [N, 3]
 * @return TM-score value (0-1)
 */
float calculate_tm_score(const torch::Tensor& coords1, const torch::Tensor& coords2);

/**
 * @brief Get current memory usage in MB
 * @return Memory usage in megabytes
 */
float get_memory_usage_mb();

/**
 * @brief Get GPU memory usage in MB
 * @return GPU memory usage in megabytes
 */
float get_gpu_memory_usage_mb();

/**
 * @brief Split string by delimiter
 * @param str Input string
 * @param delimiter Character to split on
 * @return Vector of string tokens
 */
std::vector<std::string> split_string(const std::string& str, char delimiter);

/**
 * @brief Trim whitespace from string
 * @param str Input string
 * @return Trimmed string
 */
std::string trim_string(const std::string& str);

/**
 * @brief Check if file exists
 * @param filename Path to file
 * @return True if file exists
 */
bool file_exists(const std::string& filename);

/**
 * @brief Get file extension
 * @param filename Path to file
 * @return File extension (without dot)
 */
std::string get_file_extension(const std::string& filename);

/**
 * @brief Create directory recursively
 * @param path Directory path to create
 */
void create_directory(const std::string& path);

/**
 * @brief Format time duration for display
 * @param milliseconds Time in milliseconds
 * @return Formatted time string
 */
std::string format_time(float milliseconds);

/**
 * @brief Format memory size for display
 * @param megabytes Memory in megabytes
 * @return Formatted memory string
 */
std::string format_memory(float megabytes);

/**
 * @brief Print progress bar to console
 * @param current Current progress value
 * @param total Total progress value
 * @param width Width of progress bar in characters
 */
void print_progress_bar(int current, int total, int width = 50);

/**
 * @brief Validate amino acid sequence
 * @param sequence Protein sequence to validate
 * @return True if sequence is valid
 */
bool validate_amino_acid_sequence(const std::string& sequence);

/**
 * @brief Generate timestamp string
 * @return Timestamp in YYYYMMDD_HHMMSS format
 */
std::string generate_timestamp();

/**
 * @brief Log message with timestamp and level
 * @param level Log level (INFO, WARN, ERROR)
 * @param message Message to log
 */
void log_message(const std::string& level, const std::string& message);

/**
 * @brief Log info message
 * @param message Message to log
 */
void log_info(const std::string& message);

/**
 * @brief Log warning message
 * @param message Message to log
 */
void log_warning(const std::string& message);

/**
 * @brief Log error message
 * @param message Message to log
 */
void log_error(const std::string& message);

/**
 * @brief Timer class for performance measurement
 */
class Timer {
public:
    Timer() : start_time_(std::chrono::high_resolution_clock::now()) {}
    
    void reset() {
        start_time_ = std::chrono::high_resolution_clock::now();
    }
    
    float elapsed_ms() const {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time_).count();
        return static_cast<float>(duration) / 1000.0f;
    }
    
    float elapsed_seconds() const {
        return elapsed_ms() / 1000.0f;
    }

private:
    std::chrono::high_resolution_clock::time_point start_time_;
};

/**
 * @brief RAII class for automatic timing
 */
class ScopedTimer {
public:
    explicit ScopedTimer(const std::string& name) 
        : name_(name), timer_() {
        log_info("Starting " + name_);
    }
    
    ~ScopedTimer() {
        float elapsed = timer_.elapsed_ms();
        log_info("Completed " + name_ + " in " + format_time(elapsed));
    }

private:
    std::string name_;
    Timer timer_;
};

/**
 * @brief Memory usage tracker
 */
class MemoryTracker {
public:
    MemoryTracker() : initial_memory_(get_memory_usage_mb()) {}
    
    float current_usage_mb() const {
        return get_memory_usage_mb();
    }
    
    float peak_usage_mb() const {
        float current = current_usage_mb();
        if (current > peak_memory_) {
            peak_memory_ = current;
        }
        return peak_memory_;
    }
    
    float delta_mb() const {
        return current_usage_mb() - initial_memory_;
    }
    
    void reset() {
        initial_memory_ = get_memory_usage_mb();
        peak_memory_ = initial_memory_;
    }

private:
    float initial_memory_;
    mutable float peak_memory_ = 0.0f;
};

} // namespace utils
} // namespace fold_engine
