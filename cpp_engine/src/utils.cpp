/**
 * @file utils.cpp
 * @brief Utility functions for FoldEngine
 */

#include "utils.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <sys/resource.h>

#ifdef __APPLE__
#include <mach/mach.h>
#elif defined(__linux__)
#include <sys/sysinfo.h>
#include <unistd.h>
#endif

#ifdef CUDA_AVAILABLE
#include <cuda_runtime.h>
#endif

namespace fold_engine {
namespace utils {

std::string load_fasta(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open FASTA file: " + filename);
    }
    
    std::string line;
    std::string sequence;
    bool first_sequence = true;
    
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        if (line[0] == '>') {
            if (!first_sequence) {
                // Multiple sequences - return first one
                break;
            }
            first_sequence = false;
            continue;
        }
        
        // Remove whitespace and convert to uppercase
        for (char c : line) {
            if (!std::isspace(c)) {
                sequence += std::toupper(c);
            }
        }
    }
    
    if (sequence.empty()) {
        throw std::runtime_error("No sequence found in FASTA file: " + filename);
    }
    
    return sequence;
}

void save_pdb(const FoldingResult& result, const std::string& filename, bool include_confidence) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot create PDB file: " + filename);
    }
    
    // PDB header
    file << "HEADER    PROTEIN STRUCTURE PREDICTION             " 
         << std::setfill('0') << std::setw(2) << 1 << "-JAN-24   FOLD" << std::endl;
    file << "TITLE     STRUCTURE PREDICTED BY ODINFOLD" << std::endl;
    file << "REMARK   1 PREDICTION CONFIDENCE: " << std::fixed << std::setprecision(3) 
         << result.confidence.mean().item<float>() << std::endl;
    file << "REMARK   1 PREDICTED TM-SCORE: " << result.tm_score_pred << std::endl;
    
    // Coordinates
    auto coords_accessor = result.coordinates.accessor<float, 2>();
    auto conf_accessor = result.confidence.accessor<float, 1>();
    
    for (int i = 0; i < result.sequence_length; ++i) {
        char aa = (i < static_cast<int>(result.sequence.length())) ? result.sequence[i] : 'X';
        
        float x = coords_accessor[i][0];
        float y = coords_accessor[i][1];
        float z = coords_accessor[i][2];
        
        float b_factor = include_confidence ? (conf_accessor[i] * 100.0f) : 50.0f;
        
        file << "ATOM  " << std::setw(5) << (i + 1) << "  CA  " 
             << aa << " A" << std::setw(4) << (i + 1) << "    "
             << std::fixed << std::setprecision(3)
             << std::setw(8) << x << std::setw(8) << y << std::setw(8) << z
             << std::setw(6) << "1.00" << std::setw(6) << b_factor
             << "           C" << std::endl;
    }
    
    file << "END" << std::endl;
    file.close();
}

float calculate_rmsd(const torch::Tensor& coords1, const torch::Tensor& coords2) {
    if (coords1.sizes() != coords2.sizes()) {
        throw std::runtime_error("Coordinate tensors must have same shape");
    }
    
    // Calculate RMSD
    auto diff = coords1 - coords2;
    auto squared_diff = diff * diff;
    auto msd = squared_diff.sum() / coords1.numel();
    
    return std::sqrt(msd.item<float>());
}

float calculate_tm_score(const torch::Tensor& coords1, const torch::Tensor& coords2) {
    if (coords1.sizes() != coords2.sizes()) {
        throw std::runtime_error("Coordinate tensors must have same shape");
    }
    
    int n = coords1.size(0);
    float d0 = 1.24f * std::pow(n - 15, 1.0f/3.0f) - 1.8f;
    if (d0 < 0.5f) d0 = 0.5f;
    
    // Calculate distances
    auto diff = coords1 - coords2;
    auto distances = torch::norm(diff, 2, 1);
    
    // Calculate TM-score
    auto tm_terms = 1.0f / (1.0f + (distances / d0).pow(2));
    float tm_score = tm_terms.sum().item<float>() / n;
    
    return tm_score;
}

float get_memory_usage_mb() {
#ifdef __APPLE__
    struct mach_task_basic_info info;
    mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  (task_info_t)&info, &infoCount) != KERN_SUCCESS) {
        return 0.0f;
    }
    return static_cast<float>(info.resident_size) / (1024.0f * 1024.0f);
    
#elif defined(__linux__)
    std::ifstream file("/proc/self/status");
    std::string line;
    
    while (std::getline(file, line)) {
        if (line.substr(0, 6) == "VmRSS:") {
            std::istringstream iss(line);
            std::string label;
            float value;
            std::string unit;
            iss >> label >> value >> unit;
            
            if (unit == "kB") {
                return value / 1024.0f;
            }
        }
    }
    return 0.0f;
    
#else
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return static_cast<float>(usage.ru_maxrss) / 1024.0f;
#endif
}

float get_gpu_memory_usage_mb() {
#ifdef CUDA_AVAILABLE
    if (!torch::cuda::is_available()) {
        return 0.0f;
    }
    
    size_t free_bytes, total_bytes;
    cudaError_t cuda_status = cudaMemGetInfo(&free_bytes, &total_bytes);
    
    if (cuda_status != cudaSuccess) {
        return 0.0f;
    }
    
    size_t used_bytes = total_bytes - free_bytes;
    return static_cast<float>(used_bytes) / (1024.0f * 1024.0f);
#else
    return 0.0f;
#endif
}

std::vector<std::string> split_string(const std::string& str, char delimiter) {
    std::vector<std::string> tokens;
    std::stringstream ss(str);
    std::string token;
    
    while (std::getline(ss, token, delimiter)) {
        if (!token.empty()) {
            tokens.push_back(token);
        }
    }
    
    return tokens;
}

std::string trim_string(const std::string& str) {
    size_t start = str.find_first_not_of(" \t\n\r");
    if (start == std::string::npos) {
        return "";
    }
    
    size_t end = str.find_last_not_of(" \t\n\r");
    return str.substr(start, end - start + 1);
}

bool file_exists(const std::string& filename) {
    std::ifstream file(filename);
    return file.good();
}

std::string get_file_extension(const std::string& filename) {
    size_t dot_pos = filename.find_last_of('.');
    if (dot_pos == std::string::npos) {
        return "";
    }
    return filename.substr(dot_pos + 1);
}

void create_directory(const std::string& path) {
    // Simple directory creation (platform-specific implementation would be better)
    std::string command = "mkdir -p " + path;
    system(command.c_str());
}

std::string format_time(float milliseconds) {
    if (milliseconds < 1000.0f) {
        return std::to_string(static_cast<int>(milliseconds)) + " ms";
    } else if (milliseconds < 60000.0f) {
        return std::to_string(milliseconds / 1000.0f) + " s";
    } else {
        int minutes = static_cast<int>(milliseconds / 60000.0f);
        float seconds = (milliseconds - minutes * 60000.0f) / 1000.0f;
        return std::to_string(minutes) + "m " + std::to_string(seconds) + "s";
    }
}

std::string format_memory(float megabytes) {
    if (megabytes < 1024.0f) {
        return std::to_string(static_cast<int>(megabytes)) + " MB";
    } else {
        float gigabytes = megabytes / 1024.0f;
        return std::to_string(gigabytes) + " GB";
    }
}

void print_progress_bar(int current, int total, int width) {
    float progress = static_cast<float>(current) / total;
    int filled = static_cast<int>(progress * width);
    
    std::cout << "\r[";
    for (int i = 0; i < width; ++i) {
        if (i < filled) {
            std::cout << "=";
        } else if (i == filled) {
            std::cout << ">";
        } else {
            std::cout << " ";
        }
    }
    std::cout << "] " << static_cast<int>(progress * 100) << "% (" 
              << current << "/" << total << ")";
    std::cout.flush();
    
    if (current == total) {
        std::cout << std::endl;
    }
}

bool validate_amino_acid_sequence(const std::string& sequence) {
    const std::string valid_aa = "ARNDCQEGHILKMFPSTWYV";
    
    for (char c : sequence) {
        char upper_c = std::toupper(c);
        if (valid_aa.find(upper_c) == std::string::npos && upper_c != 'X') {
            return false;
        }
    }
    
    return !sequence.empty() && sequence.length() <= 2048;
}

std::string generate_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S");
    return ss.str();
}

void log_message(const std::string& level, const std::string& message) {
    auto timestamp = generate_timestamp();
    std::cout << "[" << timestamp << "] " << level << ": " << message << std::endl;
}

void log_info(const std::string& message) {
    log_message("INFO", message);
}

void log_warning(const std::string& message) {
    log_message("WARN", message);
}

void log_error(const std::string& message) {
    log_message("ERROR", message);
}

} // namespace utils
} // namespace fold_engine
