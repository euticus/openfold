/**
 * @file fold_engine.cpp
 * @brief Implementation of the FoldEngine class
 */

#include "fold_engine.h"
#include "attention.h"
#include "structure_module.h"
#include "ligand_encoder.h"
#include "mutation_predictor.h"
#include "utils.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <chrono>

namespace fold_engine {

// Amino acid mapping
static const std::unordered_map<char, int> AA_TO_IDX = {
    {'A', 0}, {'R', 1}, {'N', 2}, {'D', 3}, {'C', 4}, {'Q', 5}, {'E', 6}, {'G', 7},
    {'H', 8}, {'I', 9}, {'L', 10}, {'K', 11}, {'M', 12}, {'F', 13}, {'P', 14},
    {'S', 15}, {'T', 16}, {'W', 17}, {'Y', 18}, {'V', 19}, {'X', 20}
};

static const std::vector<char> IDX_TO_AA = {
    'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
    'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'X'
};

FoldEngine::FoldEngine(const FoldConfig& config) 
    : config_(config), device_(torch::kCPU) {
    
    // Set device
    if (config_.use_cuda && torch::cuda::is_available()) {
        device_ = torch::kCUDA;
        std::cout << "Using CUDA device" << std::endl;
    } else {
        device_ = torch::kCPU;
        std::cout << "Using CPU device" << std::endl;
    }
    
    // Initialize components
    attention_module_ = std::make_unique<AttentionModule>(config_.d_model, config_.num_heads);
    structure_module_ = std::make_unique<StructureModule>(config_.d_model);
    ligand_encoder_ = std::make_unique<LigandEncoder>(128);
    mutation_predictor_ = std::make_unique<MutationPredictor>(config_.d_model);
}

FoldEngine::~FoldEngine() = default;

bool FoldEngine::initialize() {
    try {
        std::cout << "Initializing FoldEngine..." << std::endl;
        
        // Load models
        if (!load_models()) {
            std::cerr << "Failed to load models" << std::endl;
            return false;
        }
        
        // Move models to device
        model_.to(device_);
        if (!esm_model_.is_empty()) {
            esm_model_.to(device_);
        }
        
        // Set evaluation mode
        model_.eval();
        if (!esm_model_.is_empty()) {
            esm_model_.eval();
        }
        
        // Enable optimizations
        if (config_.use_mixed_precision && device_.is_cuda()) {
            torch::autocast::set_enabled(true);
        }
        
        initialized_ = true;
        std::cout << "FoldEngine initialized successfully" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Initialization failed: " << e.what() << std::endl;
        return false;
    }
}

bool FoldEngine::load_models() {
    try {
        // Load main model
        std::cout << "Loading main model from: " << config_.model_path << std::endl;
        model_ = torch::jit::load(config_.model_path, device_);
        
        // Load ESM model if available
        if (!config_.esm_path.empty()) {
            std::cout << "Loading ESM model from: " << config_.esm_path << std::endl;
            try {
                esm_model_ = torch::jit::load(config_.esm_path, device_);
            } catch (const std::exception& e) {
                std::cout << "Warning: Could not load ESM model: " << e.what() << std::endl;
                // Continue without ESM model
            }
        }
        
        return true;
        
    } catch (const std::exception& e) {
        throw ModelLoadException("Failed to load models: " + std::string(e.what()));
    }
}

FoldingResult FoldEngine::fold_protein(const ProteinInput& input) {
    if (!initialized_) {
        throw InferenceException("Engine not initialized");
    }
    
    if (!validate_sequence(input.sequence)) {
        throw ValidationException("Invalid protein sequence");
    }
    
    start_timer();
    
    try {
        // Preprocess input
        auto sequence_tensor = preprocess_sequence(input.sequence);
        record_time("preprocessing");
        
        // Extract ESM features if available
        torch::Tensor esm_features;
        if (!esm_model_.is_empty()) {
            esm_features = extract_esm_features(input.sequence);
        } else {
            // Use simple embedding
            esm_features = torch::randn({1, static_cast<int>(input.sequence.length()), config_.d_model}, 
                                      torch::TensorOptions().device(device_));
        }
        
        // Process ligands if provided
        torch::Tensor ligand_features;
        if (!input.ligand_smiles.empty()) {
            ligand_features = process_ligands(input.ligand_smiles);
        }
        
        record_time("feature_extraction");
        
        // Run inference
        std::vector<torch::jit::IValue> inputs_vec;
        inputs_vec.push_back(sequence_tensor);
        inputs_vec.push_back(esm_features);
        
        if (!ligand_features.numel() == 0) {
            inputs_vec.push_back(ligand_features);
        }
        
        torch::NoGradGuard no_grad;
        auto outputs = model_.forward(inputs_vec);
        
        record_time("inference");
        
        // Extract outputs
        auto output_dict = outputs.toGenericDict();
        auto coordinates = output_dict.at("coordinates").toTensor();
        auto confidence = output_dict.at("confidence").toTensor();
        
        // Postprocess results
        auto result = postprocess_output(coordinates, confidence, input);
        record_time("postprocessing");
        
        // Update metrics
        result.inference_time_ms = metrics_.total_time_ms;
        update_memory_usage();
        result.memory_usage_mb = metrics_.memory_current_mb;
        
        return result;
        
    } catch (const std::exception& e) {
        throw InferenceException("Folding failed: " + std::string(e.what()));
    }
}

std::vector<FoldingResult> FoldEngine::fold_batch(const std::vector<ProteinInput>& inputs) {
    std::vector<FoldingResult> results;
    results.reserve(inputs.size());
    
    // For now, process sequentially
    // TODO: Implement true batch processing
    for (const auto& input : inputs) {
        results.push_back(fold_protein(input));
    }
    
    return results;
}

torch::Tensor FoldEngine::predict_mutations(const std::string& sequence, 
                                           const std::vector<std::pair<int, char>>& mutations) {
    if (!initialized_) {
        throw InferenceException("Engine not initialized");
    }
    
    // Use mutation predictor
    return mutation_predictor_->predict_effects(sequence, mutations, device_);
}

void FoldEngine::set_device(const std::string& device_str) {
    if (device_str == "cpu") {
        device_ = torch::kCPU;
    } else if (device_str.substr(0, 4) == "cuda") {
        if (!torch::cuda::is_available()) {
            throw std::runtime_error("CUDA not available");
        }
        device_ = torch::Device(device_str);
    } else {
        throw std::runtime_error("Invalid device string: " + device_str);
    }
    
    // Move models to new device if initialized
    if (initialized_) {
        model_.to(device_);
        if (!esm_model_.is_empty()) {
            esm_model_.to(device_);
        }
    }
}

std::string FoldEngine::get_device() const {
    if (device_.is_cuda()) {
        return "cuda:" + std::to_string(device_.index());
    } else {
        return "cpu";
    }
}

std::string FoldEngine::get_model_info() const {
    std::stringstream ss;
    ss << "FoldEngine Model Information:\n";
    ss << "  Device: " << get_device() << "\n";
    ss << "  Model dimension: " << config_.d_model << "\n";
    ss << "  Number of heads: " << config_.num_heads << "\n";
    ss << "  Number of layers: " << config_.num_layers << "\n";
    ss << "  Max sequence length: " << config_.seq_len_max << "\n";
    ss << "  Mixed precision: " << (config_.use_mixed_precision ? "enabled" : "disabled") << "\n";
    ss << "  Flash attention: " << (config_.use_flash_attention ? "enabled" : "disabled") << "\n";
    return ss.str();
}

bool FoldEngine::validate_sequence(const std::string& sequence) {
    if (sequence.empty() || sequence.length() > 2048) {
        return false;
    }
    
    for (char c : sequence) {
        if (AA_TO_IDX.find(std::toupper(c)) == AA_TO_IDX.end()) {
            return false;
        }
    }
    
    return true;
}

torch::Tensor FoldEngine::sequence_to_indices(const std::string& sequence) {
    std::vector<int> indices;
    indices.reserve(sequence.length());
    
    for (char c : sequence) {
        char upper_c = std::toupper(c);
        auto it = AA_TO_IDX.find(upper_c);
        if (it != AA_TO_IDX.end()) {
            indices.push_back(it->second);
        } else {
            indices.push_back(AA_TO_IDX.at('X')); // Unknown amino acid
        }
    }
    
    return torch::tensor(indices, torch::kLong);
}

std::string FoldEngine::indices_to_sequence(const torch::Tensor& indices) {
    std::string sequence;
    auto accessor = indices.accessor<long, 1>();
    
    for (int i = 0; i < indices.size(0); ++i) {
        int idx = accessor[i];
        if (idx >= 0 && idx < static_cast<int>(IDX_TO_AA.size())) {
            sequence += IDX_TO_AA[idx];
        } else {
            sequence += 'X';
        }
    }
    
    return sequence;
}

torch::Tensor FoldEngine::preprocess_sequence(const std::string& sequence) {
    auto indices = sequence_to_indices(sequence);
    return indices.unsqueeze(0).to(device_); // Add batch dimension
}

torch::Tensor FoldEngine::extract_esm_features(const std::string& sequence) {
    // Simplified ESM feature extraction
    // In practice, this would use the full ESM model
    auto seq_tensor = preprocess_sequence(sequence);
    
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(seq_tensor);
    
    auto output = esm_model_.forward(inputs);
    return output.toTensor();
}

torch::Tensor FoldEngine::process_ligands(const std::vector<std::string>& smiles) {
    // Use ligand encoder to process SMILES
    return ligand_encoder_->encode_batch(smiles, device_);
}

FoldingResult FoldEngine::postprocess_output(const torch::Tensor& coords, 
                                            const torch::Tensor& confidence,
                                            const ProteinInput& input) {
    FoldingResult result;
    
    // Move tensors to CPU for output
    result.coordinates = coords.to(torch::kCPU);
    result.confidence = confidence.to(torch::kCPU);
    
    // Generate all-atom coordinates (simplified)
    result.all_atom_coords = torch::zeros({coords.size(0), 37, 3});
    result.all_atom_coords.select(1, 1) = result.coordinates; // CA atoms
    
    // Calculate pLDDT scores (simplified)
    result.plddt = result.confidence * 100.0f;
    
    // Predict TM-score (simplified)
    result.tm_score_pred = result.confidence.mean().item<float>();
    
    // Set metadata
    result.sequence = input.sequence;
    result.sequence_length = input.sequence.length();
    
    return result;
}

void FoldEngine::start_timer() {
    start_time_ = std::chrono::high_resolution_clock::now();
    metrics_ = PerformanceMetrics(); // Reset metrics
}

void FoldEngine::record_time(const std::string& stage) {
    auto current_time = std::chrono::high_resolution_clock::now();
    auto elapsed_duration = std::chrono::duration_cast<std::chrono::microseconds>(
        current_time - start_time_).count() / 1000.0f;

    metrics_.layer_times[stage] = elapsed_duration;
    metrics_.total_time_ms = elapsed_duration;

    if (stage == "preprocessing") {
        metrics_.preprocessing_time_ms = elapsed_duration;
    } else if (stage == "inference") {
        metrics_.inference_time_ms = elapsed_duration;
    } else if (stage == "postprocessing") {
        metrics_.postprocessing_time_ms = elapsed_duration;
    }
}

void FoldEngine::update_memory_usage() {
    metrics_.memory_current_mb = utils::get_memory_usage_mb();
    if (device_.is_cuda()) {
        metrics_.gpu_memory_mb = utils::get_gpu_memory_usage_mb();
    }
}

void PerformanceMetrics::print_summary() const {
    std::cout << "\n=== Performance Metrics ===" << std::endl;
    std::cout << "Total time: " << total_time_ms << " ms" << std::endl;
    std::cout << "  Preprocessing: " << preprocessing_time_ms << " ms" << std::endl;
    std::cout << "  Inference: " << inference_time_ms << " ms" << std::endl;
    std::cout << "  Postprocessing: " << postprocessing_time_ms << " ms" << std::endl;
    std::cout << "Memory usage: " << memory_current_mb << " MB" << std::endl;
    if (gpu_memory_mb > 0) {
        std::cout << "GPU memory: " << gpu_memory_mb << " MB" << std::endl;
    }
    std::cout << "===========================" << std::endl;
}

void PerformanceMetrics::save_to_file(const std::string& filename) const {
    std::ofstream file(filename);
    if (file.is_open()) {
        file << "total_time_ms," << total_time_ms << std::endl;
        file << "preprocessing_time_ms," << preprocessing_time_ms << std::endl;
        file << "inference_time_ms," << inference_time_ms << std::endl;
        file << "postprocessing_time_ms," << postprocessing_time_ms << std::endl;
        file << "memory_current_mb," << memory_current_mb << std::endl;
        file << "gpu_memory_mb," << gpu_memory_mb << std::endl;
        
        for (const auto& [stage, time] : layer_times) {
            file << stage << "_time_ms," << time << std::endl;
        }
        
        file.close();
    }
}

} // namespace fold_engine
