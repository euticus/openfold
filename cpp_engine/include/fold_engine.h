/**
 * @file fold_engine.h
 * @brief Main header for the FoldEngine C++ inference system
 * 
 * High-performance C++ implementation of OdinFold for production deployment.
 * Provides fast protein structure prediction with minimal dependencies.
 */

#pragma once

#include <torch/torch.h>
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <chrono>

namespace fold_engine {

// Forward declarations
class AttentionModule;
class StructureModule;
class LigandEncoder;
class MutationPredictor;

/**
 * @brief Configuration for FoldEngine inference
 */
struct FoldConfig {
    // Model parameters
    int seq_len_max = 2048;
    int d_model = 256;
    int num_heads = 8;
    int num_layers = 24;
    
    // Performance settings
    bool use_cuda = true;
    bool use_mixed_precision = true;
    bool use_flash_attention = true;
    int batch_size = 1;
    
    // Memory optimization
    bool use_gradient_checkpointing = false;
    bool use_quantization = false;
    
    // Output settings
    bool output_confidence = true;
    bool output_plddt = true;
    bool output_binding_pockets = false;
    
    // Paths
    std::string model_path = "models/odinfold.pt";
    std::string esm_path = "models/esm2_650m.pt";
    
    FoldConfig() = default;
};

/**
 * @brief Input data for protein folding
 */
struct ProteinInput {
    std::string sequence;
    std::vector<std::string> ligand_smiles;
    std::vector<std::pair<int, char>> mutations;  // (position, new_aa)
    
    // Optional precomputed features
    torch::Tensor esm_embeddings;
    torch::Tensor msa_features;
    
    ProteinInput() = default;
    explicit ProteinInput(const std::string& seq) : sequence(seq) {}
};

/**
 * @brief Output from protein folding
 */
struct FoldingResult {
    // Structure prediction
    torch::Tensor coordinates;      // [seq_len, 3] - CA coordinates
    torch::Tensor all_atom_coords;  // [seq_len, 37, 3] - All atom coordinates
    
    // Confidence scores
    torch::Tensor confidence;       // [seq_len] - Per-residue confidence
    torch::Tensor plddt;           // [seq_len] - pLDDT scores
    float tm_score_pred = 0.0f;    // Predicted TM-score
    
    // Binding analysis (if ligands provided)
    torch::Tensor binding_pockets;  // [seq_len] - Binding pocket scores
    torch::Tensor ligand_interactions; // [seq_len, num_ligands] - Interaction scores
    
    // Metadata
    std::string sequence;
    int sequence_length = 0;
    float inference_time_ms = 0.0f;
    float memory_usage_mb = 0.0f;
    
    FoldingResult() = default;
};

/**
 * @brief Performance metrics for benchmarking
 */
struct PerformanceMetrics {
    float total_time_ms = 0.0f;
    float preprocessing_time_ms = 0.0f;
    float inference_time_ms = 0.0f;
    float postprocessing_time_ms = 0.0f;
    
    float memory_peak_mb = 0.0f;
    float memory_current_mb = 0.0f;
    
    int gpu_utilization_percent = 0;
    float gpu_memory_mb = 0.0f;
    
    std::unordered_map<std::string, float> layer_times;
    
    void print_summary() const;
    void save_to_file(const std::string& filename) const;
};

/**
 * @brief Main FoldEngine class for protein structure prediction
 */
class FoldEngine {
public:
    /**
     * @brief Constructor
     * @param config Configuration for the engine
     */
    explicit FoldEngine(const FoldConfig& config = FoldConfig());
    
    /**
     * @brief Destructor
     */
    ~FoldEngine();
    
    /**
     * @brief Initialize the engine and load models
     * @return true if successful, false otherwise
     */
    bool initialize();
    
    /**
     * @brief Check if the engine is ready for inference
     * @return true if ready, false otherwise
     */
    bool is_ready() const { return initialized_; }
    
    /**
     * @brief Fold a single protein
     * @param input Protein input data
     * @return Folding result
     */
    FoldingResult fold_protein(const ProteinInput& input);
    
    /**
     * @brief Fold multiple proteins in batch
     * @param inputs Vector of protein inputs
     * @return Vector of folding results
     */
    std::vector<FoldingResult> fold_batch(const std::vector<ProteinInput>& inputs);
    
    /**
     * @brief Predict mutation effects
     * @param sequence Wild-type sequence
     * @param mutations List of mutations to analyze
     * @return Mutation effect predictions
     */
    torch::Tensor predict_mutations(const std::string& sequence, 
                                   const std::vector<std::pair<int, char>>& mutations);
    
    /**
     * @brief Get performance metrics from last inference
     * @return Performance metrics
     */
    const PerformanceMetrics& get_metrics() const { return metrics_; }
    
    /**
     * @brief Set device for computation
     * @param device_str Device string (e.g., "cuda:0", "cpu")
     */
    void set_device(const std::string& device_str);
    
    /**
     * @brief Get current device
     * @return Device string
     */
    std::string get_device() const;
    
    /**
     * @brief Enable/disable profiling
     * @param enable Whether to enable profiling
     */
    void set_profiling(bool enable) { profiling_enabled_ = enable; }
    
    /**
     * @brief Get model information
     * @return Model info as string
     */
    std::string get_model_info() const;
    
    /**
     * @brief Validate input sequence
     * @param sequence Protein sequence
     * @return true if valid, false otherwise
     */
    static bool validate_sequence(const std::string& sequence);
    
    /**
     * @brief Convert amino acid sequence to indices
     * @param sequence Protein sequence
     * @return Tensor of amino acid indices
     */
    static torch::Tensor sequence_to_indices(const std::string& sequence);
    
    /**
     * @brief Convert indices back to sequence
     * @param indices Tensor of amino acid indices
     * @return Protein sequence string
     */
    static std::string indices_to_sequence(const torch::Tensor& indices);

private:
    // Configuration
    FoldConfig config_;
    
    // Model components
    std::unique_ptr<AttentionModule> attention_module_;
    std::unique_ptr<StructureModule> structure_module_;
    std::unique_ptr<LigandEncoder> ligand_encoder_;
    std::unique_ptr<MutationPredictor> mutation_predictor_;
    
    // PyTorch modules
    torch::jit::script::Module model_;
    torch::jit::script::Module esm_model_;
    
    // State
    bool initialized_ = false;
    torch::Device device_ = torch::kCPU;
    bool profiling_enabled_ = false;
    
    // Performance tracking
    mutable PerformanceMetrics metrics_;
    std::chrono::high_resolution_clock::time_point start_time_;
    
    // Private methods
    bool load_models();
    torch::Tensor preprocess_sequence(const std::string& sequence);
    torch::Tensor extract_esm_features(const std::string& sequence);
    torch::Tensor process_ligands(const std::vector<std::string>& smiles);
    FoldingResult postprocess_output(const torch::Tensor& coords, 
                                   const torch::Tensor& confidence,
                                   const ProteinInput& input);
    
    void start_timer();
    void record_time(const std::string& stage);
    void update_memory_usage();
    
    // Disable copy constructor and assignment
    FoldEngine(const FoldEngine&) = delete;
    FoldEngine& operator=(const FoldEngine&) = delete;
};

/**
 * @brief Utility functions
 */
namespace utils {
    /**
     * @brief Load protein sequence from FASTA file
     * @param filename Path to FASTA file
     * @return Protein sequence
     */
    std::string load_fasta(const std::string& filename);
    
    /**
     * @brief Save structure to PDB file
     * @param result Folding result
     * @param filename Output PDB filename
     * @param include_confidence Whether to include confidence in B-factors
     */
    void save_pdb(const FoldingResult& result, const std::string& filename, 
                  bool include_confidence = true);
    
    /**
     * @brief Calculate RMSD between two structures
     * @param coords1 First structure coordinates
     * @param coords2 Second structure coordinates
     * @return RMSD value
     */
    float calculate_rmsd(const torch::Tensor& coords1, const torch::Tensor& coords2);
    
    /**
     * @brief Calculate TM-score between two structures
     * @param coords1 First structure coordinates
     * @param coords2 Second structure coordinates
     * @return TM-score value
     */
    float calculate_tm_score(const torch::Tensor& coords1, const torch::Tensor& coords2);
    
    /**
     * @brief Get system memory usage
     * @return Memory usage in MB
     */
    float get_memory_usage_mb();
    
    /**
     * @brief Get GPU memory usage
     * @return GPU memory usage in MB
     */
    float get_gpu_memory_usage_mb();
}

/**
 * @brief Exception classes for FoldEngine
 */
class FoldEngineException : public std::exception {
public:
    explicit FoldEngineException(const std::string& message) : message_(message) {}
    const char* what() const noexcept override { return message_.c_str(); }
private:
    std::string message_;
};

class ModelLoadException : public FoldEngineException {
public:
    explicit ModelLoadException(const std::string& message)
        : FoldEngineException("Model load error: " + message) {}
};

class InferenceException : public FoldEngineException {
public:
    explicit InferenceException(const std::string& message)
        : FoldEngineException("Inference error: " + message) {}
};

class ValidationException : public FoldEngineException {
public:
    explicit ValidationException(const std::string& message)
        : FoldEngineException("Validation error: " + message) {}
};

} // namespace fold_engine
