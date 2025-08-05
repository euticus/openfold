/**
 * @file odinfold_wasm.cpp
 * @brief WebAssembly interface for OdinFold++ browser deployment
 * 
 * Provides a lightweight C++ interface for protein folding in web browsers.
 * Optimized for sequences up to 200 residues with minimal memory footprint.
 */

#include <emscripten.h>
#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include <chrono>

using namespace emscripten;

/**
 * @brief Lightweight protein folding engine for WASM
 */
class OdinFoldWASM {
public:
    /**
     * @brief Constructor
     */
    OdinFoldWASM() : initialized_(false), max_seq_len_(200) {
        // Initialize amino acid mapping
        initializeAminoAcidMapping();
    }
    
    /**
     * @brief Initialize the folding engine
     * @return true if successful
     */
    bool initialize() {
        if (initialized_) return true;
        
        try {
            // Initialize mock model (replace with actual ONNX runtime)
            initializeMockModel();
            
            initialized_ = true;
            return true;
            
        } catch (const std::exception& e) {
            return false;
        }
    }
    
    /**
     * @brief Check if engine is initialized
     */
    bool isInitialized() const {
        return initialized_;
    }
    
    /**
     * @brief Get maximum supported sequence length
     */
    int getMaxSequenceLength() const {
        return max_seq_len_;
    }
    
    /**
     * @brief Validate protein sequence
     * @param sequence Protein sequence string
     * @return true if valid
     */
    bool validateSequence(const std::string& sequence) const {
        if (sequence.empty() || sequence.length() > max_seq_len_) {
            return false;
        }
        
        for (char c : sequence) {
            if (aa_to_idx_.find(std::toupper(c)) == aa_to_idx_.end()) {
                return false;
            }
        }
        
        return true;
    }
    
    /**
     * @brief Fold protein sequence
     * @param sequence Protein sequence
     * @param progress_callback Optional progress callback
     * @return Folding result as JavaScript object
     */
    val foldProtein(const std::string& sequence, val progress_callback = val::null()) {
        if (!initialized_) {
            return createErrorResult("Engine not initialized");
        }
        
        if (!validateSequence(sequence)) {
            return createErrorResult("Invalid sequence");
        }
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        try {
            // Convert sequence to indices
            std::vector<int> seq_indices = sequenceToIndices(sequence);
            
            // Report progress
            if (!progress_callback.isNull()) {
                progress_callback(10);
            }
            
            // Mock folding computation (replace with actual inference)
            auto coordinates = mockFoldingComputation(seq_indices, progress_callback);
            
            // Generate confidence scores
            auto confidence = generateConfidenceScores(seq_indices);
            
            // Report completion
            if (!progress_callback.isNull()) {
                progress_callback(100);
            }
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                end_time - start_time).count();
            
            // Create result object
            val result = val::object();
            result.set("success", true);
            result.set("sequence", sequence);
            result.set("sequence_length", static_cast<int>(sequence.length()));
            result.set("coordinates", vectorToJSArray(coordinates));
            result.set("confidence", vectorToJSArray(confidence));
            result.set("inference_time_ms", duration);
            result.set("memory_usage_mb", getMemoryUsageMB());
            
            return result;
            
        } catch (const std::exception& e) {
            return createErrorResult(std::string("Folding failed: ") + e.what());
        }
    }
    
    /**
     * @brief Generate PDB format output
     * @param coordinates Protein coordinates
     * @param sequence Protein sequence
     * @param confidence Confidence scores
     * @return PDB format string
     */
    std::string generatePDB(const std::vector<std::vector<float>>& coordinates,
                           const std::string& sequence,
                           const std::vector<float>& confidence) const {
        
        std::string pdb;
        pdb.reserve(sequence.length() * 100); // Estimate size
        
        // PDB header
        pdb += "HEADER    PROTEIN STRUCTURE PREDICTION             01-JAN-24   WASM\n";
        pdb += "TITLE     STRUCTURE PREDICTED BY ODINFOLD++ WASM\n";
        pdb += "REMARK   1 PREDICTION METHOD: ODINFOLD++ WEBASSEMBLY\n";
        
        float avg_confidence = 0.0f;
        for (float conf : confidence) avg_confidence += conf;
        avg_confidence /= confidence.size();
        
        char remark_line[100];
        snprintf(remark_line, sizeof(remark_line), 
                "REMARK   1 AVERAGE CONFIDENCE: %.3f\n", avg_confidence);
        pdb += remark_line;
        
        // Atom records
        for (size_t i = 0; i < sequence.length() && i < coordinates.size(); ++i) {
            char aa = sequence[i];
            const auto& coord = coordinates[i];
            float b_factor = confidence[i] * 100.0f;
            
            char atom_line[100];
            snprintf(atom_line, sizeof(atom_line),
                    "ATOM  %5zu  CA  %c A%4zu    %8.3f%8.3f%8.3f%6.2f%6.2f           C\n",
                    i + 1, aa, i + 1, coord[0], coord[1], coord[2], 1.00f, b_factor);
            pdb += atom_line;
        }
        
        pdb += "END\n";
        return pdb;
    }
    
    /**
     * @brief Get memory usage in MB
     */
    float getMemoryUsageMB() const {
        // Mock memory usage calculation
        return 128.0f; // Placeholder
    }
    
    /**
     * @brief Get model information
     */
    val getModelInfo() const {
        val info = val::object();
        info.set("name", "OdinFold++ WASM");
        info.set("version", "1.0.0");
        info.set("max_sequence_length", max_seq_len_);
        info.set("model_size_mb", 45.0); // Placeholder
        info.set("initialized", initialized_);
        
        val features = val::array();
        features.call<void>("push", "single_chain_folding");
        features.call<void>("push", "confidence_scoring");
        features.call<void>("push", "pdb_output");
        info.set("supported_features", features);
        
        return info;
    }

private:
    bool initialized_;
    int max_seq_len_;
    std::unordered_map<char, int> aa_to_idx_;
    std::vector<char> idx_to_aa_;
    
    void initializeAminoAcidMapping() {
        const std::string amino_acids = "ACDEFGHIKLMNPQRSTVWY";
        
        for (size_t i = 0; i < amino_acids.length(); ++i) {
            char aa = amino_acids[i];
            aa_to_idx_[aa] = static_cast<int>(i);
            idx_to_aa_.push_back(aa);
        }
        
        // Add unknown amino acid
        aa_to_idx_['X'] = 20;
        idx_to_aa_.push_back('X');
    }
    
    void initializeMockModel() {
        // Mock model initialization
        // In real implementation, this would load ONNX model
    }
    
    std::vector<int> sequenceToIndices(const std::string& sequence) const {
        std::vector<int> indices;
        indices.reserve(sequence.length());
        
        for (char c : sequence) {
            char upper_c = std::toupper(c);
            auto it = aa_to_idx_.find(upper_c);
            if (it != aa_to_idx_.end()) {
                indices.push_back(it->second);
            } else {
                indices.push_back(20); // Unknown
            }
        }
        
        return indices;
    }
    
    std::vector<std::vector<float>> mockFoldingComputation(
        const std::vector<int>& seq_indices,
        val progress_callback) {
        
        std::vector<std::vector<float>> coordinates;
        coordinates.reserve(seq_indices.size());
        
        // Mock folding with realistic-looking coordinates
        float x = 0.0f, y = 0.0f, z = 0.0f;
        
        for (size_t i = 0; i < seq_indices.size(); ++i) {
            // Simulate computation time
            if (i % 10 == 0) {
                // Small delay to simulate computation
                auto start = std::chrono::high_resolution_clock::now();
                while (std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::high_resolution_clock::now() - start).count() < 1000) {
                    // Busy wait for 1ms
                }
                
                // Report progress
                if (!progress_callback.isNull()) {
                    int progress = 10 + static_cast<int>((i * 80) / seq_indices.size());
                    progress_callback(progress);
                }
            }
            
            // Generate mock coordinates with some structure
            float angle = static_cast<float>(i) * 0.1f;
            x += 3.8f * std::cos(angle) + (rand() % 100 - 50) * 0.01f;
            y += 3.8f * std::sin(angle) + (rand() % 100 - 50) * 0.01f;
            z += (rand() % 100 - 50) * 0.02f;
            
            coordinates.push_back({x, y, z});
        }
        
        return coordinates;
    }
    
    std::vector<float> generateConfidenceScores(const std::vector<int>& seq_indices) const {
        std::vector<float> confidence;
        confidence.reserve(seq_indices.size());
        
        // Mock confidence scores with some variation
        for (size_t i = 0; i < seq_indices.size(); ++i) {
            // Higher confidence in the middle, lower at ends
            float pos_factor = 1.0f - std::abs(static_cast<float>(i) - seq_indices.size() / 2.0f) / (seq_indices.size() / 2.0f);
            float base_confidence = 0.7f + 0.2f * pos_factor;
            
            // Add some noise
            float noise = (rand() % 100 - 50) * 0.002f;
            float conf = std::max(0.0f, std::min(1.0f, base_confidence + noise));
            
            confidence.push_back(conf);
        }
        
        return confidence;
    }
    
    val createErrorResult(const std::string& error_message) const {
        val result = val::object();
        result.set("success", false);
        result.set("error", error_message);
        return result;
    }
    
    val vectorToJSArray(const std::vector<std::vector<float>>& vec) const {
        val array = val::array();
        for (const auto& inner_vec : vec) {
            val inner_array = val::array();
            for (float value : inner_vec) {
                inner_array.call<void>("push", value);
            }
            array.call<void>("push", inner_array);
        }
        return array;
    }
    
    val vectorToJSArray(const std::vector<float>& vec) const {
        val array = val::array();
        for (float value : vec) {
            array.call<void>("push", value);
        }
        return array;
    }
};

// Emscripten bindings
EMSCRIPTEN_BINDINGS(odinfold_wasm) {
    class_<OdinFoldWASM>("OdinFoldWASM")
        .constructor<>()
        .function("initialize", &OdinFoldWASM::initialize)
        .function("isInitialized", &OdinFoldWASM::isInitialized)
        .function("getMaxSequenceLength", &OdinFoldWASM::getMaxSequenceLength)
        .function("validateSequence", &OdinFoldWASM::validateSequence)
        .function("foldProtein", &OdinFoldWASM::foldProtein)
        .function("generatePDB", &OdinFoldWASM::generatePDB)
        .function("getMemoryUsageMB", &OdinFoldWASM::getMemoryUsageMB)
        .function("getModelInfo", &OdinFoldWASM::getModelInfo);
}

// C-style API for direct JavaScript access
extern "C" {
    
    EMSCRIPTEN_KEEPALIVE
    OdinFoldWASM* create_folder() {
        return new OdinFoldWASM();
    }
    
    EMSCRIPTEN_KEEPALIVE
    void destroy_folder(OdinFoldWASM* folder) {
        delete folder;
    }
    
    EMSCRIPTEN_KEEPALIVE
    int initialize_folder(OdinFoldWASM* folder) {
        return folder->initialize() ? 1 : 0;
    }
    
    EMSCRIPTEN_KEEPALIVE
    int validate_sequence(OdinFoldWASM* folder, const char* sequence) {
        return folder->validateSequence(std::string(sequence)) ? 1 : 0;
    }
    
    EMSCRIPTEN_KEEPALIVE
    int get_max_sequence_length(OdinFoldWASM* folder) {
        return folder->getMaxSequenceLength();
    }
    
    EMSCRIPTEN_KEEPALIVE
    float get_memory_usage_mb(OdinFoldWASM* folder) {
        return folder->getMemoryUsageMB();
    }
}

// Module initialization
extern "C" {
    EMSCRIPTEN_KEEPALIVE
    void wasm_module_ready() {
        // Called when WASM module is ready
        EM_ASM({
            if (typeof Module !== 'undefined' && Module.onRuntimeInitialized) {
                Module.onRuntimeInitialized();
            }
        });
    }
}
