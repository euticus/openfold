/**
 * @file mutation_predictor.cpp
 * @brief Implementation of mutation prediction modules
 */

#include "mutation_predictor.h"
#include <torch/torch.h>
#include <unordered_map>

namespace fold_engine {

// MutationPredictor implementation
struct MutationPredictor::Impl {
    int d_model;
    
    torch::nn::Embedding aa_embedding{nullptr};
    torch::nn::Linear mutation_proj{nullptr};
    torch::nn::Linear ddg_head{nullptr};
    torch::nn::Dropout dropout{nullptr};
    
    Impl(int d_model) : d_model(d_model) {
        aa_embedding = torch::nn::Embedding(21, d_model); // 20 AAs + unknown
        mutation_proj = torch::nn::Linear(d_model * 2, d_model);
        ddg_head = torch::nn::Linear(d_model, 1);
        dropout = torch::nn::Dropout(0.1);
    }
};

MutationPredictor::MutationPredictor(int d_model) 
    : pImpl(std::make_unique<Impl>(d_model)) {}

MutationPredictor::~MutationPredictor() = default;

torch::Tensor MutationPredictor::predict_effects(const std::string& sequence,
                                                const std::vector<std::pair<int, char>>& mutations,
                                                const torch::Device& device) {
    std::vector<float> ddg_predictions;
    
    for (const auto& [position, new_aa] : mutations) {
        if (position >= 0 && position < static_cast<int>(sequence.length())) {
            char wt_aa = sequence[position];
            
            // Encode amino acids
            int wt_idx = mutation_utils::aa_to_index(wt_aa);
            int mut_idx = mutation_utils::aa_to_index(new_aa);
            
            auto wt_tensor = torch::tensor({wt_idx}, torch::kLong).to(device);
            auto mut_tensor = torch::tensor({mut_idx}, torch::kLong).to(device);
            
            // Get embeddings
            auto wt_emb = pImpl->aa_embedding(wt_tensor);
            auto mut_emb = pImpl->aa_embedding(mut_tensor);
            
            // Combine embeddings
            auto combined = torch::cat({wt_emb, mut_emb}, -1);
            auto projected = pImpl->mutation_proj(combined);
            projected = pImpl->dropout(projected);
            
            // Predict ΔΔG
            auto ddg = pImpl->ddg_head(projected);
            ddg_predictions.push_back(ddg.item<float>());
        } else {
            ddg_predictions.push_back(0.0f); // Invalid position
        }
    }
    
    return torch::tensor(ddg_predictions, torch::TensorOptions().device(device));
}

torch::Tensor MutationPredictor::predict_stability(const std::string& sequence,
                                                  const std::vector<std::pair<int, char>>& mutations,
                                                  const torch::Device& device) {
    // For now, just return negative ΔΔG as stability
    auto ddg = predict_effects(sequence, mutations, device);
    return -ddg;
}

void MutationPredictor::to_device(const torch::Device& device) {
    pImpl->aa_embedding->to(device);
    pImpl->mutation_proj->to(device);
    pImpl->ddg_head->to(device);
}

// Mutation utilities implementation
namespace mutation_utils {

static const std::unordered_map<char, int> AA_TO_IDX = {
    {'A', 0}, {'R', 1}, {'N', 2}, {'D', 3}, {'C', 4}, {'Q', 5}, {'E', 6}, {'G', 7},
    {'H', 8}, {'I', 9}, {'L', 10}, {'K', 11}, {'M', 12}, {'F', 13}, {'P', 14},
    {'S', 15}, {'T', 16}, {'W', 17}, {'Y', 18}, {'V', 19}, {'X', 20}
};

static const std::vector<char> IDX_TO_AA = {
    'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
    'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'X'
};

int aa_to_index(char aa) {
    char upper_aa = std::toupper(aa);
    auto it = AA_TO_IDX.find(upper_aa);
    return (it != AA_TO_IDX.end()) ? it->second : 20; // 20 for unknown
}

char index_to_aa(int index) {
    if (index >= 0 && index < static_cast<int>(IDX_TO_AA.size())) {
        return IDX_TO_AA[index];
    }
    return 'X';
}

std::string apply_mutation(const std::string& sequence, int position, char new_aa) {
    if (position < 0 || position >= static_cast<int>(sequence.length())) {
        return sequence; // Invalid position
    }
    
    std::string mutated = sequence;
    mutated[position] = std::toupper(new_aa);
    return mutated;
}

torch::Tensor calculate_property_difference(char wt_aa, char mut_aa) {
    // Simplified amino acid properties (hydrophobicity, charge, size, etc.)
    static const std::unordered_map<char, std::vector<float>> AA_PROPERTIES = {
        {'A', {1.8f, 0.0f, 1.0f}},   // Hydrophobicity, charge, size
        {'R', {-4.5f, 1.0f, 3.0f}},
        {'N', {-3.5f, 0.0f, 2.0f}},
        {'D', {-3.5f, -1.0f, 2.0f}},
        {'C', {2.5f, 0.0f, 1.5f}},
        {'Q', {-3.5f, 0.0f, 2.5f}},
        {'E', {-3.5f, -1.0f, 2.5f}},
        {'G', {-0.4f, 0.0f, 0.5f}},
        {'H', {-3.2f, 0.5f, 2.5f}},
        {'I', {4.5f, 0.0f, 2.0f}},
        {'L', {3.8f, 0.0f, 2.0f}},
        {'K', {-3.9f, 1.0f, 2.5f}},
        {'M', {1.9f, 0.0f, 2.5f}},
        {'F', {2.8f, 0.0f, 3.0f}},
        {'P', {-1.6f, 0.0f, 1.5f}},
        {'S', {-0.8f, 0.0f, 1.0f}},
        {'T', {-0.7f, 0.0f, 1.5f}},
        {'W', {-0.9f, 0.0f, 4.0f}},
        {'Y', {-1.3f, 0.0f, 3.5f}},
        {'V', {4.2f, 0.0f, 1.5f}}
    };
    
    auto wt_it = AA_PROPERTIES.find(std::toupper(wt_aa));
    auto mut_it = AA_PROPERTIES.find(std::toupper(mut_aa));
    
    std::vector<float> wt_props = (wt_it != AA_PROPERTIES.end()) ? 
        wt_it->second : std::vector<float>{0.0f, 0.0f, 0.0f};
    std::vector<float> mut_props = (mut_it != AA_PROPERTIES.end()) ? 
        mut_it->second : std::vector<float>{0.0f, 0.0f, 0.0f};
    
    std::vector<float> diff;
    for (size_t i = 0; i < wt_props.size(); ++i) {
        diff.push_back(mut_props[i] - wt_props[i]);
    }
    
    return torch::tensor(diff);
}

} // namespace mutation_utils

} // namespace fold_engine
