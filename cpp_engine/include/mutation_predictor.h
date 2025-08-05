/**
 * @file mutation_predictor.h
 * @brief Mutation effect prediction module for FoldEngine
 */

#pragma once

#include <torch/torch.h>
#include <string>
#include <vector>
#include <memory>

namespace fold_engine {

/**
 * @brief Mutation effect predictor
 */
class MutationPredictor {
public:
    /**
     * @brief Constructor
     * @param d_model Model dimension
     */
    explicit MutationPredictor(int d_model);
    
    /**
     * @brief Destructor
     */
    ~MutationPredictor();
    
    /**
     * @brief Predict mutation effects (ΔΔG)
     * @param sequence Wild-type protein sequence
     * @param mutations List of mutations (position, new_aa)
     * @param device Target device
     * @return ΔΔG predictions [num_mutations]
     */
    torch::Tensor predict_effects(const std::string& sequence,
                                 const std::vector<std::pair<int, char>>& mutations,
                                 const torch::Device& device);
    
    /**
     * @brief Predict stability changes
     * @param sequence Protein sequence
     * @param mutations List of mutations
     * @param device Target device
     * @return Stability predictions [num_mutations]
     */
    torch::Tensor predict_stability(const std::string& sequence,
                                   const std::vector<std::pair<int, char>>& mutations,
                                   const torch::Device& device);
    
    /**
     * @brief Set device for computation
     * @param device Target device
     */
    void to_device(const torch::Device& device);

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};

/**
 * @brief Mutation utilities
 */
namespace mutation_utils {
    
    /**
     * @brief Convert amino acid character to index
     * @param aa Amino acid character
     * @return Amino acid index
     */
    int aa_to_index(char aa);
    
    /**
     * @brief Convert amino acid index to character
     * @param index Amino acid index
     * @return Amino acid character
     */
    char index_to_aa(int index);
    
    /**
     * @brief Apply mutation to sequence
     * @param sequence Original sequence
     * @param position Mutation position (0-based)
     * @param new_aa New amino acid
     * @return Mutated sequence
     */
    std::string apply_mutation(const std::string& sequence, 
                              int position, 
                              char new_aa);
    
    /**
     * @brief Calculate amino acid properties difference
     * @param wt_aa Wild-type amino acid
     * @param mut_aa Mutant amino acid
     * @return Property difference vector
     */
    torch::Tensor calculate_property_difference(char wt_aa, char mut_aa);
}

} // namespace fold_engine
