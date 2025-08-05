/**
 * @file structure_module.h
 * @brief Structure prediction module for FoldEngine
 */

#pragma once

#include <torch/torch.h>
#include <memory>

namespace fold_engine {

/**
 * @brief Structure module for protein coordinate prediction
 */
class StructureModule {
public:
    /**
     * @brief Constructor
     * @param d_model Model dimension
     */
    explicit StructureModule(int d_model);
    
    /**
     * @brief Destructor
     */
    ~StructureModule();
    
    /**
     * @brief Predict protein structure coordinates
     * @param single_features Single representation [batch, seq_len, d_model]
     * @param pair_features Pair representation [batch, seq_len, seq_len, d_pair]
     * @return Predicted coordinates [batch, seq_len, 3]
     */
    torch::Tensor forward(const torch::Tensor& single_features,
                         const torch::Tensor& pair_features);
    
    /**
     * @brief Predict all-atom coordinates
     * @param backbone_coords Backbone coordinates [batch, seq_len, 3]
     * @param sequence_features Sequence features [batch, seq_len, d_model]
     * @return All-atom coordinates [batch, seq_len, 37, 3]
     */
    torch::Tensor predict_all_atom(const torch::Tensor& backbone_coords,
                                   const torch::Tensor& sequence_features);
    
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
 * @brief Invariant Point Attention for structure prediction
 */
class InvariantPointAttention {
public:
    /**
     * @brief Constructor
     * @param d_model Model dimension
     * @param num_heads Number of attention heads
     * @param num_qk_points Number of query/key points
     * @param num_v_points Number of value points
     */
    InvariantPointAttention(int d_model, int num_heads, 
                           int num_qk_points, int num_v_points);
    
    /**
     * @brief Destructor
     */
    ~InvariantPointAttention();
    
    /**
     * @brief Apply IPA
     * @param features Input features [batch, seq_len, d_model]
     * @param rigids Rigid transformations [batch, seq_len, 4, 4]
     * @return Updated features [batch, seq_len, d_model]
     */
    torch::Tensor forward(const torch::Tensor& features,
                         const torch::Tensor& rigids);
    
    /**
     * @brief Set device for computation
     * @param device Target device
     */
    void to_device(const torch::Device& device);

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace fold_engine
