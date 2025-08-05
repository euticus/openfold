/**
 * @file attention.h
 * @brief Attention module implementations for FoldEngine
 */

#pragma once

#include <torch/torch.h>
#include <memory>

namespace fold_engine {

/**
 * @brief Multi-head attention module optimized for protein sequences
 */
class AttentionModule {
public:
    /**
     * @brief Constructor
     * @param d_model Model dimension
     * @param num_heads Number of attention heads
     */
    AttentionModule(int d_model, int num_heads);
    
    /**
     * @brief Destructor
     */
    ~AttentionModule();
    
    /**
     * @brief Apply attention to input features
     * @param query Query tensor [batch, seq_len, d_model]
     * @param key Key tensor [batch, seq_len, d_model]
     * @param value Value tensor [batch, seq_len, d_model]
     * @param mask Attention mask [batch, seq_len, seq_len] (optional)
     * @return Attention output [batch, seq_len, d_model]
     */
    torch::Tensor forward(const torch::Tensor& query,
                         const torch::Tensor& key,
                         const torch::Tensor& value,
                         const torch::Tensor& mask = torch::Tensor());
    
    /**
     * @brief Apply self-attention
     * @param input Input tensor [batch, seq_len, d_model]
     * @param mask Attention mask [batch, seq_len, seq_len] (optional)
     * @return Self-attention output [batch, seq_len, d_model]
     */
    torch::Tensor self_attention(const torch::Tensor& input,
                                const torch::Tensor& mask = torch::Tensor());
    
    /**
     * @brief Apply cross-attention between protein and ligand features
     * @param protein_features Protein features [batch, seq_len, d_model]
     * @param ligand_features Ligand features [batch, num_atoms, d_model]
     * @return Cross-attention output [batch, seq_len, d_model]
     */
    torch::Tensor cross_attention(const torch::Tensor& protein_features,
                                 const torch::Tensor& ligand_features);
    
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
 * @brief Triangle attention for structure prediction
 */
class TriangleAttention {
public:
    /**
     * @brief Constructor
     * @param d_model Model dimension
     * @param num_heads Number of attention heads
     */
    TriangleAttention(int d_model, int num_heads);
    
    /**
     * @brief Destructor
     */
    ~TriangleAttention();
    
    /**
     * @brief Apply triangle attention
     * @param pair_features Pair features [batch, seq_len, seq_len, d_model]
     * @return Updated pair features [batch, seq_len, seq_len, d_model]
     */
    torch::Tensor forward(const torch::Tensor& pair_features);
    
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
 * @brief Flash attention implementation for memory efficiency
 */
class FlashAttention {
public:
    /**
     * @brief Apply flash attention
     * @param query Query tensor
     * @param key Key tensor
     * @param value Value tensor
     * @param mask Attention mask (optional)
     * @return Attention output
     */
    static torch::Tensor forward(const torch::Tensor& query,
                                const torch::Tensor& key,
                                const torch::Tensor& value,
                                const torch::Tensor& mask = torch::Tensor());
    
    /**
     * @brief Check if flash attention is available
     * @return True if available
     */
    static bool is_available();
};

} // namespace fold_engine
