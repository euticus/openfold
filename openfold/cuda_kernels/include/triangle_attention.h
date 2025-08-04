#pragma once

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Forward declarations for triangle attention kernels
torch::Tensor triangle_attention_forward(
    torch::Tensor query,
    torch::Tensor key, 
    torch::Tensor value,
    torch::Tensor bias_mask,
    torch::Tensor triangle_bias,
    bool starting_node = true
);

torch::Tensor triangle_attention_backward(
    torch::Tensor grad_output,
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor attention_weights,
    torch::Tensor bias_mask,
    torch::Tensor triangle_bias,
    bool starting_node = true
);

// CUDA kernel declarations
void launch_triangle_attention_forward_kernel(
    const float* query,
    const float* key,
    const float* value,
    const float* bias_mask,
    const float* triangle_bias,
    float* output,
    float* attention_weights,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim,
    bool starting_node,
    cudaStream_t stream
);

void launch_triangle_attention_backward_kernel(
    const float* grad_output,
    const float* query,
    const float* key,
    const float* value,
    const float* attention_weights,
    const float* bias_mask,
    const float* triangle_bias,
    float* grad_query,
    float* grad_key,
    float* grad_value,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim,
    bool starting_node,
    cudaStream_t stream
);
