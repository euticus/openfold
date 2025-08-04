#pragma once

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Forward declarations for triangle multiplication kernels
torch::Tensor triangle_multiply_forward(
    torch::Tensor input_tensor,
    torch::Tensor mask,
    bool outgoing = true
);

torch::Tensor triangle_multiply_backward(
    torch::Tensor grad_output,
    torch::Tensor input_tensor,
    torch::Tensor mask,
    torch::Tensor projections_a,
    torch::Tensor projections_b,
    bool outgoing = true
);

// CUDA kernel declarations
void launch_triangle_multiply_forward_kernel(
    const float* input,
    const float* mask,
    float* projections_a,
    float* projections_b,
    float* output,
    int batch_size,
    int seq_len_i,
    int seq_len_j,
    int channels,
    int hidden_dim,
    bool outgoing,
    cudaStream_t stream
);

void launch_triangle_multiply_backward_kernel(
    const float* grad_output,
    const float* input,
    const float* mask,
    const float* projections_a,
    const float* projections_b,
    float* grad_input,
    int batch_size,
    int seq_len_i,
    int seq_len_j,
    int channels,
    int hidden_dim,
    bool outgoing,
    cudaStream_t stream
);

// Utility functions
void launch_projection_kernel(
    const float* input,
    const float* mask,
    const float* linear_g_weights,
    const float* linear_p_weights,
    float* output,
    int batch_size,
    int seq_len_i,
    int seq_len_j,
    int input_channels,
    int output_channels,
    cudaStream_t stream
);
