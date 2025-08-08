#include "triangle_multiply.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>
#include <algorithm>
#include <c10/cuda/CUDAStream.h>

// CUDA kernel constants
#define BLOCK_SIZE 16
#define WARP_SIZE 32
#define TILE_SIZE 16

// Triangle multiplication forward kernel
__global__ void triangle_multiply_forward_kernel(
    const float* __restrict__ input,      // [B, I, J, C]
    const float* __restrict__ mask,       // [B, I, J, 1]
    const float* __restrict__ proj_a,     // [B, I, J, H] - projection A
    const float* __restrict__ proj_b,     // [B, I, J, H] - projection B
    float* __restrict__ output,           // [B, I, J, C]
    int batch_size,
    int seq_len_i,
    int seq_len_j, 
    int channels,
    int hidden_dim,
    bool outgoing
) {
    // Block and thread indices
    int batch_idx = blockIdx.x;
    int i_idx = blockIdx.y;
    int j_idx = blockIdx.z;
    
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    
    // Bounds checking
    if (batch_idx >= batch_size || i_idx >= seq_len_i || j_idx >= seq_len_j) {
        return;
    }
    
    // Shared memory for tile-based computation
    __shared__ float shared_a[TILE_SIZE][TILE_SIZE];
    __shared__ float shared_b[TILE_SIZE][TILE_SIZE];
    
    // Calculate tensor strides
    int input_stride_batch = seq_len_i * seq_len_j * channels;
    int input_stride_i = seq_len_j * channels;
    int input_stride_j = channels;
    
    int proj_stride_batch = seq_len_i * seq_len_j * hidden_dim;
    int proj_stride_i = seq_len_j * hidden_dim;
    int proj_stride_j = hidden_dim;
    
    // Base pointers for current batch
    const float* input_base = input + batch_idx * input_stride_batch;
    const float* mask_base = mask + batch_idx * seq_len_i * seq_len_j;
    const float* proj_a_base = proj_a + batch_idx * proj_stride_batch;
    const float* proj_b_base = proj_b + batch_idx * proj_stride_batch;
    float* output_base = output + batch_idx * input_stride_batch;
    
    // Get mask value for current position
    float mask_val = mask_base[i_idx * seq_len_j + j_idx];
    
    // Process each channel
    for (int c = tid_x; c < channels; c += blockDim.x) {
        float result = 0.0f;
        
        // Triangle multiplication computation
        if (outgoing) {
            // Outgoing edges: sum over k dimension
            for (int k = 0; k < seq_len_j; k++) {
                // Load projections for current positions
                float a_ik = proj_a_base[i_idx * proj_stride_i + k * proj_stride_j + (c % hidden_dim)];
                float b_kj = proj_b_base[k * proj_stride_i + j_idx * proj_stride_j + (c % hidden_dim)];
                
                // Apply gating (simplified)
                float gate_ik = tanhf(a_ik);
                float gate_kj = tanhf(b_kj);
                
                // Accumulate contribution
                result += gate_ik * gate_kj * mask_base[i_idx * seq_len_j + k] * mask_base[k * seq_len_j + j_idx];
            }
        } else {
            // Incoming edges: sum over k dimension (different indexing)
            for (int k = 0; k < seq_len_i; k++) {
                // Load projections for current positions
                float a_ki = proj_a_base[k * proj_stride_i + i_idx * proj_stride_j + (c % hidden_dim)];
                float b_kj = proj_b_base[k * proj_stride_i + j_idx * proj_stride_j + (c % hidden_dim)];
                
                // Apply gating
                float gate_ki = tanhf(a_ki);
                float gate_kj = tanhf(b_kj);
                
                // Accumulate contribution
                result += gate_ki * gate_kj * mask_base[k * seq_len_j + i_idx] * mask_base[k * seq_len_j + j_idx];
            }
        }
        
        // Apply final gating and mask
        result = tanhf(result) * mask_val;
        
        // Add to input (residual connection)
        float input_val = input_base[i_idx * input_stride_i + j_idx * input_stride_j + c];
        output_base[i_idx * input_stride_i + j_idx * input_stride_j + c] = input_val + result;
    }
}

// Optimized projection kernel
__global__ void projection_kernel(
    const float* __restrict__ input,      // [B, I, J, C_in]
    const float* __restrict__ mask,       // [B, I, J, 1]
    const float* __restrict__ weights,    // [C_in, C_out]
    const float* __restrict__ bias,       // [C_out]
    float* __restrict__ output,           // [B, I, J, C_out]
    int batch_size,
    int seq_len_i,
    int seq_len_j,
    int input_channels,
    int output_channels
) {
    // Block and thread indices
    int batch_idx = blockIdx.x;
    int i_idx = blockIdx.y;
    int j_idx = blockIdx.z;
    
    int tid = threadIdx.x;
    
    // Bounds checking
    if (batch_idx >= batch_size || i_idx >= seq_len_i || j_idx >= seq_len_j) {
        return;
    }
    
    // Calculate strides
    int input_stride_batch = seq_len_i * seq_len_j * input_channels;
    int input_stride_i = seq_len_j * input_channels;
    int input_stride_j = input_channels;
    
    int output_stride_batch = seq_len_i * seq_len_j * output_channels;
    int output_stride_i = seq_len_j * output_channels;
    int output_stride_j = output_channels;
    
    // Base pointers
    const float* input_base = input + batch_idx * input_stride_batch + i_idx * input_stride_i + j_idx * input_stride_j;
    float* output_base = output + batch_idx * output_stride_batch + i_idx * output_stride_i + j_idx * output_stride_j;
    float mask_val = mask[batch_idx * seq_len_i * seq_len_j + i_idx * seq_len_j + j_idx];
    
    // Compute linear projection
    for (int out_c = tid; out_c < output_channels; out_c += blockDim.x) {
        float result = bias[out_c];
        
        // Matrix multiplication
        for (int in_c = 0; in_c < input_channels; in_c++) {
            result += input_base[in_c] * weights[in_c * output_channels + out_c];
        }
        
        // Apply mask and store
        output_base[out_c] = result * mask_val;
    }
}

// Triangle multiply backward kernel
__global__ void triangle_multiply_backward_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ input,
    const float* __restrict__ mask,
    const float* __restrict__ projections_a,
    const float* __restrict__ projections_b,
    float* __restrict__ grad_input,
    int batch_size,
    int seq_len_i,
    int seq_len_j,
    int channels,
    int hidden_dim,
    bool outgoing
) {
    // Block and thread indices
    int batch_idx = blockIdx.x;
    int i_idx = blockIdx.y;
    int j_idx = blockIdx.z;

    int tid = threadIdx.x;

    // Bounds checking
    if (batch_idx >= batch_size || i_idx >= seq_len_i || j_idx >= seq_len_j) {
        return;
    }

    // Calculate tensor strides
    int input_stride = seq_len_i * seq_len_j * channels;
    int mask_stride = seq_len_i * seq_len_j;
    int proj_stride = seq_len_i * seq_len_j * hidden_dim;

    // Base pointers for current batch
    const float* grad_out_base = grad_output + batch_idx * input_stride;
    const float* input_base = input + batch_idx * input_stride;
    const float* mask_base = mask + batch_idx * mask_stride;
    const float* proj_a_base = projections_a + batch_idx * proj_stride;
    const float* proj_b_base = projections_b + batch_idx * proj_stride;
    float* grad_in_base = grad_input + batch_idx * input_stride;

    // Current position indices
    int pos_idx = i_idx * seq_len_j + j_idx;
    float mask_val = mask_base[pos_idx];

    // Skip if masked
    if (mask_val == 0.0f) {
        return;
    }

    // Compute gradient for each channel
    for (int c = tid; c < channels; c += blockDim.x) {
        int input_idx = pos_idx * channels + c;
        float grad_sum = 0.0f;

        // Accumulate gradients from all positions that used this input
        for (int k = 0; k < seq_len_j; k++) {
            int out_pos = i_idx * seq_len_j + k;
            int proj_idx = out_pos * hidden_dim;

            // Gradient contribution from projection A
            for (int h = 0; h < hidden_dim; h++) {
                grad_sum += grad_out_base[out_pos * channels + c] *
                           proj_a_base[proj_idx + h] * proj_b_base[proj_idx + h];
            }
        }

        grad_in_base[input_idx] = grad_sum * mask_val;
    }
}

// Launcher functions
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
) {
    // Launch configuration
    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_dim(batch_size, seq_len_i, seq_len_j);
    
    // Launch kernel
    triangle_multiply_forward_kernel<<<grid_dim, block_dim, 0, stream>>>(
        input, mask, projections_a, projections_b, output,
        batch_size, seq_len_i, seq_len_j, channels, hidden_dim, outgoing
    );
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Triangle multiply kernel launch error: %s\n", cudaGetErrorString(err));
    }
}

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
) {
    // Launch configuration
    dim3 block_dim(256);
    dim3 grid_dim(batch_size, seq_len_i, seq_len_j);
    
    // Launch projection kernel
    projection_kernel<<<grid_dim, block_dim, 0, stream>>>(
        input, mask, linear_g_weights, nullptr, output,
        batch_size, seq_len_i, seq_len_j, input_channels, output_channels
    );
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Projection kernel launch error: %s\n", cudaGetErrorString(err));
    }
}

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
) {
    // Kernel launch configuration
    dim3 block_dim(min(channels, 256));
    dim3 grid_dim(batch_size, seq_len_i, seq_len_j);

    // Launch backward kernel
    triangle_multiply_backward_kernel<<<grid_dim, block_dim, 0, stream>>>(
        grad_output, input, mask, projections_a, projections_b, grad_input,
        batch_size, seq_len_i, seq_len_j, channels, hidden_dim, outgoing
    );

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Triangle multiply backward kernel launch error: %s\n", cudaGetErrorString(err));
    }
}

// PyTorch interface function
torch::Tensor triangle_multiply_forward(
    torch::Tensor input_tensor,
    torch::Tensor mask,
    bool outgoing
) {
    // Input validation
    TORCH_CHECK(input_tensor.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(mask.is_cuda(), "Mask tensor must be on CUDA device");
    TORCH_CHECK(input_tensor.dtype() == torch::kFloat32, "Only float32 supported");
    
    // Get tensor dimensions
    auto sizes = input_tensor.sizes();
    int batch_size = sizes[0];
    int seq_len_i = sizes[1];
    int seq_len_j = sizes[2];
    int channels = sizes[3];
    int hidden_dim = channels / 2; // Simplified assumption
    
    // Create output tensor
    auto output = torch::zeros_like(input_tensor);
    
    // Create temporary projection tensors (in real implementation, these would be computed)
    auto projections_a = torch::randn({batch_size, seq_len_i, seq_len_j, hidden_dim}, 
                                     input_tensor.options());
    auto projections_b = torch::randn({batch_size, seq_len_i, seq_len_j, hidden_dim}, 
                                     input_tensor.options());
    
    // Get CUDA stream using C10 API
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    
    // Launch kernel
    launch_triangle_multiply_forward_kernel(
        input_tensor.data_ptr<float>(),
        mask.data_ptr<float>(),
        projections_a.data_ptr<float>(),
        projections_b.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        seq_len_i,
        seq_len_j,
        channels,
        hidden_dim,
        outgoing,
        stream
    );
    
    return output;
}

torch::Tensor triangle_multiply_backward(
    torch::Tensor grad_output,
    torch::Tensor input_tensor,
    torch::Tensor mask,
    torch::Tensor projections_a,
    torch::Tensor projections_b,
    bool outgoing
) {
    // Get tensor dimensions
    auto batch_size = input_tensor.size(0);
    auto seq_len_i = input_tensor.size(1);
    auto seq_len_j = input_tensor.size(2);
    auto channels = input_tensor.size(3);
    auto hidden_dim = projections_a.size(-1);

    // Create gradient tensor for input
    auto grad_input = torch::zeros_like(input_tensor);

    // Get CUDA stream using C10 API
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    // Launch backward kernel
    launch_triangle_multiply_backward_kernel(
        grad_output.data_ptr<float>(),
        input_tensor.data_ptr<float>(),
        mask.data_ptr<float>(),
        projections_a.data_ptr<float>(),
        projections_b.data_ptr<float>(),
        grad_input.data_ptr<float>(),
        batch_size,
        seq_len_i,
        seq_len_j,
        channels,
        hidden_dim,
        outgoing,
        stream
    );

    return grad_input;
}
