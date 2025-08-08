#include "triangle_attention.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>
#include <algorithm>
#include <c10/cuda/CUDAStream.h>

// CUDA kernel constants
#define BLOCK_SIZE 16
#define WARP_SIZE 32
#define MAX_THREADS_PER_BLOCK 1024

// Shared memory triangle attention kernel
__global__ void triangle_attention_forward_kernel(
    const float* __restrict__ query,      // [B, H, I, J, D]
    const float* __restrict__ key,        // [B, H, I, J, D] 
    const float* __restrict__ value,      // [B, H, I, J, D]
    const float* __restrict__ bias_mask,  // [B, I, 1, 1, J]
    const float* __restrict__ triangle_bias, // [B, 1, H, I, J]
    float* __restrict__ output,           // [B, H, I, J, D]
    float* __restrict__ attention_weights, // [B, H, I, J, J] (optional)
    int batch_size,
    int seq_len,
    int num_heads, 
    int head_dim,
    bool starting_node
) {
    // Block and thread indices
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int i_idx = blockIdx.z;
    
    int tid = threadIdx.x;
    int j_idx = threadIdx.y;
    
    // Bounds checking
    if (batch_idx >= batch_size || head_idx >= num_heads || i_idx >= seq_len) {
        return;
    }
    
    // Shared memory for attention computation
    extern __shared__ float shared_mem[];
    float* shared_query = shared_mem;
    float* shared_key = shared_query + BLOCK_SIZE * head_dim;
    float* shared_value = shared_key + BLOCK_SIZE * head_dim;
    float* shared_scores = shared_value + BLOCK_SIZE * head_dim;
    
    // Calculate tensor strides
    int query_stride = num_heads * seq_len * seq_len * head_dim;
    int head_stride = seq_len * seq_len * head_dim;
    int i_stride = seq_len * head_dim;
    int j_stride = head_dim;
    
    // Base pointers for current batch and head
    const float* query_base = query + batch_idx * query_stride + head_idx * head_stride;
    const float* key_base = key + batch_idx * query_stride + head_idx * head_stride;
    const float* value_base = value + batch_idx * query_stride + head_idx * head_stride;
    float* output_base = output + batch_idx * query_stride + head_idx * head_stride;
    
    // Bias pointers
    const float* mask_bias_base = bias_mask + batch_idx * seq_len * seq_len;
    const float* triangle_bias_base = triangle_bias + batch_idx * num_heads * seq_len * seq_len + head_idx * seq_len * seq_len;
    
    // Load query for current i position into shared memory
    if (j_idx < seq_len && tid < head_dim) {
        shared_query[j_idx * head_dim + tid] = query_base[i_idx * i_stride + j_idx * j_stride + tid];
    }
    __syncthreads();
    
    // Process attention for each j position
    for (int j_block = 0; j_block < (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE; j_block++) {
        int j_start = j_block * BLOCK_SIZE;
        int j_end = min(j_start + BLOCK_SIZE, seq_len);
        
        // Load keys and values for current block
        if (j_idx < BLOCK_SIZE && tid < head_dim) {
            int global_j = j_start + j_idx;
            if (global_j < seq_len) {
                shared_key[j_idx * head_dim + tid] = key_base[i_idx * i_stride + global_j * j_stride + tid];
                shared_value[j_idx * head_dim + tid] = value_base[i_idx * i_stride + global_j * j_stride + tid];
            } else {
                shared_key[j_idx * head_dim + tid] = 0.0f;
                shared_value[j_idx * head_dim + tid] = 0.0f;
            }
        }
        __syncthreads();
        
        // Compute attention scores for this block
        if (j_idx < BLOCK_SIZE && tid == 0) {
            int global_j = j_start + j_idx;
            if (global_j < seq_len) {
                float score = 0.0f;
                
                // Compute dot product between query[i] and key[j]
                for (int d = 0; d < head_dim; d++) {
                    score += shared_query[i_idx * head_dim + d] * shared_key[j_idx * head_dim + d];
                }
                
                // Apply scaling
                score /= sqrtf((float)head_dim);
                
                // Add biases
                score += mask_bias_base[i_idx * seq_len + global_j];
                score += triangle_bias_base[i_idx * seq_len + global_j];
                
                shared_scores[j_idx] = score;
            } else {
                shared_scores[j_idx] = -INFINITY;
            }
        }
        __syncthreads();
        
        // Softmax computation (simplified for demonstration)
        if (tid == 0) {
            // Find max for numerical stability
            float max_score = -INFINITY;
            for (int k = 0; k < min(BLOCK_SIZE, seq_len - j_start); k++) {
                max_score = fmaxf(max_score, shared_scores[k]);
            }
            
            // Compute exponentials and sum
            float sum_exp = 0.0f;
            for (int k = 0; k < min(BLOCK_SIZE, seq_len - j_start); k++) {
                shared_scores[k] = expf(shared_scores[k] - max_score);
                sum_exp += shared_scores[k];
            }
            
            // Normalize
            for (int k = 0; k < min(BLOCK_SIZE, seq_len - j_start); k++) {
                shared_scores[k] /= sum_exp;
            }
        }
        __syncthreads();
        
        // Compute weighted sum of values
        if (j_idx < BLOCK_SIZE && tid < head_dim) {
            int global_j = j_start + j_idx;
            if (global_j < seq_len) {
                float weighted_value = 0.0f;
                for (int k = 0; k < min(BLOCK_SIZE, seq_len - j_start); k++) {
                    weighted_value += shared_scores[k] * shared_value[k * head_dim + tid];
                }
                
                // Accumulate to output (this is simplified - real implementation needs proper accumulation)
                atomicAdd(&output_base[i_idx * i_stride + global_j * j_stride + tid], weighted_value);
                
                // Store attention weights if requested
                if (attention_weights != nullptr) {
                    attention_weights[batch_idx * num_heads * seq_len * seq_len * seq_len + 
                                   head_idx * seq_len * seq_len * seq_len +
                                   i_idx * seq_len * seq_len + 
                                   global_j * seq_len + j_idx] = shared_scores[j_idx];
                }
            }
        }
        __syncthreads();
    }
}

// Launcher function
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
) {
    // Calculate shared memory size
    size_t shared_mem_size = (3 * BLOCK_SIZE * head_dim + BLOCK_SIZE) * sizeof(float);
    
    // Launch configuration
    dim3 block_dim(head_dim, BLOCK_SIZE);
    dim3 grid_dim(batch_size, num_heads, seq_len);
    
    // Launch kernel
    triangle_attention_forward_kernel<<<grid_dim, block_dim, shared_mem_size, stream>>>(
        query, key, value, bias_mask, triangle_bias, output, attention_weights,
        batch_size, seq_len, num_heads, head_dim, starting_node
    );
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
}

// PyTorch interface function
torch::Tensor triangle_attention_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor bias_mask,
    torch::Tensor triangle_bias,
    bool starting_node
) {
    // Input validation
    TORCH_CHECK(query.is_cuda(), "Query tensor must be on CUDA device");
    TORCH_CHECK(key.is_cuda(), "Key tensor must be on CUDA device");
    TORCH_CHECK(value.is_cuda(), "Value tensor must be on CUDA device");
    TORCH_CHECK(query.dtype() == torch::kFloat32, "Only float32 supported");
    
    // Get tensor dimensions
    auto sizes = query.sizes();
    int batch_size = sizes[0];
    int num_heads = sizes[1];
    int seq_len_i = sizes[2];
    int seq_len_j = sizes[3];
    int head_dim = sizes[4];
    
    // Create output tensor
    auto output = torch::zeros_like(query);
    
    // Get CUDA stream using C10 API
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    
    // Launch kernel
    launch_triangle_attention_forward_kernel(
        query.data_ptr<float>(),
        key.data_ptr<float>(),
        value.data_ptr<float>(),
        bias_mask.data_ptr<float>(),
        triangle_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        nullptr, // attention_weights
        batch_size,
        seq_len_i, // Using seq_len_i as the primary sequence length
        num_heads,
        head_dim,
        starting_node,
        stream
    );
    
    return output;
}

// Triangle attention backward kernel
__global__ void triangle_attention_backward_kernel(
    const float* __restrict__ grad_output,    // [B, H, I, J, D]
    const float* __restrict__ query,          // [B, H, I, J, D]
    const float* __restrict__ key,            // [B, H, I, J, D]
    const float* __restrict__ value,          // [B, H, I, J, D]
    const float* __restrict__ attention_weights, // [B, H, I, J, J]
    float* __restrict__ grad_query,           // [B, H, I, J, D]
    float* __restrict__ grad_key,             // [B, H, I, J, D]
    float* __restrict__ grad_value,           // [B, H, I, J, D]
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim,
    bool starting_node
) {
    // Block and thread indices
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int i_idx = blockIdx.z;

    int tid = threadIdx.x;
    int j_idx = threadIdx.y;

    // Bounds checking
    if (batch_idx >= batch_size || head_idx >= num_heads || i_idx >= seq_len) {
        return;
    }

    // Calculate tensor strides
    int tensor_stride = num_heads * seq_len * seq_len * head_dim;
    int head_stride = seq_len * seq_len * head_dim;
    int i_stride = seq_len * head_dim;
    int j_stride = head_dim;

    // Base pointers for current batch and head
    const float* grad_out_base = grad_output + batch_idx * tensor_stride + head_idx * head_stride;
    const float* query_base = query + batch_idx * tensor_stride + head_idx * head_stride;
    const float* key_base = key + batch_idx * tensor_stride + head_idx * head_stride;
    const float* value_base = value + batch_idx * tensor_stride + head_idx * head_stride;

    float* grad_q_base = grad_query + batch_idx * tensor_stride + head_idx * head_stride;
    float* grad_k_base = grad_key + batch_idx * tensor_stride + head_idx * head_stride;
    float* grad_v_base = grad_value + batch_idx * tensor_stride + head_idx * head_stride;

    // Compute gradients for each dimension
    if (j_idx < seq_len && tid < head_dim) {
        int pos_idx = i_idx * i_stride + j_idx * j_stride + tid;

        // Simplified gradient computation (real implementation would be more complex)
        float grad_q = 0.0f, grad_k = 0.0f, grad_v = 0.0f;

        // Accumulate gradients from all attention positions
        for (int k = 0; k < seq_len; k++) {
            float grad_out_val = grad_out_base[i_idx * i_stride + k * j_stride + tid];

            // Gradient w.r.t. value (simplified)
            grad_v += grad_out_val * 0.1f; // Placeholder computation

            // Gradient w.r.t. query and key (simplified)
            grad_q += grad_out_val * key_base[i_idx * i_stride + k * j_stride + tid] * 0.1f;
            grad_k += grad_out_val * query_base[i_idx * i_stride + k * j_stride + tid] * 0.1f;
        }

        // Store gradients
        grad_q_base[pos_idx] = grad_q;
        grad_k_base[pos_idx] = grad_k;
        grad_v_base[pos_idx] = grad_v;
    }
}

// Backward launcher function
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
) {
    // Launch configuration
    dim3 block_dim(head_dim, BLOCK_SIZE);
    dim3 grid_dim(batch_size, num_heads, seq_len);

    // Launch backward kernel
    triangle_attention_backward_kernel<<<grid_dim, block_dim, 0, stream>>>(
        grad_output, query, key, value, attention_weights,
        grad_query, grad_key, grad_value,
        batch_size, seq_len, num_heads, head_dim, starting_node
    );

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Triangle attention backward kernel launch error: %s\n", cudaGetErrorString(err));
    }
}

// PyTorch backward interface function
torch::Tensor triangle_attention_backward(
    torch::Tensor grad_output,
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor attention_weights,
    torch::Tensor bias_mask,
    torch::Tensor triangle_bias,
    bool starting_node
) {
    // Input validation
    TORCH_CHECK(grad_output.is_cuda(), "Gradient tensor must be on CUDA device");
    TORCH_CHECK(grad_output.dtype() == torch::kFloat32, "Only float32 supported");

    // Get tensor dimensions
    auto sizes = grad_output.sizes();
    int batch_size = sizes[0];
    int num_heads = sizes[1];
    int seq_len_i = sizes[2];
    int seq_len_j = sizes[3];
    int head_dim = sizes[4];

    // Create gradient tensors
    auto grad_query = torch::zeros_like(query);
    auto grad_key = torch::zeros_like(key);
    auto grad_value = torch::zeros_like(value);

    // Get CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    // Launch backward kernel
    launch_triangle_attention_backward_kernel(
        grad_output.data_ptr<float>(),
        query.data_ptr<float>(),
        key.data_ptr<float>(),
        value.data_ptr<float>(),
        attention_weights.data_ptr<float>(),
        bias_mask.data_ptr<float>(),
        triangle_bias.data_ptr<float>(),
        grad_query.data_ptr<float>(),
        grad_key.data_ptr<float>(),
        grad_value.data_ptr<float>(),
        batch_size,
        seq_len_i,
        num_heads,
        head_dim,
        starting_node,
        stream
    );

    // For simplicity, return concatenated gradients or just grad_query
    // In a real implementation, you might want to return all gradients differently
    return grad_query;
}
