/*
OpenFold++ CUDA Triangle Kernels
High-performance GPU implementations of triangle attention and multiplication.
*/

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#define BLOCK_SIZE 16
#define WARP_SIZE 32

// Triangle Attention Kernel
__global__ void triangle_attention_kernel(
    const float* __restrict__ query,
    const float* __restrict__ key, 
    const float* __restrict__ value,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int seq_len,
    const int head_dim,
    const int num_heads
) {
    // Shared memory for tile-based computation
    __shared__ float query_tile[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float key_tile[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float value_tile[BLOCK_SIZE][BLOCK_SIZE];
    
    int batch_idx = blockIdx.z;
    int head_idx = blockIdx.y;
    int row = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.y;
    
    if (batch_idx >= batch_size || head_idx >= num_heads) return;
    
    // Triangle attention computation
    float attention_sum = 0.0f;
    float max_val = -INFINITY;
    
    // Load query tile
    if (row < seq_len && threadIdx.y < head_dim) {
        int q_idx = batch_idx * num_heads * seq_len * head_dim + 
                   head_idx * seq_len * head_dim + 
                   row * head_dim + threadIdx.y;
        query_tile[threadIdx.x][threadIdx.y] = query[q_idx];
    }
    
    __syncthreads();
    
    // Compute attention scores for triangle
    for (int k = 0; k < seq_len; k += BLOCK_SIZE) {
        // Load key tile
        if (k + threadIdx.y < seq_len && threadIdx.x < head_dim) {
            int k_idx = batch_idx * num_heads * seq_len * head_dim + 
                       head_idx * seq_len * head_dim + 
                       (k + threadIdx.y) * head_dim + threadIdx.x;
            key_tile[threadIdx.y][threadIdx.x] = key[k_idx];
        }
        
        __syncthreads();
        
        // Compute dot product for this tile
        if (row < seq_len && col < seq_len) {
            float score = 0.0f;
            for (int d = 0; d < min(BLOCK_SIZE, head_dim); d++) {
                score += query_tile[threadIdx.x][d] * key_tile[col - k][d];
            }
            
            // Add bias if provided
            if (bias != nullptr) {
                int bias_idx = batch_idx * seq_len * seq_len + row * seq_len + col;
                score += bias[bias_idx];
            }
            
            // Triangle masking (upper triangular)
            if (row <= col) {
                max_val = fmaxf(max_val, score);
                attention_sum += expf(score - max_val);
            }
        }
        
        __syncthreads();
    }
    
    // Normalize attention weights
    if (row < seq_len && col < seq_len && row <= col) {
        float attention_weight = expf(-max_val) / attention_sum;
        
        // Apply attention to values
        float output_val = 0.0f;
        for (int v = 0; v < seq_len; v++) {
            if (row <= v) {
                int v_idx = batch_idx * num_heads * seq_len * head_dim + 
                           head_idx * seq_len * head_dim + 
                           v * head_dim + threadIdx.y;
                output_val += attention_weight * value[v_idx];
            }
        }
        
        int out_idx = batch_idx * num_heads * seq_len * head_dim + 
                     head_idx * seq_len * head_dim + 
                     row * head_dim + threadIdx.y;
        output[out_idx] = output_val;
    }
}

// Triangle Multiplication Kernel
__global__ void triangle_multiplication_kernel(
    const float* __restrict__ input_a,
    const float* __restrict__ input_b,
    float* __restrict__ output,
    const int batch_size,
    const int seq_len,
    const int channels
) {
    // Shared memory for efficient computation
    __shared__ float tile_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float tile_b[BLOCK_SIZE][BLOCK_SIZE];
    
    int batch_idx = blockIdx.z;
    int row = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int col = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    
    if (batch_idx >= batch_size || row >= seq_len || col >= seq_len) return;
    
    float result = 0.0f;
    
    // Triangle multiplication: C[i,j] = sum_k A[i,k] * B[k,j] for k <= min(i,j)
    for (int k_block = 0; k_block <= min(row, col); k_block += BLOCK_SIZE) {
        // Load tiles
        int k = k_block + threadIdx.x;
        if (k <= min(row, col) && k < seq_len) {
            int a_idx = batch_idx * seq_len * seq_len * channels + 
                       row * seq_len * channels + k * channels + threadIdx.y;
            int b_idx = batch_idx * seq_len * seq_len * channels + 
                       k * seq_len * channels + col * channels + threadIdx.y;
            
            tile_a[threadIdx.x][threadIdx.y] = input_a[a_idx];
            tile_b[threadIdx.x][threadIdx.y] = input_b[b_idx];
        } else {
            tile_a[threadIdx.x][threadIdx.y] = 0.0f;
            tile_b[threadIdx.x][threadIdx.y] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial result
        for (int k_local = 0; k_local < BLOCK_SIZE; k_local++) {
            int k_global = k_block + k_local;
            if (k_global <= min(row, col) && k_global < seq_len) {
                result += tile_a[k_local][threadIdx.y] * tile_b[k_local][threadIdx.y];
            }
        }
        
        __syncthreads();
    }
    
    // Store result
    int out_idx = batch_idx * seq_len * seq_len * channels + 
                 row * seq_len * channels + col * channels + threadIdx.y;
    output[out_idx] = result;
}

// Fused Triangle Update Kernel
__global__ void fused_triangle_update_kernel(
    const float* __restrict__ pair_repr,
    const float* __restrict__ gate_values,
    float* __restrict__ output,
    const int batch_size,
    const int seq_len,
    const int channels
) {
    int batch_idx = blockIdx.z;
    int row = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int col = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int chan = threadIdx.z;
    
    if (batch_idx >= batch_size || row >= seq_len || col >= seq_len || chan >= channels) return;
    
    int idx = batch_idx * seq_len * seq_len * channels + 
             row * seq_len * channels + col * channels + chan;
    
    // Apply gating and residual connection
    float gated_value = pair_repr[idx] * gate_values[idx];
    output[idx] = pair_repr[idx] + gated_value;
}

// C++ Interface Functions
torch::Tensor triangle_attention_cuda(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor bias
) {
    auto batch_size = query.size(0);
    auto num_heads = query.size(1);
    auto seq_len = query.size(2);
    auto head_dim = query.size(3);
    
    auto output = torch::zeros_like(query);
    
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE, num_heads, batch_size);
    
    triangle_attention_kernel<<<grid, block>>>(
        query.data_ptr<float>(),
        key.data_ptr<float>(),
        value.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, seq_len, head_dim, num_heads
    );
    
    cudaDeviceSynchronize();
    return output;
}

torch::Tensor triangle_multiplication_cuda(
    torch::Tensor input_a,
    torch::Tensor input_b
) {
    auto batch_size = input_a.size(0);
    auto seq_len = input_a.size(1);
    auto channels = input_a.size(3);
    
    auto output = torch::zeros_like(input_a);
    
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE, 
              (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE, 
              batch_size);
    
    triangle_multiplication_kernel<<<grid, block>>>(
        input_a.data_ptr<float>(),
        input_b.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, seq_len, channels
    );
    
    cudaDeviceSynchronize();
    return output;
}

torch::Tensor fused_triangle_update_cuda(
    torch::Tensor pair_repr,
    torch::Tensor gate_values
) {
    auto batch_size = pair_repr.size(0);
    auto seq_len = pair_repr.size(1);
    auto channels = pair_repr.size(3);
    
    auto output = torch::zeros_like(pair_repr);
    
    dim3 block(BLOCK_SIZE, BLOCK_SIZE, min(channels, 8));
    dim3 grid((seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE, 
              (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE, 
              batch_size);
    
    fused_triangle_update_kernel<<<grid, block>>>(
        pair_repr.data_ptr<float>(),
        gate_values.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, seq_len, channels
    );
    
    cudaDeviceSynchronize();
    return output;
}

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("triangle_attention", &triangle_attention_cuda, "Triangle Attention CUDA");
    m.def("triangle_multiplication", &triangle_multiplication_cuda, "Triangle Multiplication CUDA");
    m.def("fused_triangle_update", &fused_triangle_update_cuda, "Fused Triangle Update CUDA");
}
