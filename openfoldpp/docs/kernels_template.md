OPENFOLD KERNELS

No problem! Below are the full contents of the scaffolded CUDA kernel source files, headers, and bindings so you can copy and paste them directly into your project.

📁 include/triangle_attention.h
#pragma once

#include <torch/extension.h>

torch::Tensor triangle_attention_forward(
    torch::Tensor pair_rep);

📁 include/triangle_multiply.h
#pragma once

#include <torch/extension.h>

torch::Tensor triangle_multiply_forward(
    torch::Tensor pair_rep);

📁 src/triangle_attention.cu
#include <torch/extension.h>
#include "triangle_attention.h"

// Placeholder CUDA kernel
__global__ void triangle_attention_kernel(const float* __restrict__ input,
                                          float* __restrict__ output,
                                          int N, int C) {
    // TODO: Implement tile-based attention kernel
}

torch::Tensor triangle_attention_forward(torch::Tensor pair_rep) {
    // TODO: Launch CUDA kernel with torch::Tensor input
    TORCH_CHECK(pair_rep.is_cuda(), "Input must be a CUDA tensor");
    return pair_rep;  // placeholder
}

📁 src/triangle_multiply.cu
#include <torch/extension.h>
#include "triangle_multiply.h"

// Placeholder CUDA kernel
__global__ void triangle_multiply_kernel(const float* __restrict__ A,
                                         const float* __restrict__ B,
                                         float* __restrict__ output,
                                         int N, int C) {
    // TODO: Implement triangle multiplication kernel
}

torch::Tensor triangle_multiply_forward(torch::Tensor pair_rep) {
    // TODO: Launch CUDA kernel
    TORCH_CHECK(pair_rep.is_cuda(), "Input must be a CUDA tensor");
    return pair_rep;  // placeholder
}

📁 src/bindings.cpp
#include <torch/extension.h>
#include "triangle_attention.h"
#include "triangle_multiply.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("triangle_attention_forward", &triangle_attention_forward,
          "Triangle Attention Forward (CUDA)");
    m.def("triangle_multiply_forward", &triangle_multiply_forward,
          "Triangle Multiply Forward (CUDA)");
}

📁 CMakeLists.txt
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(triangle_kernels LANGUAGES CXX CUDA)

find_package(Torch REQUIRED)

add_library(triangle_kernels SHARED
    src/triangle_attention.cu
    src/triangle_multiply.cu
    src/bindings.cpp
)

target_include_directories(triangle_kernels PRIVATE include)
target_link_libraries(triangle_kernels PRIVATE ${TORCH_LIBRARIES})
set_property(TARGET triangle_kernels PROPERTY CXX_STANDARD 14)

Would you like me to:
* Help fill in the CUDA logic inside triangle_attention_kernel and triangle_multiply_kernel?
* Write a setup.py build system for PyTorch directly?
* Show how to test these kernels on dummy input tensors?
Let’s make it run.
