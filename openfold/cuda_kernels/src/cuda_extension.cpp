#include <torch/extension.h>
#include "triangle_attention.h"
#include "triangle_multiply.h"

// Python bindings for triangle attention
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "OpenFold++ CUDA kernels for triangle operations";
    
    // Triangle attention functions
    m.def("triangle_attention_forward", &triangle_attention_forward, 
          "Triangle attention forward pass",
          py::arg("query"), py::arg("key"), py::arg("value"), 
          py::arg("bias_mask"), py::arg("triangle_bias"), 
          py::arg("starting_node") = true);
    
    m.def("triangle_attention_backward", &triangle_attention_backward,
          "Triangle attention backward pass", 
          py::arg("grad_output"), py::arg("query"), py::arg("key"), py::arg("value"),
          py::arg("attention_weights"), py::arg("bias_mask"), py::arg("triangle_bias"),
          py::arg("starting_node") = true);
    
    // Triangle multiplication functions  
    m.def("triangle_multiply_forward", &triangle_multiply_forward,
          "Triangle multiplication forward pass",
          py::arg("input_tensor"), py::arg("mask"), py::arg("outgoing") = true);
    
    m.def("triangle_multiply_backward", &triangle_multiply_backward,
          "Triangle multiplication backward pass",
          py::arg("grad_output"), py::arg("input_tensor"), py::arg("mask"),
          py::arg("projections_a"), py::arg("projections_b"), py::arg("outgoing") = true);
}
