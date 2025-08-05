# FoldEngine - High-Performance C++ Inference Engine

FoldEngine is a high-performance C++ implementation of OdinFold for production deployment. It provides fast protein structure prediction with minimal dependencies and optimized performance.

## Features

- **High Performance**: Optimized C++ implementation with CUDA acceleration
- **Low Latency**: Minimal overhead for real-time inference
- **Memory Efficient**: Optimized memory usage with optional quantization
- **Production Ready**: Robust error handling and comprehensive logging
- **CLI Interface**: Easy-to-use command-line interface
- **Python Bindings**: Seamless integration with Python workflows
- **Batch Processing**: Efficient batch inference for multiple proteins
- **Ligand Awareness**: Support for ligand-conditioned folding
- **Mutation Prediction**: Built-in ΔΔG mutation effect prediction

## Requirements

### System Requirements
- **OS**: Linux (Ubuntu 18.04+), macOS (10.15+), Windows (WSL2)
- **CPU**: x86_64 with AVX2 support
- **Memory**: 8GB RAM minimum, 16GB recommended
- **GPU**: NVIDIA GPU with Compute Capability 7.5+ (optional but recommended)

### Software Dependencies
- **CMake**: 3.18 or higher
- **C++ Compiler**: GCC 9+ or Clang 10+ with C++17 support
- **PyTorch**: 1.12+ with C++ API (libtorch)
- **CUDA**: 11.0+ (for GPU acceleration)
- **OpenMP**: For CPU parallelization

## Installation

### 1. Install Dependencies

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install cmake build-essential libomp-dev
```

#### macOS
```bash
brew install cmake libomp
```

### 2. Install PyTorch C++
```bash
# Download libtorch
wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcu118.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.0.0+cu118.zip
export TORCH_ROOT=$(pwd)/libtorch
```

### 3. Build FoldEngine
```bash
cd cpp_engine
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=$TORCH_ROOT
make -j$(nproc)
```

### 4. Install
```bash
sudo make install
```

## Quick Start

### Command Line Usage

#### Fold a protein from sequence
```bash
fold_engine fold -s "MKWVTFISLLFLFSSAYSRGVFRRDAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPFEDHVKLVNEVTEFAKTCVADESAENCDKSLHTLFGDKLCTVATLRETYGEMADCCAKQEPERNECFLQHKDDNPNLPRLVRPEVDVMCTAFHDNEETFLKKYLYEIARRHPYFYAPELLFFAKRYKAAFTECCQAADKAACLLPKLDELRDEGKASSAKQRLKCASLQKFGERAFKAWAVARLSQRFPKAEFAEVSKLVTDLTKVHTECCHGDLLECADDRADLAKYICENQDSISSKLKECCEKPLLEKSHCIAEVENDEMPADLPSLAADFVESKDVCKNYAEAKDVFLGMFLYEYARRHPDYSVVLLLRLAKTYETTLEKCCAAADPHECYAKVFDEFKPLVEEPQNLIKQNCELFEQLGEYKFQNALLVRYTKKVPQVSTPTLVEVSRNLGKVGSKCCKHPEAKRMPCAEDYLSVVLNQLCVLHEKTPVSDRVTKCCTESLVNRRPCFSALEVDETYVPKEFNAETFTFHADICTLSEKERQIKKQTALVELVKHKPKATKEQLKAVMDDFAAFVEKCCKADDKETCFAEEGKKLVAASQAALGL" -o structure.pdb
```

#### Fold from FASTA file
```bash
fold_engine fold -i protein.fasta -o structure.pdb --device cuda
```

#### Predict mutation effects
```bash
fold_engine mutate -s "MKWVTFISLLFLFSSAYS" -m "A1V,L5P,F10Y"
```

#### Ligand-aware folding
```bash
fold_engine fold -s "MKWVTFISLLFLFSSAYS" -l "CCO" -l "CC(=O)O" -o complex.pdb
```

#### Get model information
```bash
fold_engine info
```

### Python Integration

```python
import fold_engine_py

# Create engine
config = fold_engine_py.FoldConfig()
config.use_cuda = True
config.use_mixed_precision = True

engine = fold_engine_py.FoldEngine(config)
engine.initialize()

# Fold protein
input_data = fold_engine_py.ProteinInput("MKWVTFISLLFLFSSAYS")
result = engine.fold_protein(input_data)

print(f"Coordinates shape: {result.coordinates.shape}")
print(f"Average confidence: {result.confidence.mean():.3f}")
print(f"Inference time: {result.inference_time_ms:.1f} ms")
```

## Configuration

### FoldConfig Options

```cpp
struct FoldConfig {
    // Model parameters
    int seq_len_max = 2048;        // Maximum sequence length
    int d_model = 256;             // Model dimension
    int num_heads = 8;             // Number of attention heads
    int num_layers = 24;           // Number of transformer layers
    
    // Performance settings
    bool use_cuda = true;          // Enable CUDA acceleration
    bool use_mixed_precision = true;  // Enable mixed precision
    bool use_flash_attention = true;  // Enable flash attention
    int batch_size = 1;            // Batch size
    
    // Memory optimization
    bool use_gradient_checkpointing = false;
    bool use_quantization = false;
    
    // Output settings
    bool output_confidence = true;
    bool output_plddt = true;
    bool output_binding_pockets = false;
    
    // Model paths
    std::string model_path = "models/odinfold.pt";
    std::string esm_path = "models/esm2_650m.pt";
};
```

## Performance Optimization

### GPU Acceleration
- **CUDA**: Automatic GPU detection and usage
- **Mixed Precision**: 2x speedup with minimal accuracy loss
- **Flash Attention**: Memory-efficient attention for long sequences
- **Tensor Cores**: Automatic utilization on supported GPUs

### CPU Optimization
- **OpenMP**: Multi-threaded CPU inference
- **SIMD**: Vectorized operations with AVX2/AVX-512
- **Memory Layout**: Cache-friendly data structures
- **Compiler Optimizations**: -O3 with native architecture targeting

### Memory Management
- **Memory Pooling**: Reduced allocation overhead
- **Gradient Checkpointing**: Trade compute for memory
- **Quantization**: INT8 inference for reduced memory usage
- **Batch Processing**: Efficient memory utilization

## Benchmarks

### Performance Comparison (RTX 4090, Protein Length 256)

| Implementation | Inference Time | Memory Usage | Throughput |
|---------------|----------------|--------------|------------|
| Python (PyTorch) | 1,250 ms | 8.2 GB | 0.8 proteins/s |
| **FoldEngine C++** | **185 ms** | **3.1 GB** | **5.4 proteins/s** |
| FoldEngine + Mixed Precision | **98 ms** | **1.8 GB** | **10.2 proteins/s** |

### Scalability (Batch Processing)

| Batch Size | Time per Protein | Memory Usage | GPU Utilization |
|------------|------------------|--------------|-----------------|
| 1 | 98 ms | 1.8 GB | 65% |
| 4 | 28 ms | 4.2 GB | 85% |
| 8 | 18 ms | 7.1 GB | 92% |
| 16 | 15 ms | 12.8 GB | 95% |

## API Reference

### Core Classes

#### FoldEngine
Main inference engine class.

```cpp
class FoldEngine {
public:
    explicit FoldEngine(const FoldConfig& config = FoldConfig());
    bool initialize();
    FoldingResult fold_protein(const ProteinInput& input);
    std::vector<FoldingResult> fold_batch(const std::vector<ProteinInput>& inputs);
    torch::Tensor predict_mutations(const std::string& sequence, 
                                   const std::vector<std::pair<int, char>>& mutations);
};
```

#### ProteinInput
Input data structure for protein folding.

```cpp
struct ProteinInput {
    std::string sequence;
    std::vector<std::string> ligand_smiles;
    std::vector<std::pair<int, char>> mutations;
    torch::Tensor esm_embeddings;  // Optional precomputed features
    torch::Tensor msa_features;    // Optional MSA features
};
```

#### FoldingResult
Output structure containing folding results.

```cpp
struct FoldingResult {
    torch::Tensor coordinates;      // [seq_len, 3] CA coordinates
    torch::Tensor all_atom_coords;  // [seq_len, 37, 3] All atoms
    torch::Tensor confidence;       // [seq_len] Per-residue confidence
    torch::Tensor plddt;           // [seq_len] pLDDT scores
    float tm_score_pred;           // Predicted TM-score
    float inference_time_ms;       // Inference time
    float memory_usage_mb;         // Memory usage
};
```

## Error Handling

FoldEngine provides comprehensive error handling with specific exception types:

```cpp
try {
    auto result = engine.fold_protein(input);
} catch (const fold_engine::ModelLoadException& e) {
    std::cerr << "Model loading failed: " << e.what() << std::endl;
} catch (const fold_engine::InferenceException& e) {
    std::cerr << "Inference failed: " << e.what() << std::endl;
} catch (const fold_engine::ValidationException& e) {
    std::cerr << "Input validation failed: " << e.what() << std::endl;
}
```

## Logging and Monitoring

### Performance Metrics
```cpp
auto metrics = engine.get_metrics();
metrics.print_summary();
metrics.save_to_file("performance.csv");
```

### Memory Monitoring
```cpp
#include "utils.h"

float cpu_memory = fold_engine::utils::get_memory_usage_mb();
float gpu_memory = fold_engine::utils::get_gpu_memory_usage_mb();
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size or sequence length
   - Enable gradient checkpointing
   - Use mixed precision

2. **Model Loading Fails**
   - Check model file path and permissions
   - Verify PyTorch version compatibility
   - Ensure sufficient disk space

3. **Slow Performance**
   - Verify CUDA installation and GPU detection
   - Enable mixed precision and flash attention
   - Check CPU/GPU utilization

### Debug Mode
```bash
export FOLD_ENGINE_DEBUG=1
fold_engine fold -s "SEQUENCE" -v
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

## License

FoldEngine is licensed under the Apache License 2.0. See LICENSE file for details.

## Citation

If you use FoldEngine in your research, please cite:

```bibtex
@software{foldengine2024,
  title={FoldEngine: High-Performance C++ Inference Engine for Protein Structure Prediction},
  author={OdinFold Team},
  year={2024},
  url={https://github.com/odinfold/foldengine}
}
```
