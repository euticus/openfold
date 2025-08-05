# OdinFold++ WASM Build

Browser-compatible WebAssembly build of OdinFold for client-side protein folding inference.

## Overview

The WASM build enables protein folding directly in web browsers without server dependencies. Perfect for:
- **Free tier users** with sequence length limits (50-200 residues)
- **Educational demos** and interactive tutorials
- **Offline usage** when server access is unavailable
- **Privacy-sensitive** applications requiring local processing

## Features

### ✅ Supported
- Protein sequences up to 200 residues
- Single-chain folding (no multimers)
- Basic confidence scoring
- PDB output generation
- Real-time progress updates
- Memory-efficient inference

### ❌ Not Supported (Server-only)
- Sequences > 200 residues
- Multimer/complex folding
- Ligand-aware folding
- MSA generation
- Advanced refinement
- Batch processing

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web Browser   │    │   WASM Module    │    │  Optimized      │
│                 │    │                  │    │  Model Weights  │
│ ┌─────────────┐ │    │ ┌──────────────┐ │    │                 │
│ │ JavaScript  │◄┼────┼►│ C++ Runtime  │◄┼────┤ • Quantized     │
│ │ Interface   │ │    │ │              │ │    │ • Pruned        │
│ └─────────────┘ │    │ └──────────────┘ │    │ • Compressed    │
│                 │    │                  │    │                 │
│ ┌─────────────┐ │    │ ┌──────────────┐ │    │ Size: ~50MB     │
│ │ Web Worker  │ │    │ │ Model Core   │ │    │                 │
│ │ (Threading) │ │    │ │              │ │    │                 │
│ └─────────────┘ │    │ └──────────────┘ │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| **Model Size** | < 50MB | Quantized + compressed |
| **Memory Usage** | < 512MB | Browser heap limit |
| **Inference Time** | < 30s | 100-residue protein |
| **Load Time** | < 10s | Initial model download |
| **Browser Support** | 95%+ | Chrome, Firefox, Safari, Edge |

## Build Process

### 1. Model Optimization
```bash
# Quantize model to INT8
python scripts/quantize_for_wasm.py --input models/odinfold.pt --output wasm_build/model_int8.pt

# Prune unnecessary layers
python scripts/prune_model.py --input wasm_build/model_int8.pt --output wasm_build/model_pruned.pt

# Convert to ONNX
python scripts/export_onnx.py --input wasm_build/model_pruned.pt --output wasm_build/model.onnx
```

### 2. WASM Compilation
```bash
# Install Emscripten
git clone https://github.com/emscripten-core/emsdk.git
cd emsdk && ./emsdk install latest && ./emsdk activate latest

# Build WASM module
cd wasm_build
emcmake cmake -DCMAKE_BUILD_TYPE=Release .
emmake make -j4

# Generate optimized WASM
wasm-opt -O3 --enable-simd odinfold.wasm -o odinfold_optimized.wasm
```

### 3. Web Integration
```bash
# Bundle with web assets
npm run build:wasm

# Test in browser
npm run serve:wasm
```

## Usage

### JavaScript API
```javascript
import { OdinFoldWASM } from './odinfold-wasm.js';

// Initialize
const folder = new OdinFoldWASM();
await folder.initialize();

// Fold protein
const result = await folder.foldProtein({
    sequence: "MKWVTFISLLFLFSSAYS",
    onProgress: (progress) => console.log(`${progress}% complete`)
});

// Get results
console.log("Coordinates:", result.coordinates);
console.log("Confidence:", result.confidence);
console.log("PDB:", result.toPDB());
```

### Web Worker Integration
```javascript
// main.js
const worker = new Worker('odinfold-worker.js');
worker.postMessage({
    action: 'fold',
    sequence: 'MKWVTFISLLFLFSSAYS'
});

worker.onmessage = (e) => {
    if (e.data.type === 'progress') {
        updateProgressBar(e.data.progress);
    } else if (e.data.type === 'result') {
        displayStructure(e.data.result);
    }
};
```

## Browser Compatibility

| Browser | Version | Support | Notes |
|---------|---------|---------|-------|
| Chrome | 91+ | ✅ Full | Best performance |
| Firefox | 89+ | ✅ Full | Good performance |
| Safari | 14+ | ✅ Full | Requires SharedArrayBuffer |
| Edge | 91+ | ✅ Full | Same as Chrome |
| Mobile Chrome | 91+ | ⚠️ Limited | Memory constraints |
| Mobile Safari | 14+ | ⚠️ Limited | Performance issues |

## Deployment

### CDN Hosting
```html
<script src="https://cdn.odinfold.com/wasm/v1.0/odinfold-wasm.js"></script>
<script>
    OdinFoldWASM.initialize().then(folder => {
        // Ready to fold!
    });
</script>
```

### Self-Hosted
```bash
# Copy WASM files to web server
cp wasm_build/dist/* /var/www/html/odinfold-wasm/

# Serve with proper MIME types
# .wasm -> application/wasm
# .js -> application/javascript
```

## Development

### Prerequisites
- Emscripten SDK 3.1.0+
- Node.js 16+
- Python 3.8+
- CMake 3.18+

### Build Commands
```bash
# Development build (with debug symbols)
npm run build:dev

# Production build (optimized)
npm run build:prod

# Test suite
npm run test:wasm

# Benchmark
npm run benchmark:wasm
```

## Limitations

### Technical Constraints
- **Memory**: Limited to browser heap (~2GB on desktop, ~1GB mobile)
- **Threading**: Limited Web Worker support
- **File I/O**: No direct file system access
- **Networking**: CORS restrictions for model loading

### Model Constraints
- **Sequence Length**: 50-200 residues (configurable limit)
- **Model Size**: Heavily quantized and pruned
- **Accuracy**: ~5-10% lower than full server model
- **Features**: Subset of full OdinFold capabilities

## Security & Privacy

### Advantages
- **Local Processing**: No data sent to servers
- **Privacy**: Sequences never leave user's browser
- **Offline**: Works without internet after initial load
- **Compliance**: GDPR/HIPAA friendly for sensitive data

### Considerations
- **Model Protection**: WASM can be reverse-engineered
- **Resource Usage**: May impact browser performance
- **Updates**: Manual model updates required

## Roadmap

### Phase 1: Basic Folding (Current)
- [x] Model quantization and pruning
- [x] WASM compilation pipeline
- [x] JavaScript API
- [x] Basic web interface

### Phase 2: Enhanced Features
- [ ] Multi-threading with Web Workers
- [ ] Progressive model loading
- [ ] Advanced visualization
- [ ] Mobile optimization

### Phase 3: Advanced Capabilities
- [ ] Small multimer support (2-3 chains)
- [ ] Basic ligand docking
- [ ] Confidence-based early stopping
- [ ] Streaming inference

## Support

For WASM-specific issues:
- Check browser console for errors
- Verify WebAssembly support: `typeof WebAssembly`
- Test with smaller sequences first
- Monitor memory usage in DevTools

For general OdinFold support, see main documentation.
