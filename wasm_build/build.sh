#!/bin/bash

# OdinFold++ WASM Build Script
# Automated build system for WebAssembly deployment

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BUILD_TYPE=${BUILD_TYPE:-Release}
BUILD_DIR=${BUILD_DIR:-build}
CLEAN_BUILD=false
OPTIMIZE_MODEL=false
RUN_TESTS=false
SERVE_DEMO=false
VERBOSE=false

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    cat << EOF
OdinFold++ WASM Build Script

Usage: $0 [OPTIONS]

Options:
    -h, --help              Show this help message
    -c, --clean             Clean build directory before building
    -o, --optimize-model    Run model optimization for WASM
    -t, --test              Run tests after building
    -s, --serve             Serve demo after building
    -v, --verbose           Verbose output
    --debug                 Build in Debug mode
    --build-dir DIR         Build directory (default: build)

Environment Variables:
    EMSDK_ROOT              Path to Emscripten SDK
    MODEL_PATH              Path to original model for optimization

Examples:
    $0                      # Basic build
    $0 -c -o -t            # Clean build with model optimization and tests
    $0 --debug --verbose   # Debug build with verbose output
    $0 -s                  # Build and serve demo

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -c|--clean)
            CLEAN_BUILD=true
            shift
            ;;
        -o|--optimize-model)
            OPTIMIZE_MODEL=true
            shift
            ;;
        -t|--test)
            RUN_TESTS=true
            shift
            ;;
        -s|--serve)
            SERVE_DEMO=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        --debug)
            BUILD_TYPE=Debug
            shift
            ;;
        --build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Print build configuration
print_status "OdinFold++ WASM Build Configuration"
echo "  Build Type: $BUILD_TYPE"
echo "  Build Directory: $BUILD_DIR"
echo "  Clean Build: $CLEAN_BUILD"
echo "  Optimize Model: $OPTIMIZE_MODEL"
echo "  Run Tests: $RUN_TESTS"
echo "  Serve Demo: $SERVE_DEMO"
echo ""

# Check system requirements
print_status "Checking system requirements..."

# Check Emscripten
if ! command -v emcc &> /dev/null; then
    print_error "Emscripten not found. Please install and activate Emscripten SDK."
    echo "Instructions:"
    echo "  git clone https://github.com/emscripten-core/emsdk.git"
    echo "  cd emsdk"
    echo "  ./emsdk install latest"
    echo "  ./emsdk activate latest"
    echo "  source ./emsdk_env.sh"
    exit 1
fi

EMCC_VERSION=$(emcc --version | head -n1)
print_success "Emscripten found: $EMCC_VERSION"

# Check CMake
if ! command -v cmake &> /dev/null; then
    print_error "CMake not found. Please install CMake 3.18 or higher."
    exit 1
fi

CMAKE_VERSION=$(cmake --version | head -n1 | cut -d' ' -f3)
print_success "CMake found: $CMAKE_VERSION"

# Check Node.js (for testing)
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    print_success "Node.js found: $NODE_VERSION"
else
    print_warning "Node.js not found. Some tests may not work."
fi

# Check Python (for model optimization)
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    print_success "Python found: $PYTHON_VERSION"
else
    print_warning "Python not found. Model optimization will be skipped."
    OPTIMIZE_MODEL=false
fi

# Model optimization
if [[ "$OPTIMIZE_MODEL" == "true" ]]; then
    print_status "Running model optimization..."
    
    MODEL_PATH=${MODEL_PATH:-"../models/odinfold.pt"}
    OUTPUT_DIR="optimized_models"
    
    if [[ ! -f "$MODEL_PATH" ]]; then
        print_warning "Model file not found at $MODEL_PATH. Skipping optimization."
        print_warning "Set MODEL_PATH environment variable to specify model location."
    else
        mkdir -p "$OUTPUT_DIR"
        
        python3 scripts/quantize_for_wasm.py \
            --input "$MODEL_PATH" \
            --output "$OUTPUT_DIR" \
            --max-seq-len 200 \
            --pruning-ratio 0.3 \
            $([ "$VERBOSE" == "true" ] && echo "--verbose")
        
        print_success "Model optimization completed"
    fi
fi

# Clean build directory if requested
if [[ "$CLEAN_BUILD" == "true" ]]; then
    print_status "Cleaning build directory..."
    rm -rf "$BUILD_DIR"
    print_success "Build directory cleaned"
fi

# Create build directory
print_status "Creating build directory..."
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure CMake
print_status "Configuring CMake..."

CMAKE_ARGS=(
    "-DCMAKE_BUILD_TYPE=$BUILD_TYPE"
    "-DCMAKE_TOOLCHAIN_FILE=$EMSDK/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake"
)

if [[ "$VERBOSE" == "true" ]]; then
    CMAKE_ARGS+=("-DCMAKE_VERBOSE_MAKEFILE=ON")
fi

# Run CMake
if emcmake cmake "${CMAKE_ARGS[@]}" ..; then
    print_success "CMake configuration completed"
else
    print_error "CMake configuration failed"
    exit 1
fi

# Build
print_status "Building OdinFold++ WASM..."

if [[ "$VERBOSE" == "true" ]]; then
    MAKE_ARGS=("VERBOSE=1")
else
    MAKE_ARGS=()
fi

if emmake make "${MAKE_ARGS[@]}"; then
    print_success "Build completed successfully"
else
    print_error "Build failed"
    exit 1
fi

# Create package
print_status "Creating distribution package..."
if make package; then
    print_success "Distribution package created"
else
    print_warning "Package creation failed"
fi

# Run tests if requested
if [[ "$RUN_TESTS" == "true" ]]; then
    print_status "Running tests..."
    
    cd ..  # Go back to project root
    
    if python3 -m pytest tests/test_wasm_build.py -v; then
        print_success "All tests passed"
    else
        print_warning "Some tests failed"
    fi
    
    cd "$BUILD_DIR"  # Return to build directory
fi

# Check output files
print_status "Checking output files..."

REQUIRED_FILES=("odinfold.js" "odinfold.wasm" "odinfold-wasm.js")
MISSING_FILES=()

for file in "${REQUIRED_FILES[@]}"; do
    if [[ -f "$file" ]]; then
        SIZE=$(du -h "$file" | cut -f1)
        print_success "✓ $file ($SIZE)"
    else
        MISSING_FILES+=("$file")
        print_error "✗ $file (missing)"
    fi
done

if [[ ${#MISSING_FILES[@]} -gt 0 ]]; then
    print_error "Build incomplete. Missing files: ${MISSING_FILES[*]}"
    exit 1
fi

# Check WASM file size
WASM_SIZE=$(stat -f%z odinfold.wasm 2>/dev/null || stat -c%s odinfold.wasm 2>/dev/null || echo "0")
WASM_SIZE_MB=$((WASM_SIZE / 1024 / 1024))

if [[ $WASM_SIZE_MB -gt 50 ]]; then
    print_warning "WASM file is large: ${WASM_SIZE_MB}MB (target: <50MB)"
else
    print_success "WASM file size OK: ${WASM_SIZE_MB}MB"
fi

# Serve demo if requested
if [[ "$SERVE_DEMO" == "true" ]]; then
    print_status "Starting demo server..."
    
    if command -v python3 &> /dev/null; then
        print_success "Demo server starting at http://localhost:8000"
        print_status "Press Ctrl+C to stop the server"
        python3 -m http.server 8000
    elif command -v python &> /dev/null; then
        print_success "Demo server starting at http://localhost:8000"
        print_status "Press Ctrl+C to stop the server"
        python -m SimpleHTTPServer 8000
    else
        print_error "Python not found. Cannot start demo server."
        print_status "You can serve the files manually with any HTTP server."
    fi
fi

# Print summary
print_success "OdinFold++ WASM build completed!"
echo ""
echo "Build artifacts:"
echo "  JavaScript: $BUILD_DIR/odinfold.js"
echo "  WebAssembly: $BUILD_DIR/odinfold.wasm"
echo "  API Wrapper: $BUILD_DIR/odinfold-wasm.js"
echo "  Demo: $BUILD_DIR/index.html"

if [[ -d "$BUILD_DIR/dist" ]]; then
    echo "  Package: $BUILD_DIR/dist/"
fi

echo ""
echo "Next steps:"
echo "  1. Test the demo: python3 -m http.server 8000 (in build directory)"
echo "  2. Open http://localhost:8000 in your browser"
echo "  3. Try folding a protein sequence!"

if [[ "$OPTIMIZE_MODEL" == "false" ]]; then
    echo ""
    print_warning "Model optimization was skipped. For production deployment:"
    echo "  Run: $0 --optimize-model --clean"
fi

echo ""
print_success "Happy folding! 🧬"
