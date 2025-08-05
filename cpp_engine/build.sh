#!/bin/bash

# FoldEngine Build Script
# Automated build system for the C++ inference engine

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
INSTALL_PREFIX=${INSTALL_PREFIX:-/usr/local}
NUM_JOBS=${NUM_JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}

# Flags
CLEAN_BUILD=false
INSTALL=false
RUN_TESTS=false
VERBOSE=false
CUDA_ENABLED=true
PYTHON_BINDINGS=true

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
FoldEngine Build Script

Usage: $0 [OPTIONS]

Options:
    -h, --help              Show this help message
    -c, --clean             Clean build directory before building
    -i, --install           Install after building
    -t, --test              Run tests after building
    -v, --verbose           Verbose output
    --debug                 Build in Debug mode
    --no-cuda               Disable CUDA support
    --no-python             Disable Python bindings
    --build-dir DIR         Build directory (default: build)
    --install-prefix DIR    Install prefix (default: /usr/local)
    --jobs N                Number of parallel jobs (default: auto-detect)

Environment Variables:
    TORCH_ROOT              Path to libtorch installation
    CUDA_ROOT               Path to CUDA installation
    BUILD_TYPE              Build type (Release/Debug/RelWithDebInfo)

Examples:
    $0                      # Basic build
    $0 -c -i -t            # Clean build, install, and test
    $0 --debug --verbose   # Debug build with verbose output
    $0 --no-cuda           # CPU-only build

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
        -i|--install)
            INSTALL=true
            shift
            ;;
        -t|--test)
            RUN_TESTS=true
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
        --no-cuda)
            CUDA_ENABLED=false
            shift
            ;;
        --no-python)
            PYTHON_BINDINGS=false
            shift
            ;;
        --build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        --install-prefix)
            INSTALL_PREFIX="$2"
            shift 2
            ;;
        --jobs)
            NUM_JOBS="$2"
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
print_status "FoldEngine Build Configuration"
echo "  Build Type: $BUILD_TYPE"
echo "  Build Directory: $BUILD_DIR"
echo "  Install Prefix: $INSTALL_PREFIX"
echo "  Parallel Jobs: $NUM_JOBS"
echo "  CUDA Enabled: $CUDA_ENABLED"
echo "  Python Bindings: $PYTHON_BINDINGS"
echo "  Clean Build: $CLEAN_BUILD"
echo "  Install: $INSTALL"
echo "  Run Tests: $RUN_TESTS"
echo ""

# Check system requirements
print_status "Checking system requirements..."

# Check CMake
if ! command -v cmake &> /dev/null; then
    print_error "CMake not found. Please install CMake 3.18 or higher."
    exit 1
fi

CMAKE_VERSION=$(cmake --version | head -n1 | cut -d' ' -f3)
print_success "CMake found: $CMAKE_VERSION"

# Check compiler
if command -v g++ &> /dev/null; then
    GCC_VERSION=$(g++ --version | head -n1)
    print_success "GCC found: $GCC_VERSION"
elif command -v clang++ &> /dev/null; then
    CLANG_VERSION=$(clang++ --version | head -n1)
    print_success "Clang found: $CLANG_VERSION"
else
    print_error "No suitable C++ compiler found. Please install GCC 9+ or Clang 10+."
    exit 1
fi

# Check PyTorch
if [[ -z "$TORCH_ROOT" ]]; then
    print_warning "TORCH_ROOT not set. Trying to find PyTorch automatically..."
    
    # Try to find PyTorch in common locations
    TORCH_CANDIDATES=(
        "/usr/local/libtorch"
        "/opt/libtorch"
        "$HOME/libtorch"
        "$(pwd)/../libtorch"
    )
    
    for candidate in "${TORCH_CANDIDATES[@]}"; do
        if [[ -d "$candidate" && -f "$candidate/share/cmake/Torch/TorchConfig.cmake" ]]; then
            export TORCH_ROOT="$candidate"
            print_success "Found PyTorch at: $TORCH_ROOT"
            break
        fi
    done
    
    if [[ -z "$TORCH_ROOT" ]]; then
        print_error "PyTorch not found. Please set TORCH_ROOT or install libtorch."
        echo "Download from: https://pytorch.org/get-started/locally/"
        exit 1
    fi
else
    print_success "Using PyTorch at: $TORCH_ROOT"
fi

# Check CUDA if enabled
if [[ "$CUDA_ENABLED" == "true" ]]; then
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | cut -d' ' -f6 | cut -d',' -f1)
        print_success "CUDA found: $CUDA_VERSION"
    else
        print_warning "CUDA not found. Building CPU-only version."
        CUDA_ENABLED=false
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
    "-DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX"
    "-DCMAKE_PREFIX_PATH=$TORCH_ROOT"
)

if [[ "$CUDA_ENABLED" == "false" ]]; then
    CMAKE_ARGS+=("-DUSE_CUDA=OFF")
fi

if [[ "$PYTHON_BINDINGS" == "false" ]]; then
    CMAKE_ARGS+=("-DBUILD_PYTHON_BINDINGS=OFF")
fi

if [[ "$VERBOSE" == "true" ]]; then
    CMAKE_ARGS+=("-DCMAKE_VERBOSE_MAKEFILE=ON")
fi

# Run CMake
if cmake "${CMAKE_ARGS[@]}" ..; then
    print_success "CMake configuration completed"
else
    print_error "CMake configuration failed"
    exit 1
fi

# Build
print_status "Building FoldEngine with $NUM_JOBS parallel jobs..."

if [[ "$VERBOSE" == "true" ]]; then
    MAKE_ARGS=("VERBOSE=1")
else
    MAKE_ARGS=()
fi

if make -j"$NUM_JOBS" "${MAKE_ARGS[@]}"; then
    print_success "Build completed successfully"
else
    print_error "Build failed"
    exit 1
fi

# Run tests if requested
if [[ "$RUN_TESTS" == "true" ]]; then
    print_status "Running tests..."
    
    if make test; then
        print_success "All tests passed"
    else
        print_warning "Some tests failed"
    fi
fi

# Install if requested
if [[ "$INSTALL" == "true" ]]; then
    print_status "Installing FoldEngine..."
    
    if [[ "$EUID" -ne 0 && "$INSTALL_PREFIX" == "/usr/local" ]]; then
        print_warning "Installing to system directory. You may need sudo privileges."
        if sudo make install; then
            print_success "Installation completed"
        else
            print_error "Installation failed"
            exit 1
        fi
    else
        if make install; then
            print_success "Installation completed"
        else
            print_error "Installation failed"
            exit 1
        fi
    fi
fi

# Print summary
print_success "FoldEngine build completed!"
echo ""
echo "Build artifacts:"
echo "  Executable: $BUILD_DIR/fold_engine"
echo "  Library: $BUILD_DIR/libfold_engine_lib.a"

if [[ "$PYTHON_BINDINGS" == "true" ]]; then
    echo "  Python module: $BUILD_DIR/fold_engine_py*.so"
fi

echo ""
echo "Next steps:"
echo "  1. Test the executable: ./$BUILD_DIR/fold_engine info"
echo "  2. Run a folding example: ./$BUILD_DIR/fold_engine fold -s 'MKWVTFISLLFLFSSAYS' -o test.pdb"

if [[ "$INSTALL" == "true" ]]; then
    echo "  3. Use system-wide: fold_engine --help"
fi

echo ""
print_success "Happy folding! 🧬"
