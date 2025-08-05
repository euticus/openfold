cmake_minimum_required(VERSION 3.18)
project(FoldEngine VERSION 1.0.0 LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Torch REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDA REQUIRED)

# Set CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 75 80 86)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${TORCH_INCLUDE_DIRS})

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -fopenmp")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")

# Source files
set(SOURCES
    src/fold_engine.cpp
    src/tensor_ops.cpp
    src/attention.cpp
    src/structure_module.cpp
    src/ligand_encoder.cpp
    src/mutation_predictor.cpp
    src/utils.cpp
    src/cli.cpp
)

# CUDA source files
set(CUDA_SOURCES
    src/cuda/attention_kernels.cu
    src/cuda/triangle_kernels.cu
    src/cuda/structure_kernels.cu
)

# Create the main library
add_library(fold_engine_lib STATIC ${SOURCES} ${CUDA_SOURCES})

# Link libraries
target_link_libraries(fold_engine_lib ${TORCH_LIBRARIES})
target_link_libraries(fold_engine_lib OpenMP::OpenMP_CXX)
target_link_libraries(fold_engine_lib ${CUDA_LIBRARIES})

# Create the CLI executable
add_executable(fold_engine src/main.cpp)
target_link_libraries(fold_engine fold_engine_lib)

# Create Python bindings
find_package(pybind11 REQUIRED)
pybind11_add_module(fold_engine_py src/python_bindings.cpp)
target_link_libraries(fold_engine_py PRIVATE fold_engine_lib)

# Installation
install(TARGETS fold_engine DESTINATION bin)
install(TARGETS fold_engine_lib DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)

# Testing
enable_testing()
add_subdirectory(tests)

# Documentation
find_package(Doxygen)
if(DOXYGEN_FOUND)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
    add_custom_target(doc
        ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen" VERBATIM
    )
endif()

# Package configuration
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/FoldEngineConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/FoldEngineConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/FoldEngineConfig.cmake"
    INSTALL_DESTINATION lib/cmake/FoldEngine
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/FoldEngineConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/FoldEngineConfigVersion.cmake"
    DESTINATION lib/cmake/FoldEngine
)
