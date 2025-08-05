cmake_minimum_required(VERSION 3.18)
project(OdinFoldWASM VERSION 1.0.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Emscripten-specific settings
if(EMSCRIPTEN)
    message(STATUS "Building for WebAssembly with Emscripten")
    
    # Emscripten compiler flags
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s WASM=1")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s ALLOW_MEMORY_GROWTH=1")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s INITIAL_MEMORY=67108864")  # 64MB initial
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s MAXIMUM_MEMORY=536870912") # 512MB max
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s STACK_SIZE=8388608")       # 8MB stack
    
    # Enable SIMD for better performance
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msimd128")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s SIMD=1")
    
    # Threading support (optional)
    # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s USE_PTHREADS=1")
    # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s PTHREAD_POOL_SIZE=4")
    
    # Emscripten bindings
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --bind")
    
    # Export functions
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s EXPORTED_FUNCTIONS='[\"_malloc\",\"_free\"]'")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s EXPORTED_RUNTIME_METHODS='[\"ccall\",\"cwrap\",\"addFunction\",\"removeFunction\"]'")
    
    # Optimize for size and speed
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s AGGRESSIVE_VARIABLE_ELIMINATION=1")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s ELIMINATE_DUPLICATE_FUNCTIONS=1")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s SINGLE_FILE=0")  # Separate .wasm file
    
    # Debug options (disable for production)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s ASSERTIONS=1")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s SAFE_HEAP=1")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s ASSERTIONS=0")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --closure 1")
    endif()
    
else()
    message(FATAL_ERROR "This project requires Emscripten. Use: emcmake cmake ..")
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Source files
set(SOURCES
    src/odinfold_wasm.cpp
)

# Create the WASM library
add_executable(odinfold ${SOURCES})

# Set output names
set_target_properties(odinfold PROPERTIES
    OUTPUT_NAME "odinfold"
    SUFFIX ".js"
)

# Custom post-build steps
add_custom_command(TARGET odinfold POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "Optimizing WASM output..."
    
    # Copy JavaScript wrapper
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/src/odinfold-wasm.js
        ${CMAKE_CURRENT_BINARY_DIR}/odinfold-wasm.js
    
    # Create web demo files
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/web/index.html
        ${CMAKE_CURRENT_BINARY_DIR}/index.html
    
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/web/demo.js
        ${CMAKE_CURRENT_BINARY_DIR}/demo.js
    
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/web/style.css
        ${CMAKE_CURRENT_BINARY_DIR}/style.css
    
    COMMENT "Copying web assets"
)

# Optional: WASM optimization with wasm-opt (if available)
find_program(WASM_OPT wasm-opt)
if(WASM_OPT AND NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_custom_command(TARGET odinfold POST_BUILD
        COMMAND ${WASM_OPT} -O3 --enable-simd odinfold.wasm -o odinfold_optimized.wasm
        COMMAND ${CMAKE_COMMAND} -E rename odinfold_optimized.wasm odinfold.wasm
        COMMENT "Optimizing WASM with wasm-opt"
    )
endif()

# Create package structure
add_custom_target(package
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/dist
    COMMAND ${CMAKE_COMMAND} -E copy odinfold.js ${CMAKE_CURRENT_BINARY_DIR}/dist/
    COMMAND ${CMAKE_COMMAND} -E copy odinfold.wasm ${CMAKE_CURRENT_BINARY_DIR}/dist/
    COMMAND ${CMAKE_COMMAND} -E copy odinfold-wasm.js ${CMAKE_CURRENT_BINARY_DIR}/dist/
    COMMAND ${CMAKE_COMMAND} -E copy index.html ${CMAKE_CURRENT_BINARY_DIR}/dist/
    COMMAND ${CMAKE_COMMAND} -E copy demo.js ${CMAKE_CURRENT_BINARY_DIR}/dist/
    COMMAND ${CMAKE_COMMAND} -E copy style.css ${CMAKE_CURRENT_BINARY_DIR}/dist/
    
    # Create manifest
    COMMAND ${CMAKE_COMMAND} -E echo "Creating package manifest..."
    COMMAND ${CMAKE_COMMAND} -E echo "{" > ${CMAKE_CURRENT_BINARY_DIR}/dist/manifest.json
    COMMAND ${CMAKE_COMMAND} -E echo "  \"name\": \"OdinFold++ WASM\"," >> ${CMAKE_CURRENT_BINARY_DIR}/dist/manifest.json
    COMMAND ${CMAKE_COMMAND} -E echo "  \"version\": \"${PROJECT_VERSION}\"," >> ${CMAKE_CURRENT_BINARY_DIR}/dist/manifest.json
    COMMAND ${CMAKE_COMMAND} -E echo "  \"files\": [" >> ${CMAKE_CURRENT_BINARY_DIR}/dist/manifest.json
    COMMAND ${CMAKE_COMMAND} -E echo "    \"odinfold.js\"," >> ${CMAKE_CURRENT_BINARY_DIR}/dist/manifest.json
    COMMAND ${CMAKE_COMMAND} -E echo "    \"odinfold.wasm\"," >> ${CMAKE_CURRENT_BINARY_DIR}/dist/manifest.json
    COMMAND ${CMAKE_COMMAND} -E echo "    \"odinfold-wasm.js\"" >> ${CMAKE_CURRENT_BINARY_DIR}/dist/manifest.json
    COMMAND ${CMAKE_COMMAND} -E echo "  ]" >> ${CMAKE_CURRENT_BINARY_DIR}/dist/manifest.json
    COMMAND ${CMAKE_COMMAND} -E echo "}" >> ${CMAKE_CURRENT_BINARY_DIR}/dist/manifest.json
    
    DEPENDS odinfold
    COMMENT "Creating distribution package"
)

# Install targets
install(FILES 
    ${CMAKE_CURRENT_BINARY_DIR}/odinfold.js
    ${CMAKE_CURRENT_BINARY_DIR}/odinfold.wasm
    ${CMAKE_CURRENT_BINARY_DIR}/odinfold-wasm.js
    DESTINATION lib/odinfold-wasm
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/index.html
    ${CMAKE_CURRENT_BINARY_DIR}/demo.js
    ${CMAKE_CURRENT_BINARY_DIR}/style.css
    DESTINATION share/odinfold-wasm/demo
)

# Testing
enable_testing()

# Add a simple test to verify WASM module loads
add_test(NAME wasm_module_test
    COMMAND node -e "
        const fs = require('fs');
        if (!fs.existsSync('odinfold.js')) {
            console.error('WASM JavaScript file not found');
            process.exit(1);
        }
        if (!fs.existsSync('odinfold.wasm')) {
            console.error('WASM binary file not found');
            process.exit(1);
        }
        console.log('WASM files found successfully');
    "
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# Size check test
add_test(NAME wasm_size_test
    COMMAND node -e "
        const fs = require('fs');
        const wasmSize = fs.statSync('odinfold.wasm').size;
        const maxSize = 50 * 1024 * 1024; // 50MB
        if (wasmSize > maxSize) {
            console.error(\`WASM file too large: \${wasmSize} bytes > \${maxSize} bytes\`);
            process.exit(1);
        }
        console.log(\`WASM size OK: \${(wasmSize / 1024 / 1024).toFixed(1)}MB\`);
    "
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# Print build information
message(STATUS "OdinFold WASM Build Configuration:")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Emscripten: ${EMSCRIPTEN}")
if(WASM_OPT)
    message(STATUS "  wasm-opt: Found (${WASM_OPT})")
else()
    message(STATUS "  wasm-opt: Not found (optional optimization disabled)")
endif()

# Build summary
add_custom_target(info
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "OdinFold++ WASM Build"
    COMMAND ${CMAKE_COMMAND} -E echo "===================="
    COMMAND ${CMAKE_COMMAND} -E echo "Build Type: ${CMAKE_BUILD_TYPE}"
    COMMAND ${CMAKE_COMMAND} -E echo "Output: odinfold.js + odinfold.wasm"
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "Build commands:"
    COMMAND ${CMAKE_COMMAND} -E echo "  make          - Build WASM module"
    COMMAND ${CMAKE_COMMAND} -E echo "  make package  - Create distribution package"
    COMMAND ${CMAKE_COMMAND} -E echo "  make test     - Run tests"
    COMMAND ${CMAKE_COMMAND} -E echo "  make install  - Install files"
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "Usage:"
    COMMAND ${CMAKE_COMMAND} -E echo "  python -m http.server 8000"
    COMMAND ${CMAKE_COMMAND} -E echo "  Open http://localhost:8000"
    COMMAND ${CMAKE_COMMAND} -E echo ""
)
