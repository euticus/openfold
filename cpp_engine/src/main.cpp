/**
 * @file main.cpp
 * @brief Main entry point for FoldEngine CLI
 */

#include "fold_engine.h"
#include <iostream>
#include <exception>

// Forward declaration from cli.cpp
namespace fold_engine {
    int run_cli(int argc, char* argv[]);
}

int main(int argc, char* argv[]) {
    try {
        // Set up signal handling for graceful shutdown
        std::cout << "FoldEngine v1.0.0 - High-Performance Protein Structure Prediction" << std::endl;
        std::cout << "Copyright (c) 2024 OdinFold Team" << std::endl;
        std::cout << std::endl;
        
        // Run CLI
        return fold_engine::run_cli(argc, argv);
        
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown fatal error occurred" << std::endl;
        return 1;
    }
}
