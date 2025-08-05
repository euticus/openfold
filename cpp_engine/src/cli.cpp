/**
 * @file cli.cpp
 * @brief Command-line interface for FoldEngine
 */

#include "fold_engine.h"
#include "utils.h"

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <getopt.h>
#include <filesystem>

namespace fold_engine {

/**
 * @brief CLI configuration
 */
struct CLIConfig {
    std::string command;
    std::string input_file;
    std::string output_file;
    std::string sequence;
    std::vector<std::string> ligands;
    std::string mutations;
    std::string model_path = "models/odinfold.pt";
    std::string device = "auto";
    bool verbose = false;
    bool benchmark = false;
    bool confidence = true;
    bool plddt = true;
    int batch_size = 1;
};

/**
 * @brief Print usage information
 */
void print_usage(const char* program_name) {
    std::cout << "FoldEngine - High-Performance Protein Structure Prediction\n\n";
    std::cout << "Usage: " << program_name << " <command> [options]\n\n";
    std::cout << "Commands:\n";
    std::cout << "  fold      Fold protein structure from sequence\n";
    std::cout << "  mutate    Predict mutation effects\n";
    std::cout << "  batch     Process multiple sequences\n";
    std::cout << "  benchmark Run performance benchmark\n";
    std::cout << "  info      Show model information\n\n";
    std::cout << "Options:\n";
    std::cout << "  -i, --input FILE      Input FASTA file or sequence\n";
    std::cout << "  -o, --output FILE     Output PDB file\n";
    std::cout << "  -s, --sequence SEQ    Protein sequence (alternative to -i)\n";
    std::cout << "  -l, --ligand SMILES   Ligand SMILES string (can be repeated)\n";
    std::cout << "  -m, --mutations MUT   Mutations in format A123V,B456C\n";
    std::cout << "  --model PATH          Path to model file\n";
    std::cout << "  --device DEVICE       Device (cpu, cuda, cuda:0, auto)\n";
    std::cout << "  --batch-size N        Batch size for processing\n";
    std::cout << "  --no-confidence       Disable confidence prediction\n";
    std::cout << "  --no-plddt           Disable pLDDT calculation\n";
    std::cout << "  -v, --verbose         Verbose output\n";
    std::cout << "  -b, --benchmark       Enable benchmarking\n";
    std::cout << "  -h, --help           Show this help\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << program_name << " fold -s MKWVTFISLLFLFSSAYS -o output.pdb\n";
    std::cout << "  " << program_name << " fold -i protein.fasta -o structure.pdb --device cuda\n";
    std::cout << "  " << program_name << " mutate -s MKWVTFISLLFLFSSAYS -m A1V,L5P\n";
    std::cout << "  " << program_name << " batch -i sequences.fasta --batch-size 4\n";
}

/**
 * @brief Parse command line arguments
 */
CLIConfig parse_arguments(int argc, char* argv[]) {
    CLIConfig config;
    
    if (argc < 2) {
        print_usage(argv[0]);
        exit(1);
    }
    
    config.command = argv[1];
    
    static struct option long_options[] = {
        {"input", required_argument, 0, 'i'},
        {"output", required_argument, 0, 'o'},
        {"sequence", required_argument, 0, 's'},
        {"ligand", required_argument, 0, 'l'},
        {"mutations", required_argument, 0, 'm'},
        {"model", required_argument, 0, 1001},
        {"device", required_argument, 0, 1002},
        {"batch-size", required_argument, 0, 1003},
        {"no-confidence", no_argument, 0, 1004},
        {"no-plddt", no_argument, 0, 1005},
        {"verbose", no_argument, 0, 'v'},
        {"benchmark", no_argument, 0, 'b'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };
    
    int option_index = 0;
    int c;
    
    while ((c = getopt_long(argc - 1, argv + 1, "i:o:s:l:m:vbh", long_options, &option_index)) != -1) {
        switch (c) {
            case 'i':
                config.input_file = optarg;
                break;
            case 'o':
                config.output_file = optarg;
                break;
            case 's':
                config.sequence = optarg;
                break;
            case 'l':
                config.ligands.push_back(optarg);
                break;
            case 'm':
                config.mutations = optarg;
                break;
            case 'v':
                config.verbose = true;
                break;
            case 'b':
                config.benchmark = true;
                break;
            case 'h':
                print_usage(argv[0]);
                exit(0);
            case 1001:
                config.model_path = optarg;
                break;
            case 1002:
                config.device = optarg;
                break;
            case 1003:
                config.batch_size = std::stoi(optarg);
                break;
            case 1004:
                config.confidence = false;
                break;
            case 1005:
                config.plddt = false;
                break;
            default:
                print_usage(argv[0]);
                exit(1);
        }
    }
    
    return config;
}

/**
 * @brief Parse mutations string
 */
std::vector<std::pair<int, char>> parse_mutations(const std::string& mutations_str) {
    std::vector<std::pair<int, char>> mutations;
    
    std::stringstream ss(mutations_str);
    std::string mutation;
    
    while (std::getline(ss, mutation, ',')) {
        if (mutation.length() >= 3) {
            char from_aa = mutation[0];
            char to_aa = mutation.back();
            std::string pos_str = mutation.substr(1, mutation.length() - 2);
            int position = std::stoi(pos_str) - 1; // Convert to 0-based
            
            mutations.emplace_back(position, to_aa);
        }
    }
    
    return mutations;
}

/**
 * @brief Execute fold command
 */
int execute_fold(const CLIConfig& config) {
    try {
        // Setup FoldEngine configuration
        FoldConfig fold_config;
        fold_config.model_path = config.model_path;
        fold_config.output_confidence = config.confidence;
        fold_config.output_plddt = config.plddt;
        
        // Create and initialize engine
        FoldEngine engine(fold_config);
        
        if (config.device != "auto") {
            engine.set_device(config.device);
        }
        
        if (!engine.initialize()) {
            std::cerr << "Failed to initialize FoldEngine" << std::endl;
            return 1;
        }
        
        if (config.verbose) {
            std::cout << engine.get_model_info() << std::endl;
        }
        
        // Get input sequence
        std::string sequence;
        if (!config.sequence.empty()) {
            sequence = config.sequence;
        } else if (!config.input_file.empty()) {
            sequence = utils::load_fasta(config.input_file);
        } else {
            std::cerr << "No input sequence provided" << std::endl;
            return 1;
        }
        
        if (config.verbose) {
            std::cout << "Folding sequence of length " << sequence.length() << std::endl;
        }
        
        // Setup input
        ProteinInput input(sequence);
        input.ligand_smiles = config.ligands;
        
        // Run folding
        auto result = engine.fold_protein(input);
        
        // Output results
        if (!config.output_file.empty()) {
            utils::save_pdb(result, config.output_file, config.confidence);
            std::cout << "Structure saved to: " << config.output_file << std::endl;
        }
        
        // Print summary
        std::cout << "Folding completed successfully" << std::endl;
        std::cout << "Sequence length: " << result.sequence_length << std::endl;
        std::cout << "Average confidence: " << result.confidence.mean().item<float>() << std::endl;
        std::cout << "Predicted TM-score: " << result.tm_score_pred << std::endl;
        std::cout << "Inference time: " << result.inference_time_ms << " ms" << std::endl;
        
        if (config.benchmark) {
            engine.get_metrics().print_summary();
        }
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}

/**
 * @brief Execute mutate command
 */
int execute_mutate(const CLIConfig& config) {
    try {
        // Setup engine
        FoldConfig fold_config;
        fold_config.model_path = config.model_path;
        
        FoldEngine engine(fold_config);
        
        if (config.device != "auto") {
            engine.set_device(config.device);
        }
        
        if (!engine.initialize()) {
            std::cerr << "Failed to initialize FoldEngine" << std::endl;
            return 1;
        }
        
        // Get sequence
        std::string sequence;
        if (!config.sequence.empty()) {
            sequence = config.sequence;
        } else if (!config.input_file.empty()) {
            sequence = utils::load_fasta(config.input_file);
        } else {
            std::cerr << "No input sequence provided" << std::endl;
            return 1;
        }
        
        // Parse mutations
        auto mutations = parse_mutations(config.mutations);
        
        if (config.verbose) {
            std::cout << "Analyzing " << mutations.size() << " mutations" << std::endl;
        }
        
        // Predict mutation effects
        auto effects = engine.predict_mutations(sequence, mutations);
        
        // Output results
        std::cout << "Mutation effects (ΔΔG in kJ/mol):" << std::endl;
        auto effects_accessor = effects.accessor<float, 1>();
        
        for (size_t i = 0; i < mutations.size(); ++i) {
            auto [pos, new_aa] = mutations[i];
            char old_aa = sequence[pos];
            float ddg = effects_accessor[i];
            
            std::cout << old_aa << (pos + 1) << new_aa << ": " << ddg << std::endl;
        }
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}

/**
 * @brief Execute info command
 */
int execute_info(const CLIConfig& config) {
    try {
        FoldConfig fold_config;
        fold_config.model_path = config.model_path;
        
        FoldEngine engine(fold_config);
        
        if (config.device != "auto") {
            engine.set_device(config.device);
        }
        
        if (!engine.initialize()) {
            std::cerr << "Failed to initialize FoldEngine" << std::endl;
            return 1;
        }
        
        std::cout << engine.get_model_info() << std::endl;
        
        // System information
        std::cout << "\nSystem Information:" << std::endl;
        std::cout << "  PyTorch version: " << TORCH_VERSION << std::endl;
        std::cout << "  CUDA available: " << (torch::cuda::is_available() ? "Yes" : "No") << std::endl;
        if (torch::cuda::is_available()) {
            std::cout << "  CUDA devices: " << torch::cuda::device_count() << std::endl;
        }
        std::cout << "  Memory usage: " << utils::get_memory_usage_mb() << " MB" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}

/**
 * @brief Main CLI entry point
 */
int run_cli(int argc, char* argv[]) {
    auto config = parse_arguments(argc, argv);
    
    if (config.command == "fold") {
        return execute_fold(config);
    } else if (config.command == "mutate") {
        return execute_mutate(config);
    } else if (config.command == "info") {
        return execute_info(config);
    } else if (config.command == "batch") {
        std::cerr << "Batch processing not yet implemented" << std::endl;
        return 1;
    } else if (config.command == "benchmark") {
        std::cerr << "Benchmark not yet implemented" << std::endl;
        return 1;
    } else {
        std::cerr << "Unknown command: " << config.command << std::endl;
        print_usage(argv[0]);
        return 1;
    }
}

} // namespace fold_engine
