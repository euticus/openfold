/**
 * @file ligand_encoder.h
 * @brief Ligand encoding module for FoldEngine
 */

#pragma once

#include <torch/torch.h>
#include <string>
#include <vector>
#include <memory>

namespace fold_engine {

/**
 * @brief Ligand encoder for processing molecular graphs
 */
class LigandEncoder {
public:
    /**
     * @brief Constructor
     * @param d_model Model dimension
     */
    explicit LigandEncoder(int d_model);
    
    /**
     * @brief Destructor
     */
    ~LigandEncoder();
    
    /**
     * @brief Encode single SMILES string
     * @param smiles SMILES string
     * @param device Target device
     * @return Ligand embedding [d_model]
     */
    torch::Tensor encode_smiles(const std::string& smiles, 
                               const torch::Device& device);
    
    /**
     * @brief Encode batch of SMILES strings
     * @param smiles_list Vector of SMILES strings
     * @param device Target device
     * @return Batch of ligand embeddings [batch_size, max_atoms, d_model]
     */
    torch::Tensor encode_batch(const std::vector<std::string>& smiles_list,
                              const torch::Device& device);
    
    /**
     * @brief Set device for computation
     * @param device Target device
     */
    void to_device(const torch::Device& device);

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};

/**
 * @brief Molecular graph utilities
 */
namespace molecular_utils {
    
    /**
     * @brief Parse SMILES string to molecular graph
     * @param smiles SMILES string
     * @return Molecular graph data
     */
    struct MolecularGraph {
        torch::Tensor atom_types;
        torch::Tensor bond_indices;
        torch::Tensor bond_types;
        int num_atoms;
        int num_bonds;
    };
    
    MolecularGraph parse_smiles(const std::string& smiles);
    
    /**
     * @brief Convert molecular graph to tensor representation
     * @param graph Molecular graph
     * @param d_model Model dimension
     * @param device Target device
     * @return Tensor representation [num_atoms, d_model]
     */
    torch::Tensor graph_to_tensor(const MolecularGraph& graph, 
                                  int d_model, 
                                  const torch::Device& device);
}

} // namespace fold_engine
